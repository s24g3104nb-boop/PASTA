import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from pasta_weight_app_v2.experiments.sheetless_weight import _load_matches, compare, features


def single_layer():
    mask = np.zeros((128, 192), bool)
    mask[32:39, 24:168] = True
    mask[49:56, 24:168] = True
    return mask


def test_object_ratios_track_scale_and_rotation_without_a_ruler():
    mask = single_layer()
    image = np.zeros((*mask.shape, 3), np.uint8)
    original = features(image, mask)
    smaller = cv2.resize(mask.astype(np.uint8), None, fx=.5, fy=.5, interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros_like(mask)
    canvas[32:96, 48:144] = smaller > 0
    distant = features(image, canvas)
    rotated = features(np.rot90(image), np.rot90(mask))
    assert distant['frame_geometry'][0] < .35 * original['frame_geometry'][0]
    assert distant['object_geometry'][:2] == pytest.approx(original['object_geometry'][:2], rel=.12)
    assert rotated['object_geometry'][:2] == pytest.approx(original['object_geometry'][:2], rel=.03)


def test_frozen_features_and_reviewed_product_are_explicit(tmp_path):
    profile = tmp_path/'profile.json'
    profile.write_text(json.dumps(dict(product_name='fixture', diameter_mm=1.6, length_mm=250,
        measured_strand_count=50, measured_total_weight_g=34, scale_resolution_g=.1)), encoding='utf-8')
    digest = hashlib.sha256(b'fixture').hexdigest()
    match = tmp_path/'match.json'
    matches = {digest: {'product_name':'fixture', 'product_and_intact_single_layer_reviewed':True}}
    match.write_text(json.dumps(matches), encoding='utf-8')
    product = _load_matches([profile], match, [digest])[digest]
    mask = single_layer()
    image = np.full((*mask.shape,3), 255, np.uint8)
    image[mask] = (248,184,55)
    def encoder(pil):
        pixels = np.asarray(pil)
        return [float(pixels.mean()), float(pixels.std())]

    result = features(image, mask, encoder, product)
    assert result['object_with_product'][-1] == pytest.approx(result['object_geometry'][0] * 250/1.6*.68)
    assert result['object_with_embedding'][-2:] == result['foreground_embedding']
    assert result['full_image_embedding'] != result['foreground_embedding']
    assert 'object_with_product' not in features(image, mask, encoder)
    mask[:,0] = True
    assert 'object_with_product' not in features(image, mask, encoder, product)
    assert set(features(image, np.zeros_like(mask), encoder)) == {'full_image_embedding'}

    matches[digest]['product_and_intact_single_layer_reviewed'] = False
    match.write_text(json.dumps(matches), encoding='utf-8')
    with pytest.raises(ValueError, match='requires a reviewed'):
        _load_matches([profile], match, [digest])
    with pytest.raises(ValueError, match='verified original'):
        _load_matches([profile], match, [])


def test_fit_uses_only_weighed_originals_and_retains_derivative_failures():
    def entry(name, label, value, transformed):
        return dict(original_sha256=name, measured_g=label, features={'geom':[value]},
                    views={'resize50': {'geom':[transformed]} if transformed is not None else {}})
    originals = [entry('first',20,1,1), entry('second',80,3,None),
                 entry('unweighed',None,1000,0.1)]
    report = compare(originals, ['geom', 'absent'])
    results = report['available']
    assert report['common_originals'] == 2
    assert results['geom']['fit_images'] == 2
    assert results['absent']['training_mae_g'] is None
    rescaled = results['geom']['derivatives']['resize50']
    assert rescaled['pairs'] == 2
    assert rescaled['changed_to_rejected'] == 1
    assert rescaled['max_absolute_delta_g'] == 0


def test_public_capture_and_guide_exclude_sheet_with_model_setting(tmp_path):
    script = r'''
from io import BytesIO
import sys
from streamlit.testing.v1 import AppTest
from PIL import Image
at = AppTest.from_file('pasta_weight_app_v2/app.py',default_timeout=20).run()
assert not at.exception
from storage import save_product
save_product('試料の製品', .68, 350, 12, 2, 70)
at.button(key='home_camera').click().run()
assert any(c.value=='乾麺をできるだけ平らに広げ、重なりを少なくして、真上から撮影してください。' for c in at.caption)
photo = BytesIO()
Image.new('RGB',(500,450),(196,164,55)).save(photo, format='PNG')
at.session_state['capture_photo'] = photo
at.run()
assert not at.exception
assert any(metric.label == '面積からの推定重量' for metric in at.metric)
assert all(b.key!='analyze_photo' for b in at.button)
assert not any('A4' in e.label for e in at.expander)
assert not at.get('download_button')
assert 'area_ui' not in sys.modules
at.radio(key='main_nav').set_value('設定').run()
assert not at.exception
assert not any('A4' in e.label for e in at.expander)
assert not at.get('download_button')
assert 'capture_guide_ui' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', script], cwd=Path(__file__).resolve().parents[2],
                            env=dict(os.environ, PASTA_DATA_DIR=str(tmp_path/'app'),
                                     PASTA_MOBILESAM_CHECKPOINT=str(tmp_path/'optional.pt')),
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout+result.stderr
