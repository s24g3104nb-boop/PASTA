from dataclasses import replace
import hashlib
import json

import cv2
import numpy as np
from PIL import Image
import pytest

from pasta_weight_app_v2.experiments.area_proportion import (
    OUTSIDE_EVALUATED_RANGE, PRIMARY, analyze, calibrate, paired_color_deltas,
    patch_colors, predict, yellow_masks,
)
from pasta_weight_app_v2.experiments.compare_area_proportion import compare, load_originals


def _spread(lines=16):
    rgb = np.full((720, 960, 3), 37, np.uint8)
    for y in range(80, 80 + lines*18, 18):
        rgb[y:y+5, 100:850] = [225, 189, 87]
    return rgb


def test_20g_reference_uses_the_median_and_marks_120g_as_unverified():
    evidence = analyze(_spread())
    assert not evidence.capture_reason and evidence.track_px > 0
    f = evidence.features[PRIMARY]
    a = replace(evidence, features=dict(evidence.features, track_area=2*f))
    b = replace(evidence, features=dict(evidence.features, track_area=f/2))
    model = calibrate([a, evidence, b])
    assert model[PRIMARY] == f
    assert predict(model, evidence)['estimated_g'] == 20
    sixfold = replace(evidence, features=dict(evidence.features, track_area=6*f))
    result = predict(model, sixfold)
    assert result['estimated_g'] == pytest.approx(120)
    assert result['status'] == OUTSIDE_EVALUATED_RANGE
    stopped = predict(model, replace(sixfold, capture_reason='見切れ'))
    assert stopped['estimated_g'] is None and stopped['reason'] == '見切れ'
    assert stopped['diagnostic_g'] == pytest.approx(120)


def test_track_area_reduces_scale_dependence_and_rejects_missing_length():
    rgb = _spread()
    small = np.full_like(rgb, 37)
    small[90:630, 120:840] = cv2.resize(rgb, (720, 540), interpolation=cv2.INTER_AREA)
    reference, resized = analyze(rgb), analyze(small)
    assert resized.features['frame_area'] / reference.features['frame_area'] == pytest.approx(.75**2, abs=.06)
    assert resized.features[PRIMARY] / reference.features[PRIMARY] == pytest.approx(1, abs=.12)
    model = calibrate([reference]*3)
    assert predict(model, replace(resized, features=dict(resized.features, track_area=None)))['estimated_g'] is None
    with pytest.raises(ValueError, match='基準写真'):
        calibrate([reference, reference, replace(resized, capture_reason='見切れ')])


def test_patch_colors_pair_against_the_same_photo_and_report_mask_inclusion():
    rgb = np.full((24, 24, 3), 38, np.uint8)
    rgb[1:8, 1:8] = [221, 190, 81]
    rgb[11:17, 11:17] = [205, 149, 24]
    sites = [dict(x=4, y=4, kind='single'), dict(x=14, y=14, kind='crossing'),
             dict(x=20, y=20, kind='tray_background')]
    patches = patch_colors(rgb, yellow_masks(rgb), sites)
    assert patches[0]['rgb'] == [221, 190, 81]
    assert patches[0]['mask_fraction']['HSV+Lab'] == 1
    assert patches[2]['mask_fraction']['HSV+Lab'] == 0
    deltas = paired_color_deltas(patches)
    assert deltas['crossing']['S'] > 0
    assert deltas['crossing']['V'] < 0
    assert 'dense' not in deltas
    with pytest.raises(ValueError, match='座標'):
        patch_colors(rgb, {}, [dict(x=0, y=0, kind='single')])


def _make_batch(root):
    photos = root / 'photos'
    photos.mkdir()
    rows = []
    for group, g in enumerate(range(20, 101, 20)):
        for index, variant in enumerate('ABC'):
            rgb = _spread(16 + group*2)
            # Unique bytes and slight app-independent changes within each group.
            rgb[:4, :4] = 30 + group*3 + index*11
            name = f'photos/fixture_{g}_{variant}.jpeg'
            Image.fromarray(rgb).save(root / name, quality=97)
            rows.append(dict(file=name, sha256=hashlib.sha256((root / name).read_bytes()).hexdigest(),
                             width_px=960, height_px=720, series_id=f'series-{g}',
                             photo_placement_variant=variant, weight_label_g=g,
                             weight_label_source='user_reported_weight_in_chat',
                             physical_independence_between_weight_series='unknown',
                             usable_for_independent_test_accuracy=False))
    manifest = dict(schema_version='pasta_lens_user_photo_manifest_v1', photo_count=15, items=rows)
    (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return manifest


def test_batch_checks_original_bytes_and_higher_labels_only_score_predictions(tmp_path):
    manifest = _make_batch(tmp_path)
    original_content = (tmp_path / 'manifest.json').read_bytes()
    first = compare(tmp_path, tmp_path / 'first')
    assert first['reference_specimens'] == 1 and first['independent_evaluation'] is None
    assert len(first['rows']) == 15
    assert first['groups']['40-100'][PRIMARY]['diagnostic_g']['images'] == 12
    assert all(row['color_corrected_g'] is None for row in first['rows'])
    assert (tmp_path / 'manifest.json').read_bytes() == original_content
    assert (tmp_path / 'first/images/01_hsv_lab.png').is_file()

    for item in manifest['items']:
        if item['weight_label_g'] in (40, 60):
            item['weight_label_g'] = 100 - item['weight_label_g']
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    second = compare(tmp_path, tmp_path / 'second')
    assert second['groups']['40-100'][PRIMARY] != first['groups']['40-100'][PRIMARY]
    assert second['reference_features'] == first['reference_features']
    for row, old in zip(second['rows'], first['rows']):
        assert row['outcomes'][PRIMARY]['diagnostic_g'] == old['outcomes'][PRIMARY]['diagnostic_g']
    path = tmp_path / manifest['items'][5]['file']
    path.write_bytes(path.read_bytes() + b'changed after receipt')
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        list(load_originals(tmp_path))
