from contextlib import nullcontext
from datetime import date
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))

from local_dish_vision import suggest_from_scores
from pasta_weight_app_v2.experiments.area_calibration import fit_area_profile, predict_area, product_geometry_grams
from pasta_weight_app_v2.experiments.capture_sheet import render_sheet
from pasta_weight_app_v2.experiments.foundation_backends import Sam31, embedding_image, mask_union
from pasta_weight_app_v2.experiments.foundation_masks import a4_projection, annotate_mask, mask_scores
from cooking import Ingredient
from cooking_storage import save_cooking_session
from meal_diary import save
from standard_recipes import standard_recipe
from storage import init_db, save_measurement


def test_a4_geometry_requires_verified_print_and_uncropped_foreground():
    rgb = render_sheet()
    mask = np.zeros(rgb.shape[:2], bool)
    mask[200:240, 200:600] = True  # 100mm by 10mm on the synthetic 4px/mm sheet
    assert a4_projection(rgb, mask, None)['status'] == 'print_not_verified'
    assert a4_projection(rgb, mask, dict(printed_line_mm=96, flat_sheet_confirmed=True))['features'] is None
    review = dict(printed_line_mm=100, flat_sheet_confirmed=True)
    projected = a4_projection(rgb, mask, review)
    assert projected['status'] == 'sheet_plane_projection'
    assert projected['features'][0] == pytest.approx(1000, rel=.04)
    mask[0, :10] = True
    assert a4_projection(rgb, mask, review)['status'] == 'foreground_at_image_edge'
    rgb[:130, :130] = 255
    assert a4_projection(rgb, mask, review)['status'] == 'markers_invalid'


def test_area_fit_does_not_mix_shared_materials_or_extrapolate():
    backend = {'method': 'mobile_sam', 'version': 'synthetic'}
    rows = [dict(original_sha256=hashlib.sha256(str(i).encode()).hexdigest(), physical_sample_id=f'portion{i}',
                 disjoint_material_confirmed=True, flat_layer_and_mask_confirmed=True, backend=backend,
                 projection_status='sheet_plane_projection', area_mm2=g*100, measured_g=g,
                 split='fit' if i<3 else 'check')
            for i, g in enumerate((20, 50, 100, 30, 80))]
    profile = fit_area_profile('製品', backend, rows)
    assert profile['checked_mae_g'] == pytest.approx(0., abs=.0001)
    assert predict_area(profile,'製品',backend,4300) == pytest.approx(43)
    assert predict_area(profile,'別製品',backend,4300) is None
    assert predict_area(profile,'製品',backend,11000) is None
    assert product_geometry_grams(16000, strand_length_mm=250, strand_diameter_mm=1.6, grams_per_strand=.68) == pytest.approx(27.2)
    rows[-1]['measured_g'] = 130
    assert predict_area(profile,'製品',backend,4300) is None
    rows[-1]['physical_sample_id'] = rows[0]['physical_sample_id']
    with pytest.raises(ValueError, match='separately weighed'):
        fit_area_profile('製品', backend, rows)


def test_annotations_and_crop_exclude_background():
    ref = annotate_mask(dict(schema='pasta-visible-mask-v1', image_shape=[20,20],
                             strokes=[dict(tool='polygon', mode='add', points=[[3,3],[15,3],[15,15],[3,15]]),
                                      dict(tool='line', mode='erase', width_px=2, points=[[9,4],[9,14]])]),
                        (20,20),(20,20))
    image = np.full((20,20,3),255,np.uint8)
    image[ref] = [200,150,90]
    assert not ref[9,9]
    assert mask_scores(ref,ref)['iou'] == 1
    score = mask_scores(~ref,ref)
    assert score['background_in_predicted_fraction'] == score['missed_visible_pasta_fraction'] == 1
    assert np.array_equal(embedding_image(image,ref),embedding_image(np.where(ref[...,None],image,3).astype(np.uint8),ref))
    assert mask_union([], ref.shape).sum() == 0


def test_sam31_uses_a_local_unlabelled_frame_and_closes_session():
    sam = object.__new__(Sam31)
    types = []
    shape = (20,20)
    class Fake:
        def handle_request(self, request):
            types.append(request['type'])
            if request['type'] == 'start_session':
                assert [p.name for p in Path(request['resource_path']).iterdir()] == ['00000.jpg']
                return {'session_id':'fixture'}
            if request['type'] == 'add_prompt':
                return {'frame_index': 0, 'outputs': {'out_binary_masks': np.ones((1,20,20), bool)}}
    sam.torch = type('Torch', (), dict(inference_mode=staticmethod(nullcontext)))
    sam.predictor = Fake()
    assert sam(np.zeros((*shape,3), np.uint8)).all()
    assert types == ['start_session', 'add_prompt', 'close_session']


def test_editable_recipe_and_duplicate_meal_protection(tmp_path):
    db = tmp_path/'isolated.sqlite3'
    init_db(db)
    id = save_measurement(product_name='食品テスト', mode='manual', estimator='manual_weight',
                          detected_count=0, estimated_g=None, actual_g=80, source_kind='real',
                          nutrition=dict(kcal=280, protein_g=9.6, fat_g=1.6, carb_g=56),
                          image_rgb=None, db_path=db)
    recipe = standard_recipe(-3)
    ingredients = [Ingredient(**row) for row in recipe['ingredients']]
    # Eat half of the dry pasta and the actual edited amount of ingredients.
    session = save_cooking_session(measurement_id=id, selected_dish='カルボナーラ', confirmed_same_batch=True,
                                   ingredients=ingredients[:1], servings=2, recipe_template_id=-3,
                                   classifier_status='user_confirmed_vision', db_path=db)
    first = save('first', date(2026, 11, 22), None, 'カルボナーラ', db=db, cooking_session_id=session)
    assert first['nutrition_status'] == 'template_estimate'
    assert first['nutrition_snapshot']['kcal'] == pytest.approx((280+71.5)/2)
    assert '利用者が確認' in first['nutrition_source']
    assert save('first', date(2026, 11, 22), None, 'カルボナーラ', db=db, cooking_session_id=session)['id'] == 'first'
    with pytest.raises(ValueError, match='使用済み'):
        save('second', date(2026, 11, 22), None, '二重', db=db, measurement_id=id)


def test_dish_abstains_on_tie_and_does_not_autofill_ingredients():
    scores = {'トマトパスタ':.30,'ジェノベーゼ':.30,'カルボナーラ':.21,'その他':.20}
    ingredients = {'ベーコン片':(.32,.27)}
    assert suggest_from_scores(scores, ingredients,'緑').name is None
    scores['カルボナーラ'] = .35
    item = suggest_from_scores(scores,ingredients,'緑')
    assert item.name == 'カルボナーラ' and item.visible_options == ('ベーコン片',)
    assert not hasattr(item,'calories') and not hasattr(item,'amount_g')


def test_admin_entry_requires_loopback_password_and_separate_storage(tmp_path):
    script = r'''
import os, sys
from pathlib import Path
from streamlit import config
from streamlit.testing.v1 import AppTest
app = Path("pasta_weight_app_v2")
root = Path(os.environ["PASTA_ADMIN_DATA_DIR"])
normal=AppTest.from_file(app/"app.py").run()
assert not normal.exception
assert not any(b.key=="home_demo" for b in normal.button)
normal.session_state['_app_admin']=True
normal.session_state['settings_section']='開発・検証ツール'
normal.radio(key="main_nav").set_value("設定").run()
assert normal.selectbox(key="appearance")
assert not any(w.key in {"settings_section","product_edit_mode","cooking_section"} for w in normal.selectbox)
assert "developer_ui" not in sys.modules and "training_manager" not in sys.modules
assert "pasta_weight_app_v2.experiments.area_proportion_ui" not in sys.modules
assert "pasta_weight_app_v2.experiments.practical_capture_ui" not in sys.modules
config.set_option("server.address","0.0.0.0")
denied=AppTest.from_file(app/"admin_app.py").run()
assert denied.error and not denied.text_input and not root.exists()
config.set_option("server.address","127.0.0.1")
login=AppTest.from_file(app/"admin_app.py").run()
assert login.text_input and not root.exists()
login.text_input[0].set_value("wrong")
login.button[0].click().run()
assert login.error and not root.exists()
public=os.environ["PASTA_DATA_DIR"]
os.environ["PASTA_ADMIN_DATA_DIR"]=public
same=AppTest.from_file(app/"admin_app.py").run()
assert same.error and not same.text_input
os.environ["PASTA_ADMIN_DATA_DIR"]=str(root)
# Clear modules deliberately in this one test only to emulate a fresh process.
for name in ("app_runtime","storage","admin_access"):
    sys.modules.pop(name,None)
fresh=AppTest.from_file(app/"admin_app.py").run()
fresh.text_input[0].set_value(os.environ["PASTA_ADMIN_PASSWORD"])
fresh.button[0].click().run()
assert not fresh.exception and fresh.button(key="home_demo") and root.exists()
fresh.radio(key="main_nav").set_value("設定").run()
assert any("A4撮影補助シート" in e.label for e in fresh.expander)
assert fresh.get("download_button")
assert fresh.selectbox(key="settings_section").options == ["表示・使い方","製品・校正","開発・検証ツール"]
fresh.selectbox(key="settings_section").set_value("製品・校正").run()
assert fresh.selectbox(key="product_edit_mode")
assert any(w.label == "1本あたりの重量（g）" for w in fresh.number_input)
fresh.selectbox(key="settings_section").set_value("開発・検証ツール").run()
fresh.selectbox(key="developer_section").set_value("20g基準の面積換算").run()
assert not fresh.exception and len(fresh.get('file_uploader')) == 2
next(b for b in fresh.button if b.label == '面積と重量を比較する').click().run()
assert any('基準写真を3枚' in msg.value for msg in fresh.info)
fresh.selectbox(key='developer_section').set_value('概算重量の撮影試験').run()
assert not fresh.exception and len(fresh.get('file_uploader')) == 1
assert any('写真1枚' in msg.value for msg in fresh.info)
from pasta_weight_app_v2.experiments.practical_capture import WidthReference
fresh.session_state['practical_width_profile'] = WidthReference(2000., 2100.)
fresh.run()
assert not fresh.exception and len(fresh.get('file_uploader')) == 2
fresh.radio(key='practical_photo_source').set_value('カメラで撮影する').run()
assert not fresh.exception and len(fresh.get('camera_input')) == 1
'''
    result = subprocess.run([sys.executable,'-c',script], cwd=APP.parent, capture_output=True, text=True,
                            env=dict(os.environ, PASTA_DATA_DIR=str(tmp_path/'public'),
                                     PASTA_ADMIN_DATA_DIR=str(tmp_path/'admin'),
                                     PASTA_ADMIN_PASSWORD='fixture-only-phrase'), timeout=75)
    assert result.returncode == 0, result.stdout + result.stderr


def test_future_group_split_keeps_stops_in_the_denominator_and_rejects_shared_lineage():
    from pasta_weight_app_v2.experiments.foundation_evaluation import evaluate
    hashes = [hashlib.sha256(f'fresh-{i}'.encode()).hexdigest() for i in range(5)]
    groups = ('train', 'train', 'validation', 'test', 'test')
    records = [dict(image_id=f'photo-{i}', original_id=f'original-{i}', original_sha256=digest,
                    sample_id=f'sample-{i}', capture_group_id=f'capture-{i}', physical_family_id=f'family-{i}',
                    parent_material_ids=[f'material-{i}'], lineage_verified=True, weight_label_verified=True,
                    lineage_evidence='verified-for-test-fixture', lineage_confirmed_by='tester', measured_g=20.+20*i)
               for i,digest in enumerate(hashes)]
    plan = dict(records=records, assignments={rec['image_id']:group for rec,group in zip(records,groups)},
                material_parents={rec['parent_material_ids'][0]:[] for rec in records},
                feature_variant='area', target_weights_g=[80,100], frozen_before_test=True)
    rows = [dict(original_sha256=digest, measured_g=rec['measured_g'],
                 features=dict(area=[i*100.]) if i<4 else {}) for i,(digest,rec) in enumerate(zip(hashes,records))]
    result = evaluate(rows,plan,'test')
    assert result['overall']['attempts'] == 2 and result['overall']['accepted'] == 1
    assert result['overall']['rejection_rate'] == .5
    assert result['training_feature_vectors'] == 2 and result['production_approved'] is False
    rows[-1]['physical_family_id'] = 'old-cumulative'
    with pytest.raises(ValueError, match='cumulative'):
        evaluate(rows,plan,'test')
    rows[-1].pop('physical_family_id')
    records[-1]['parent_material_ids'] = records[0]['parent_material_ids']
    with pytest.raises(ValueError, match='crosses splits'):
        evaluate(rows,plan,'test')
