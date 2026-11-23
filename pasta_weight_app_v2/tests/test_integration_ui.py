"""Exercise recovery and navigation with isolated, synthetic data only."""
import os
from pathlib import Path
import subprocess
import sys


def run_app_script(script, tmp_path):
    result = subprocess.run(
        [sys.executable, '-c', script], cwd=Path(__file__).resolve().parents[2],
        env=dict(os.environ, PASTA_DATA_DIR=str(tmp_path / 'app'),
                 PASTA_RESEARCH_DATA_DIR=str(tmp_path / 'research')),
        capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_calibration_return_refreshes_product_and_reuses_detection(tmp_path):
    run_app_script(r'''
import sys
sys.path.insert(0,'pasta_weight_app_v2')
from streamlit.testing.v1 import AppTest
import capture_ui
from storage import save_product
at=AppTest.from_string("import sys; from pathlib import Path; sys.path.insert(0, str(Path.cwd()/'pasta_weight_app_v2')); from app_runtime import render_application; render_application(admin=True)", default_timeout=20).run()
save_product('合成テスト用の製品',0,350,12,2,70)
at.button(key='home_demo').click().run()
at.selectbox(key='which_product').set_value(2).run()
at.button(key='analyze_photo').click().run()
assert at.button(key='recovery_calibration')
def never_recalculate(*args,**kwargs):
    raise AssertionError('unchanged photo was analyzed twice')
capture_ui.detect_spread=never_recalculate
at.button(key='recovery_calibration').click().run()
at.selectbox(key='product_edit_mode').set_value(2).run()
next(w for w in at.number_input if w.label=='1本あたりの重量（g）').set_value(.68)
next(b for b in at.button if b.label=='製品データを保存').click().run()
assert not at.exception,at.exception
assert any('保存しました' in s.value for s in at.success)
assert at.session_state['result_request']['product']['g_per_strand']==.68
at.button(key='calibration_return').click().run()
assert any(m.value=='8.2 g' for m in at.metric)
at.checkbox(key='actual_known').check().run()
at.number_input(key='actual_weight').set_value(10.).run()
assert not at.exception
assert any(m.value=='8.2 g' for m in at.metric)
assert not any('解析できません' in e.value for e in at.error)
at.radio(key='main_nav').set_value('設定').run()
at.radio(key='main_nav').set_value('ホーム').run()
at.button(key='home_resume_capture').click().run()
assert any(m.value=='8.2 g' for m in at.metric)
assert at.number_input(key='actual_weight').value==10.
''', tmp_path)


def test_unreadable_photo_offers_manual_recovery_without_fake_estimate(tmp_path):
    run_app_script(r'''
from io import BytesIO
from streamlit.testing.v1 import AppTest
at=AppTest.from_file('pasta_weight_app_v2/app.py',default_timeout=20).run()
at.button(key='home_camera').click().run()
at.session_state['capture_photo']=BytesIO(b'not an image')
at.run()
assert not at.exception,at.exception
assert any('写真を読み込めませんでした' in e.value for e in at.error)
assert at.button(key='decode_retake')
assert not at.metric
at.button(key='decode_manual').click().run()
assert not at.exception,at.exception
assert at.button(key='manual_weight_save')
assert all(b.key != 'manual_register_product' for b in at.button)
assert not any(m.label=='推定重量' for m in at.metric)
''', tmp_path)


def test_unregistered_upload_automatically_extracts_then_saves_standard_nutrition(tmp_path):
    run_app_script(r'''
import sys
from io import BytesIO
from streamlit.testing.v1 import AppTest
from PIL import Image
import numpy as np
sys.path.insert(0,'pasta_weight_app_v2')
import capture_ui
from storage import list_measurements, list_products
from meal_diary import entries
from cooking import dry_snapshot

at=AppTest.from_file('pasta_weight_app_v2/app.py',default_timeout=20).run()
old_products=list_products()
at.button(key='home_camera').click().run()
image=np.full((480,640,3),42,np.uint8)
for i in range(6):
    image[90+i*40:94+i*40,100:530]=(220,180,40)
photo=BytesIO()
Image.fromarray(image).save(photo,'PNG')
at.session_state['capture_photo']=photo
at.run()
assert not at.exception
request=at.session_state['result_request']
assert request['_detection_result'].count==6
geometry=request['_geometry']
assert geometry['mask_area_px']>0 and geometry['quantile_length_px']>0 and geometry['width_over_length']>0
assert request['product']['id'] is None
assert any('面積から重量を推定できませんでした' in w.value for w in at.warning)
assert not at.error and not any(m.label=='推定重量' for m in at.metric)
assert not any('製品を登録' in b.label or b.key=='analyze_photo' for b in at.button)

def must_not_repeat(*args, **kwargs):
    raise AssertionError('画像を再解析しました')
capture_ui.detect_spread=must_not_repeat
next(w for w in at.number_input if w.label=='はかりで測った実測重量（g）').set_value(100.).run()
assert any(m.value=='371.0 kcal' for m in at.metric)
assert any('一般的な乾麺の概算' in c.value for c in at.caption)
at.button(key='manual_weight_save').click().run()
assert not at.error and at.button(key='manual_weight_save').disabled
row, = list_measurements()
assert row['estimated_g'] is None and row['actual_g']==100 and row['g_per_strand'] is None
assert row['estimator']=='manual_weight' and row['kcal']==371 and row['carb_g']==74.7
assert row['carb_100']==74.67
assert list_products()==old_products
assert '一般的な乾麺の概算' in dry_snapshot(row)['nutrition_source']

at.button(key='manual_weight_to_diary').click().run()
next(w for w in at.text_input if w.label=='料理名').set_value('乾麺100gの記録').run()
at.button(key='meal_save').click().run()
meal, = entries()
assert meal['measurement_id']==row['id'] and meal['nutrition_snapshot']['kcal']==371
assert '一般的な乾麺の概算' in meal['nutrition_source']
at.button(key='meal_detail_back').click().run()
at.radio(key='diary_view').set_value('カレンダー').run()
assert at.button(key='diary_add_selected_day') and not at.exception
''', tmp_path)


def test_no_foreground_and_read_failure_are_distinct_from_missing_generic_model(tmp_path):
    run_app_script(r'''
import sys
from io import BytesIO
from streamlit.testing.v1 import AppTest
from PIL import Image, ImageDraw
sys.path.insert(0,'pasta_weight_app_v2')
import capture_ui
from sqlite3 import OperationalError
at=AppTest.from_file('pasta_weight_app_v2/app.py',default_timeout=20).run()
at.button(key='home_camera').click().run()
image=Image.new('RGB',(680,470),(42,42,42))
for x in range(0,680,40):
    ImageDraw.Draw(image).line((x,0,x,469),fill=(65,65,65))
photo=BytesIO()
image.save(photo,'PNG')
at.session_state['capture_photo']=photo
def read_products():
    raise OperationalError('fixture')
capture_ui.list_products=read_products
at.run()
assert not at.exception and not at.error
assert at.session_state['result_request']['_geometry']['mask_area_px']==0
assert any('面積から重量を推定できませんでした' in w.value for w in at.warning)
assert any('登録済みの製品を読み込めませんでした' in w.value for w in at.warning)
assert not any('汎用重量推定モデル' in i.value for i in at.info)
assert not any(m.label=='推定重量' for m in at.metric)
assert at.button(key='manual_weight_save') and at.button(key='recovery_retake')
''', tmp_path)


def test_invalid_recent_photo_does_not_block_home(tmp_path):
    run_app_script(r'''
import sys
sys.path.insert(0,'pasta_weight_app_v2')
from streamlit.testing.v1 import AppTest
import home_ui
at=AppTest.from_file('pasta_weight_app_v2/app.py').run()
home_ui.entries=lambda: [dict(id='synthetic-test',photo='../invalid.png',source_kind='real',dish_name='写真を開けない記録',eaten_date='2026-11-11',eaten_time=None)]
at.run()
assert not at.exception,at.exception
assert at.button(key='home_camera')
assert at.button(key='recent_synthetic-test')
assert any('写真なしの記録' in m.value for m in at.markdown)
''', tmp_path)


def test_area_estimate_flows_to_measured_nutrition_and_calendar(tmp_path):
    run_app_script(r'''
import sys
from io import BytesIO
from streamlit.testing.v1 import AppTest
from PIL import Image
import numpy as np
sys.path.insert(0, 'pasta_weight_app_v2')
import capture_ui
from cooking import dry_snapshot
from meal_diary import entries
from storage import list_measurements, list_products

at = AppTest.from_file('pasta_weight_app_v2/app.py', default_timeout=20).run()
old_products = list_products()
at.button(key='home_camera').click().run()
image = np.full((480, 640, 3), 42, np.uint8)
for y in range(60, 420, 10):
    image[y:y + 5, 64:576] = (220, 180, 40)
photo = BytesIO()
Image.fromarray(image).save(photo, 'PNG')
at.session_state['capture_photo'] = photo

def must_not_count(*args, **kwargs):
    raise AssertionError('面積から重量を出せる場合は本数検出が不要です')

capture_ui.detect_spread = must_not_count
at.run()
assert not at.exception and not at.error
area = at.session_state['result_request']['_area_estimate']
assert area.weight_g is not None
assert any(m.label == '面積からの推定重量' and m.value == f'約{area.weight_g:.0f} g' for m in at.metric)
assert any('精度は未検証' in c.value for c in at.caption)
assert not any('製品を登録' in b.label or b.key == 'analyze_photo' for b in at.button)
assert any(m.label == '乾麺分のカロリー' and m.value == f'{area.weight_g * 3.71:.1f} kcal'
           for m in at.metric)

at.checkbox(key='actual_known').check().run()
at.number_input(key='actual_weight').set_value(40.).run()
assert any(m.value == '148.4 kcal' for m in at.metric)
assert at.session_state['result_request']['_area_estimate'] == area
at.button(key='measurement_save').click().run()
assert at.button(key='measurement_save').disabled and not at.error
row, = list_measurements()
assert row['estimated_g'] == area.weight_g and row['actual_g'] == 40.
assert row['estimator'] == 'sheetless_area_v2' and row['detected_count'] == 0
assert row['kcal'] == 148.4 and list_products() == old_products
snapshot = dry_snapshot(row)
assert snapshot['weight_source'] == 'actual' and snapshot['dry_weight_g'] == 40.
assert snapshot['nutrition']['kcal'] == 148.4

at.button(key='measurement_to_diary').click().run()
assert any('面積からの推定重量 約' in m.value for m in at.markdown)
at.button(key='meal_open_cooking').click().run()
assert at.selectbox(key='record_kind').value == '料理・栄養'
assert any('実測 40 g' in m.value for m in at.markdown)
servings = f"cooking_servings_{row['id']}_None"
fraction = f"cooking_fraction_{row['id']}"
assert at.number_input(key=servings).value == 1
assert at.number_input(key=fraction).value == 1.0
at.number_input(key=servings).set_value(3).run()
at.number_input(key=fraction).set_value(0.4).run()
at.selectbox(key='cooking_section').set_value('保存した料理・レシピ').run()
at.selectbox(key='cooking_section').set_value('料理を作る').run()
assert at.number_input(key=servings).value == 3
assert at.number_input(key=fraction).value == 0.4
assert not at.exception
at.button(key='back_to_meal_draft').click().run()
assert any('面積からの推定重量 約' in m.value for m in at.markdown)
next(w for w in at.text_input if w.label == '料理名').set_value('面積推定の動作確認').run()
at.button(key='meal_save').click().run()
meal, = entries()
assert meal['measurement_id'] == row['id'] and meal['nutrition_snapshot']['kcal'] == 148.4
at.button(key='meal_detail_back').click().run()
at.radio(key='diary_view').set_value('カレンダー').run()
assert at.button(key='diary_add_selected_day') and not at.exception
assert len(list_measurements()) == 1
''', tmp_path)


def test_public_settings_migrate_old_cooking_route_and_link_to_photo(tmp_path):
    run_app_script(r'''
from streamlit.testing.v1 import AppTest

at = AppTest.from_file('pasta_weight_app_v2/app.py', default_timeout=20).run()
assert not any('きれいに撮るコツ' in m.value for m in at.markdown)
assert not any(b.key == 'home_upload' for b in at.button)
at.session_state['settings_section'] = '製品・校正'
at.radio(key='main_nav').set_value('設定').run()
assert at.selectbox(key='appearance')
assert not any(w.key in {'settings_section', 'product_edit_mode', 'cooking_section'} for w in at.selectbox)
at.selectbox(key='appearance').set_value('ダーク').run()
at.session_state['settings_section'] = '料理の栄養計算'
at.run()
assert not at.exception and at.radio(key='main_nav').value == '記録'
assert at.selectbox(key='record_kind').value == '料理・栄養'
assert not at.warning
at.button(key='cooking_capture').click().run()
assert at.radio(key='main_nav').value == 'ホーム'
assert at.get('file_uploader')
assert any('平らに広げ' in c.value for c in at.caption)
''', tmp_path)
