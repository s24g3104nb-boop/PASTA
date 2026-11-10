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
at=AppTest.from_file('pasta_weight_app_v2/app.py',default_timeout=20).run()
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
at.button(key='home_upload').click().run()
at.session_state['capture_photo']=BytesIO(b'not an image')
at.run()
assert not at.exception,at.exception
assert any('写真を読み込めませんでした' in e.value for e in at.error)
assert at.button(key='decode_retake')
assert not at.metric
at.button(key='decode_manual').click().run()
assert not at.exception,at.exception
assert at.button(key='manual_weight_save')
assert not any(m.label=='推定重量' for m in at.metric)
''', tmp_path)


def test_invalid_recent_photo_does_not_block_home(tmp_path):
    run_app_script(r'''
import sys
sys.path.insert(0,'pasta_weight_app_v2')
from streamlit.testing.v1 import AppTest
import home_ui
at=AppTest.from_file('pasta_weight_app_v2/app.py').run()
home_ui.entries=lambda: [dict(id='synthetic-test',photo='../invalid.png',dish_name='写真を開けない記録',eaten_date='2026-11-11',eaten_time=None)]
at.run()
assert not at.exception,at.exception
assert at.button(key='home_camera')
assert at.button(key='recent_synthetic-test')
assert any('写真なしの記録' in m.value for m in at.markdown)
''', tmp_path)
