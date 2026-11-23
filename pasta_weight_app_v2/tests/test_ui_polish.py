"""UI routes use temporary data and keep research behind settings."""
import os,sys,subprocess
from pathlib import Path

def test_upload_shortcut_and_shared_sheet_guide(tmp_path):
    script=r'''
from streamlit.testing.v1 import AppTest
at=AppTest.from_string("import sys; from pathlib import Path; sys.path.insert(0, str(Path.cwd()/'pasta_weight_app_v2')); from app_runtime import render_application; render_application(admin=True)", default_timeout=20).run()
assert not at.exception
assert not any(b.key=='sheet_guide_download' for b in at.get('download_button'))
at.button(key='home_camera').click().run()
assert at.session_state['capture_source']=='写真をアップロード'
assert at.get('file_uploader')
at.radio(key='main_nav').set_value('設定').run()
assert at.get('download_button')
at.selectbox(key='settings_section').set_value('開発・検証ツール').run()
at.selectbox(key='developer_section').set_value('独立試料・撮影データ登録').run()
assert not at.exception
assert at.get('download_button')
assert at.button(key='collection_save').disabled
'''
    result=subprocess.run([sys.executable,'-c',script],cwd=Path(__file__).resolve().parents[2],env=dict(os.environ,PASTA_DATA_DIR=str(tmp_path/'app'),PASTA_RESEARCH_DATA_DIR=str(tmp_path/'research')),capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr


def test_public_home_opens_either_diary_view_without_discarding_draft(tmp_path):
    script = r'''
from streamlit.testing.v1 import AppTest

at = AppTest.from_file('pasta_weight_app_v2/app.py', default_timeout=20).run()
assert not at.exception
assert at.button(key='home_camera').label == '乾麺を測定'
assert at.button(key='home_meal').label == '完成料理を記録'
assert at.button(key='home_calendar').label == 'カレンダー'
assert at.button(key='home_history').label == '食事日記'
assert not any(button.key == 'home_demo' for button in at.button)

at.button(key='home_meal').click().run()
name = next(widget for widget in at.text_input if widget.label == '料理名')
name.set_value('入力途中のパスタ').run()
at.radio(key='main_nav').set_value('ホーム').run()
at.button(key='home_calendar').click().run()
assert at.radio(key='main_nav').value == '記録'
assert at.selectbox(key='record_kind').value == '食事日記'
assert at.radio(key='diary_view').value == 'カレンダー'

at.radio(key='main_nav').set_value('ホーム').run()
at.button(key='home_history').click().run()
assert at.radio(key='diary_view').value == '写真付き一覧'
at.radio(key='main_nav').set_value('ホーム').run()
at.button(key='home_meal').click().run()
assert next(widget for widget in at.text_input if widget.label == '料理名').value == '入力途中のパスタ'
assert not at.exception
'''
    result = subprocess.run(
        [sys.executable, '-c', script],
        cwd=Path(__file__).resolve().parents[2],
        env=dict(os.environ, PASTA_DATA_DIR=str(tmp_path / 'app')),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
