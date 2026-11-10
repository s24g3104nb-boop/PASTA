"""UI routes use temporary data and keep research behind settings."""
import os,sys,subprocess
from pathlib import Path

def test_upload_shortcut_and_shared_sheet_guide(tmp_path):
    script=r'''
from streamlit.testing.v1 import AppTest
at=AppTest.from_file('pasta_weight_app_v2/app.py',default_timeout=20).run()
assert not at.exception
assert not any(b.key=='sheet_guide_download' for b in at.get('download_button'))
at.button(key='home_upload').click().run()
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
