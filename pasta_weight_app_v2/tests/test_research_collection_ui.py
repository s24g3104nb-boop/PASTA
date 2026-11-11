import os,subprocess,sys
from pathlib import Path

def test_collection_draft_and_save(tmp_path):
    script='''
from streamlit.testing.v1 import AppTest
from PIL import Image
from io import BytesIO
import sys
sys.path.insert(0,'pasta_weight_app_v2')
from research_collection import CollectionStore
at=AppTest.from_file('pasta_weight_app_v2/app.py',default_timeout=20).run()
at.radio(key='main_nav').set_value('設定').run()
at.selectbox(key='settings_section').set_value('開発・検証ツール').run()
at.selectbox(key='developer_section').set_value('独立試料・撮影データ登録').run()
b=BytesIO();Image.new('RGB',(400,400),'gold').save(b,format='PNG')
d=at.session_state['research_collection_draft'];d['photo_bytes']=b.getvalue();d['photo_name']='synthetic.png'
at.run()
at.button(key='collection_save').click().run()
assert at.error and not CollectionStore().list_records()
next(x for x in at.number_input if x.label=='はかりに表示された実測重量（g）').set_value(40.).run()
next(x for x in at.text_input if x.label=='パスタの製品名').set_value('synthetic').run()
at.radio(key='main_nav').set_value('ホーム').run()
at.radio(key='main_nav').set_value('設定').run()
at.selectbox(key='settings_section').set_value('開発・検証ツール').run()
at.selectbox(key='developer_section').set_value('独立試料・撮影データ登録').run()
assert at.session_state['research_collection_draft']['photo_bytes']==b.getvalue()
at.button(key='collection_save').click().run()
assert not at.exception,at.exception
rows=CollectionStore().list_records()
assert len(rows)==1 and rows[0]['metadata']['measured_weight_g']==40
assert not rows[0]['training_approved']
assert rows[0]['quality']['capture_diagnostic']['whole_pasta_verified'] is False
'''
    result=subprocess.run([sys.executable,'-c',script],cwd=Path(__file__).resolve().parents[2],env=dict(os.environ,PASTA_DATA_DIR=str(tmp_path/'app'),PASTA_RESEARCH_DATA_DIR=str(tmp_path/'research')),capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
