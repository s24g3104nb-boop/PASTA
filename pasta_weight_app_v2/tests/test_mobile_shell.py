"""Navigation and capture boundaries; no claims of device camera accuracy."""
import os
from pathlib import Path
import subprocess
import sys
from streamlit.testing.v1 import AppTest


def test_navigation_theme_and_retake(tmp_path):
    script=r'''
from streamlit.testing.v1 import AppTest
at=AppTest.from_file('pasta_weight_app_v2/app.py').run()
assert not at.exception
assert not any(w.key=='training_upload' for w in at.get('file_uploader'))
at.radio(key='main_nav').set_value('設定').run()
at.selectbox(key='appearance').set_value('ダーク').run()
at.radio(key='main_nav').set_value('履歴').run()
assert at.session_state['theme_preference']=='ダーク'
at.radio(key='main_nav').set_value('ホーム').run()
at.button(key='home_demo').click().run()
assert not at.metric
at.button(key='analyze_photo').click().run()
assert any(m.value=='12.0 g' for m in at.metric)
at.button(key='result_retake').click().run()
assert not at.exception and not at.metric
assert at.session_state['capture_source']=='スマホで撮影'
assert at.button(key='camera_fallback')
at.button(key='camera_fallback').click().run()
assert at.session_state['capture_source']=='写真をアップロード'
at.radio(key='main_nav').set_value('設定').run()
at.selectbox(key='settings_section').set_value('AI学習管理（管理者向け）').run()
assert not at.exception
assert at.button(key='training_start').disabled
'''
    result=subprocess.run([sys.executable,'-c',script],cwd=Path(__file__).resolve().parents[2],env=dict(os.environ,PASTA_DATA_DIR=str(tmp_path/'data')),capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr


def test_mobile_ingredient_add_edit_remove():
    app_dir=str(Path(__file__).resolve().parents[1])
    at=AppTest.from_string(f'''
import sys
sys.path.insert(0,{app_dir!r})
from ingredient_ui import render_ingredients
import streamlit as st
frame=render_ingredients([], 'test_mobile')
st.metric('食材数',len(frame))
''').run()
    assert not at.exception
    at.text_input[0].set_value('トマト')
    for widget,value in zip(at.number_input,[100.,20.,1.,0.,4.]):widget.set_value(value)
    next(b for b in at.button if b.label=='この食材を保存').click().run()
    assert not at.exception and at.metric[0].value=='1'
    at.button(key='test_mobile_edit_0').click().run()
    at.number_input[0].set_value(50.)
    next(b for b in at.button if b.label=='この食材を保存').click().run()
    assert at.session_state['test_mobile_cards'][0]['amount_g']==50
    at.button(key='test_mobile_remove_0').click().run()
    assert at.metric[0].value=='0'


def test_missing_ingredient_nutrition_not_silently_zero():
    app_dir=str(Path(__file__).resolve().parents[1])
    at=AppTest.from_string(f'''
import sys
sys.path.insert(0,{app_dir!r})
from ingredient_ui import render_ingredients
render_ingredients([], 'missing')
''').run()
    at.text_input[0].set_value('unknown')
    at.button[0].click().run()
    assert at.error and not at.session_state['missing_cards']
