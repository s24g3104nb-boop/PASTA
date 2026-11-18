"""Isolated app flows for live product rows and immutable saved receipts."""
from pathlib import Path
import os
import subprocess
import sys


def test_registration_recalibration_and_saved_receipt(tmp_path):
    script = r'''
import sys
from io import BytesIO
sys.path.insert(0,'pasta_weight_app_v2')
from streamlit.testing.v1 import AppTest
from storage import save_product,list_measurements
from PIL import Image
import numpy as np
at=AppTest.from_file('pasta_weight_app_v2/app.py',default_timeout=25).run()
p=save_product('最初の製品',.68,350,12,2,70)
at.button(key='home_upload').click().run()
rgb=np.full((480,640,3),42,np.uint8)
for i in range(6):rgb[90+i*40:94+i*40,100:530]=(220,180,40)
b=BytesIO();Image.fromarray(rgb).save(b,'PNG')
at.session_state['capture_photo']=b
at.run()
assert at.selectbox(key='which_product').value == p['id']
at.button(key='analyze_photo').click().run()
assert any(m.value=='4.1 g' for m in at.metric),at.exception
save_product('最初の製品',.88,350,12,2,70,product_id=p['id'])
at.run()
assert any(m.value=='5.3 g' for m in at.metric),at.exception
at.button(key='measurement_save').click().run()
assert len(list_measurements())==1
save_product('最初の製品',.99,350,12,2,70,product_id=p['id'])
at.run()
assert any(m.value=='5.3 g' for m in at.metric)
assert at.button(key='measurement_save').disabled
assert list_measurements()[0]['g_per_strand']==.88
at.button(key='edit_capture').click().run()
at.button(key='analyze_photo').click().run()
assert any(m.value=='5.9 g' for m in at.metric)
at.radio(key='main_nav').set_value('設定').run()
at.selectbox(key='settings_section').set_value('製品・校正').run()
at.selectbox(key='product_edit_mode').set_value(0).run()
next(w for w in at.text_input if w.label=='製品名').set_value('新しい製品')
next(w for w in at.number_input if w.label=='1本あたりの重量（g）').set_value(.75)
next(w for w in at.button if w.label=='製品データを保存').click().run()
at.button(key='calibration_return').click().run()
at.button(key='edit_capture').click().run()
assert not at.exception,at.exception
# Returning to an explicit existing measurement preserves its product selection.
assert at.selectbox(key='which_product').value == p['id']
'''
    result=subprocess.run([sys.executable,'-c',script],cwd=Path(__file__).parents[2],
        env=dict(os.environ,PASTA_DATA_DIR=str(tmp_path/'app'),PASTA_RESEARCH_DATA_DIR=str(tmp_path/'research')),
        capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
