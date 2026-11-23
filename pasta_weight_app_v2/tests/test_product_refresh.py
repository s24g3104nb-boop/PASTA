"""Isolated app flows for live product rows and immutable saved receipts."""
from pathlib import Path
import os
import subprocess
import sys


def test_public_reads_existing_calibration_and_preserves_saved_receipt(tmp_path):
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
at.button(key='home_camera').click().run()
rgb=np.full((480,640,3),42,np.uint8)
for i in range(6):rgb[90+i*40:94+i*40,100:530]=(220,180,40)
b=BytesIO();Image.fromarray(rgb).save(b,'PNG')
at.session_state['capture_photo']=b
at.run()
assert at.selectbox(key='which_product').value is None
assert not any(m.label=='推定重量' for m in at.metric)
at.selectbox(key='which_product').set_value(p['id']).run()
assert any(m.value=='4.1 g' for m in at.metric),at.exception
at.checkbox(key='actual_known').check().run()
at.number_input(key='actual_weight').set_value(10.).run()
assert any(m.value=='35.0 kcal' for m in at.metric)
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
at.button(key='result_retake').click().run()
at.session_state['capture_photo']=b
at.run()
assert any(m.value=='5.9 g' for m in at.metric)
at.radio(key='main_nav').set_value('設定').run()
assert not any(w.key == 'product_edit_mode' for w in at.selectbox)
at.radio(key='main_nav').set_value('ホーム').run()
at.button(key='home_resume_capture').click().run()
assert not at.exception,at.exception
# Returning to an explicit existing measurement preserves its product selection.
assert at.selectbox(key='which_product').value == p['id']
'''
    result=subprocess.run([sys.executable,'-c',script],cwd=Path(__file__).parents[2],
        env=dict(os.environ,PASTA_DATA_DIR=str(tmp_path/'app'),PASTA_RESEARCH_DATA_DIR=str(tmp_path/'research')),
        capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
