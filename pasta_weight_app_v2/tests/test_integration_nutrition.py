"""Regression tests for manual-to-cooking and absent-nutrition integration."""
import sys
from pathlib import Path
from datetime import date
import sqlite3
from io import BytesIO
from contextlib import contextmanager
import numpy as np
import pytest
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from storage import init_db, save_measurement, list_measurements
from cooking import dry_snapshot
from dish_classifier import DISH_NAMES
from cooking_storage import save_cooking_session, list_cooking_sessions
from nutrition_state import has_nutrition, snapshot_for_nutrition
import meal_diary


@pytest.mark.parametrize('known', [True, False])
def test_prediction_unknown_nutrition_and_manual_cooking(tmp_path, known):
    db = tmp_path/'app.sqlite3';init_db(db)
    product = dict(name='検証製品', kcal_100=350 if known else 0, protein_100=12 if known else 0,
                   fat_100=2 if known else 0, carb_100=70 if known else 0, g_per_strand=0.68)
    assert has_nutrition(product) is known
    nutrition = dict(kcal=350, protein_g=12, fat_g=2, carb_g=70) if known else None
    mid=save_measurement(product_name=product['name'], mode='実測重量の入力',detected_count=0,
          estimated_g=None,actual_g=100,nutrition=nutrition,image_rgb=None,estimator='manual_weight',
          db_path=db,product_snapshot=snapshot_for_nutrition(product),submission_token='manual')
    row=list_measurements(db)[0]
    if known:
        sid=save_cooking_session(measurement_id=mid,selected_dish=DISH_NAMES[0],confirmed_same_batch=True,
                                ingredients=[],servings=1,db_path=db)
        assert list_cooking_sessions(db)[0]['calculation']['portion']['kcal']==350
        saved=meal_diary.save('meal',date.today(),None,'パスタ',db=db,cooking_session_id=sid)
        assert saved['nutrition_snapshot']['kcal']==350
    else:
        with pytest.raises(ValueError,match='栄養未計算'):dry_snapshot(row)
        pid=save_measurement(product_name='栄養未登録',mode='広げた状態',detected_count=10,
              estimated_g=50,actual_g=None,nutrition=None,image_rgb=np.zeros((24,24,3),np.uint8),
              db_path=db,product_snapshot=snapshot_for_nutrition(product))
        saved=meal_diary.save('meal',date.today(),None,'パスタ',db=db,measurement_id=pid)
        assert saved['nutrition_status']=='unknown' and saved['nutrition_snapshot'] is None
        assert list_measurements(db)[0]['kcal'] is None
    assert row['estimated_g'] is None and row['actual_g']==100


def test_meal_commit_ack_failure_preserves_photo(tmp_path, monkeypatch):
    db=tmp_path/'test.sqlite3';init_db(db);meal_diary.initialize(db)
    original=sqlite3.connect;calls=[0]
    @contextmanager
    def connect(*args,**kwargs):
        with original(*args,**kwargs) as con:
            yield con
        calls[0]+=1
        if calls[0]==2:raise OSError('commit acknowledgement lost')
    monkeypatch.setattr(meal_diary.sqlite3,'connect',connect)
    buf=BytesIO();Image.new('RGB',(32,32),'orange').save(buf,'PNG')
    with pytest.raises(OSError):
        meal_diary.save('durable',date.today(),None,'パスタ',photo=buf.getvalue(),db=db)
    with original(db) as con:
        key=con.execute("SELECT photo FROM meal_entries WHERE id='durable'").fetchone()[0]
    assert (tmp_path/'meal_photos'/key).is_file()


def test_estimated_result_without_nutrition_displays_unknown_and_saves(tmp_path):
    from streamlit.testing.v1 import AppTest
    db=tmp_path/'ui.sqlite3';init_db(db)
    script='''
import sys
sys.path.insert(0,'pasta_weight_app_v2')
import numpy as np
from measurement_ui import render_weight_result
render_weight_result(weight=50,product=dict(id=2,name='栄養未登録',g_per_strand=.68,kcal_100=0,protein_100=0,fat_100=0,carb_100=0),
 rgb=np.zeros((24,24,3),np.uint8),mode='広げた状態',using_demo=False,detected_count=73,
 estimator='opencv',explanation='synthetic UI contract test',db_path=DB_PATH)
'''
    script=script.replace('DB_PATH',repr(str(db)))
    at=AppTest.from_string(script).run()
    assert not at.exception
    assert any('未計算' in item.value for item in at.info)
    assert not any('kcal' in item.value for item in at.metric)
    at.button(key='measurement_save').click().run()
    assert not at.exception and list_measurements(db)[0]['kcal'] is None
    assert at.button(key='measurement_to_diary')
