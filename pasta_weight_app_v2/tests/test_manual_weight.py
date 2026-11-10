import sys,sqlite3
from pathlib import Path
import pytest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from storage import init_db,save_measurement,list_measurements
from manual_measurement_storage import allow_unestimated_measurements


def save(db,token='one',nutrition=None):
    return save_measurement(product_name='test',mode='実測重量の入力',detected_count=0,estimated_g=None,actual_g=100.,nutrition=nutrition,image_rgb=None,db_path=db,source_kind='real',estimator='manual_weight',submission_token=token)


def test_manual_null_backup_idempotency_and_foreign_keys(tmp_path):
    db=tmp_path/'test.sqlite3';init_db(db)
    with sqlite3.connect(db) as c:
        c.execute('CREATE TABLE child(mid INTEGER REFERENCES measurements(id))')
        c.execute('CREATE INDEX custom_name ON measurements(product_name)')
    ident=save(db)
    assert save(db)==ident
    row=list_measurements(db)[0]
    assert row['estimated_g'] is None and row['actual_g']==100 and row['kcal'] is None
    assert row['image_filename'] is None and row['estimator']=='manual_weight'
    assert list(tmp_path.glob('*.before-manual-weight-*.backup'))
    with sqlite3.connect(db) as c:
        c.execute('PRAGMA foreign_keys=ON');c.execute('INSERT INTO child VALUES(?)',(ident,))
        assert not c.execute('PRAGMA foreign_key_check').fetchall()
        assert c.execute("SELECT 1 FROM sqlite_master WHERE name='custom_name'").fetchone()


def test_manual_nutrition_and_failure_rollback(tmp_path):
    db=tmp_path/'test.sqlite3';init_db(db);allow_unestimated_measurements(db)
    with sqlite3.connect(db) as c:c.execute("CREATE TRIGGER reject_insert BEFORE INSERT ON measurements BEGIN SELECT RAISE(ABORT,'test'); END")
    with pytest.raises(sqlite3.Error):save(db)
    assert not list_measurements(db)
    with sqlite3.connect(db) as c:c.execute('DROP TRIGGER reject_insert')
    save(db,nutrition=dict(kcal=350,protein_g=12,fat_g=2,carb_g=70))
    assert list_measurements(db)[0]['kcal']==350


def test_legacy_measurements_survive_migration(tmp_path):
    db=tmp_path/'legacy.sqlite3';init_db(db)
    save_measurement(product_name='old',mode='spread',detected_count=10,estimated_g=20,actual_g=None,nutrition=dict(kcal=70,protein_g=2,fat_g=1,carb_g=10),image_rgb=np.zeros((32,32,3),np.uint8),db_path=db)
    before=list_measurements(db)[0];save(db)
    assert list_measurements(db)[1]==before


def test_manual_never_accepts_fake_estimate(tmp_path):
    db=tmp_path/'test.sqlite3';init_db(db)
    with pytest.raises(ValueError):
        save_measurement(product_name='test',mode='manual',detected_count=0,estimated_g=0,actual_g=100,nutrition=None,image_rgb=None,db_path=db,estimator='manual_weight')

def test_manual_ui_save_failure_retry_diary_without_photo(tmp_path):
    import subprocess,os
    script=r'''
from streamlit.testing.v1 import AppTest
import sys,sqlite3
sys.path.insert(0,'pasta_weight_app_v2')
from storage import DEFAULT_DB,list_measurements,save_product
from meal_diary import entries
at=AppTest.from_file('pasta_weight_app_v2/app.py',default_timeout=20).run()
save_product('本番テスト製品',0,350,12,2,70)
at.button(key='home_manual_weight').click().run()
at.selectbox(key='manual_only_product').set_value(2).run()
next(x for x in at.number_input if x.label=='はかりで測った実測重量（g）').set_value(100.).run()
with sqlite3.connect(DEFAULT_DB) as c:c.execute("CREATE TRIGGER reject_manual BEFORE INSERT ON measurements BEGIN SELECT RAISE(ABORT,'test'); END")
at.button(key='manual_weight_save').click().run()
assert at.error and not list_measurements()
assert next(x for x in at.number_input if x.label=='はかりで測った実測重量（g）').value==100.
with sqlite3.connect(DEFAULT_DB) as c:c.execute('DROP TRIGGER reject_manual')
at.button(key='manual_weight_save').click().run()
assert not at.exception,at.exception
rows=list_measurements();assert len(rows)==1 and rows[0]['actual_g']==100 and rows[0]['estimated_g'] is None and rows[0]['kcal']==350
assert at.button(key='manual_weight_save').disabled
at.button(key='manual_weight_to_diary').click().run()
next(x for x in at.text_input if x.label=='料理名').set_value('実測のパスタ').run()
at.button(key='meal_save').click().run()
assert not at.exception,at.exception
assert entries()[0]['measurement_snapshot']['estimated_g'] is None
assert entries()[0]['nutrition_snapshot']['kcal']==350
at.radio(key='main_nav').set_value('記録').run()
at.selectbox(key='record_kind').set_value('乾麺の測定履歴').run()
assert not at.exception,at.exception
'''
    result=subprocess.run([sys.executable,'-c',script],cwd=Path(__file__).resolve().parents[2],env=dict(os.environ,PASTA_DATA_DIR=str(tmp_path/'app')),capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr

def test_commit_ack_failure_keeps_committed_measurement_photo(tmp_path,monkeypatch):
    import storage
    from contextlib import contextmanager
    db=tmp_path/'ack.sqlite3';init_db(db)
    original=storage._connect;first=[True]
    @contextmanager
    def acknowledgement_failure(path):
        with original(path) as con:yield con
        if first[0]:
            first[0]=False
            raise OSError('commit acknowledgement lost')
    monkeypatch.setattr(storage,'_connect',acknowledgement_failure)
    with pytest.raises(OSError):
        save_measurement(product_name='photo',mode='spread',detected_count=1,estimated_g=10,actual_g=None,nutrition=dict(kcal=35,protein_g=1,fat_g=1,carb_g=5),image_rgb=np.zeros((32,32,3),np.uint8),db_path=db)
    row=list_measurements(db)[0]
    assert (tmp_path/'photos'/row['image_filename']).is_file()
