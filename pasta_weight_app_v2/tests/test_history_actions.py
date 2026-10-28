import json
import sqlite3
import numpy as np
import pytest
from storage import init_db, save_measurement, list_measurements
from history_actions import change_measurement


@pytest.fixture
def record(tmp_path):
    db=tmp_path/'history.sqlite3'
    init_db(db)
    ident=save_measurement(product_name='fixture',mode='spread',detected_count=10,estimated_g=10,actual_g=None,nutrition=dict(kcal=35,protein_g=1,fat_g=.2,carb_g=7),image_rgb=np.zeros((20,20,3),np.uint8),db_path=db)
    return db,ident


def test_edit_preserves_prediction_and_audit(record):
    db,ident=record
    change_measurement(ident,actual_g=46,db_path=db)
    row=list_measurements(db)[0]
    assert row['actual_g']==46 and row['estimated_g']==10 and row['kcal']==35
    with sqlite3.connect(db) as con:
        previous=json.loads(con.execute('SELECT previous_json FROM measurement_edits').fetchone()[0])
    assert previous['actual_g'] is None


def test_archive_removes_from_list_but_keeps_original(record):
    db,ident=record
    change_measurement(ident,archive=True,db_path=db)
    assert not list_measurements(db)
    with sqlite3.connect(db) as con:
        assert con.execute('SELECT COUNT(*) FROM measurements').fetchone()[0]==1


def test_linked_cooking_record_cannot_be_changed(record):
    db,ident=record
    with sqlite3.connect(db) as con:
        con.execute('CREATE TABLE cooking_sessions(measurement_id INTEGER)')
        con.execute('INSERT INTO cooking_sessions VALUES(?)',(ident,))
    for kwargs in [dict(archive=True),dict(actual_g=20)]:
        with pytest.raises(ValueError,match='使用中'):change_measurement(ident,db_path=db,**kwargs)
    assert list_measurements(db)[0]['actual_g'] is None


@pytest.mark.parametrize('value',[0,-1,float('nan'),float('inf')])
def test_invalid_weight_is_not_saved(record,value):
    db,ident=record
    with pytest.raises(ValueError):change_measurement(ident,actual_g=value,db_path=db)
    assert list_measurements(db)[0]['actual_g'] is None
