"""36g実機事例そのものの再現ではなく、原因候補を検出する回帰試験。"""
import hashlib
import json
import sqlite3
import numpy as np
from PIL import Image
from pasta_weight_app_v2.measurement_diagnostics import weight_gate, trace
from pasta_weight_app_v2.pasta_ml.evaluate import summary
from pasta_weight_app_v2.pasta_ml.model import predict
from pasta_weight_app_v2.storage import init_db, save_measurement


def test_uncalibrated_real_photo_rejected():
    product=dict(id=1,name='デモ用（仮）',g_per_strand=1.)
    assert weight_gate(product,using_demo=False,mode='広げた状態',manual=True)
    assert not weight_gate(product,using_demo=True,mode='広げた状態')


def test_touching_bundle_and_cropped_mask_rejected():
    product=dict(id=2,name='校正製品',g_per_strand=.8)
    mask=np.zeros((300,300),np.uint8)
    mask[80:110,30:270]=255
    assert weight_gate(product,using_demo=False,mode='広げた状態',mask=mask)
    mask[:]=0
    mask[80:83,30:270]=255
    assert not weight_gate(product,using_demo=False,mode='広げた状態',mask=mask)
    mask[80:83,:30]=255
    assert weight_gate(product,using_demo=False,mode='広げた状態',mask=mask)


def test_foreground_model_rejects_raw_phone_input():
    result=predict(dict(input_contract='reviewed_foreground'),np.zeros((32,32,3),np.uint8),'spread','2')
    assert not result['success']


def test_error_metrics_include_36g_and_failures():
    rows=[dict(success=True,predicted_g=56,actual_weight_g=20),dict(success=True,predicted_g=40,actual_weight_g=40),dict(success=False)]
    result=summary(rows)
    assert result['max_absolute_error_g']==36
    assert result['mae_g']==18
    assert result['failed']==1
    assert summary([])['max_absolute_error_g'] is None


def test_exact_inference_image_and_provenance_saved(tmp_path):
    db=tmp_path/'measurement.sqlite3'
    init_db(db)
    rgb=np.random.default_rng(42).integers(0,256,(40,40,3),dtype=np.uint8)
    product=dict(id=2,name='test',g_per_strand=.8)
    details=trace(rgb,estimator='opencv',product=product,mode='spread',parameters={'min_area':25})
    ident=save_measurement(product_name='test',mode='spread',detected_count=10,estimated_g=8,actual_g=44,nutrition=dict(kcal=0,protein_g=0,fat_g=0,carb_g=0),image_rgb=rgb,db_path=db,diagnostics=details)
    with sqlite3.connect(db) as con:
        saved=json.loads(con.execute('SELECT details_json FROM measurement_diagnostics WHERE measurement_id=?',(ident,)).fetchone()[0])
    restored=np.array(Image.open(tmp_path/'photos'/saved['inference_image_filename']))
    assert np.array_equal(rgb,restored)
    assert saved['inference_rgb_sha256']==hashlib.sha256(str(restored.shape).encode()+restored.tobytes()).hexdigest()
    assert saved['parameters']=={'min_area':25}
