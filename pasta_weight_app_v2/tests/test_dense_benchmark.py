"""Diagnostic contracts, not photo weight accuracy claims."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from pasta_weight_app_v2.experiments.dense_weight_benchmark import interior_texture,INTERIOR_KEYS,fit_exploratory
from pasta_weight_app_v2.experiments.bundle_texture import extract_bundle_texture
from pasta_weight_app_v2.pasta_vision import detect_spread
from pasta_weight_app_v2.measurement_diagnostics import weight_gate


def test_calibration_does_not_convert_merged_components_to_true_strands():
    image=np.full((300,700,3),20,np.uint8)
    for i in range(20):image[30+i*9:33+i*9,80:620]=(220,180,40)
    apart=detect_spread(image)
    assert apart.count==20
    assert not weight_gate(dict(name='synthetic',g_per_strand=.68),using_demo=False,mode='広げた状態',mask=apart.mask)
    image=np.full((300,700,3),20,np.uint8);image[50:110,80:620]=(220,180,40)
    dense=detect_spread(image)
    assert dense.count==1
    assert weight_gate(dict(name='synthetic',g_per_strand=.68),using_demo=False,mode='広げた状態',mask=dense.mask)
    assert 1*.68!=20*.68


def test_foreground_interior_features_do_not_read_background():
    rng=np.random.default_rng(7);rgb=rng.integers(40,220,(200,350,3),dtype=np.uint8)
    mask=np.zeros(rgb.shape[:2],np.uint8);mask[30:170,40:300]=255
    ref=interior_texture(rgb,mask)
    rgb[mask==0]=0
    result=interior_texture(rgb,mask)
    assert result==ref and np.isfinite(list(result.values())).all()


def test_exploratory_fit_only_originals_and_no_independent_claim(tmp_path):
    rows=[];rng=np.random.default_rng(4)
    for i in range(15):
        row=dict(image_id=str(i),transform='original',reported_g=20*(i//3+1),original_sha256=f'{i:064x}',lineage_block='one',production_approved=False,f=float(rng.random()))
        rows.append(row);rows.append(dict(row,transform='resize50',f=row['f']*.99));rows.append(dict(row,transform='canvas_scale75',f=row['f']*.5))
    fits,predictions,drift,_=fit_exploratory(rows,{'one':['f']},tmp_path/'models')
    assert len(fits)==5 and len(predictions)==225 and len(drift)==10
    assert all(f['original_images']==15 and f['independent_metrics'] is None for f in fits)
    assert all(p['public_prediction_g'] is None for p in predictions)
    with pytest.raises(FileExistsError):fit_exploratory(rows,{'one':['f']},tmp_path/'models')


def test_calibrated_recovery_does_not_send_user_back_to_calibration():
    from streamlit.testing.v1 import AppTest
    script="""
import sys
sys.path.insert(0,'pasta_weight_app_v2')
import manual_weight_ui
manual_weight_ui.render_manual_weight=lambda *a,**k:None
manual_weight_ui.recovery(dict(id=1,name='synthetic fixture',g_per_strand=.68),None)
"""
    at=AppTest.from_string(script).run()
    assert not at.exception
    assert all(b.key!='recovery_calibration' for b in at.button)
    assert at.button(key='recovery_retake')
