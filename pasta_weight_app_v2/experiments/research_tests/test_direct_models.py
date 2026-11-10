"""Research environment tests; synthetic only, separate from the app dependencies."""
import sys,json
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from pasta_weight_app_v2.experiments.direct_models import models,metrics,public_result,validate_independent_assignment,FEATURE_SETS,run


def records():
    return [dict(image_id=str(i),physical_family=str(i),capture_group=str(i),source_sha256=str(i),lineage_verified=True) for i in range(3)]


def test_unknown_lineage_cannot_generate_independent_partitions():
    r=records();r[0]['lineage_verified']=False
    with pytest.raises(ValueError):validate_independent_assignment(r,dict(zip('012',['train','validation','test'])))


@pytest.mark.parametrize('key',['physical_family','capture_group','source_sha256'])
def test_shared_material_capture_or_augmented_original_cannot_cross_split(key):
    r=records();r[1][key]=r[0][key]
    with pytest.raises(ValueError):validate_independent_assignment(r,dict(zip('012',['train','validation','test'])))


def test_explicit_independent_partitions_and_no_public_numeric_result():
    validate_independent_assignment(records(),dict(zip('012',['train','validation','test'])))
    assert public_result()['predicted_weight_g'] is None
    assert public_result()['status']=='要確認'


def test_features_cannot_include_labels_filenames_or_group_ids():
    forbidden={'reported_g','capture_group','image_id','lineage_block','source_sha256'}
    assert all(not forbidden.intersection(keys) for keys in FEATURE_SETS.values())


def test_models_are_reproducible_and_input_is_not_mutated():
    x=np.arange(45,dtype=float).reshape(15,3);y=np.repeat([20,40,60,80,100],3);old=x.copy()
    for name,model in models().items():
        p=model.fit(x,y).predict(x);q=models()[name].fit(x,y).predict(x)
        np.testing.assert_allclose(p,q)
        assert np.isfinite(p).all()
    np.testing.assert_array_equal(x,old)


def test_error_metric_reports_absolute_and_signed_cases_consistently():
    v=metrics([20,40],[30,20]);assert v['mae_g']==15 and v['max_absolute_error_g']==20
    with pytest.raises(ValueError):metrics([20],[np.nan])


def test_exploratory_run_keeps_augmentations_out_of_fit_and_no_validation_claim(tmp_path):
    rng=np.random.default_rng(9);rows=[]
    keys=set(k for group in FEATURE_SETS.values() for k in group)
    for method in ('hsv_baseline','hsv_lab'):
        for i in range(15):
            original=dict(image_id=str(i),reported_g=20*(1+i//3),capture_group=str(i//3),method=method,transform='original',**{k:float(rng.random()) for k in sorted(keys)})
            rows.extend([original,dict(original,transform='resize_half')])
    path=tmp_path/'features.json';path.write_text(json.dumps(dict(rows=rows,independent_test_images=0,supervised_approval=False)))
    out=tmp_path/'results';result=run(path,out)
    assert result['training_images']==15 and result['approved_production_training_images']==0
    assert result['independent_metrics'] is None and len(result['fits'])==30
    assert len(result['predictions'])==450 and len(result['perturbations'])==450
    assert all(f['n_images']==15 and f['independent_metrics'] is None for f in result['fits'])
    assert all(p['public_result']['predicted_weight_g'] is None for p in result['predictions'])
    assert len(list((out/'models').glob('*.joblib')))==30
    with pytest.raises(FileExistsError):run(path,out)
