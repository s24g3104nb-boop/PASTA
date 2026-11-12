"""Artificial metadata tests; no measured accuracy is inferred from these fixtures."""
import sys,copy
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from pasta_weight_app_v2.experiments.weight_evaluation import summarize_attempts,independent_evaluation
from pasta_weight_app_v2.experiments.lineage_protocol import validate_split


def manifest():
    records=[];assignment={};materials={}
    for i,split in enumerate(('train','validation','test')):
        key=f'material{i}';materials[key]=[]
        records.append(dict(image_id=f'image{i}',original_id=f'original{i}',original_sha256=f'{i+1:064x}',
            sample_id=f'sample{i}',capture_group_id=f'capture{i}',physical_family_id=f'family{i}',
            parent_material_ids=[key],lineage_verified=True,lineage_evidence='synthetic fixture only',
            lineage_confirmed_by='synthetic test',weight_label_verified=True,measured_g=40,
            distance_cm=40,arrangement='dense'))
        assignment[f'image{i}']=split
    split=validate_split(records,assignment,materials)
    card=dict(frozen_before_test=True,split_manifest_sha256=split['split_manifest_sha256'],training_original_sha256=[records[0]['original_sha256']])
    return records,assignment,materials,card


def test_equal_sample_weighting_and_rejections_not_silently_dropped():
    rows=[dict(sample_id='a',actual_g=40,predicted_g=41) for _ in range(10)]
    rows+=[dict(sample_id='b',actual_g=40,predicted_g=60),dict(sample_id='c',actual_g=40,predicted_g=None)]
    result=summarize_attempts(rows)
    assert result['attempts']==12 and result['accepted']==11
    assert result['sample_balanced_mae_g']==10.5
    assert result['mae_g']==pytest.approx(30/11)
    assert result['rejection_rate']==pytest.approx(1/12)
    assert result['failure_rate']==pytest.approx(2/12)


def test_reviewed_test_metrics_do_not_grant_automatic_approval():
    r,a,m,c=manifest()
    result=independent_evaluation(r,a,m,[dict(image_id='image2',predicted_g=43)],model_card=c)
    assert result['overall']['mae_g']==3 and result['overall']['rmse_g']==3
    assert result['by_distance']['40']['mae_g']==3
    assert result['by_arrangement']['dense']['mae_g']==3
    assert not result['proposed_adoption_gate']['passed']
    assert not result['production_approved']


def test_unknown_lineage_label_leak_and_selective_omission_fail_closed():
    r,a,m,c=manifest()
    unknown=copy.deepcopy(r);unknown[0]['lineage_verified']=False
    with pytest.raises(ValueError):independent_evaluation(unknown,a,m,[],model_card=c)
    shared=copy.deepcopy(m);shared['material2']=['material0']
    with pytest.raises(ValueError,match='crosses splits'):independent_evaluation(r,a,shared,[],model_card=c)
    with pytest.raises(ValueError,match='Every test'):independent_evaluation(r,a,m,[],model_card=c)
    c['training_original_sha256'].append(r[2]['original_sha256'])
    with pytest.raises(ValueError,match='provenance'):independent_evaluation(r,a,m,[],model_card=c)


@pytest.mark.parametrize('invalid',[0,-1,float('nan'),float('inf'),True])
def test_invalid_numeric_predictions_are_not_treated_as_success(invalid):
    r,a,m,c=manifest()
    with pytest.raises(ValueError):independent_evaluation(r,a,m,[dict(image_id='image2',predicted_g=invalid)],model_card=c)


def test_all_rejected_still_counts_failure_and_preserves_unknown_strata():
    r,a,m,c=manifest();r[2].pop('distance_cm');r[2].pop('arrangement')
    split=validate_split(r,a,m);c['split_manifest_sha256']=split['split_manifest_sha256']
    result=independent_evaluation(r,a,m,[dict(image_id='image2',predicted_g=None,reason='not usable')],model_card=c)
    assert result['overall']['failure_rate']==1 and result['overall']['mae_g'] is None
    assert result['by_distance']['unknown']['rejection_rate']==1
    assert not result['proposed_adoption_gate']['passed']


def test_many_good_photos_do_not_conceal_rejected_independent_samples():
    rows=[dict(sample_id='a',actual_g=40,predicted_g=41) for _ in range(100)]
    rows += [dict(sample_id=f'b{i}',actual_g=40,predicted_g=None) for i in range(10)]
    result=summarize_attempts(rows)
    assert result['rejection_rate']<.1
    assert result['sample_balanced_rejection_rate']> .9


def test_shared_lineage_inside_test_is_not_counted_as_independent_sample_ids():
    r,a,m,c=manifest()
    another=dict(r[2],image_id='extra',original_id='extra',original_sha256='e'*64,sample_id='different-name')
    r.append(another);a['extra']='test'
    split=validate_split(r,a,m);c['split_manifest_sha256']=split['split_manifest_sha256']
    with pytest.raises(ValueError,match='not independent units'):
        independent_evaluation(r,a,m,[dict(image_id='image2',predicted_g=40),dict(image_id='extra',predicted_g=40)],model_card=c)


def test_nominal_weight_strata_do_not_replace_actual_decimal_measurement():
    r,a,m,c=manifest();r[2]['measured_g']=40.2;r[2]['nominal_weight_g']=40
    c['split_manifest_sha256']=validate_split(r,a,m)['split_manifest_sha256']
    result=independent_evaluation(r,a,m,[dict(image_id='image2',predicted_g=42)],model_card=c)
    assert result['overall']['mae_g']==pytest.approx(1.8)
    assert result['by_nominal_weight']['40']['mae_g']==pytest.approx(1.8)
    assert '40.2' in result['by_weight']


def test_sufficient_declared_dense_trials_can_pass_screen_but_not_auto_deploy():
    # All values/evidence are synthetic metadata used to exercise the evaluator.
    records=[];assignment={};materials={}
    for i in range(60):
        part='train' if i<5 else 'validation' if i<10 else 'test'
        nominal=(20,40,60,80,100)[i%5];materials[f'm{i}']=[]
        for j,distance in enumerate((35,70)):
            ident=f'i{i}-{j}'
            records.append(dict(image_id=ident,original_id=ident,original_sha256=f'{i*2+j+1:064x}',
                sample_id=f's{i}',capture_group_id=f'c{i}',physical_family_id=f'f{i}',parent_material_ids=[f'm{i}'],
                lineage_verified=True,lineage_evidence='synthetic fixture',lineage_confirmed_by='synthetic tester',
                weight_label_verified=True,measured_g=nominal+.2,nominal_weight_g=nominal,distance_cm=distance,arrangement='dense'))
            assignment[ident]=part
    split=validate_split(records,assignment,materials)
    card=dict(frozen_before_test=True,split_manifest_sha256=split['split_manifest_sha256'],
        training_original_sha256=[r['original_sha256'] for r in records if assignment[r['image_id']]=='train'])
    predictions=[dict(image_id=r['image_id'],predicted_g=r['measured_g']) for r in records if assignment[r['image_id']]=='test']
    result=independent_evaluation(records,assignment,materials,predictions,model_card=card)
    assert result['proposed_adoption_gate']['passed'] and not result['production_approved']
