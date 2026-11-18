"""Sample-balanced, reject-aware evaluation after explicit physical-lineage review.

No model fitting, automatic split, database writes or production approval.
Metadata checks cannot prove the truth of a user's physical-sample declarations.
"""
from collections import defaultdict
from collections.abc import Mapping
import math
import numpy as np
from .lineage_protocol import validate_split

DEFAULT_TARGET_WEIGHTS_G = (20, 40, 60, 80, 100)


def _number(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(label + ': finite numeric value required')
    if value < 0 or (positive and value == 0):
        raise ValueError(label + ': invalid range')
    return float(value)


def _stratum_key(value):
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def summarize_attempts(rows, *, error_limit_g=5.):
    """All attempts retained; errors are conditional on a returned valid number.

    Sample MAE gives each physical sample equal weight, regardless of photo count.
    Failure rate here means rejection OR absolute error > error_limit_g. Neither
    rejected images nor unknown distance/layout labels silently disappear.
    """
    if not rows:
        return dict(attempts=0, accepted=0, success_rate=None, rejection_rate=None, failure_rate=None,
                    sample_count=0,sample_balanced_rejection_rate=None,sample_balanced_failure_rate=None,
                    mae_g=None, rmse_g=None, max_absolute_error_g=None,
                    sample_balanced_mae_g=None, sample_balanced_rmse_g=None)
    errors = []; grouped=defaultdict(list); outcomes=defaultdict(list); rejected=0
    for row in rows:
        actual=_number(row['actual_g'], 'actual_g', positive=True)
        predicted=row.get('predicted_g')
        if predicted is None:
            rejected+=1
            outcomes[row['sample_id']].append((1,1))
            continue
        predicted=_number(predicted, 'predicted_g', positive=True)
        error=predicted-actual;errors.append(error);grouped[row['sample_id']].append(error)
        outcomes[row['sample_id']].append((0,int(abs(error)>error_limit_g)))
    e=np.array(errors, float)
    return dict(attempts=len(rows), accepted=len(errors), success_rate=len(errors)/len(rows),
        sample_count=len(outcomes),
        sample_balanced_rejection_rate=float(np.mean([np.mean([p[0] for p in v]) for v in outcomes.values()])),
        sample_balanced_failure_rate=float(np.mean([np.mean([p[1] for p in v]) for v in outcomes.values()])),
        rejection_rate=rejected/len(rows), failure_rate=(rejected+sum(abs(x)>error_limit_g for x in errors))/len(rows),
        mae_g=float(np.mean(abs(e))) if len(e) else None,
        rmse_g=float(np.sqrt(np.mean(e**2))) if len(e) else None,
        max_absolute_error_g=float(np.max(abs(e))) if len(e) else None,
        sample_balanced_mae_g=float(np.mean([np.mean(np.abs(v)) for v in grouped.values()])) if grouped else None,
        sample_balanced_rmse_g=float(np.sqrt(np.mean([np.mean(np.square(v)) for v in grouped.values()]))) if grouped else None)


def prediction_drift(rows):
    """Compare transformed predictions with the same original without adding attempts."""
    by_transform = defaultdict(list)
    for row in rows:
        original = row.get('predicted_g')
        if original is not None:
            _number(original, 'original prediction', positive=True)
        transforms = row.get('transformed_predictions', {})
        if not isinstance(transforms, Mapping):
            raise ValueError('transformed_predictions must map transform names to predictions or null')
        for name, transformed in transforms.items():
            if not isinstance(name, str) or not name.strip() or name == 'original':
                raise ValueError('A named derivative, not the original, is required')
            if transformed is not None:
                _number(transformed, 'transformed prediction', positive=True)
            by_transform[name].append((original, transformed))
    result = {}
    for name, pairs in sorted(by_transform.items()):
        comparable = [abs(a-b) for a,b in pairs if a is not None and b is not None]
        result[name] = dict(pairs=len(pairs), both_accepted=len(comparable),
                            changed_to_rejected=sum(a is not None and b is None for a,b in pairs),
                            changed_to_accepted=sum(a is None and b is not None for a,b in pairs),
                            mean_absolute_delta_g=float(np.mean(comparable)) if comparable else None,
                            max_absolute_delta_g=max(comparable) if comparable else None)
    return result


def independent_evaluation(records, assignments, material_parents, predictions, *, model_card,
                           target_weights_g=DEFAULT_TARGET_WEIGHTS_G):
    """Evaluate exact frozen test attempts only after reviewed split/labels.

    Required extra record fields: measured_g, weight_label_verified=True.
    Distance/layout and optional nominal_weight_g are reporting metadata only.
    Exact measured_g is always used for errors; nominal targets never replace it.
    model_card lists training original hashes and the fixed split fingerprint;
    it is a declaration, not cryptographic proof of correct model construction.
    """
    return _partition_evaluation(records, assignments, material_parents, predictions, model_card,
                                 'test', target_weights_g)


def validation_evaluation(records, assignments, material_parents, predictions, *, model_card,
                          target_weights_g=DEFAULT_TARGET_WEIGHTS_G):
    """Report validation results separately. Do not compute a deployment screen."""
    return _partition_evaluation(records, assignments, material_parents, predictions, model_card,
                                 'validation', target_weights_g)


def _partition_evaluation(records, assignments, material_parents, predictions, model_card,
                          partition, target_weights_g):
    targets = tuple(_number(weight, 'target weight', positive=True) for weight in target_weights_g)
    if not targets or len(set(targets)) != len(targets):
        raise ValueError('Target weight strata must be explicit and distinct')
    split = validate_split(records, assignments, material_parents)
    for row in records:
        if row.get('weight_label_verified') is not True:
            raise ValueError('Reviewed measured-weight labels required')
        _number(row.get('measured_g'), 'measured_g', positive=True)
    expected_train={r['original_sha256'].lower() for r in records if assignments[r['image_id']]=='train'}
    if not isinstance(model_card.get('training_original_sha256'), (list,tuple)):
        raise ValueError('Explicit training provenance required')
    if ((partition == 'test' and model_card.get('frozen_before_test') is not True) or
        model_card.get('split_manifest_sha256')!=split['split_manifest_sha256'] or
        set(model_card.get('training_original_sha256', []))!=expected_train):
        raise ValueError('Frozen model training provenance/split mismatch')
    selected={r['image_id']:r for r in records if assignments[r['image_id']]==partition}
    if split['groups_by_split'][partition]!=len({r['sample_id'] for r in selected.values()}):
        raise ValueError('Evaluation sample IDs share physical lineage; they are not independent units')
    if len(selected) != len({row['original_sha256'].lower() for row in selected.values()}):
        raise ValueError('Repeated original in evaluation; provide derivatives as transformed_predictions')
    if len(predictions)!=len(selected) or {p['image_id'] for p in predictions}!=set(selected):
        raise ValueError(f'Every {partition} attempt must occur exactly once; no selective omission')
    rows=[]; labels=defaultdict(set); nominal_labels=defaultdict(set)
    for prediction in predictions:
        original=selected[prediction['image_id']];p=prediction.get('predicted_g')
        if p is not None:_number(p,'predicted_g',positive=True)
        if p is None and not prediction.get('reason'):
            raise ValueError('Rejected attempt requires a reason')
        labels[original['sample_id']].add(original['measured_g'])
        nominal=original.get('nominal_weight_g')
        if nominal is None and original['measured_g'] in targets:nominal=original['measured_g']
        if nominal is not None:
            _number(nominal, 'nominal weight', positive=True)
            nominal_labels[original['sample_id']].add(nominal)
        angle = original.get('tilt_deg')
        if angle is not None and _number(angle, 'tilt_deg') > 90:
            raise ValueError('tilt_deg must be 0 to 90 degrees from overhead')
        rows.append(dict(image_id=original['image_id'],sample_id=original['sample_id'],
            actual_g=original['measured_g'],predicted_g=p,reason=prediction.get('reason'),
            nominal_weight_g=nominal if nominal is not None else 'unknown',
            distance_cm=original.get('distance_cm', 'unknown'),tilt_deg=angle if angle is not None else 'unknown',
            arrangement=original.get('arrangement', 'unknown'),
            transformed_predictions=prediction.get('transformed_predictions', {})))
    if any(len(weights)!=1 for weights in labels.values()):
        raise ValueError('A physical sample cannot have contradictory measured weights')
    if any(len(weights)>1 for weights in nominal_labels.values()):
        raise ValueError('A physical sample cannot belong to multiple nominal weight strata')
    scope = 'independent_declared_sample_test' if partition == 'test' else 'validation'
    result=dict(scope=scope,split=split,target_weights_g=list(targets),overall=summarize_attempts(rows),
        by_weight={},by_nominal_weight={},by_distance={},by_angle={},by_arrangement={},
        transformed_prediction_drift=prediction_drift(rows),production_approved=False)
    for field,target in [('actual_g','by_weight'),('nominal_weight_g','by_nominal_weight'),('distance_cm','by_distance'),
                         ('tilt_deg','by_angle'),('arrangement','by_arrangement')]:
        for key in sorted({_stratum_key(r[field]) for r in rows}):
            result[target][key]=summarize_attempts([r for r in rows if _stratum_key(r[field])==key])
    if partition == 'validation':
        return result
    by_sample=defaultdict(list)
    for row in rows:
        if row['predicted_g'] is not None:by_sample[row['sample_id']].append(abs(row['predicted_g']-row['actual_g']))
    sample_errors=np.array([np.mean(by_sample[k]) for k in sorted(by_sample)],float)
    if len(sample_errors)>=2:
        rng=np.random.default_rng(42)
        boot=np.mean(rng.choice(sample_errors,(2000,len(sample_errors)),replace=True),axis=1)
        result['sample_mae_bootstrap_95_g']=[float(v) for v in np.quantile(boot,[.025,.975])]
    else:result['sample_mae_bootstrap_95_g']=None
    # A proposed final-goal screen, NOT an automatic deployment authorization.
    reasons=[];overall=result['overall'];weights=result['by_nominal_weight']
    test_samples={r['sample_id'] for r in rows}
    if len({r['sample_id'] for r in rows if r['nominal_weight_g'] in targets})<50:
        reasons.append('at_least_50_independent_test_samples_required')
    for w in targets:
        if len({r['sample_id'] for r in rows if r['nominal_weight_g']==w})<10:
            reasons.append(f'insufficient_test_samples_at_{w:g}g')
    if max(overall['rejection_rate'],overall['sample_balanced_rejection_rate'])>.1:reasons.append('coverage_below_90_percent')
    if overall['sample_balanced_mae_g'] is None or overall['sample_balanced_mae_g']>5:reasons.append('mae_over_5g_or_unavailable')
    if result['sample_mae_bootstrap_95_g'] is None or result['sample_mae_bootstrap_95_g'][1]>5:reasons.append('uncertainty_does_not_support_5g_goal')
    if overall['max_absolute_error_g'] is None or overall['max_absolute_error_g']>20:reasons.append('maximum_error_over_20g_or_unavailable')
    if any(v['sample_balanced_mae_g'] is None or v['sample_balanced_mae_g']>10 for v in weights.values()):reasons.append('weight_stratum_error_over_10g_or_unavailable')
    distances=defaultdict(set)
    for row in rows:
        distance=row['distance_cm']
        if type(distance) in (int,float) and math.isfinite(distance) and distance>0:
            distances[row['sample_id']].add(distance)
    if any(len(distances[sample])<2 for sample in test_samples):reasons.append('multiple_measured_distances_per_sample_required')
    dense=[r for r in rows if r['arrangement']=='dense']
    dense_metrics=summarize_attempts(dense)
    if (dense_metrics['sample_count'] if 'sample_count' in dense_metrics else 0)<25:
        reasons.append('dense_test_sample_coverage_insufficient')
    if dense_metrics['sample_balanced_mae_g'] is None or dense_metrics['sample_balanced_mae_g']>5:
        reasons.append('dense_mae_over_5g_or_unavailable')
    if dense_metrics['sample_balanced_rejection_rate'] is None or dense_metrics['sample_balanced_rejection_rate']>.1:
        reasons.append('dense_coverage_below_90_percent')
    if any(v['sample_balanced_mae_g'] is None or v['sample_balanced_mae_g']>10 for v in result['by_distance'].values()):
        reasons.append('distance_stratum_error_over_10g_or_unavailable')
    result['proposed_adoption_gate']=dict(passed=not reasons,reasons=reasons,requires_human_review=True,
        note='Declared-lineage checks and a statistical screen do not independently establish real physical independence.')
    return result


def main():
    """Evaluate a locally reviewed manifest and predictions for one explicit split."""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--partition', choices=('validation', 'test'), required=True)
    args = parser.parse_args()
    bundle = json.loads(args.input.read_text(encoding='utf-8-sig'))
    evaluator = independent_evaluation if args.partition == 'test' else validation_evaluation
    result = evaluator(bundle['records'], bundle['assignments'], bundle['material_parents'],
                       bundle['predictions'], model_card=bundle['model_card'],
                       target_weights_g=bundle.get('target_weights_g', DEFAULT_TARGET_WEIGHTS_G))
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(result['scope'], result['overall']['attempts'], 'attempts')


if __name__ == '__main__':
    main()
