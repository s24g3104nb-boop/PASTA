import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pasta_weight_app_v2.experiments.capture_readiness import (
    summarize_predictions, validate_comparison_inputs, PIPELINES, MODELS)


def row(image='a', transform='original', prediction=20):
    return dict(pipeline='B_normalized5', model='ridge', image_id=image,
                transform=transform, nominal_g=20, research_prediction_g=prediction,
                original_sha256=image * 64)


def test_stability_and_training_error_are_separate_and_not_independent_mae():
    result = summarize_predictions([row(prediction=30), row(transform='resize50', prediction=30.2),
                                   row(transform='canvas_scale75', prediction=60)])[0]
    assert result['training_mae_g'] == 10
    assert result['ordinary_max_delta_g'] == pytest.approx(.2)
    assert result['transform_max_delta_g']['canvas_scale75'] == 30
    assert result['independent_mae_g'] is None
    assert result['production_eligible'] is False


def test_duplicate_or_missing_source_cannot_be_counted_as_more_samples():
    with pytest.raises(ValueError, match='Duplicate'):
        summarize_predictions([row(), row()])
    with pytest.raises(ValueError, match='Missing original'):
        summarize_predictions([row(transform='resize50')])
    with pytest.raises(ValueError, match='provenance'):
        summarize_predictions([row(), row('b', 'resize50')])


def test_changed_labels_or_nonfinite_predictions_are_rejected():
    changed = row(transform='resize50'); changed['nominal_g'] = 40
    with pytest.raises(ValueError, match='nominal label'):
        summarize_predictions([row(), changed])
    with pytest.raises(ValueError, match='Nonfinite'):
        summarize_predictions([row(prediction=float('nan'))])


def test_all_models_must_use_exactly_same_inputs_and_labels():
    rows = [dict(row(), pipeline=p, model=m) for p in PIPELINES for m in MODELS]
    document = {'inputs': [row()], 'rows': [dict(row(), reported_g=20)]}
    validate_comparison_inputs(rows, document)
    with pytest.raises(ValueError, match='identical complete'):
        validate_comparison_inputs(rows[:-1], document)
    changed = [dict(r) for r in rows]; changed[0]['nominal_g'] = 40
    with pytest.raises(ValueError, match='provenance or label'):
        validate_comparison_inputs(changed, document)
    document['inputs'][0]['image_id'] = '../outside'
    document['rows'][0]['image_id'] = '../outside'
    with pytest.raises(ValueError, match='Unsafe'):
        validate_comparison_inputs(rows, document)
