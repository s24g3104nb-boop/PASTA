"""Synthetic metadata tests; no real images, model loading, or database writes."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pasta_weight_app_v2.experiments.research_contract import (
    validate_research_artifact, propagate_capture_qc,
)


class Predictor:
    def predict(self, _):
        raise AssertionError('Validation must happen before inference')


def contract():
    config = dict(version='synthetic-v1', target_long_side=512,
                  down_interpolation='INTER_AREA', up_interpolation='INTER_CUBIC')
    expected = dict(expected_version='synthetic-v1', expected_config=config,
                    expected_feature_keys=['fraction', 'relative_width'],
                    expected_source_hashes={'synthetic_pipeline.py': 'a'*64},
                    expected_image_versions={'numpy': 'test-np', 'opencv': 'test-cv'})
    artifact = dict(model=Predictor(), pipeline_version='synthetic-v1',
                    pipeline_config=copy.deepcopy(config),
                    config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                    feature_keys=list(expected['expected_feature_keys']),
                    source_hashes=dict(expected['expected_source_hashes']),
                    image_versions=dict(expected['expected_image_versions']),
                    approval='EXPLORATORY_ONLY', production_eligible=False,
                    independent_validation=False)
    return artifact, expected


def test_exact_research_contract_passes_without_calling_model_or_mutating_input():
    artifact, expected = contract()
    old_config = copy.deepcopy(artifact['pipeline_config'])
    assert validate_research_artifact(artifact, **expected) is None
    assert artifact['pipeline_config'] == old_config


@pytest.mark.parametrize('change', [
    lambda a: a.update(pipeline_version='obsolete'),
    lambda a: a['pipeline_config'].update(version='obsolete'),
    lambda a: a['pipeline_config'].update(down_interpolation='INTER_NEAREST'),
    lambda a: a['pipeline_config'].update(unknown_setting=1),
    lambda a: a['pipeline_config'].update(target_long_side=512.0),
    lambda a: a.update(config_sha256='0'*64),
    lambda a: a.update(feature_keys=['relative_width', 'fraction']),
    lambda a: a['source_hashes'].update({'synthetic_pipeline.py': 'b'*64}),
    lambda a: a['source_hashes'].update({'extra.py': 'c'*64}),
    lambda a: a.update(source_hashes={}),
    lambda a: a['image_versions'].update(numpy='different-np'),
    lambda a: a.update(image_versions={'numpy': 'test-np'}),
    lambda a: a.update(production_eligible=True),
    lambda a: a.update(independent_validation=True),
    lambda a: a.update(production_eligible=0),
    lambda a: a.update(approval='APPROVED'),
    lambda a: a.update(public_prediction_g=50),
    lambda a: a.pop('config_sha256'),
    lambda a: a.update(model=None),
])
def test_altered_missing_or_unsupported_metadata_is_rejected(change):
    artifact, expected = contract(); change(artifact)
    with pytest.raises(ValueError): validate_research_artifact(artifact, **expected)


def test_recomputed_hash_cannot_make_unsupported_interpolation_acceptable():
    artifact, expected = contract()
    artifact['pipeline_config']['down_interpolation'] = 'INTER_NEAREST'
    artifact['config_sha256'] = hashlib.sha256(json.dumps(artifact['pipeline_config'], sort_keys=True).encode()).hexdigest()
    with pytest.raises(ValueError, match='configuration'):
        validate_research_artifact(artifact, **expected)


def test_padding_preserves_original_capture_clipping_warning_and_public_rejection():
    original = ['foreground_touches_frame', 'unrecognized_reference_warning']
    current = ['low_detail_review', 'low_detail_review']
    result = propagate_capture_qc(original, current)
    assert result['qc_flags'] == ['foreground_touches_frame', 'low_detail_review', 'unrecognized_reference_warning']
    assert len(result['capture_reasons']) == 1
    assert result['public_prediction_g'] is None and result['status'] == '要確認'
    assert original == ['foreground_touches_frame', 'unrecognized_reference_warning']
    assert current == ['low_detail_review', 'low_detail_review']


def test_no_flags_still_does_not_approve_an_unvalidated_model():
    result = propagate_capture_qc([], [])
    assert result == dict(qc_flags=[], capture_reasons=[], public_prediction_g=None, status='要確認')


@pytest.mark.parametrize('bad_flags', ['one_string', None, [None], ['']])
def test_malformed_capture_flags_are_not_silently_dropped(bad_flags):
    with pytest.raises(ValueError): propagate_capture_qc(bad_flags, [])
