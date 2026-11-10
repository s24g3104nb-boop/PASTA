"""Fail-closed metadata checks for offline research artifacts, not deployment approval.

Expected values must be constructed from trusted current code, configuration and
runtime versions. Reading both the artifact and its expected values from the same
untrusted artifact would not establish integrity. This is not a signature check or
a safe loader for pickle/joblib; only trusted locally created artifacts may be read.
"""
from collections.abc import Mapping
import hashlib
import json
import re


ARTIFACT_FIELDS = frozenset({
    'model', 'pipeline_version', 'pipeline_config', 'config_sha256',
    'feature_keys', 'source_hashes', 'image_versions', 'approval',
    'production_eligible', 'independent_validation',
})
CLIPPING_FLAGS = frozenset({
    'foreground_touches_frame', 'foreground_at_image_border',
    'capture_content_clipped', 'reference_touches_frame', 'reference_clipped',
})


def _canonical_config(config):
    if not isinstance(config, Mapping) or not config:
        raise ValueError('pipeline_config must be a nonempty mapping')
    try:
        return json.dumps(dict(config), sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError('pipeline_config must contain finite JSON values') from error


def _hashes(values, name):
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f'{name} must be a nonempty source-hash mapping')
    if any(not isinstance(k, str) or not k.strip() or
           not isinstance(v, str) or not re.fullmatch(r'[0-9a-f]{64}', v)
           for k, v in values.items()):
        raise ValueError(f'{name} must use named lowercase SHA-256 values')
    return dict(values)


def _versions(values, name):
    if not isinstance(values, Mapping) or not {'numpy', 'opencv'} <= set(values):
        raise ValueError(f'{name} must explicitly include numpy and opencv')
    if any(not isinstance(k, str) or not k.strip() or
           not isinstance(v, str) or not v.strip() for k, v in values.items()):
        raise ValueError(f'{name} must contain named nonempty version strings')
    return dict(values)


def validate_research_artifact(
    artifact, *, expected_version, expected_config, expected_feature_keys,
    expected_source_hashes, expected_image_versions,
):
    """Validate an exact research contract; return None or raise ValueError.

    This never invokes ``predict`` or grants permission to expose a public weight.
    The caller-supplied expected_config is the complete supported configuration,
    including interpolation choices. Extra fields, changed types/values, reordered
    features, obsolete sources, and dependency mismatches all fail closed.
    """
    if not isinstance(artifact, Mapping) or set(artifact) != ARTIFACT_FIELDS:
        raise ValueError('Unsupported or missing research artifact fields')
    if not isinstance(expected_version, str) or not expected_version.strip():
        raise ValueError('An explicit expected pipeline version is required')
    expected_json = _canonical_config(expected_config)
    actual_json = _canonical_config(artifact['pipeline_config'])
    if (expected_config.get('version') != expected_version or
            artifact['pipeline_version'] != expected_version or
            artifact['pipeline_config'].get('version') != expected_version):
        raise ValueError('Top-level or nested pipeline version mismatch')
    if actual_json != expected_json:
        raise ValueError('Unsupported preprocessing configuration')
    digest = hashlib.sha256(expected_json.encode('utf-8')).hexdigest()
    if artifact['config_sha256'] != digest:
        raise ValueError('Preprocessing configuration hash mismatch')
    if (not isinstance(expected_feature_keys, (list, tuple)) or
            not expected_feature_keys or
            any(not isinstance(k, str) or not k.strip() for k in expected_feature_keys) or
            len(set(expected_feature_keys)) != len(expected_feature_keys)):
        raise ValueError('Expected feature order must contain unique names')
    if (not isinstance(artifact['feature_keys'], list) or
            artifact['feature_keys'] != list(expected_feature_keys)):
        raise ValueError('Feature definition/order mismatch')
    if (_hashes(artifact['source_hashes'], 'source_hashes') !=
            _hashes(expected_source_hashes, 'expected_source_hashes')):
        raise ValueError('Preprocessing source hash mismatch')
    if (_versions(artifact['image_versions'], 'image_versions') !=
            _versions(expected_image_versions, 'expected_image_versions')):
        raise ValueError('Image dependency version mismatch')
    if (artifact['approval'] != 'EXPLORATORY_ONLY' or
            artifact['production_eligible'] is not False or
            artifact['independent_validation'] is not False):
        raise ValueError('Research artifacts cannot claim production approval or independent validation')
    if not callable(getattr(artifact['model'], 'predict', None)):
        raise ValueError('Research artifact must contain a predictor')


def _flags(flags):
    if isinstance(flags, (str, bytes)):
        raise ValueError('QC flags must be a collection of strings')
    try:
        result = list(flags)
    except TypeError as error:
        raise ValueError('QC flags must be a collection of strings') from error
    if any(not isinstance(flag, str) or not flag.strip() for flag in result):
        raise ValueError('QC flags must be nonempty strings')
    return set(result)


def propagate_capture_qc(original_flags, current_flags):
    """Retain source-capture warnings after padding, resizing or other derivatives.

    A clear derivative frame cannot recover content missing in its source capture.
    Unknown flags are retained too. Even an empty union is not an approval or a
    claim that an unvalidated weight model is usable.
    """
    original, current = _flags(original_flags), _flags(current_flags)
    reasons = []
    if original & CLIPPING_FLAGS:
        reasons.append('元画像で画面端への接触・欠けが疑われます。余白追加では失われた内容を復元できないため、対象全体が収まるよう再撮影してください。')
    return dict(qc_flags=sorted(original | current), capture_reasons=reasons,
                public_prediction_g=None, status='要確認')
