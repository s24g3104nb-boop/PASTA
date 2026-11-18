"""One-photo feature and count stress tests on a verified, private photo batch.

Cumulative material records are kept together. The experiment fits on known-
weight originals to inspect behavior and confounding, with no independent split
or deployable model artifact. Each transformation uses the same extractor.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform

import cv2
import numpy as np
from PIL import Image

from ..measurement_diagnostics import decide_count_weight
from ..pasta_vision import image_quality_warnings, image_to_rgb
from .bundle_texture import extract_bundle_texture
from .dense_weight_benchmark import interior_texture
from .diagnose_series import runtime_parameters
from .lineage_revision import verify_revision
from .photo_batch_diagnosis import (FEATURE_GROUPS, image_features, load_diagnostic_entries,
                                   regression_model_factories, write_csv)
from .spread_count_candidate import compare_spread_counts
from .stability_experiment import TRANSFORM_SETTINGS, transformations
from .stability_pipeline import PipelineConfig, clean_mask, normalize_size, raw_mask, shape_features

VERSION = 'weight-robustness-v1'
PREPROCESS = PipelineConfig('candidate_object', 960, 'hsv_lab', True, 'object4')
OBJECT_FEATURES = ('area_over_length_squared', 'width_over_length', 'quantile_density', 'profile_width_cv')
TEXTURE_FEATURES = ('interior_luminance_cv', 'interior_structure_coherence')
BACKGROUND_FEATURES = ('background_light_mean', 'background_light_sd', 'background_a_mean', 'background_b_mean')
GROUPS = {**FEATURE_GROUPS, 'object_geometry': OBJECT_FEATURES,
          'object_interior': (*OBJECT_FEATURES, *TEXTURE_FEATURES), 'background_control': BACKGROUND_FEATURES}
WEIGHT_TRANSFORMS = ('resize50', 'rotate90', 'rotate3', 'jpeg60', 'brightness90', 'brightness110', 'canvas_scale75')
COUNT_TRANSFORMS = ('original', 'resize50', 'rotate90', 'rotate3', 'jpeg60', 'brightness90', 'brightness110')


def feature_vectors(rgb):
    """Extract both legacy and cleaned object ratios. No label or scene ID enters."""
    normalized = normalize_size(rgb, PREPROCESS.target_long_side)
    candidate = raw_mask(normalized, PREPROCESS)
    baseline = image_features(normalized, candidate)
    cleaned = clean_mask(candidate, PREPROCESS)
    obj = shape_features(cleaned, PREPROCESS)
    obj.update(interior_texture(normalized, cleaned))
    # Exclude the whole original color candidate plus a margin from the control.
    # Do not expose foreground area or the count of background pixels to it.
    background = cv2.dilate(candidate, np.ones((9, 9), np.uint8)) == 0
    lab = cv2.cvtColor(normalized, cv2.COLOR_RGB2LAB)
    pixels = lab[background]
    if len(pixels):
        light, a, b = map(float, pixels.mean(axis=0))
        spread = float(pixels[:, 0].std())
    else:
        light = spread = a = b = 0.
    negative = dict(zip(BACKGROUND_FEATURES, (light, spread, a, b)))
    sources = dict(scene=baseline, shape_texture=baseline, object_geometry=obj,
                   object_interior=obj, background_control=negative)
    return {group: {name: float(sources[group][name]) for name in names} for group, names in GROUPS.items()}


def _fit_scope(rows, keys, model):
    train = rows['original']
    x = [[row['vectors'][key] for key in keys] for row in train]
    y = np.array([row['reported_g'] for row in train], dtype=float)
    model.fit(x, y)
    fitted = model.predict(x)
    errors = abs(fitted - y)
    training = dict(mae_g=float(errors.mean()), rmse_g=float(np.sqrt((errors**2).mean())),
                    max_absolute_error_g=float(errors.max()))
    training['by_nominal_weight'] = {f'{w:g}': dict(n=int((y == w).sum()), mae_g=float(errors[y == w].mean()),
                                                   mean_fitted_g=float(fitted[y == w].mean())) for w in np.unique(y)}
    indexed = {row['sha256']: float(estimate) for row, estimate in zip(train, fitted)}
    details = [dict(original_sha256=row['sha256'], reported_g=row['reported_g'], transform='original',
                    fitted_g=float(estimate), absolute_error_g=float(error), absolute_delta_g=0.)
               for row, estimate, error in zip(train, fitted, errors)]
    stress = {}
    for name, transformed in rows.items():
        if name == 'original':
            continue
        estimates = model.predict([[row['vectors'][key] for key in keys] for row in transformed])
        deltas = [abs(float(value) - indexed[row['sha256']]) for row, value in zip(transformed, estimates)]
        stress[name] = dict(pairs=len(deltas), max_absolute_delta_g=max(deltas), mean_absolute_delta_g=float(np.mean(deltas)))
        details.extend(dict(original_sha256=row['sha256'], reported_g=row['reported_g'], transform=name,
                            fitted_g=float(value), absolute_error_g=None, absolute_delta_g=delta)
                       for row, value, delta in zip(transformed, estimates, deltas))
    return training, stress, details


def compare_models(feature_rows):
    fits, predictions = [], []
    for group, keys in GROUPS.items():
        rows = {}
        for row in feature_rows:
            if row['reported_g'] is not None:
                rows.setdefault(row['transform'], []).append(dict(row, vectors=row['vectors'][group]))
        for name, create in regression_model_factories().items():
            training, stress, details = _fit_scope(rows, keys, create())
            fits.append(dict(feature_group=group, model=name, fitted_originals=len(rows['original']),
                             training=training, validation=None, independent_test=None, paired_stress=stress))
            predictions.extend(dict(row, feature_group=group, model=name) for row in details)
    return fits, predictions


def count_diagnostics(rgb, parameters, expected_count):
    baseline, lab = compare_spread_counts(rgb, **parameters)
    quality = image_quality_warnings(rgb)
    result = {}
    for method, detection in [('hsv', baseline), ('hsv_lab_edge_check', lab)]:
        # A positive test-only calibration checks the calculation boundary.
        # The check is reported as a boolean and releases no nominal grams.
        decision = decide_count_weight(dict(name='Research test fixture', g_per_strand=1.), detection, mode='spread')
        result[method] = dict(candidates=detection.count, error_strands=(abs(detection.count-expected_count)
                             if expected_count is not None else None), quality=quality,
                             image_gate_pass=decision.weight_g is not None and not quality,
                             reasons=decision.reason_codes, evidence=detection.evidence)
    return result, (baseline, lab)


def cumulative_feature_trends(feature_rows, revision):
    """Departure from a monotonic trend within one sequence of additions."""
    keys = {'scene': ('area_fraction', 'width_fraction'), 'object_geometry': OBJECT_FEATURES}
    staged = {stage['material_id']: [] for stage in revision['stages']}
    for row in feature_rows:
        if row['transform'] == 'original' and row['material_stage_id'] in staged:
            staged[row['material_stage_id']].append(row['vectors'])
    trends = []
    for group, names in keys.items():
        for name in names:
            means = []
            for stage in revision['stages']:
                values = [v[group][name] for v in staged[stage['material_id']]]
                means.append(dict(nominal_weight_g=stage['nominal_weight_g'], photos=len(values),
                                  mean=float(np.mean(values)), minimum=min(values), maximum=max(values)))
            trends.append(dict(feature_group=group, feature=name, stages=means,
                               increasing_steps=sum(b['mean'] > a['mean'] for a,b in zip(means, means[1:])),
                               physical_series=1))
    return trends


def run(datasets, revision_path, output):
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError('Use a new output directory')
    entries = load_diagnostic_entries(datasets, out)
    if not any(entry['reported_g'] is not None for entry in entries):
        raise ValueError('Supply a cohort with reported gram weights')
    revision_bytes = Path(revision_path).read_bytes()
    manifest_hash = json.loads(revision_bytes)['source_manifest_sha256']
    origins = {entry['dataset_root'] for entry in entries if entry['manifest_sha256'] == manifest_hash}
    if len(origins) != 1:
        raise ValueError('Exactly one supplied batch must match the cumulative revision')
    revision = verify_revision(revision_path, origins.pop())
    cumulative = {item['original_sha256']: stage['material_id'] for stage in revision['stages'] for item in stage['photos']}
    family = revision['physical_family_id']
    parameters = runtime_parameters()
    out.mkdir(parents=True)
    cv2.setNumThreads(1)
    features, counts, original_texture, image_rows = [], [], [], []
    for number, entry in enumerate(entries, start=1):
        rgb = image_to_rgb(entry['path'])
        digest = entry['sha256']
        original = dict(filename=entry['record']['filename'], sha256=digest,
                        reported_g=entry['reported_g'], reported_count=entry['reported_count'],
                        physical_family_id=family if digest in cumulative else None,
                        material_stage_id=cumulative.get(digest), eligible_for_independent_test=False)
        image_rows.append(original)
        for transform, pixels in transformations(rgb):
            if transform in ('original', *WEIGHT_TRANSFORMS):
                features.append(dict(original, transform=transform, vectors=feature_vectors(pixels)))
            if transform == 'original' or (entry['reported_count'] is not None and transform in COUNT_TRANSFORMS):
                result, detections = count_diagnostics(pixels, parameters['DEFAULT_SPREAD'], entry['reported_count'])
                counts.extend(dict(original, transform=transform, method=method, **details) for method, details in result.items())
                if transform == 'original' and entry['reported_count'] is not None:
                    for method, detection in zip(result, detections):
                        Image.fromarray(detection.overlay_rgb).save(out / f'count_{number:02d}_{method}.jpg')
            if transform == 'original' and digest in cumulative:
                normalized = normalize_size(pixels, PREPROCESS.target_long_side)
                mask = clean_mask(raw_mask(normalized, PREPROCESS), PREPROCESS)
                texture = extract_bundle_texture(normalized, mask)
                original_texture.append(dict(original, texture_status=texture['texture_status'], pitch_px=texture['pitch_px'],
                                             flags=texture['qc_flags']))
    fits, predictions = compare_models(features)
    for entry in entries:
        if hashlib.sha256(entry['path'].read_bytes()).hexdigest() != entry['sha256']:
            raise RuntimeError('Original bytes changed during the diagnostic')
    import sklearn
    report = dict(version=VERSION, originals=image_rows, verified_originals=len(image_rows),
                  confirmed_cumulative_series=1, cumulative_revision_sha256=hashlib.sha256(revision_bytes).hexdigest(),
                  cumulative_originals=len(cumulative), validation_attempts=0, independent_test_attempts=0,
                  independent_mae_g=None, independent_rmse_g=None, production_eligible=False,
                  preprocessing=asdict(PREPROCESS), features=GROUPS, transforms=TRANSFORM_SETTINGS,
                  detector_parameters=parameters['DEFAULT_SPREAD'],
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  versions=dict(python=platform.python_version(), opencv=cv2.__version__, numpy=np.__version__,
                                sklearn=sklearn.__version__), counts=counts, visible_periodic_texture=original_texture,
                  cumulative_feature_trends=cumulative_feature_trends(features, revision),
                  regression=fits)
    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    write_csv(out/'counts.csv', counts)
    write_csv(out/'features.csv', [dict(row, feature_group=group, vectors=vector)
                                  for row in features for group, vector in row['vectors'].items()])
    write_csv(out/'predictions.csv', predictions)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, action='append', required=True)
    parser.add_argument('--lineage-revision', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.dataset, args.lineage_revision, args.output)
    print(f"{result['verified_originals']} verified originals, {result['cumulative_originals']} cumulative views, "
          f"{len(result['regression'])} training-only comparisons")
