"""Evaluate one fixed Ridge recipe using a separately reviewed physical split."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .foundation_backends import sha256_file
from .foundation_benchmark import ALPHA, _write_json
from .lineage_protocol import validate_split
from .weight_evaluation import independent_evaluation, validation_evaluation


def evaluate(rows, plan, partition):
    records, assignments, parents = (plan[key] for key in ('records', 'assignments', 'material_parents'))
    method = plan['feature_variant']
    if not isinstance(method, str) or not method:
        raise ValueError('Select one feature variant before evaluation')
    reviewed = validate_split(records, assignments, parents)
    vectors = {row['original_sha256']: row for row in rows}
    originals = {rec['original_sha256'].lower() for rec in records}
    if len(vectors) != len(rows) or len(originals) != len(records) or set(vectors) != originals:
        raise ValueError('Each reviewed original requires exactly one matching feature row')
    if partition == 'test' and plan.get('frozen_before_test') is not True:
        raise ValueError('Freeze the feature variant and protocol before opening the test split')
    train, targets, trials = [], [], []
    training_originals = []
    for rec in records:
        digest = rec['original_sha256'].lower()
        row = vectors[digest]
        chosen = assignments[rec['image_id']]
        if row.get('physical_family_id') and chosen != 'train':
            raise ValueError('Existing cumulative diagnostic photos are ineligible for held-out evaluation')
        if rec.get('weight_label_verified') is not True or rec['measured_g'] != row['measured_g']:
            raise ValueError('Reviewed label must match the recorded original label')
        features = row['features'].get(method)
        if features is not None:
            features = np.asarray(features, float)
            if features.ndim != 1 or not features.size or not np.isfinite(features).all():
                raise ValueError('Malformed feature vector')
        if chosen == 'train':
            # Use the entire declared training roster for provenance, including
            # missing vectors, and report how many actually enter the fit.
            training_originals.append(digest)
            if features is not None:
                train.append(features)
                targets.append(rec['measured_g'])
        elif chosen == partition:
            trials.append((rec['image_id'], features))
    if len(train) < 2:
        raise ValueError('Fitting Ridge requires at least two valid training feature vectors')
    model = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA)).fit(train, targets)
    predictions = []
    for image_id, features in trials:
        value = float(model.predict([features])[0]) if features is not None else None
        if value is None or not np.isfinite(value) or value <= 0:
            predictions.append(dict(image_id=image_id, predicted_g=None, reason='features_or_prediction_unavailable'))
        else:
            predictions.append(dict(image_id=image_id, predicted_g=value))
    model_card = dict(training_original_sha256=sorted(training_originals),
                      frozen_before_test=plan.get('frozen_before_test') is True,
                      split_manifest_sha256=reviewed['split_manifest_sha256'])
    evaluator = independent_evaluation if partition == 'test' else validation_evaluation
    report = evaluator(records, assignments, parents, predictions, model_card=model_card,
                       target_weights_g=plan['target_weights_g'])
    return dict(report, feature_variant=method, ridge_alpha=ALPHA,
                training_feature_vectors=len(train), predictions=predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', required=True, type=Path)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--partition', required=True, choices=('validation', 'test'))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        rows = json.loads(args.features.read_text(encoding='utf-8'))
        plan = json.loads(args.plan.read_text(encoding='utf-8-sig'))
        report = evaluate(rows, plan, args.partition)
        _write_json(args.output, dict(report, features_sha256=sha256_file(args.features),
                                    evaluation_plan_sha256=sha256_file(args.plan)))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
