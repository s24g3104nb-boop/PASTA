"""Same fixed model families, paired derivative sensitivity, no independent split."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import sklearn
from .direct_models import models, metrics


def csv_write(path, records):
    keys = list(records[0])
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader(); writer.writerows(records)


def run(features_path, previous, output):
    features_path, previous, output = Path(features_path), Path(previous), Path(output)
    if output.exists(): raise FileExistsError('No overwrite')
    data = json.loads(features_path.read_text())
    if data['independent_test_images'] != 0 or data['production_approved']:
        raise ValueError('Unexpected data approval/split')
    old = json.loads((previous/'models_experiment/models_report.json').read_text())
    output.mkdir(parents=True); (output/'models').mkdir()
    results, predictions, sensitivity, feature_stability, contributions = [], [], [], [], []
    baseline_check = []
    for config in data['configuration']:
        pipeline = config['name']; keys = data['feature_keys'][pipeline]
        rows = [r for r in data['rows'] if r['pipeline'] == pipeline]
        originals = [r for r in rows if r['transform'] == 'original']
        if len(originals) != 15 or len({r['original_sha256'] for r in originals}) != 15 or len({r['capture_group'] for r in originals}) != 5:
            raise ValueError('Expected 15 unique originals, 5 dependent capture groups')
        if len({r['lineage_block'] for r in rows}) != 1 or any(r['mask_approved'] for r in rows):
            raise ValueError('Unresolved lineage and unapproved masks must be retained')
        ref = {r['image_id']: r for r in originals}
        X = np.array([[r[k] for k in keys] for r in originals]); y = np.array([r['reported_g'] for r in originals])
        all_X = np.array([[r[k] for k in keys] for r in rows])
        if not np.isfinite(all_X).all(): raise ValueError('Non-finite features')
        for key in keys:
            scale = max(float(np.std([r[key] for r in originals])), 1e-12)
            for r in rows:
                if r['transform'] == 'original': continue
                delta = r[key] - ref[r['image_id']][key]
                feature_stability.append(dict(pipeline=pipeline, feature=key, image_id=r['image_id'], transform=r['transform'],
                                              original_value=ref[r['image_id']][key], transformed_value=r[key], delta=delta,
                                              delta_over_original_image_std=delta/scale))
        for name, estimator in models().items():
            estimator.fit(X, y)
            fitted = estimator.predict(X)
            all_pred = estimator.predict(all_X)
            reference_prediction = {r['image_id']: float(v) for r, v in zip(originals, fitted)}
            ident = pipeline+'__'+name
            artifact = dict(model=estimator, pipeline_version=data['version'], pipeline_config=config,
                            config_sha256=originals[0]['config_sha256'], feature_keys=keys,
                            pipeline_source_sha256=data['source_hashes']['stability_pipeline.py'],
                            production_eligible=False, independent_validation=False, approval='EXPLORATORY_ONLY')
            joblib.dump(artifact, output/'models'/(ident+'.joblib'))
            for row, pred in zip(rows, all_pred):
                predictions.append(dict(id=ident, pipeline=pipeline, model=name, image_id=row['image_id'], transform=row['transform'],
                                        nominal_g=row['reported_g'], research_prediction_g=float(pred),
                                        delta_from_original_g=float(pred-reference_prediction[row['image_id']]),
                                        absolute_nominal_error_g=float(abs(pred-row['reported_g'])),
                                        original_sha256=row['original_sha256'], capture_group=row['capture_group'],
                                        public_prediction_g=None, independent_evaluation=False))
            for transform in sorted({r['transform'] for r in rows} - {'original'}):
                selected = [r for r in predictions if r['id'] == ident and r['transform'] == transform]
                delta = np.array([abs(r['delta_from_original_g']) for r in selected])
                sensitivity.append(dict(id=ident, pipeline=pipeline, model=name, transform=transform, n_derivatives=len(delta),
                                        mean_absolute_change_g=float(delta.mean()), p95_absolute_change_g=float(np.percentile(delta, 95)),
                                        max_absolute_change_g=float(delta.max()), change_over_5g=int(np.sum(delta > 5))))
            changed = [s for s in sensitivity if s['id'] == ident and s['transform'] != 'canvas_scale75']
            results.append(dict(id=ident, pipeline=pipeline, model=name, n_originals=15, n_features=len(keys),
                                design_rank=int(np.linalg.matrix_rank(np.column_stack([np.ones(len(X)), X]))),
                                training_metrics=metrics(y, fitted), independent_metrics=None,
                                per_nominal_weight={str(w):metrics(y[y == w], fitted[y == w]) for w in sorted(set(y))},
                                max_standard_transform_change_g=max(s['max_absolute_change_g'] for s in changed),
                                parameters={k:v for k,v in estimator.get_params(deep=True).items() if v is None or isinstance(v, (str,int,float,bool))}))
            if pipeline in ('A_hsv_legacy12', 'A_lab_legacy12'):
                old_id = ('hsv_baseline' if pipeline == 'A_hsv_legacy12' else 'hsv_lab')+'__shape12__'+name
                old_values = {r['image_id']:r['training_prediction_g'] for r in old['predictions'] if r['id'] == old_id}
                diff = max(abs(reference_prediction[k]-old_values[k]) for k in old_values)
                baseline_check.append(dict(id=ident, max_old_prediction_difference_g=diff))
                if diff > 1e-7: raise ValueError('Baseline fit changed')
            if name == 'linear':
                scaled_ref = {r['image_id']:v for r,v in zip(originals, estimator[0].transform(X))}
                for r, v in zip(rows, estimator[0].transform(all_X)):
                    if r['transform'] == 'original': continue
                    parts = (v-scaled_ref[r['image_id']])*estimator[1].coef_
                    for key, value in zip(keys, parts):
                        contributions.append(dict(id=ident, image_id=r['image_id'], transform=r['transform'], feature=key, delta_contribution_g=float(value)))
        print(pipeline, 'fitted', flush=True)
    report = dict(source_originals=15, capture_groups=5, unresolved_lineage_blocks=1, independent_test_images=0,
                  production_approved=False, validation_metrics=None, split='NONE', tuning='NONE',
                  version=data['version'], feature_keys=data['feature_keys'],
                  input_sha256=hashlib.sha256(features_path.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  versions=dict(python=platform.python_version(), sklearn=sklearn.__version__, numpy=np.__version__, joblib=joblib.__version__),
                  fits=results, sensitivity=sensitivity, baseline_prediction_reproduction=baseline_check)
    (output/'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    for filename, records in [('predictions.csv',predictions),('sensitivity.csv',sensitivity),('feature_stability.csv',feature_stability),('linear_contributions.csv',contributions)]:
        csv_write(output/filename, records)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--features',required=True); parser.add_argument('--previous',required=True); parser.add_argument('--output',required=True)
    args = parser.parse_args(); run(args.features, args.previous, args.output)
