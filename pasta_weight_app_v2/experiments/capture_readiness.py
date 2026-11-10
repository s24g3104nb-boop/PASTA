"""Read-only, frozen-model diagnostics for planning independent captures.

No fitting, dataset approval, train/test assignment or production prediction.
Real masks remain unreviewed. Derived inputs are never independent samples.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import html
import json
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image

from .stability_pipeline import configurations, process
from .stability_experiment import pixel_sha

VERSION = 'capture-readiness-v1'
PIPELINES = ('B_normalized5', 'C_clean5', 'D_object4')
MODELS = ('ridge', 'random_forest', 'gradient_boosting')


def summarize_predictions(rows):
    """Describe frozen training residuals and paired perturbations, never test MAE."""
    grouped = {}
    for row in rows:
        if row['pipeline'] not in PIPELINES or row['model'] not in MODELS:
            continue
        key = (row['pipeline'], row['model'])
        grouped.setdefault(key, []).append(row)
    results = []
    for (pipeline, model), values in sorted(grouped.items()):
        indexed = {(r['image_id'], r['transform']): r for r in values}
        if len(indexed) != len(values):
            raise ValueError('Duplicate image/transform prediction')
        originals = {r['image_id']: r for r in values if r['transform'] == 'original'}
        if not originals:
            raise ValueError('Missing original predictions')
        errors, deltas = [], {}
        for r in values:
            source = originals.get(r['image_id'])
            if source is None or r['original_sha256'] != source['original_sha256']:
                raise ValueError('Missing or inconsistent original provenance')
            weight, pred = float(r['nominal_g']), float(r['research_prediction_g'])
            if not np.isfinite([weight, pred]).all() or weight <= 0:
                raise ValueError('Nonfinite prediction or invalid nominal label')
            if weight != float(source['nominal_g']):
                raise ValueError('A derivative changed the nominal label')
            if r['transform'] == 'original':
                errors.append(abs(pred - weight))
            else:
                deltas.setdefault(r['transform'], []).append(abs(pred - float(source['research_prediction_g'])))
        ordinary = [x for k, vals in deltas.items() if k != 'canvas_scale75' for x in vals]
        by_weight = {}
        for weight in sorted({float(r['nominal_g']) for r in originals.values()}):
            e = [abs(float(r['research_prediction_g']) - weight) for r in originals.values()
                 if float(r['nominal_g']) == weight]
            by_weight[str(weight)] = {'images': len(e), 'training_mae_g': float(np.mean(e))}
        results.append(dict(pipeline=pipeline, model=model, original_images=len(originals),
            training_mae_g=float(np.mean(errors)), training_rmse_g=float(np.sqrt(np.mean(np.square(errors)))),
            training_max_error_g=max(errors), training_by_weight=by_weight,
            ordinary_max_delta_g=max(ordinary) if ordinary else None,
            ordinary_p95_delta_g=float(np.percentile(ordinary, 95)) if ordinary else None,
            transform_max_delta_g={k: max(v) for k, v in sorted(deltas.items())},
            independent_mae_g=None, independent_samples=0, production_eligible=False))
    return results


def _contained(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Cache path outside source root')
    return path


def validate_comparison_inputs(rows, document):
    """Every compared model must cover the same frozen inputs and labels."""
    inputs = {(r['image_id'], r['transform']): r for r in document['inputs']}
    if len(inputs) != len(document['inputs']):
        raise ValueError('Duplicate frozen input')
    labels = {(r['image_id'], r['transform']): r for r in document['rows']
              if r['pipeline'] == 'B_normalized5'}
    if not inputs or set(labels) != set(inputs):
        raise ValueError('Incomplete input/feature provenance')
    for image_id, _ in inputs:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', image_id):
            raise ValueError('Unsafe image identifier')
    for pipeline in PIPELINES:
        for model in MODELS:
            values = [r for r in rows if r['pipeline'] == pipeline and r['model'] == model]
            observed = {(r['image_id'], r['transform']) for r in values}
            if observed != set(inputs) or len(values) != len(inputs):
                raise ValueError('Model comparisons require identical complete input sets')
            for row in values:
                key = row['image_id'], row['transform']
                if row['original_sha256'] != inputs[key]['original_sha256'] or float(row['nominal_g']) != labels[key]['reported_g']:
                    raise ValueError('Prediction provenance or label differs from source features')


def plot_summary(report, output):
    """Optional matplotlib export; plots frozen diagnostics without model fitting."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    selected = [r for r in report['comparisons'] if r['pipeline'] == 'B_normalized5']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    labels = [r['model'].replace('_', '\n') for r in selected]
    axes[0].bar(labels, [r['training_mae_g'] for r in selected], color='#5d7765')
    axes[0].set(title='Training residuals: 15 original photos', ylabel='MAE (g), NOT independent test MAE')
    x = np.arange(len(selected))
    axes[1].bar(x - .18, [r['ordinary_max_delta_g'] for r in selected], width=.36, color='#5d7765', label='Resize / JPEG / rotation / light')
    axes[1].bar(x + .18, [r['transform_max_delta_g']['canvas_scale75'] for r in selected], width=.36, color='#c58e72', label='Synthetic canvas-scale stress')
    axes[1].set(xticks=x, xticklabels=labels, title='Same-image prediction sensitivity', ylabel='Maximum paired change (g)')
    axes[1].legend(fontsize=7)
    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='y', alpha=.2)
    fig.suptitle('Frozen B5 comparison - no new fitting, independent accuracy unknown')
    fig.tight_layout()
    fig.savefig(output / 'model_comparison.png', dpi=150)
    fig.savefig(output / 'model_comparison.pdf'); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for axis, key, title in zip(axes, ['mask_fraction', 'quantile_width_relative', 'quantile_density'],
                              ['Foreground / image area', 'Projected width / image long edge', 'Projected mask density']):
        for row in report['images']:
            axis.scatter(row['nominal_g'], row['features'][key], color='#5d7765', s=45, alpha=.75)
        axis.set(title=title, xlabel='Nominal weight (g)', xticks=[20,40,60,80,100])
        axis.spines[['top', 'right']].set_visible(False); axis.grid(alpha=.2)
    fig.suptitle('Three views per weight - shared/unknown physical lineage, unapproved masks')
    fig.tight_layout(); fig.savefig(output / 'feature_ranges.png', dpi=150)
    fig.savefig(output / 'feature_ranges.pdf'); plt.close(fig)


def run(source, output):
    """Recheck 15 frozen inputs, marker absence, and unchanged B features."""
    from .marker_reference import analyze_marker_reference
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new empty output directory; never overwrite previous results')
    feature_path = source / 'extraction/features.json'
    prediction_path = source / 'regression/predictions.csv'
    document = json.loads(feature_path.read_text())
    if document['versions']['numpy'] != np.__version__ or document['versions']['opencv'] != cv2.__version__:
        raise ValueError('Use the original NumPy/OpenCV versions for exact feature replay')
    config = next(c for c in configurations() if c.name == 'B_normalized5')
    if asdict(config) != next(c for c in document['configuration'] if c['name'] == config.name):
        raise ValueError('Frozen pipeline configuration differs')
    with prediction_path.open(newline='') as f:
        predictions = list(csv.DictReader(f))
    validate_comparison_inputs(predictions, document)
    comparisons = summarize_predictions(predictions)
    output.mkdir(parents=True, exist_ok=True)
    images = output / 'images'; images.mkdir()
    rows = []
    source_rows = {(r['image_id'], r['transform']): r for r in document['rows'] if r['pipeline'] == config.name}
    keys = document['feature_keys'][config.name]
    for entry in document['inputs']:
        if entry['transform'] != 'original':
            continue
        image = np.asarray(Image.open(_contained(source / 'extraction', entry['path'])).convert('RGB'))
        if pixel_sha(image) != entry['pixel_sha256']:
            raise ValueError('Frozen RGB cache changed')
        features, scene, mask = process(image, config)
        old = source_rows[entry['image_id'], 'original']
        if pixel_sha(mask) != old['mask_sha256']:
            raise ValueError('Frozen original mask did not reproduce exactly')
        if any(features[k] != old[k] for k in keys):
            raise ValueError('Frozen original features did not reproduce exactly')
        metadata, _, overlay = analyze_marker_reference(image)
        image_id = entry['image_id']
        preview = Image.fromarray(overlay); preview.thumbnail((640, 640))
        preview.save(images / f'{image_id}_reference.jpg')
        Image.fromarray(mask).save(images / f'{image_id}_mask.png')
        source_preview = Image.fromarray(scene); source_preview.thumbnail((640, 640))
        source_preview.save(images / f'{image_id}_input.jpg')
        rows.append(dict(image_id=image_id, nominal_g=old['reported_g'],
            original_sha256=entry['original_sha256'], capture_group=entry['capture_group'],
            lineage_block=entry['lineage_block'], features={k: features[k] for k in keys},
            qc_flags=old['qc_flags'], mask_approved=False, marker_reference=metadata,
            actual_noodle_length_mm=None, verified_physical_reference=False))
    spans = []
    for weight in sorted({r['nominal_g'] for r in rows}):
        same = [r for r in rows if r['nominal_g'] == weight]
        for key in keys:
            vals = [r['features'][key] for r in same]
            spans.append(dict(nominal_g=weight, feature=key, minimum=min(vals), maximum=max(vals),
                              range=max(vals)-min(vals), same_weight_images=len(vals)))
    report = dict(version=VERSION, original_images=len(rows),
        independent_test_samples=0, fitted_new_models=0, production_eligible=False,
        source_hashes={str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (feature_path, prediction_path)},
        versions={'numpy':np.__version__, 'opencv':cv2.__version__},
        comparisons=comparisons, within_weight_feature_spans=spans, images=rows,
        limitations=['Frozen training residuals only; labels and lineage not approved.',
                     'Digital canvas scaling is not a real camera-distance experiment.',
                     'No visible reference with measured dimensions or verified noodle length.',
                     'Mask density is projected occupancy, not hidden strand mass.'])
    (output / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    with (output / 'feature_spans.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(spans[0])); writer.writeheader(); writer.writerows(spans)
    cards = []
    for row in rows:
        name = html.escape(row['image_id'])
        cards.append(f'<section><h2>{name} · {row["nominal_g"]}g（申告ラベル）</h2>'
            f'<img alt="元入力" src="images/{name}_input.jpg">'
            f'<img alt="未承認マスク" src="images/{name}_mask.png">'
            f'<img alt="マーカー検査" src="images/{name}_reference.jpg">'
            f'<pre>{html.escape(json.dumps(row, ensure_ascii=False, indent=2))}</pre></section>')
    (output / 'index.html').write_text('<!doctype html><html lang="ja"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>撮影条件の診断</title>'
        '<style>body{background:#faf8f2;color:#29352e;font:16px sans-serif;max-width:1200px;margin:24px auto;padding:12px}section{background:#fffdf9;padding:18px;margin:18px 0;border:1px solid #e6e8df}img{width:31%;vertical-align:top}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}</style>'
        '<h1>既存15原本の再現と基準物の確認</h1><p>非公開研究用。マスク未承認・独立評価なし。'
        '左：入力、中：HSV＋Labマスク、右：印刷マーカー検査。寸法・重量を補完していません。</p>'
        '<p><a href="results.json">凍結モデル比較と全診断JSON</a> / <a href="feature_spans.csv">同重量内の特徴変動CSV</a></p>'
        + ''.join(cards) + '</html>')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); report = run(args.source, args.output)
    print(json.dumps({k:report[k] for k in ('original_images','independent_test_samples','fitted_new_models')}))
