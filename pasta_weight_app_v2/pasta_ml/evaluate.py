from __future__ import annotations

import csv

import numpy as np

from ..pasta_vision import detect_bundle_ends, detect_spread
from .dataset import dataset_hash, image_rgb, validate_splits


def load_calibrations(path):
    calibrations = {}
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not {'product_id', 'g_per_strand'} <= set(reader.fieldnames or []):
            raise ValueError('校正CSVにはproduct_id,g_per_strandが必要です')
        for row in reader:
            product = row['product_id'].strip()
            weight = float(row['g_per_strand'])
            if not product or not np.isfinite(weight) or weight <= 0 or product in calibrations:
                raise ValueError('校正値は製品ごとに一意の有限正数を入力してください')
            calibrations[product] = weight
    return calibrations


def summary(rows):
    ok = [r for r in rows if r['success']]
    result = dict(total=len(rows), succeeded=len(ok), failed=len(rows) - len(ok),
                  success_rate=len(ok) / len(rows) if rows else None,
                  mae_g=None, rmse_g=None, mape_percent=None, median_absolute_error_g=None, bias_g=None)
    if ok:
        errors = np.array([r['predicted_g'] - r['actual_weight_g'] for r in ok])
        actual = np.array([r['actual_weight_g'] for r in ok])
        result.update(mae_g=float(np.abs(errors).mean()), rmse_g=float(np.sqrt((errors ** 2).mean())),
                      mape_percent=float((np.abs(errors) / actual).mean() * 100),
                      median_absolute_error_g=float(np.median(np.abs(errors))), bias_g=float(errors.mean()))
    return result


def evaluate(samples, split, *, model=None, calibrations=None, allow_synthetic=False):
    from .model import check_source, predict
    validate_splits(samples)
    source = check_source(samples, allow_synthetic)
    if split not in ('val', 'test'):
        raise ValueError('評価対象はvalまたはtestです')
    if model is not None and (model['dataset_sha256'] != dataset_hash(samples) or model['source'] != source):
        raise ValueError('モデルに記録したデータ/分割と一致しません。別データで評価する場合は別の実験を作成してください')
    rows = []
    for s in samples:
        if s.values['split'] != split:
            continue
        rgb = image_rgb(s.path)
        if model is not None:
            prediction = predict(model, rgb, s.values['mode'], s.values['product_id'])
        elif s.values['mode'] == 'bundle_side':
            prediction = dict(success=False, predicted_g=None,
                              reason='束の側面・上面はOpenCV断面検出の適用外です。対応する回帰モデルと実測写真で評価してください')
        elif s.values['product_id'] not in (calibrations or {}):
            prediction = dict(success=False, predicted_g=None, reason='製品の1本重量の校正値がありません')
        else:
            result = (detect_spread(rgb) if s.values['mode'] == 'spread' else
                      detect_bundle_ends(rgb, min_radius=8, max_radius=27, min_distance=32, hough_sensitivity=18))
            prediction = dict(success=result.count > 0,
                              predicted_g=result.count * calibrations[s.values['product_id']] if result.count else None,
                              reason='OpenCV本数検出。精度未保証' if result.count else '本数0: 再撮影が必要です')
        rows.append(dict(image_path=s.values['image_path'], actual_weight_g=s.weight,
                         mode=s.values['mode'], product_id=s.values['product_id'],
                         lighting=s.values.get('lighting', ''), background=s.values.get('background', ''),
                         **prediction))
    grouped = {}
    for field in ('mode', 'product_id', 'actual_weight_g', 'lighting', 'background'):
        grouped[field] = {str(key): summary([r for r in rows if r[field] == key]) for key in sorted({r[field] for r in rows})}
    return dict(split=split, source=source, estimator='numpy_ridge' if model else 'opencv_count',
                dataset_sha256=dataset_hash(samples), overall=summary(rows), by=grouped, rows=rows,
                metric_scope='誤差は推定成功画像のみ。失敗数・成功率も必ず併記。syntheticの結果は実物精度ではありません。')
