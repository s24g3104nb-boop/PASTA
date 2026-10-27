from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .dataset import dataset_hash, grouping, image_rgb, validate_splits

FEATURE_VERSION = 'rgb-hsv-yellow-grid-v1'
MODEL_VERSION = 'numpy-ridge-v1'


def features(rgb):
    rgb = np.asarray(Image.fromarray(rgb).resize((128, 128)))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    yellow = cv2.inRange(hsv, np.array([7, 20, 60]), np.array([45, 255, 255])) / 255
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    out = [yellow.mean(), (cv2.Canny(gray, 60, 150) > 0).mean()]
    out.extend(rgb.mean(axis=(0, 1)) / 255)
    out.extend(rgb.std(axis=(0, 1)) / 255)
    histogram = np.histogram(hsv[:, :, 0], bins=12, range=(0, 180))[0] / hsv.shape[0] / hsv.shape[1]
    out.extend(histogram)
    for y in range(0, 128, 32):
        for x in range(0, 128, 32):
            out.append(yellow[y:y + 32, x:x + 32].mean())
    return np.asarray(out, dtype=float)


def check_source(samples, allow_synthetic=False):
    sources = {s.values['source'] for s in samples}
    if len(sources) != 1:
        raise ValueError('realとsyntheticは別CSV・別モデルで扱ってください')
    if sources != {'real'} and not allow_synthetic:
        raise ValueError('syntheticはテスト専用です。検証時だけ --allow-synthetic を明示してください')
    return next(iter(sources))


def train(samples, output, alpha=1., allow_synthetic=False):
    validate_splits(samples)
    source = check_source(samples, allow_synthetic)
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('alphaは有限の正数です')
    output = Path(output)
    if output.exists():
        raise ValueError('モデルを上書きしません。新しいモデルバージョンのパスを指定してください')
    train_rows = [s for s in samples if s.values['split'] == 'train']
    entries = {}
    for pair in sorted({s.pair for s in train_rows}):
        subset = [s for s in train_rows if s.pair == pair]
        if len(grouping(subset)) < 3 or len({s.weight for s in subset}) < 2:
            raise ValueError(pair + ': 訓練には独立した3グループ以上と2重量以上が必要です')
        x = np.stack([features(image_rgb(s.path)) for s in subset])
        y = np.array([s.weight for s in subset])
        mean, scale = x.mean(axis=0), x.std(axis=0)
        scale[scale < 1e-8] = 1
        z = (x - mean) / scale
        y_mean = float(y.mean())
        coef = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y - y_mean))
        entries[pair] = dict(mean=mean.tolist(), scale=scale.tolist(), coef=coef.tolist(),
                             intercept=y_mean, min_weight=float(y.min()), max_weight=float(y.max()),
                             max_feature_distance=float(max(3., np.linalg.norm(z, axis=1).max() * 1.5)),
                             train_rows=len(subset), train_groups=len(grouping(subset)))
    if not entries:
        raise ValueError('訓練行がありません')
    model = dict(model_version=MODEL_VERSION, feature_version=FEATURE_VERSION,
                 trained_at=datetime.now(timezone.utc).isoformat(), source=source,
                 experimental=True, alpha=alpha, dataset_sha256=dataset_hash(samples),
                 training_data_hash=dataset_hash(train_rows), entries=entries,
                 note='実験用CPU回帰。未見製品・撮影条件・範囲外重量の精度は保証しません。')
    # Validation is reported; test labels/features do not train or tune the model.
    from .evaluate import evaluate
    model['validation_report'] = evaluate(samples, 'val', model=model, allow_synthetic=allow_synthetic)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as f:
        json.dump(model, f, ensure_ascii=False, indent=2, allow_nan=False)
    return model


def load_model(path):
    path = Path(path)
    if path.stat().st_size > 5_000_000:
        raise ValueError('モデルファイルが大きすぎます')
    def reject_constant(value):
        raise ValueError('モデルJSONに有限でない値があります: ' + value)

    model = json.loads(path.read_text(encoding='utf-8'), parse_constant=reject_constant)
    try:
        return _validate_model(model)
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError('モデルJSONの構造・必須フィールドが不正です') from exc


def _validate_model(model):
    if not isinstance(model, dict):
        raise ValueError('モデルJSONはオブジェクトが必要です')
    if model.get('model_version') != MODEL_VERSION or model.get('feature_version') != FEATURE_VERSION:
        raise ValueError('モデル形式・特徴量バージョンが一致しません')
    if (model.get('source') not in ('real', 'synthetic')
            or not isinstance(model.get('entries'), dict) or not model['entries']):
        raise ValueError('モデルメタデータが不正です')
    for name in ('dataset_sha256', 'training_data_hash'):
        digest = model[name]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('モデルのデータハッシュが不正です')
    for e in model['entries'].values():
        if not isinstance(e, dict):
            raise ValueError('モデル係数はオブジェクトが必要です')
        for field in ('mean', 'scale', 'coef'):
            array = np.asarray(e[field], dtype=float)
            if array.shape != (36,) or not np.all(np.isfinite(array)):
                raise ValueError('モデル係数が不正です')
        if np.any(np.asarray(e['scale']) <= 0):
            raise ValueError('モデルの正規化尺度が不正です')
        for field in ('intercept', 'min_weight', 'max_weight', 'max_feature_distance'):
            if not np.isfinite(e[field]) or e[field] <= 0:
                raise ValueError('モデル範囲が不正です')
        if e['min_weight'] > e['max_weight']:
            raise ValueError('モデル重量範囲が不正です')
    return model


def predict(model, rgb, mode, product_id):
    mode = 'bundle_end' if mode == 'bundle' else mode
    e = model['entries'].get(mode + ':' + product_id)
    if e is None:
        return dict(success=False, predicted_g=None, reason='未学習の撮影モード/製品です')
    x = (features(rgb) - np.array(e['mean'])) / np.array(e['scale'])
    if float(np.linalg.norm(x)) > e['max_feature_distance']:
        return dict(success=False, predicted_g=None, reason='訓練写真の特徴範囲外です。再撮影・追加データが必要です')
    weight = float(x @ np.array(e['coef']) + e['intercept'])
    if not np.isfinite(weight) or not e['min_weight'] <= weight <= e['max_weight']:
        return dict(success=False, predicted_g=None, reason='訓練重量の範囲外です')
    return dict(success=True, predicted_g=weight, reason='実験用回帰の推定値。実測ではありません',
                source=model['source'], model_version=model['model_version'])
