"""Fit a one-feature area calibration from verified, user-weighed JPEG originals."""

import argparse
from dataclasses import asdict
import hashlib
from io import BytesIO
import json
from pathlib import Path

import cv2
import numpy as np

from tools.diagnose_photo_manifest import manifest_records, photo_spec
from ..pasta_vision import image_quality_warnings, image_to_rgb
from ..sheetless_area import AREA_FEATURES, ESTIMATOR, SEGMENTATION, AreaModel, observe


def fit_line(areas, weights, feature):
    x, y = np.asarray(areas, dtype=float), np.asarray(weights, dtype=float)
    if (x.ndim != 1 or x.shape != y.shape or len(x) < 3 or not np.isfinite(x).all()
            or not np.isfinite(y).all() or min(y) <= 0 or np.ptp(x) <= 0 or np.ptp(y) <= 0):
        raise ValueError('Need at least three paired images spanning area and measured weight')
    slope, intercept = np.polyfit(x, y, 1)
    model = AreaModel(feature, float(slope), float(intercept), float(x.min()), float(x.max()))
    errors = np.abs(model.predict(x) - y)
    return model, {
        'training_mae_g': float(errors.mean()),
        'training_rmse_g': float(np.sqrt(np.mean(errors ** 2))),
        'training_max_error_g': float(errors.max()),
    }


def fit_reference(originals, feature):
    root = Path(originals).resolve()
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8-sig'))
    if 'photos' not in manifest:
        raise ValueError('Provide original photos with explicit measured-weight labels')
    areas, weights, seen = [], [], set()
    for record in manifest_records(manifest):
        relative, weight, _ = photo_spec(record)
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Image is outside the original dataset')
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != record['sha256'] or digest in seen:
            raise ValueError('Original hash mismatch or duplicate original')
        seen.add(digest)
        label = record.get('label', {})
        if label.get('kind') != 'measured_weight_g' or label.get('confidence') != 'user_stated':
            continue
        rgb = image_to_rgb(BytesIO(content))
        obs = observe(rgb)
        if not obs.reason and not image_quality_warnings(rgb):
            areas.append(getattr(obs, feature))
            weights.append(weight)
    model, metrics = fit_line(areas, weights, feature)
    return {
        'version': ESTIMATOR,
        'segmentation': SEGMENTATION,
        'status': 'exploratory_unvalidated',
        'model': asdict(model),
        'training_images': len(weights),
        'training_weight_range_g': [min(weights), max(weights)],
        'training_metrics': metrics,
        'independent_evaluation': None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--originals', type=Path, required=True)
    parser.add_argument('--feature', choices=AREA_FEATURES, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.originals.resolve()):
        parser.error('Keep output separate from originals')
    cv2.setNumThreads(1)
    try:
        document = fit_reference(args.originals, args.feature)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(document, stream, indent=2, allow_nan=False)
            stream.write('\n')
    except (KeyError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
