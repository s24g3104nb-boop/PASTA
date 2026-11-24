"""Fit and compare the sheetless weight candidates on local, SHA-verified originals."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
from io import BytesIO
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..pasta_vision import image_quality_warnings, image_to_rgb
from ..sheetless_area import _load_model, canonical_image, describe_mask, estimate_checked, estimate_observation
from .sheetless_lengths import measure


# B and C use one projected length equivalent. D retains A's feature and adds
# both length equivalents and relative strand width. A stays the shipped model.
COLUMNS = {'B': (1,), 'C': (2,), 'D': (0, 1, 2, 3)}
RIDGE_ALPHA = 1.
EXPERIMENT = 'sheetless-projected-lengths-v1'


@dataclass(frozen=True)
class Regression:
    coefficients: tuple[float, ...]
    intercept_g: float
    min_features: tuple[float, ...]
    max_features: tuple[float, ...]

    def grams(self, values):
        return float(np.dot(values, self.coefficients) + self.intercept_g)


def fit(method, vectors, measured_g):
    """Fit B and C proportional to visible length. Fit D by standardized Ridge."""
    x = np.asarray(vectors, dtype=float)
    y = np.asarray(measured_g, dtype=float)
    if (y.ndim != 1 or len(y) < 3 or x.shape != (len(y), len(COLUMNS['D']))
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or np.min(x) <= 0 or np.min(y) <= 0 or np.ptp(y) == 0):
        raise ValueError('At least three finite geometry observations spanning positive measured weights are required')
    x = x[:, COLUMNS[method]]
    if method == 'D':
        center, scale = x.mean(axis=0), x.std(axis=0)
        scale = np.where(scale > 0, scale, 1)
        normalized = (x - center) / scale
        matrix = normalized.T @ normalized + RIDGE_ALPHA * np.eye(x.shape[1])
        coef = np.linalg.solve(matrix, normalized.T @ (y - y.mean())) / scale
        intercept = y.mean() - center @ coef
    else:
        # A positive scale gives 0g at zero visible strand length.
        coef = (x.T @ y) / np.sum(x ** 2)
        intercept = 0.
    return Regression(tuple(coef), float(intercept), tuple(x.min(axis=0)), tuple(x.max(axis=0)))


def report_weight(model, method, obs, *, warnings=()):
    if warnings or obs.capture_reason or obs.scale_reason:
        return None, ' '.join(warnings) or obs.capture_reason or obs.scale_reason
    agreement = obs.skeleton_length_px / obs.area_length_px
    if not .65 <= agreement <= 1.35:
        return None, '細線化と麺幅から求めた線長が大きく異なる'
    vector = obs.vector()
    values = tuple(vector[col] for col in COLUMNS[method])
    g = model.grams(values)
    if not all(lo - 1e-12 <= v <= hi + 1e-12 for lo, v, hi in zip(model.min_features, values, model.max_features)):
        return None, '換算に使った特徴量の範囲外'
    if not 0 < g <= 200:
        return None, '換算値の範囲外'
    return g, ''


def error_metrics(errors):
    if not errors:
        return None
    absolute = np.abs(errors)
    return dict(mae_g=float(absolute.mean()), rmse_g=float(np.sqrt(np.mean(absolute ** 2))),
                max_abs_g=float(absolute.max()), images=len(errors))


def _variants(rgb):
    height, width = rgb.shape[:2]
    smaller = cv2.resize(rgb, (round(.75 * width), round(.75 * height)), interpolation=cv2.INTER_AREA)
    canvas = np.full_like(rgb, 32)
    y, x = (height - smaller.shape[0]) // 2, (width - smaller.shape[1]) // 2
    canvas[y:y + smaller.shape[0], x:x + smaller.shape[1]] = smaller
    return {'half_pixels': cv2.resize(rgb, (width // 2, height // 2), interpolation=cv2.INTER_AREA),
            'rotate90': np.rot90(rgb).copy(), 'smaller_in_frame': canvas,
            'dimmer': cv2.convertScaleAbs(rgb, alpha=.87)}


def _load_verified(entry):
    content = Path(entry['path']).read_bytes()
    if hashlib.sha256(content).hexdigest() != entry['sha256']:
        raise ValueError(f"Original SHA mismatch: {entry['file']}")
    return image_to_rgb(BytesIO(content))


def compare(entries, mobile_masks=None):
    """All original images are inspected, while only eligible measured rows are fitted."""
    unique = {entry['sha256'] for entry in entries}
    if not entries or len(unique) != len(entries):
        raise ValueError('Every original must occur exactly once')
    images = [_load_verified(entry) for entry in entries]
    observations = [measure(rgb) for rgb in images]
    old = [estimate_checked(rgb) for rgb in images]
    quality = [image_quality_warnings(rgb) for rgb in images]
    # Use the same four supported measured originals for the comparisons.
    reference = [i for i, e in enumerate(entries) if e['label'] == 'user_reported_measured'
                 and old[i].weight_g is not None and not quality[i]]
    fit_indices = [i for i in reference if observations[i].vector() is not None]
    vectors = [observations[i].vector() for i in fit_indices]
    weights = [entries[i]['weight_g'] for i in fit_indices]
    models = {method: fit(method, vectors, weights) for method in COLUMNS}
    rows = []
    fit_errors = {m: [] for m in 'ABCDE'}
    displayed_errors = {m: [] for m in 'ABCDE'}
    variation = {m: {} for m in 'ABCD'}
    model_a = _load_model()
    for i, (entry, rgb, obs, v2, problems) in enumerate(zip(entries, images, observations, old, quality)):
        values = obs.vector()
        outcome = {'A': dict(estimated_g=None if problems else v2.weight_g,
                             reason=' '.join(problems) or v2.reason)}
        for method, model in models.items():
            g, reason = report_weight(model, method, obs, warnings=problems)
            outcome[method] = dict(estimated_g=g, reason=reason)
        if mobile_masks is not None:
            with Image.open(mobile_masks / (entry['sha256'] + '.png')) as image:
                alt = describe_mask(canonical_image(rgb), np.asarray(image) > 0)
            val = alt.area_over_length_squared
            prediction = estimate_observation(alt)
            reason = ' '.join(problems) or prediction.reason
            outcome['E'] = dict(estimated_g=None if reason else prediction.weight_g, reason=reason,
                                mobile_area_ratio=val, diagnostic_g=model_a.predict(val))
        if i in fit_indices:
            for method, model in models.items():
                fitted = model.grams(tuple(values[col] for col in COLUMNS[method]))
                outcome[method]['fit_g'] = fitted
                fit_errors[method].append(fitted - entry['weight_g'])
            fit_errors['A'].append(v2.weight_g - entry['weight_g'])
            if 'E' in outcome:
                fit_errors['E'].append(outcome['E']['diagnostic_g'] - entry['weight_g'])
            for name, modified in _variants(rgb).items():
                other = measure(modified)
                other_values = other.vector()
                old_variant = estimate_checked(modified)
                variation['A'].setdefault(name, []).append(
                    [abs(model_a.predict(other.area_ratio)-v2.weight_g), old_variant.weight_g is not None])
                for method, model in models.items():
                    variant_g, _ = report_weight(model, method, other)
                    delta = None if other_values is None else abs(
                        model.grams(tuple(other_values[col] for col in COLUMNS[method])) -
                        model.grams(tuple(values[col] for col in COLUMNS[method])))
                    variation[method].setdefault(name, []).append([delta, variant_g is not None])
        if entry['label'] in ('user_reported_measured', 'weighed_bundle_calibration'):
            for method, result in outcome.items():
                if result['estimated_g'] is not None:
                    displayed_errors[method].append(result['estimated_g'] - entry['weight_g'])
        rows.append(dict(name=entry['file'], sha256=entry['sha256'], label=entry['label'],
                         weight_g=entry['weight_g'], count=entry.get('strands'), series=entry['series'],
                         fit=i in fit_indices, independent_test=False,
                         observations=asdict(obs), vector=values, outcomes=outcome))
    return dict(version=EXPERIMENT, originals=len(rows), fit_images=len(fit_indices), independent_images=0,
                metrics={k: error_metrics(v) for k, v in fit_errors.items()},
                displayed_metrics={k: error_metrics(v) for k, v in displayed_errors.items()},
                variants=variation, rows=rows,
                regressions={method: asdict(model) for method, model in models.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    batch = commands.add_parser('compare')
    batch.add_argument('--receipt', type=Path, required=True, help='Original byte hashes, labels, and provenance')
    batch.add_argument('--mobile-masks', type=Path, help='MobileSAM masks named by original SHA-256')
    batch.add_argument('--output', type=Path, required=True)
    photo = commands.add_parser('predict')
    photo.add_argument('--report', type=Path, required=True)
    photo.add_argument('--method', choices=COLUMNS, required=True)
    photo.add_argument('--image', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.command == 'compare':
        entries = json.loads(args.receipt.read_text(encoding='utf-8-sig'))['entries']
        report = compare(entries, args.mobile_masks)
        with args.output.open('x', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    else:
        saved = json.loads(args.report.read_text(encoding='utf-8'))
        if saved['version'] != EXPERIMENT:
            parser.error('Wrong report version')
        rgb = image_to_rgb(args.image)
        model = Regression(**saved['regressions'][args.method])
        weight, reason = report_weight(model, args.method, measure(rgb), warnings=image_quality_warnings(rgb))
        print(json.dumps(dict(estimated_g=weight, reason=reason), ensure_ascii=False))


if __name__ == '__main__':
    main()
