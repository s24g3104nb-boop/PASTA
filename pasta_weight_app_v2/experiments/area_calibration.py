"""Versioned, product-specific area conversion from real paired observations."""
import argparse
import json
from pathlib import Path
import re

import numpy as np

SCHEMA = 'pasta-a4-area-linear-v1'
MAX_VALIDATION_MAE_G = 5.
MAX_VALIDATION_ERROR_G = 10.


def _positive(value):
    if type(value) not in (float, int) or not np.isfinite(value) or value <= 0:
        raise ValueError('Area and measured weight must be positive finite numbers')
    return value


def fit_area_profile(product_name, backend, samples):
    if not isinstance(product_name, str) or not product_name.strip() or backend.get('method') != 'mobile_sam':
        raise ValueError('A real product and the exact MobileSAM backend are required')
    training, heldout, originals, portions = [], [], set(), set()
    for sample in samples:
        digest, material = sample['original_sha256'], sample['physical_sample_id']
        if (not isinstance(digest, str) or not re.fullmatch('[a-f0-9]{64}', digest) or digest in originals
                or not isinstance(material, str) or not material.strip() or material in portions
                or sample.get('disjoint_material_confirmed') is not True
                or sample.get('flat_layer_and_mask_confirmed') is not True
                or sample.get('backend') != backend
                or sample.get('projection_status') != 'sheet_plane_projection'):
            raise ValueError('Each row must be a verified, separately weighed flat sample with matching A4/segmentation')
        originals.add(digest)
        portions.add(material)
        area = _positive(sample['area_mm2'])
        weight = _positive(sample['measured_g'])
        group = sample['split']
        if group not in ('fit', 'check'):
            raise ValueError('Choose fit or check before capture evaluation')
        (training if group == 'fit' else heldout).append((area, weight))
    if len(training) < 3 or len(heldout) < 2 or len({weight for _,weight in training}) < 3:
        raise ValueError('Use at least three fit samples at distinct weights plus two disjoint check samples')
    x, y = np.asarray(training, float).T
    if np.ptp(x) <= 0:
        raise ValueError('Fit areas must span a nonzero range')
    slope, intercept = np.polyfit(x, y, 1)
    if not np.isfinite([slope, intercept]).all() or slope <= 0 or slope*x.min()+intercept <= 0:
        raise ValueError('The observations do not support a positive, increasing area conversion')
    if not all(x.min() <= area <= x.max() for area, _ in heldout):
        raise ValueError('Check samples must fall within the measured calibration area range')
    errors = [abs(slope*area + intercept - weight) for area, weight in heldout]
    return dict(schema=SCHEMA, product_name=product_name, backend=backend, slope_g_per_mm2=float(slope),
                intercept_g=float(intercept), fit_area_mm2=[float(x.min()), float(x.max())],
                checked_mae_g=float(np.mean(errors)), checked_max_error_g=float(max(errors)),
                usable_as_experimental_estimate=np.mean(errors) <= MAX_VALIDATION_MAE_G and max(errors) <= MAX_VALIDATION_ERROR_G,
                fit_samples=len(training), check_samples=len(heldout), samples=samples,
                production_validated=False)


def predict_area(profile, product_name, backend, area):
    if profile.get('schema') != SCHEMA or profile.get('product_name') != product_name or profile.get('backend') != backend:
        return None
    # Refit from reviewed observations to keep edited coefficients from being used.
    checked = fit_area_profile(product_name, backend, profile['samples'])
    low, high = checked['fit_area_mm2']
    if checked['usable_as_experimental_estimate'] and low <= _positive(area) <= high:
        return checked['slope_g_per_mm2']*area + checked['intercept_g']
    return None


def product_geometry_grams(area_mm2, *, strand_length_mm, strand_diameter_mm, grams_per_strand):
    """Comparison C: project uniform cylinders viewed side-on in a single layer.

    Requires a separate measured product calibration. Occlusions, glancing
    camera angles and gaps removed by segmentation bias the result.
    """
    values = [_positive(v) for v in (area_mm2, strand_length_mm, strand_diameter_mm, grams_per_strand)]
    area, length, diameter, grams = values
    return area / (length * diameter) * grams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        data = json.loads(args.samples.read_text(encoding='utf-8-sig'))
        result = fit_area_profile(data['product_name'], data['backend'], data['samples'])
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
