"""Compare 20g-only proportional models against a verified local photo batch."""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
from statistics import mean, median
from time import perf_counter

import cv2
import numpy as np
from PIL import Image

from ..pasta_vision import image_to_rgb
from ..sheetless_area import canonical_image
from .area_proportion import (FEATURES, PRIMARY, REFERENCE_G, WIDTH_AREA, analyze, calibrate, overlay,
                              paired_color_deltas, patch_colors, predict, yellow_masks)
from .bundle_texture import extract_bundle_texture
from .compare_sheetless_lengths import error_metrics

VERSION = 'reference-area-proportion-v2'
METHODS = ('legacy', *FEATURES)
STRESS_SCALES = {'smaller_in_frame_75': .75, 'smaller_in_frame_60': .60, 'half_resolution': .50}


def load_originals(root):
    """Check filenames, hashes, JPEG dimensions, and per-weight group metadata."""
    document = json.loads((root / 'manifest.json').read_text(encoding='utf-8-sig'))
    items = document['items']
    if (document.get('schema_version') != 'pasta_lens_user_photo_manifest_v1'
            or document.get('photo_count') != 15 or len(items) != 15
            or Counter(item['weight_label_g'] for item in items) != {g: 3 for g in range(20, 101, 20)}):
        raise ValueError('Expected 3 original photos at each of 20, 40, 60, 80 and 100g')
    series = set()
    for weight in range(20, 101, 20):
        group = [item for item in items if item['weight_label_g'] == weight]
        ids = {item['series_id'] for item in group}
        if len(ids) != 1 or series & ids or {item['photo_placement_variant'] for item in group} != {'A', 'B', 'C'}:
            raise ValueError('Each weight needs one series with three distinct placements')
        series.update(ids)
    paths, digests = set(), set()
    for item in items:
        name = item['file']
        if (not re.fullmatch(r'photos/[^/\\]+\.jpe?g', name, re.I) or name in paths
                or item['weight_label_source'] != 'user_reported_weight_in_chat'
                or item['physical_independence_between_weight_series'] != 'unknown'
                or item['usable_for_independent_test_accuracy'] is not False):
            raise ValueError('Unexpected original or weight-label provenance')
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Original path must be inside the batch directory')
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != item['sha256'] or digest in digests:
            raise ValueError(f'Original SHA-256 mismatch or duplicate: {name}')
        with Image.open(BytesIO(content)) as im:
            if im.format != 'JPEG' or im.size != (item['width_px'], item['height_px']):
                raise ValueError(f'Original JPEG dimensions differ from the manifest: {name}')
        paths.add(name)
        digests.add(digest)
        yield item, image_to_rgb(BytesIO(content))


def _metrics(rows, method, field):
    values = [(row['outcomes'][method][field], row['reported_g']) for row in rows]
    errors = [g - weight for g, weight in values if g is not None]
    if not errors:
        return dict(images=0, bias_g=None, mae_g=None, rmse_g=None, max_abs_g=None)
    return dict(error_metrics(errors), bias_g=mean(errors))


def _summarize(rows):
    groups = {str(g): [r for r in rows if r['reported_g'] == g] for g in range(20, 101, 20)}
    groups['40-100'] = [r for r in rows if r['reported_g'] > REFERENCE_G]
    summary = {}
    for label, group in groups.items():
        common = [row for row in group if all(row['outcomes'][key]['diagnostic_g'] is not None
                                               for key in METHODS)]
        summary[label] = {key: dict(diagnostic_g=_metrics(group, key, 'diagnostic_g'),
                                    estimated_g=_metrics(group, key, 'estimated_g'),
                                    matched_diagnostic_g=_metrics(common, key, 'diagnostic_g'))
                          for key in METHODS}
    return summary


def _outcomes(model, evidence):
    return {
        'legacy': dict(diagnostic_g=evidence.legacy_diagnostic_g, estimated_g=evidence.legacy_weight_g,
                       reason=evidence.legacy_reason),
        **{key: predict(model, evidence, key) for key in FEATURES},
    }


def _scale_stress(rgb, original, model):
    height, width = rgb.shape[:2]
    results = {}
    for name, scale in STRESS_SCALES.items():
        smaller = cv2.resize(rgb, (round(width*scale), round(height*scale)), interpolation=cv2.INTER_AREA)
        if name != 'half_resolution':
            resized_h, resized_w = smaller.shape[:2]
            top, left = (height - resized_h)//2, (width - resized_w)//2
            smaller = cv2.copyMakeBorder(smaller, top, height - resized_h - top,
                                         left, width - resized_w - left, cv2.BORDER_CONSTANT,
                                         value=(128, 128, 128))
        derived = analyze(smaller)
        outcomes = _outcomes(model, derived)
        for key, outcome in outcomes.items():
            before, after = original[key]['diagnostic_g'], outcome['diagnostic_g']
            outcome['abs_change_g'] = None if before is None or after is None else abs(after - before)
        results[name] = dict(area_px=derived.area_px, strand_width=asdict(derived.strand_width), outcomes=outcomes)
    return results


def _summarize_stress(rows):
    result = {}
    for name in STRESS_SCALES:
        variants = [row['scale_stress'][name]['outcomes'] for row in rows]
        # Match to the proposed width method for direct comparisons while keeping
        # the number of available pairs for each estimator visible.
        result[name] = {}
        for key in METHODS:
            changes = [v[key]['abs_change_g'] for v in variants if v[key]['abs_change_g'] is not None]
            common = [v[key]['abs_change_g'] for v in variants
                      if v[key]['abs_change_g'] is not None and v[WIDTH_AREA]['abs_change_g'] is not None]
            result[name][key] = dict(pairs=len(changes), matched_pairs=len(common),
                                     matched_median_abs_g=median(common) if common else None,
                                     matched_max_abs_g=max(common) if common else None)
    return result


def _visible_dense_fraction(mask, width_px):
    if width_px is None:
        return None
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    # A resolved single stripe has distance ~width/2 at its center. This flag
    # instead measures a wide connected foreground interior, including contact.
    return float(np.count_nonzero(distance > .85*width_px) / np.count_nonzero(mask))


def _write_images(output, index, scene, masks, sites):
    name = f'{index:02d}'
    Image.fromarray(scene).save(output / f'{name}_original.jpg', quality=93)
    for method, mask in masks.items():
        slug = method.lower().replace('+', '_')
        Image.fromarray(mask.astype(np.uint8)*255).save(output / f'{name}_{slug}.png')
        Image.fromarray(overlay(scene, mask)).save(output / f'{name}_{slug}_overlay.jpg', quality=93)
    if sites:
        annotated = scene.copy()
        for i, site in enumerate(sites, 1):
            x, y = site['x'], site['y']
            cv2.circle(annotated, (x, y), site.get('radius', 1) + 4, (230, 60, 80), 1)
            cv2.putText(annotated, str(i), (x+4, y-4), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
        Image.fromarray(annotated).save(output / f'{name}_color_sites.jpg', quality=93)


def compare(originals, output, *, reviews=None, mobile=None, scale_stress=False):
    inputs = list(load_originals(originals))
    reviews = reviews or {}
    if not set(reviews) <= {item['sha256'] for item, _ in inputs}:
        raise ValueError('Color reviews must correspond to an original SHA-256')
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    images = output / 'images'
    images.mkdir()
    results = []
    for index, (item, rgb) in enumerate(inputs, 1):
        scene = canonical_image(rgb)
        started = perf_counter()
        masks = yellow_masks(scene)
        extraction_ms = (perf_counter() - started) * 1000
        evidence = analyze(rgb, scene=scene, mask=masks['HSV+Lab'])
        mobile_ms = None
        if mobile is not None:
            started = perf_counter()
            masks['MobileSAM'] = mobile(scene)
            mobile_ms = (perf_counter() - started) * 1000
        sites = reviews.get(item['sha256'], [])
        patches = patch_colors(scene, masks, sites)
        texture = (extract_bundle_texture(scene, evidence.mask) if min(evidence.mask.shape) >= 16
                   else dict(texture_status='unavailable', pitch_px=None, qc_flags=['insufficient_size']))
        texture.pop('profiles', None)
        _write_images(images, index, scene, masks, sites)
        results.append((evidence, dict(
            index=index, file=item['file'], sha256=item['sha256'], reported_g=item['weight_label_g'],
            placement=item['photo_placement_variant'], series=item['series_id'],
            role='reference' if item['weight_label_g'] == REFERENCE_G else 'exploratory',
            area_px=evidence.area_px, track_px=evidence.track_px, features=evidence.features,
            strand_width=asdict(evidence.strand_width), previous_width_px=evidence.previous_width_px,
            visible_dense_fraction=_visible_dense_fraction(evidence.mask, evidence.strand_width.width_px),
            visible_texture=texture,
            mask_area_px={name: int(mask.sum()) for name, mask in masks.items()},
            color_sites=patches, paired_color_deltas=paired_color_deltas(patches),
            color_extraction_ms=extraction_ms, mobilesam_ms=mobile_ms,
        )))
    model = calibrate([ev for ev, row in results if row['role'] == 'reference'])
    rows = []
    for (ev, row), (_, rgb) in zip(results, inputs, strict=True):
        row['outcomes'] = _outcomes(model, ev)
        if scale_stress:
            row['scale_stress'] = _scale_stress(rgb, row['outcomes'], model)
        row['color_corrected_g'] = None
        row['color_correction_status'] = 'not_applied'
        for outcome in row['outcomes'].values():
            for field in ('diagnostic_g', 'estimated_g'):
                value = outcome[field]
                outcome[field + '_error'] = None if value is None else value - row['reported_g']
        rows.append(row)
    return dict(version=VERSION, reference_g=REFERENCE_G, reference_images=3, reference_specimens=1,
                independence_between_groups='unknown', independent_evaluation=None,
                reference_features=model, primary=PRIMARY, width_method=WIDTH_AREA,
                scale_stress=_summarize_stress(rows) if scale_stress else None,
                mobile=None if mobile is None else mobile.metadata, groups=_summarize(rows), rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--originals', type=Path, required=True, help='Private folder with manifest.json')
    parser.add_argument('--output', type=Path, required=True, help='New private output folder')
    parser.add_argument('--color-reviews', type=Path, help='Private JSON keyed by original SHA-256')
    parser.add_argument('--mobile-checkpoint', type=Path, help='Optional locally installed official MobileSAM ViT-T')
    parser.add_argument('--scale-stress', action='store_true', help='Repeat analysis on three digital resizes')
    args = parser.parse_args()
    cv2.setNumThreads(1)
    reviews = json.loads(args.color_reviews.read_text(encoding='utf-8-sig')) if args.color_reviews else None
    mobile = None
    if args.mobile_checkpoint:
        import torch

        from .foundation_backends import MobileSam

        torch.set_num_threads(2)
        mobile = MobileSam(args.mobile_checkpoint, 'cpu', torch)
    report = compare(args.originals, args.output, reviews=reviews, mobile=mobile, scale_stress=args.scale_stress)
    with (args.output / 'comparison.json').open('x', encoding='utf-8') as file:
        json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
