"""Compare 20g-only proportional models against a verified local photo batch."""

import argparse
from collections import Counter
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
from statistics import mean
from time import perf_counter

import cv2
import numpy as np
from PIL import Image

from ..pasta_vision import image_to_rgb
from ..sheetless_area import canonical_image
from .area_proportion import (FEATURES, PRIMARY, REFERENCE_G, analyze, calibrate, overlay,
                              paired_color_deltas, patch_colors, predict, yellow_masks)
from .compare_sheetless_lengths import error_metrics

VERSION = 'reference-area-proportion-v1'
METHODS = ('legacy', *FEATURES)


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


def compare(originals, output, *, reviews=None, mobile=None):
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
        _write_images(images, index, scene, masks, sites)
        results.append((evidence, dict(
            index=index, file=item['file'], sha256=item['sha256'], reported_g=item['weight_label_g'],
            placement=item['photo_placement_variant'], series=item['series_id'],
            role='reference' if item['weight_label_g'] == REFERENCE_G else 'exploratory',
            area_px=evidence.area_px, track_px=evidence.track_px, features=evidence.features,
            mask_area_px={name: int(mask.sum()) for name, mask in masks.items()},
            color_sites=patches, paired_color_deltas=paired_color_deltas(patches),
            color_extraction_ms=extraction_ms, mobilesam_ms=mobile_ms,
        )))
    model = calibrate([ev for ev, row in results if row['role'] == 'reference'])
    rows = []
    for ev, row in results:
        row['outcomes'] = {
            'legacy': dict(diagnostic_g=ev.legacy_diagnostic_g, estimated_g=ev.legacy_weight_g,
                           reason=ev.legacy_reason),
            **{key: predict(model, ev, key) for key in FEATURES},
        }
        row['color_corrected_g'] = None
        row['color_correction_status'] = 'not_applied'
        for outcome in row['outcomes'].values():
            for field in ('diagnostic_g', 'estimated_g'):
                value = outcome[field]
                outcome[field + '_error'] = None if value is None else value - row['reported_g']
        rows.append(row)
    return dict(version=VERSION, reference_g=REFERENCE_G, reference_images=3, reference_specimens=1,
                independence_between_groups='unknown', independent_evaluation=None,
                reference_features=model, primary=PRIMARY,
                mobile=None if mobile is None else mobile.metadata, groups=_summarize(rows), rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--originals', type=Path, required=True, help='Private folder with manifest.json')
    parser.add_argument('--output', type=Path, required=True, help='New private output folder')
    parser.add_argument('--color-reviews', type=Path, help='Private JSON keyed by original SHA-256')
    parser.add_argument('--mobile-checkpoint', type=Path, help='Optional locally installed official MobileSAM ViT-T')
    args = parser.parse_args()
    cv2.setNumThreads(1)
    reviews = json.loads(args.color_reviews.read_text(encoding='utf-8-sig')) if args.color_reviews else None
    mobile = None
    if args.mobile_checkpoint:
        import torch

        from .foundation_backends import MobileSam

        torch.set_num_threads(2)
        mobile = MobileSam(args.mobile_checkpoint, 'cpu', torch)
    report = compare(args.originals, args.output, reviews=reviews, mobile=mobile)
    with (args.output / 'comparison.json').open('x', encoding='utf-8') as file:
        json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
