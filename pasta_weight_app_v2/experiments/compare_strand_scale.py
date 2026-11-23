"""Explore native width measurements with verified photos and fixed 20g references."""

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from statistics import median

import cv2
import numpy as np

from .area_proportion import PRIMARY, REFERENCE_G, WIDTH_AREA, analyze, calibrate, predict
from .compare_area_proportion import STRESS_SCALES, load_originals, scaled_photo
from .compare_sheetless_lengths import error_metrics
from .strand_scale import (AGREED, MAX_SIDE, SCALE_VARIANTS, calibrate_widths, canonical_width,
                           choose_estimator, measure_for_area, thickness_diagnostic, visible_density,
                           width_outcomes)

BASELINES = ('legacy', 'frame_area', PRIMARY, WIDTH_AREA)
METHODS = (*BASELINES, *SCALE_VARIANTS, 'selected')


def _analyze_pair(previous_rgb, native_rgb):
    evidence = analyze(previous_rgb)
    return evidence, measure_for_area(native_rgb, evidence)


def _results(pair, old_ref, width_ref):
    evidence, scale = pair
    widths = width_outcomes(evidence, scale, width_ref)
    choice = choose_estimator(evidence, scale, widths, old_ref)
    envelope_px = np.sqrt(evidence.area_px/evidence.features['pca_area']) if evidence.area_px else 0.
    radius_fraction = None
    if envelope_px > 0:
        distance = cv2.distanceTransform(evidence.mask.astype(np.uint8), cv2.DIST_L2, 5)
        radius_fraction = float(np.quantile(distance[evidence.mask], .75) / envelope_px)
    return dict(
        area_px=evidence.area_px, capture_reason=evidence.width_capture_reason,
        radius_p75_over_envelope=radius_fraction,
        track_960_px=evidence.track_px, native_track_960_px=scale.track_960_px,
        density=visible_density(evidence, scale), scale=asdict(scale),
        widths_960_px={WIDTH_AREA: evidence.strand_width.width_px,
                       **{name: canonical_width(scale, name) for name in SCALE_VARIANTS}},
        outcomes={'legacy': dict(diagnostic_g=evidence.legacy_diagnostic_g,
                                 estimated_g=evidence.legacy_weight_g, reason=evidence.legacy_reason),
                  **{key: predict(old_ref, evidence, key) for key in BASELINES if key != 'legacy'},
                  **widths, 'selected': dict(choice, diagnostic_g=None)},
        thick_counterfactual_g=thickness_diagnostic(evidence, widths),
    )


def _for_group(rows, key):
    displayed = [r['original']['outcomes'][key]['estimated_g'] for r in rows]
    available = [r['original']['outcomes'][key]['diagnostic_g'] for r in rows]
    present = [g for g in available if g is not None]
    errors = [g - row['weight_g'] for g, row in zip(displayed, rows) if g is not None]
    # Report an actual 3-placement spread only if all three diagnostic values exist.
    return dict(images=len(rows), displayed=len(errors), displayed_metrics=error_metrics(errors),
                diagnostic_available=len(present),
                placement_span_g=max(present) - min(present) if len(rows) == len(present) == 3 else None,
                displayed_placement_span_g=max(displayed) - min(displayed) if len(rows) == len(errors) == 3 else None)


def _metrics(rows):
    beyond_ref = [row for row in rows if row['weight_g'] > REFERENCE_G]
    keys = (*BASELINES, *SCALE_VARIANTS)
    common = [row for row in beyond_ref
              if all(row['original']['outcomes'][key]['diagnostic_g'] is not None for key in keys)]
    errors = {
        key: error_metrics([r['original']['outcomes'][key]['diagnostic_g'] - r['weight_g'] for r in common])
        for key in keys
    }
    grouped = {str(g): [row for row in rows if row['weight_g'] == g] for g in range(20, 101, 20)}
    grouped['all'] = rows
    grouped['40-100'] = beyond_ref
    return dict(matched_diagnostics_40_to_100=errors,
                groups={g: {key: _for_group(members, key) for key in METHODS} for g, members in grouped.items()})


def _stress_summary(rows):
    summary = {}
    for transform in STRESS_SCALES:
        estimates = {}
        for key in METHODS:
            field = 'estimated_g' if key == 'selected' else 'diagnostic_g'
            pairs = [(r['original']['outcomes'][key][field], r['variants'][transform]['outcomes'][key][field])
                     for r in rows]
            matching = [i for i, (a, b) in enumerate(pairs) if a is not None and b is not None]
            changes = [abs(pairs[i][0] - pairs[i][1]) for i in matching]
            widths = []
            for i in matching:
                first = rows[i]['original']['widths_960_px'].get(key)
                second = rows[i]['variants'][transform]['widths_960_px'].get(key)
                scale = 1. if transform == 'half_resolution' else STRESS_SCALES[transform]
                if first and second:
                    widths.append(abs(second / (first * scale) - 1) * 100)
            published = [(r['original']['outcomes'][key]['estimated_g'],
                          r['variants'][transform]['outcomes'][key]['estimated_g']) for r in rows]
            estimates[key] = dict(
                pairs=len(changes), median_abs_change_g=median(changes) if changes else None,
                max_abs_change_g=max(changes) if changes else None,
                median_width_scale_error_pct=median(widths) if widths else None,
                max_width_scale_error_pct=max(widths) if widths else None,
                displayed_before=sum(a is not None for a, _ in published),
                displayed_after=sum(b is not None for _, b in published),
                lost_display=sum(a is not None and b is None for a, b in published))
        shared = [r for r in rows if all(view['outcomes'][key]['diagnostic_g'] is not None
                                        for key in (WIDTH_AREA, AGREED)
                                        for view in (r['original'], r['variants'][transform]))]
        for key in (WIDTH_AREA, AGREED):
            changes = [abs(r['original']['outcomes'][key]['diagnostic_g'] -
                           r['variants'][transform]['outcomes'][key]['diagnostic_g']) for r in shared]
            estimates[key]['matched_width_pairs'] = len(changes)
            estimates[key]['matched_max_abs_change_g'] = max(changes) if changes else None
        estimates['selected']['method_switches'] = sum(
            initial is not None and changed is not None and initial != changed
            for r in rows
            for initial, changed in [(r['original']['outcomes']['selected']['method'],
                                      r['variants'][transform]['outcomes']['selected']['method'])])
        summary[transform] = estimates
    return summary


def _thickness_check(rows):
    counterfactual = [(r['original']['thick_counterfactual_g'], r['weight_g']) for r in rows]
    added_errors = [g - truth for g, truth in counterfactual if g is not None]
    decisions = Counter(r['original']['capture_reason'] or 'pass' for r in rows)
    return dict(original_reasons=dict(decisions), added_if_thickness_veto_removed=len(added_errors),
                added_metrics=error_metrics(added_errors), physical_layers_annotated=False)


def compare(root):
    original = list(load_originals(root))
    native = list(load_originals(root, max_side=MAX_SIDE))
    if [item for item, _ in original] != [item for item, _ in native]:
        raise ValueError('Photo metadata changed during the comparison')
    pairs = [_analyze_pair(old, full) for (_, old), (_, full) in zip(original, native, strict=True)]
    refs = [p for (item, _), p in zip(original, pairs, strict=True) if item['weight_label_g'] == REFERENCE_G]
    old_ref = calibrate([e for e, _ in refs])
    width_ref = calibrate_widths(refs)
    rows = []
    for (item, old), (_, full), sample in zip(original, native, pairs, strict=True):
        variants = {}
        for transform in STRESS_SCALES:
            derived = _analyze_pair(scaled_photo(old, transform), scaled_photo(full, transform))
            variants[transform] = _results(derived, old_ref, width_ref)
        rows.append(dict(file=item['file'], sha256=item['sha256'], weight_g=item['weight_label_g'],
                         series=item['series_id'], placement=item['photo_placement_variant'],
                         original=_results(sample, old_ref, width_ref), variants=variants))
    return dict(version='native-strand-width-v1', reference_specimens=1, reference_placements=3,
                independent_evaluation=None, original_resolution_cap=MAX_SIDE,
                reference_features=dict(old_ref, **width_ref), rows=rows, **_metrics(rows),
                stress=_stress_summary(rows), thickness_check=_thickness_check(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--originals', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='New private JSON output path')
    args = parser.parse_args()
    cv2.setNumThreads(1)
    report = compare(args.originals)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
