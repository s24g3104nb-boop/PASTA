"""Local color edges and same-location checks for research-only strand scale."""

from dataclasses import dataclass

import cv2
import numpy as np

from ..geometry_candidate import skeletonize
from ..sheetless_area import THICK_CAPTURE_REASON, pasta_mask
from .area_proportion import (EVALUATED_MAX_G, EXPLORATORY, OUTSIDE_EVALUATED_RANGE, PRIMARY,
                              REFERENCE_G, predict)
from .isolated_strand_width import WidthMeasurement, separated_sections, summarize_sections
from .sheetless_lengths import projected_reference_length

# Protect the local experiment from full-resolution 25MP working arrays.
# Supplied originals have a long side of 1536px, below this cap.
MAX_SIDE = 2000
NATIVE_MASK = 'native_mask'
COLOR_EDGE = 'color_edge'
AGREED = 'agreed'
SCALE_VARIANTS = (NATIVE_MASK, COLOR_EDGE, AGREED)


@dataclass(frozen=True)
class StrandScale:
    long_side_px: int
    check_long_side_px: int
    widths: dict
    track_960_px: float = 0.


def _centerline(mask):
    center = np.zeros(mask.shape, dtype=np.uint8)
    y, x = np.nonzero(mask)
    if len(x):
        roi = np.s_[y.min():y.max()+1, x.min():x.max()+1]
        center[roi] = skeletonize(mask[roi])
    return center


def _half_contrast_edges(chroma, xs, ys, nx, ny, seed_width):
    """Find the two local 50% Lab b* transitions around a separated segment."""
    offsets = np.arange(0, 16.25, .25, dtype=np.float32)
    half = seed_width[:, None] * .5
    widths = []
    for sign in (-1, 1):
        dx, dy = sign * nx[:, None] * offsets, sign * ny[:, None] * offsets
        samples = cv2.remap(chroma, (xs[:, None] + dx).astype(np.float32),
                            (ys[:, None] + dy).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        inside = np.max(np.where(offsets <= .55 * half, samples, -np.inf), axis=1)
        outer = (offsets >= half + .75) & (offsets <= half + 2.5)
        outside = np.min(np.where(outer, samples, np.inf), axis=1)
        contrast = inside - outside
        threshold = (inside + outside) * .5
        below = samples < threshold[:, None]
        first = below.argmax(axis=1)
        rows = np.arange(len(first))
        start = samples[rows, np.maximum(first - 1, 0)]
        end = samples[rows, first]
        edge = (first - 1 + (start - threshold) / np.maximum(start - end, 1e-6)) * .25
        valid = (contrast >= 10) & (first > 0) & (np.abs(edge - half[:, 0]) <= 2.)
        widths.append(np.where(valid, edge, np.nan))
    return widths[0] + widths[1]


def measure_strand_scale(rgb, envelope_960_px=0.):
    """Use the actual supplied resolution and check the same sites after downsampling."""
    side = max(rgb.shape[:2])
    check_side = min(960, round(side * .75))
    empty = dict.fromkeys(SCALE_VARIANTS, WidthMeasurement())
    if min(rgb.shape[:2]) < 40:
        return StrandScale(side, check_side, empty)
    mask = pasta_mask(rgb)
    centerline = _centerline(mask)
    track = (projected_reference_length(centerline, envelope_960_px * side/960) * 960/side
             if envelope_960_px > 0 else 0.)
    x, y, nx, ny, mask_width = separated_sections(mask, centerline)
    if not len(mask_width):
        return StrandScale(side, check_side, empty, track)
    native = summarize_sections(x, y, mask_width, mask.shape)
    b = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[:, :, 2].astype(np.float32)
    width = _half_contrast_edges(b, x, y, nx, ny, mask_width)
    valid = np.isfinite(width)
    refined = summarize_sections(x[valid], y[valid], width[valid], mask.shape)

    ratio = check_side / side
    smaller = cv2.resize(rgb, (max(1, round(rgb.shape[1]*ratio)), max(1, round(rgb.shape[0]*ratio))),
                         interpolation=cv2.INTER_AREA)
    check_b = cv2.cvtColor(smaller, cv2.COLOR_RGB2LAB)[:, :, 2].astype(np.float32)
    lx = (x+.5)*smaller.shape[1]/rgb.shape[1] - .5
    ly = (y+.5)*smaller.shape[0]/rgb.shape[0] - .5
    check = _half_contrast_edges(check_b, lx, ly, nx, ny, mask_width*ratio)
    # One pixel of uncertainty or a modest fractional deviation can be caused by
    # sampling. Larger discrepancies often indicate a neighboring strand.
    agree = valid & (np.abs(width - check/ratio) <= np.maximum(1., .15*width))
    consistent = summarize_sections(x[agree], y[agree], width[agree], mask.shape)
    return StrandScale(side, check_side, {NATIVE_MASK: native, COLOR_EDGE: refined, AGREED: consistent}, track)


def canonical_width(scale, method):
    width = scale.widths[method].width_px
    return None if width is None else width * 960 / scale.long_side_px


def _area_feature(evidence, scale, method):
    width = canonical_width(scale, method)
    return evidence.area_px / width**2 if width and evidence.area_px else None


def measure_for_area(rgb, evidence):
    ratio = evidence.features['pca_area']
    extent = np.sqrt(evidence.area_px / ratio) if ratio else 0.
    return measure_strand_scale(rgb, extent)


def calibrate_widths(pairs):
    """The same three 20g placements anchor all variants and all resize tests."""
    if len(pairs) != 3 or any(e.width_capture_reason for e, _ in pairs):
        raise ValueError('撮影条件を満たす20gの基準写真3枚を選んでください。')
    reference = {}
    for method in SCALE_VARIANTS:
        areas = [_area_feature(ev, scale, method) for ev, scale in pairs]
        if all(value is not None for value in areas):
            reference[method] = float(np.median(areas))
    return reference


def width_outcomes(evidence, scale, reference):
    results = {}
    for method in SCALE_VARIANTS:
        area = _area_feature(evidence, scale, method)
        anchor = reference.get(method)
        diagnostic = REFERENCE_G * area / anchor if anchor and area else None
        reason = evidence.width_capture_reason
        if diagnostic is None:
            missing = '面積換算の基準を確認できません。' if not anchor else '分離した麺の幅を十分に測れません。'
            reason = ' '.join(filter(None, (reason, missing)))
        status = ''
        if diagnostic is not None:
            status = EXPLORATORY if REFERENCE_G <= diagnostic <= EVALUATED_MAX_G else OUTSIDE_EVALUATED_RANGE
        results[method] = dict(diagnostic_g=diagnostic, estimated_g=None if reason else diagnostic,
                               reason=reason, status=status)
    return results


def visible_density(evidence, scale):
    """Wide connected foreground relative to a strand, independent of weight."""
    width = canonical_width(scale, AGREED)
    if not width or not evidence.area_px:
        return None
    distance = cv2.distanceTransform(evidence.mask.astype(np.uint8), cv2.DIST_L2, 5)
    return float(np.count_nonzero(distance > .85*width) / evidence.area_px)


def choose_estimator(evidence, scale, outcomes, legacy_reference):
    """Use one geometrically supported result, retaining the thickness veto."""
    if evidence.width_capture_reason:
        return dict(method=None, estimated_g=None, reason=evidence.width_capture_reason)
    width = scale.widths[AGREED]
    density = visible_density(evidence, scale)
    # A widespread connected interior can bias the local width. A stable
    # long track is a separate option for usable captures, without averaging.
    width_reliable = (density is not None and density <= 2/3 and width.spread_px is not None
                      and width.spread_px <= .25*width.width_px)
    if width_reliable and outcomes[AGREED]['estimated_g'] is not None:
        return dict(method=AGREED, estimated_g=outcomes[AGREED]['estimated_g'], reason='')
    track = predict(legacy_reference, evidence, PRIMARY)
    if (track['estimated_g'] is not None and evidence.track_px > 0
            and abs(scale.track_960_px / evidence.track_px - 1) <= .10):
        return dict(method=PRIMARY, estimated_g=track['estimated_g'], reason='')
    if evidence.legacy_weight_g is not None:
        return dict(method='legacy', estimated_g=evidence.legacy_weight_g, reason='')
    return dict(method=None, estimated_g=None, reason='麺幅または長い線の測定が安定しません。')


def thickness_diagnostic(evidence, outcomes):
    """Offline-only counterfactual. Never a displayed value or a layer label."""
    return (outcomes[AGREED]['diagnostic_g']
            if evidence.width_capture_reason == THICK_CAPTURE_REASON else None)
