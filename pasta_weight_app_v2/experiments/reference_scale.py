"""Conservative, label-blind *candidate* reference detection for offline research.

A dark convex rectangle is not proof of a tray's identity or physical size. This
module measures only the observed candidate in the same 512 px coordinate system
as ``B_normalized5``. It never fills a missing reference with the image frame,
estimates real dimensions, approves a mask, or releases a weight prediction.

The threshold/geometry rules are deliberately fixed, not fitted against weights.
Partial occlusion, dark backgrounds, glare and perspective can defeat these
rules. In particular, padding cannot restore a tray edge absent from an original
capture: callers must retain original capture rejection across derivatives.
"""
from dataclasses import asdict, dataclass
import hashlib
import json

import cv2
import numpy as np

from .stability_pipeline import normalize_size


VERSION = 'reference-scale-candidate-v1'


@dataclass(frozen=True)
class ReferenceConfig:
    version: str = VERSION
    target_long_side: int = 512
    gray_max_exclusive: int = 120
    closing_kernel_px: int = 5
    min_hull_fraction: float = 0.10
    frame_clearance_px: int = 2
    min_hull_rectangularity: float = 0.86
    min_contour_solidity: float = 0.65
    min_short_side_fraction: float = 0.15
    max_aspect_ratio: float = 3.0
    polygon_epsilon_fraction: float = 0.02
    ambiguous_relative_area: float = 0.40
    down_interpolation: str = 'INTER_AREA'
    up_interpolation: str = 'INTER_CUBIC'


CONFIG = ReferenceConfig()


def _candidate(contour, shape):
    h, w = shape
    hull = cv2.convexHull(contour)
    area = float(cv2.contourArea(hull))
    if area < CONFIG.min_hull_fraction * h * w:
        return None
    contour_area = float(cv2.contourArea(contour))
    rect = cv2.minAreaRect(hull)
    short, long = sorted(float(v) for v in rect[1])
    perimeter = float(cv2.arcLength(hull, True))
    polygon = cv2.approxPolyDP(hull, CONFIG.polygon_epsilon_fraction * perimeter, True)
    points = hull.reshape(-1, 2)
    margin = CONFIG.frame_clearance_px
    edges = []
    for name, touched in [('left', points[:, 0].min() <= margin),
                          ('right', points[:, 0].max() >= w - 1 - margin),
                          ('top', points[:, 1].min() <= margin),
                          ('bottom', points[:, 1].max() >= h - 1 - margin)]:
        if touched:
            edges.append(name)
    rectangularity = area / max(short * long, 1.0)
    solidity = contour_area / max(area, 1.0)
    reasons = []
    if edges:
        reasons.append('candidate_touches_frame')
    if rectangularity < CONFIG.min_hull_rectangularity or not 4 <= len(polygon) <= 8:
        reasons.append('non_rectangular_candidate')
    if solidity < CONFIG.min_contour_solidity:
        reasons.append('fragmented_candidate')
    if short < CONFIG.min_short_side_fraction * max(h, w):
        reasons.append('reference_too_narrow')
    if long / max(short, 1.0) > CONFIG.max_aspect_ratio:
        reasons.append('reference_too_elongated')
    return {
        'candidate_area_px': area,
        'candidate_contour_area_px': contour_area,
        'candidate_fraction': area / (h * w),
        'candidate_long_side_px': long,
        'candidate_short_side_px': short,
        'hull_rectangularity': rectangularity,
        'contour_solidity': solidity,
        'polygon_px': points.tolist(),
        'quadrilateral_px': polygon.reshape(-1, 2).tolist() if len(polygon) == 4 else None,
        'approximation_vertex_count': len(polygon),
        'frame_contact_edges': edges,
        'reasons': reasons,
    }


def analyze_reference(rgb):
    """Return ``(metadata, canonical_rgb, candidate_mask, overlay_rgb)``.

    Only RGB uint8 image pixels are accepted; there is no weight, path, filename
    or label input. ``metadata`` is strictly JSON serializable. ``area_px`` and
    side lengths are ``None`` on failure, while ``candidate_area_px`` preserves
    diagnostic visible geometry. The mask is an **unapproved convex candidate**
    including its interior, not a pasta mask or segmentation ground truth.

    ``usable_for_relative_scale`` permits an exploratory, same-reference-object
    ratio only. It does not verify that object identity across photographs. Even
    an accepted candidate is never eligible for production weight inference.
    """
    scene = normalize_size(rgb, CONFIG.target_long_side)
    gray = cv2.cvtColor(scene, cv2.COLOR_RGB2GRAY)
    dark = np.where(gray < CONFIG.gray_max_exclusive, 255, 0).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE,
                           np.ones((CONFIG.closing_kernel_px, CONFIG.closing_kernel_px), np.uint8))
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = [c for contour in contours if (c := _candidate(contour, dark.shape)) is not None]
    candidates.sort(key=lambda c: (-c['candidate_area_px'], c['polygon_px'][0]))
    config = asdict(CONFIG)
    metadata = {
        'version': VERSION,
        'config': config,
        'config_sha256': hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        'status': 'unavailable',
        'candidate_found': bool(candidates),
        'candidate_count': len(candidates),
        'approval_status': 'UNAPPROVED',
        'usable_for_relative_scale': False,
        'physical_calibration': False,
        'eligible_for_weight_inference': False,
        'same_reference_object_verified': False,
        'perspective_rectified': False,
        'scale_kind': 'observed_candidate_relative_only',
        'polygon_kind': 'observed_convex_hull',
        'canonical_shape_hw': [int(v) for v in scene.shape[:2]],
        'area_px': None,
        'long_side_px': None,
        'short_side_px': None,
        'candidate_area_px': None,
        'polygon_px': None,
        'quadrilateral_px': None,
        'reasons': ['no_large_dark_region'],
        'candidates': candidates,
        'limitations': [
            'A dark rectangle does not establish tray identity or real dimensions.',
            'Occlusion, reflections, shadows and perspective may alter the candidate.',
            'Preserve original capture rejection when processing padded derivatives.',
            'No independent real-photo validation or physical calibration is available.',
        ],
    }
    mask = np.zeros(scene.shape[:2], np.uint8)
    overlay = scene.copy()
    if candidates:
        best = candidates[0]
        # Copy the diagnostics so appending ambiguity does not mutate candidates.
        metadata.update({key: value for key, value in best.items() if key != 'reasons'})
        reasons = list(best['reasons'])
        if any(c['candidate_area_px'] >= CONFIG.ambiguous_relative_area * best['candidate_area_px']
               for c in candidates[1:]):
            reasons.append('ambiguous_dark_regions')
        metadata['reasons'] = reasons
        if not reasons:
            metadata.update(status='candidate_unapproved', usable_for_relative_scale=True,
                            area_px=best['candidate_area_px'],
                            long_side_px=best['candidate_long_side_px'],
                            short_side_px=best['candidate_short_side_px'])
        cv2.fillConvexPoly(mask, np.asarray(best['polygon_px'], dtype=np.int32), 255)
        tint = np.array((50, 180, 100) if not reasons else (225, 70, 55), dtype=np.float32)
        selected = mask > 0
        overlay[selected] = np.rint(0.80 * overlay[selected] + 0.20 * tint).astype(np.uint8)
        for c in candidates:
            cv2.polylines(overlay, [np.asarray(c['polygon_px'], dtype=np.int32)], True,
                          (50, 180, 100) if not reasons else (225, 70, 55), 2)
    return metadata, scene, mask, overlay
