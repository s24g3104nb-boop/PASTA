"""Reject-first, experimental A4 reference detector. Never predicts weight."""
import hashlib
import json

import cv2
import numpy as np

from .capture_sheet import (WIDTH_MM, HEIGHT_MM, MARKER_POSITIONS, DICTIONARY_ID,
                            PIXELS_PER_MM, SHEET_VERSION, marker_corners_mm)

VERSION = 'a4-marker-reference-v1'
CONFIG = dict(max_long_side=1600, min_marker_edge_px=20., edge_margin_px=3.,
              max_marker_area_ratio=4., max_reprojection_error_fraction=.04,
              max_reprojection_error_px=3., min_sheet_area_fraction=.15,
              min_marker_angle_degrees=25., pixels_per_mm=PIXELS_PER_MM,
              expected_ids=sorted(MARKER_POSITIONS), sheet_version=SHEET_VERSION,
              dictionary='DICT_4X4_50')


def _angles(corners):
    a = np.roll(corners, 1, axis=0) - corners
    b = np.roll(corners, -1, axis=0) - corners
    cosine = (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def analyze_marker_reference(rgb, *, print_scale_verified=False, coplanar_confirmed=False):
    """Return metadata, rectified RGB (or None), and normalized detection overlay.

    Confirmation booleans are user declarations, not measurements by software.
    Metric scale describes the SHEET PLANE; elevated/hidden pasta remains unknown.
    Input must already have its EXIF orientation applied. Original is not changed.
    """
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('RGB uint8 image required')
    if min(rgb.shape[:2]) < 16 or rgb.shape[0] * rgb.shape[1] > 50_000_000:
        raise ValueError('Image dimensions outside supported research range')
    if type(print_scale_verified) is not bool or type(coplanar_confirmed) is not bool:
        raise ValueError('Reference confirmations must be explicit booleans')
    factor = min(1., CONFIG['max_long_side'] / max(rgb.shape[:2]))
    scene = cv2.resize(rgb, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA) if factor < 1 else rgb.copy()
    overlay = scene.copy()
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY_ID)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, rejected = detector.detectMarkers(cv2.cvtColor(scene, cv2.COLOR_RGB2GRAY))
    observed = [] if ids is None else [int(value) for value in ids.ravel()]
    if observed:
        cv2.aruco.drawDetectedMarkers(overlay, corners, ids)
    metadata = dict(version=VERSION, config=dict(CONFIG),
        config_sha256=hashlib.sha256(json.dumps(CONFIG, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        status='retake', detected_ids=observed, detected_marker_count=len(observed),
        rejected_marker_candidate_count=len(rejected), geometry_valid=False,
        usable_for_metric_scale=False, print_scale_verified=print_scale_verified,
        coplanar_confirmed=coplanar_confirmed, sheet_pixels_per_mm=None,
        homography_image_to_sheet_px=None, rectified_shape=None,
        normalized_shape=list(scene.shape[:2]), input_scale_factor=factor,
        reasons=[], production_eligible=False, weight_prediction_g=None,
        physical_scale_verified_by_software=False,
        limitation='Sheet-plane research geometry only; printer scale, coplanarity, hidden quantity and weight accuracy are not verified by software.')
    reasons = metadata['reasons']
    if len(observed) != len(set(observed)):
        reasons.append('duplicate_marker_ids')
    if set(observed) != set(MARKER_POSITIONS):
        reasons.append('all_four_expected_markers_required')
    if reasons:
        return metadata, None, overlay
    by_id = {marker_id: corner.reshape(4, 2) for marker_id, corner in zip(observed, corners)}
    areas, edges = [], []
    for corner in by_id.values():
        edge = np.linalg.norm(np.roll(corner, -1, axis=0) - corner, axis=1)
        edges.extend(edge.tolist())
        areas.append(abs(cv2.contourArea(corner)))
        if not cv2.isContourConvex(corner) or np.min(_angles(corner)) < CONFIG['min_marker_angle_degrees']:
            reasons.append('marker_too_oblique')
        margin = CONFIG['edge_margin_px']
        if np.any(corner < margin) or np.any(corner[:, 0] > scene.shape[1] - 1 - margin) or np.any(corner[:, 1] > scene.shape[0] - 1 - margin):
            reasons.append('marker_near_image_edge')
    metadata['minimum_marker_edge_px'] = min(edges)
    if min(edges) < CONFIG['min_marker_edge_px']:
        reasons.append('markers_too_small')
    if min(areas) <= 0 or max(areas) / min(areas) > CONFIG['max_marker_area_ratio']:
        reasons.append('perspective_or_marker_size_inconsistent')
    source = np.concatenate([by_id[index] for index in sorted(by_id)]).astype(np.float32)
    target_mm = np.concatenate([marker_corners_mm(index) for index in sorted(by_id)])
    # Use every decoded corner, not only four centers: swapped/moved/rotated
    # individual markers must disagree with the printed board geometry.
    homography, _ = cv2.findHomography(target_mm, source, method=0)
    if homography is None or not np.all(np.isfinite(homography)):
        reasons.append('invalid_homography')
        return metadata, None, overlay
    errors = np.linalg.norm(cv2.perspectiveTransform(target_mm[None, ...], homography)[0] - source, axis=1)
    metadata['max_reprojection_error_px'] = float(max(errors))
    if max(errors) > min(CONFIG['max_reprojection_error_px'], np.median(edges) * CONFIG['max_reprojection_error_fraction']):
        reasons.append('printed_geometry_inconsistent')
    sheet_mm = np.array([[[0, 0], [WIDTH_MM, 0], [WIDTH_MM, HEIGHT_MM], [0, HEIGHT_MM]]], np.float32)
    projected_sheet = cv2.perspectiveTransform(sheet_mm, homography)[0]
    metadata['projected_sheet_corners'] = projected_sheet.tolist()
    if not np.all(np.isfinite(projected_sheet)) or not cv2.isContourConvex(projected_sheet):
        reasons.append('invalid_sheet_polygon')
    else:
        fraction = abs(cv2.contourArea(projected_sheet)) / (scene.shape[0] * scene.shape[1])
        metadata['projected_sheet_fraction'] = float(fraction)
        if fraction < CONFIG['min_sheet_area_fraction']:
            reasons.append('sheet_too_small')
        # Require the whole sheet, not extrapolation outside the photograph.
        if np.any(projected_sheet < -2) or np.any(projected_sheet[:, 0] > scene.shape[1] + 1) or np.any(projected_sheet[:, 1] > scene.shape[0] + 1):
            reasons.append('sheet_clipped')
        cv2.polylines(overlay, [np.round(projected_sheet).astype(np.int32)], True, (30, 170, 70), 2)
    metadata['reasons'] = sorted(set(reasons))
    if reasons:
        return metadata, None, overlay
    metadata['geometry_valid'] = True
    if not print_scale_verified or not coplanar_confirmed:
        metadata['status'] = 'confirmation_required'
        metadata['reasons'] = ([] if print_scale_verified else ['verify_printed_100mm_line']) + ([] if coplanar_confirmed else ['confirm_flat_sheet_same_plane'])
        return metadata, None, overlay
    scale = np.diag([PIXELS_PER_MM, PIXELS_PER_MM, 1.])
    inverse = scale @ np.linalg.inv(homography)
    rectified = cv2.warpPerspective(scene, inverse, (round(WIDTH_MM * PIXELS_PER_MM), round(HEIGHT_MM * PIXELS_PER_MM)), flags=cv2.INTER_LINEAR)
    metadata.update(status='usable_research_reference', usable_for_metric_scale=True,
                    sheet_pixels_per_mm=PIXELS_PER_MM,
                    homography_image_to_sheet_px=inverse.tolist(), rectified_shape=list(rectified.shape[:2]))
    return metadata, rectified, overlay
