"""A4 capture-feasibility diagnostics for a dense, unchanged pasta bundle.

The image is projected onto the sheet plane only to compare repeated captures.
Elevated noodles are never declared coplanar. No true 3D area, mass, learning
approval, sample independence or camera/printer accuracy is inferred here.
"""
from __future__ import annotations

from io import BytesIO
import math

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .capture_sheet import WIDTH_MM, HEIGHT_MM, PIXELS_PER_MM
from .marker_reference import analyze_marker_reference
from .research_contract import CLIPPING_FLAGS
from .stability_pipeline import configurations, raw_mask, shape_features

VERSION = 'dense-pilot-sheet-projection-v1'
CONFIG = dict(max_image_bytes=20 * 1024 * 1024, max_pixels=40_000_000,
              min_side_px=16, outside_sheet_review_fraction=.01,
              minimum_candidate_fraction=.005, projection_pixels_per_nominal_mm=PIXELS_PER_MM,
              height_approximation_max_tilt_deg=10.)


def _rgb(source):
    if isinstance(source, bytes):
        if not source or len(source) > CONFIG['max_image_bytes']:
            raise ValueError('Expected an image of at most 20MiB')
        try:
            with Image.open(BytesIO(source)) as image:
                if (image.format not in {'JPEG', 'PNG', 'WEBP'} or getattr(image, 'n_frames', 1) != 1
                        or min(image.size) < CONFIG['min_side_px']
                        or image.width * image.height > CONFIG['max_pixels']):
                    raise ValueError('Unsupported image format or dimensions')
                image.load()
                return np.asarray(ImageOps.exif_transpose(image).convert('RGB'))
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ValueError('Image cannot be decoded') from exc
    if (not isinstance(source, np.ndarray) or source.dtype != np.uint8
            or source.ndim != 3 or source.shape[2] != 3
            or min(source.shape[:2]) < CONFIG['min_side_px']
            or source.shape[0] * source.shape[1] > CONFIG['max_pixels']):
        raise ValueError('Expected a bounded RGB uint8 image')
    return source


def _optional_number(value, name, *, positive=False):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name + ' must be finite or None')
    if value < 0 or (positive and value == 0):
        raise ValueError(name + ' has an invalid range')
    return float(value)


def ideal_height_bias(camera_distance_cm=None, bundle_height_mm=None, tilt_deg=None):
    """Illustrative parallel-plane pinhole bias, not a measured or general bound.

    D is distance to the sheet, h is bundle height. The formula assumes a nearly
    normal camera and a parallel surface. It cannot correct unknown layers,
    real-camera lens distortion or oblique bundle geometry.
    """
    distance = _optional_number(camera_distance_cm, 'camera_distance_cm', positive=True)
    height = _optional_number(bundle_height_mm, 'bundle_height_mm')
    tilt = _optional_number(tilt_deg, 'tilt_deg')
    if tilt is not None and tilt > 90:
        raise ValueError('tilt_deg must be between 0 and 90')
    result = dict(applicable_idealization=False, linear_bias_fraction=None,
                  area_bias_fraction=None, camera_distance_cm=distance,
                  bundle_height_mm=height, tilt_deg=tilt,
                  verified_real_error_bound=False,
                  assumption='Ideal pinhole, near-normal camera, parallel elevated surface; not a weight correction')
    if distance is None or height is None or tilt is None:
        result['reason'] = 'distance_height_or_tilt_unmeasured'
    elif distance * 10 <= height:
        result['reason'] = 'bundle_height_not_below_camera_distance'
    elif tilt > CONFIG['height_approximation_max_tilt_deg']:
        result['reason'] = 'oblique_view_outside_overhead_idealization'
    else:
        distance_mm = distance * 10
        if not math.isfinite(distance_mm):
            result['reason'] = 'height_idealization_numeric_range_invalid'
            return result
        scale = distance_mm / (distance_mm - height)
        if not math.isfinite(scale * scale):
            result['reason'] = 'height_idealization_numeric_range_invalid'
            return result
        result.update(applicable_idealization=True, linear_bias_fraction=scale-1,
                      area_bias_fraction=scale*scale-1,
                      reason='idealized_height_diagnostic_only')
    return result


def _capture_border_fraction(rgb, config):
    """Inspect native border pixels before shrinking; padding cannot undo flags."""
    border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]), axis=0)[None, ...]
    return float(np.count_nonzero(raw_mask(border, config)) / border.shape[1])


def analyze_pilot_geometry(source, *, photo_role='dense', print_scale_verified=False,
                           camera_distance_cm=None, bundle_height_mm=None, tilt_deg=None,
                           inherited_flags=()):
    """Return ``(JSON-compatible metadata, RGB overlay)`` for one photo.

    photo_role is ``dense`` or ``empty_sheet``. A printed-scale confirmation is
    only the operator's declaration. Nominal paper-coordinate features may be
    used for repeatability comparisons but are not true elevated-pasta dimensions.
    Missing/invalid markers return no projection. Original border warnings and
    supplied capture flags survive rectification and padding.
    """
    if photo_role not in {'dense', 'empty_sheet'} or type(print_scale_verified) is not bool:
        raise ValueError('Explicit photo role and boolean printing declaration required')
    if isinstance(inherited_flags, (str, bytes)):
        raise ValueError('inherited_flags must be a collection of strings')
    try:
        flags = set(inherited_flags)
    except TypeError as exc:
        raise ValueError('inherited_flags must be a collection of strings') from exc
    if any(not isinstance(flag, str) or not flag.strip() for flag in flags):
        raise ValueError('Invalid inherited capture flags')
    rgb = _rgb(source)
    if round(min(rgb.shape[:2]) * min(1., 1600/max(rgb.shape[:2]))) < 16:
        raise ValueError('Image aspect ratio leaves too few pixels for reference geometry')
    height_diagnostic = ideal_height_bias(camera_distance_cm, bundle_height_mm, tilt_deg)
    config = next(c for c in configurations() if c.name == 'B_normalized5')
    source_border = _capture_border_fraction(rgb, config)
    if source_border > 0:
        flags.add('foreground_touches_frame')
    # Geometry-only analysis intentionally does not assert coplanarity or trust
    # a dense bundle as an object on the paper plane.
    reference, _, overlay = analyze_marker_reference(rgb, print_scale_verified=False,
                                                     coplanar_confirmed=False)
    shape = reference['normalized_shape']
    scene = cv2.resize(rgb, (shape[1], shape[0]), interpolation=cv2.INTER_AREA) if list(rgb.shape[:2]) != shape else rgb.copy()
    mask = raw_mask(scene, config)
    fraction = float(np.count_nonzero(mask)/mask.size)
    result = dict(version=VERSION, config=dict(CONFIG), photo_role=photo_role,
                  original_shape=list(rgb.shape[:2]),
                  status='retake_reference', reference=reference,
                  reference_geometry_valid=reference['geometry_valid'],
                  print_scale_verified_by_operator=print_scale_verified,
                  projection_scale_status='operator_print_verified_sheet_plane_only' if print_scale_verified else 'nominal_print_unverified',
                  candidate_foreground_fraction=fraction,
                  candidate_foreground_outside_sheet_fraction=None,
                  source_border_candidate_fraction=source_border,
                  sheet_plane_projection=None, homography_normalized_image_to_nominal_sheet_px=None,
                  qc_flags=[], whole_pasta_verified=False, pasta_coplanar_verified=False,
                  physical_scale_verified_by_software=False, height_diagnostic=height_diagnostic,
                  public_prediction_g=None, production_eligible=False, training_approved=False,
                  limitation='Visible candidate projection onto the A4 plane only; dense hidden/elevated quantity and weight remain unknown')
    if not print_scale_verified:
        flags.add('printed_scale_unverified')
    if not reference['geometry_valid']:
        flags.add('reference_geometry_unavailable')
        result['qc_flags'] = sorted(flags)
        return result, overlay
    result['status'] = 'sheet_geometry_only' if photo_role == 'empty_sheet' else 'candidate_projection_review'
    paper = np.asarray(reference['projected_sheet_corners'], np.float32)
    destination = np.array([[0, 0], [WIDTH_MM * PIXELS_PER_MM, 0],
                            [WIDTH_MM * PIXELS_PER_MM, HEIGHT_MM * PIXELS_PER_MM],
                            [0, HEIGHT_MM * PIXELS_PER_MM]], np.float32)
    homography = cv2.getPerspectiveTransform(paper, destination)
    if not np.isfinite(homography).all():
        flags.add('invalid_sheet_projection')
        result.update(status='retake_reference', qc_flags=sorted(flags))
        return result, overlay
    result['homography_normalized_image_to_nominal_sheet_px'] = homography.tolist()
    if photo_role == 'empty_sheet':
        # A tan ruler or another calibration aid is not a pasta candidate to
        # accept/reject. The empty-board role only checks the reference plane.
        result['qc_flags'] = sorted(flags)
        return result, overlay
    if fraction < CONFIG['minimum_candidate_fraction']:
        if photo_role == 'dense':
            flags.add('insufficient_foreground')
            result['status'] = 'review_foreground_missing'
        if flags & CLIPPING_FLAGS:
            result['status'] = 'retake_capture'
        result['qc_flags'] = sorted(flags)
        return result, overlay
    inside = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(inside, np.round(paper).astype(np.int32), 255)
    outside = float(np.count_nonzero((mask > 0) & (inside == 0))/max(1, np.count_nonzero(mask)))
    result['candidate_foreground_outside_sheet_fraction'] = outside
    if outside > CONFIG['outside_sheet_review_fraction']:
        flags.add('candidate_foreground_outside_sheet')
        result['status'] = 'review_foreground_outside_sheet'
    if flags & CLIPPING_FLAGS:
        result['status'] = 'retake_capture'
    # Never report a truncated paper crop as the whole bundle's projection.
    if not (flags & CLIPPING_FLAGS) and 'candidate_foreground_outside_sheet' not in flags:
        projected = cv2.warpPerspective(mask, homography,
            (round(WIDTH_MM*PIXELS_PER_MM), round(HEIGHT_MM*PIXELS_PER_MM)), flags=cv2.INTER_NEAREST)
        features = shape_features(projected, config)
        if features['mask_fraction'] > 0:
            result['sheet_plane_projection'] = dict(
                nominal_area_mm2=float(features['mask_area_px']/PIXELS_PER_MM**2),
                nominal_length_mm=float(features['quantile_length_px']/PIXELS_PER_MM),
                nominal_width_mm=float(features['quantile_width_px']/PIXELS_PER_MM),
                # Inclusive raster endpoints can put the pixel ratio slightly above one.
                candidate_density=min(1., float(features['quantile_density'])),
                definition='Candidate mask projected onto nominal sheet plane; NOT true 3D pasta dimensions')
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 20, 180), 1)
    result['qc_flags'] = sorted(flags)
    return result, overlay


def measure_pilot_reference_segment(metadata, original_points_xy, expected_mm):
    """Compare a declared ruler segment with the sheet-plane homography.

    Points must be exactly two EXIF-oriented ORIGINAL-image pixel coordinates,
    not coordinates in an uploaded thumbnail. The physical ruler, its indicated
    length, paper print size and its coplanarity must still be checked by a human.
    This is an external geometry check, not validation of the printed 100mm line
    by the same marker geometry, and never establishes weight accuracy.
    """
    expected = _optional_number(expected_mm, 'expected_mm', positive=True)
    if expected is None:
        raise ValueError('An independently known ruler length is required')
    if not isinstance(metadata, dict) or metadata.get('reference_geometry_valid') is not True:
        raise ValueError('Valid four-marker sheet geometry required')
    reference = metadata.get('reference', {})
    if reference.get('geometry_valid') is not True:
        raise ValueError('Reference metadata does not contain valid geometry')
    raw = np.asarray(original_points_xy)
    if raw.shape != (2, 2) or raw.dtype.kind not in 'iuf':
        raise ValueError('Exactly two numeric original-image points are required')
    points = raw.astype(float)
    if not np.isfinite(points).all() or np.linalg.norm(points[1]-points[0]) <= 0:
        raise ValueError('Ruler endpoints must be distinct finite points')
    try:
        h, w = metadata['original_shape']
        factor = float(reference['input_scale_factor'])
        homography = np.asarray(metadata['homography_normalized_image_to_nominal_sheet_px'], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Incomplete sheet coordinate mapping') from exc
    if (type(h) is not int or type(w) is not int or min(h,w) < 16
            or not np.isfinite(factor) or not 0 < factor <= 1 or homography.shape != (3,3)
            or not np.isfinite(homography).all()):
        raise ValueError('Invalid sheet coordinate mapping')
    if (np.any(points < 0) or np.any(points[:,0] > w-1) or np.any(points[:,1] > h-1)):
        raise ValueError('Ruler endpoints must be inside the original image')
    projected = cv2.perspectiveTransform((points*factor)[None, ...], homography)[0]
    if (not np.isfinite(projected).all() or np.any(projected < 0)
            or np.any(projected[:,0] > WIDTH_MM*PIXELS_PER_MM)
            or np.any(projected[:,1] > HEIGHT_MM*PIXELS_PER_MM)):
        raise ValueError('Ruler endpoints must lie on the visible A4 sheet')
    measured = float(np.linalg.norm(projected[1]-projected[0])/PIXELS_PER_MM)
    if measured <= 0 or not math.isfinite(measured):
        raise ValueError('Invalid projected ruler length')
    relative_error = float(abs(measured-expected)/expected)
    if not math.isfinite(relative_error):
        raise ValueError('Expected ruler length is outside numeric range')
    return dict(expected_mm=expected, projected_length_mm=measured,
                relative_error=relative_error,
                signed_error_mm=measured-expected, original_points_xy=points.tolist(),
                projected_points_nominal_sheet_mm=(projected/PIXELS_PER_MM).tolist(),
                print_scale_verified_by_operator=metadata.get('print_scale_verified_by_operator') is True,
                physical_scale_verified_by_software=False, weight_prediction_g=None,
                purpose='External ruler segment geometry comparison only; physical ruler/print/plane require human confirmation')
