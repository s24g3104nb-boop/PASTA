"""Experimental sheet-plane features. No guessed reference and no weight output."""
from dataclasses import asdict
import cv2
import numpy as np
from .marker_reference import analyze_marker_reference
from .stability_pipeline import configurations, process
from .research_contract import propagate_capture_qc

METRIC_KEYS = ('visible_area_mm2', 'projection_length_mm', 'projection_width_mm',
               'visible_perimeter_mm', 'projection_density')


def extract_metric_foreground(rgb, *, print_scale_verified=False, coplanar_confirmed=False):
    """Require validated markers and explicit confirmations; retain source clipping.

    These are visible foreground projections, not total pasta surface/volume.
    A thick bundle is not coplanar with the sheet. A successful homography cannot
    verify printer dimensions, strand diameter, overlap or true weight.
    """
    config = next(c for c in configurations() if c.name == 'B_normalized5')
    original, _, _ = process(rgb, config)
    metadata, rectified, overlay = analyze_marker_reference(
        rgb, print_scale_verified=print_scale_verified, coplanar_confirmed=coplanar_confirmed)
    result = dict(reference=metadata, features=None, mask=None, rectified=None, overlay=overlay,
                  original_qc_flags=original['qc_flags'], public_prediction_g=None,
                  production_eligible=False)
    if not metadata['usable_for_metric_scale'] or rectified is None:
        result['status'] = 'reference_unavailable'
        return result
    from dataclasses import replace
    # Do not resize an already metric-calibrated image before measuring pixels.
    physical_config = replace(config, name='A4_sheet_native', target_long_side=None)
    features, scene, mask = process(rectified, physical_config)
    combined = propagate_capture_qc(original['qc_flags'], features['qc_flags'])
    result.update(mask=mask, rectified=scene, qc_flags=combined['qc_flags'])
    if 'foreground_touches_frame' in combined['qc_flags'] or features['mask_fraction'] < .005:
        result['status'] = 'foreground_incomplete_or_missing'
        return result
    ratio = metadata['sheet_pixels_per_mm']
    if not isinstance(ratio, (int, float)) or not np.isfinite(ratio) or ratio <= 0:
        result['status'] = 'reference_unavailable'
        return result
    values = dict(visible_area_mm2=features['mask_area_px']/ratio**2,
                  projection_length_mm=features['quantile_length_px']/ratio,
                  projection_width_mm=features['quantile_width_px']/ratio,
                  visible_perimeter_mm=features['contour_perimeter_px']/ratio,
                  projection_density=features['quantile_density'])
    result.update(status='research_features_only', features=values, pipeline_config=asdict(physical_config))
    return result
