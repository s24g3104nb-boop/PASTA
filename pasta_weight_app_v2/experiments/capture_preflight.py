"""Review A4 geometry from one preflight photo and prepare the fixed pilot ledger."""
from hashlib import sha256
import csv
from io import StringIO
import math

from .pilot_geometry import analyze_pilot_geometry, measure_pilot_reference_segment

PRINT_TOLERANCE_MM = 1.0
PILOT_WEIGHTS_G = (120, 160, 200)
PILOT_VIEWS = (('near', 35, 0), ('far', 55, 0), ('tilt', 55, 15))


def review_preflight(content, *, photo_role, printed_line_mm=None, flat_sheet_confirmed=False,
                     camera_distance_cm=None, tilt_deg=None, bundle_height_mm=None,
                     ruler_points_xy=None, ruler_length_mm=None):
    """An operator's ruler measurement is required to treat the print as 100%.

    The optional independent ruler check compares the scale after rectification.
    Its endpoints are EXIF-oriented source pixel coordinates. A dense bundle's
    projection stays nominal, even when all the sheet-plane checks pass.
    """
    if not isinstance(content, bytes):
        raise ValueError('Original photo bytes required')
    if type(flat_sheet_confirmed) is not bool:
        raise ValueError('Sheet flatness requires an explicit confirmation')
    if printed_line_mm is not None and (type(printed_line_mm) not in (int, float)
                                       or not math.isfinite(printed_line_mm) or printed_line_mm <= 0):
        raise ValueError('Measure the actual printed line in millimeters')
    if (ruler_points_xy is None) != (ruler_length_mm is None):
        raise ValueError('Both the independent ruler endpoints and their length are required')
    print_verified = printed_line_mm is not None and abs(printed_line_mm - 100) <= PRINT_TOLERANCE_MM
    metadata, overlay = analyze_pilot_geometry(content, photo_role=photo_role,
        print_scale_verified=print_verified and flat_sheet_confirmed, camera_distance_cm=camera_distance_cm,
        bundle_height_mm=bundle_height_mm, tilt_deg=tilt_deg)
    ruler = None
    if ruler_points_xy is not None and metadata['reference_geometry_valid']:
        ruler = measure_pilot_reference_segment(metadata, ruler_points_xy, ruler_length_mm)
    return dict(original_sha256=sha256(content).hexdigest(), printed_line_mm=printed_line_mm,
                within_print_line_tolerance=print_verified, flat_sheet_confirmed=flat_sheet_confirmed,
                distance_cm=camera_distance_cm, tilt_deg=tilt_deg, bundle_height_mm=bundle_height_mm,
                geometry=metadata, independent_ruler=ruler), overlay


def pilot_rows():
    """Two empty-sheet checks plus three photos of each of six separately weighed samples."""
    yield dict(shot_id='sheet-near', role='empty_sheet', sample_id='', nominal_weight_g='', target_distance_cm=35, target_tilt_deg=0)
    yield dict(shot_id='sheet-far', role='empty_sheet', sample_id='', nominal_weight_g='', target_distance_cm=55, target_tilt_deg=0)
    for weight in PILOT_WEIGHTS_G:
        for sample_index in (1, 2):
            sample = f'{weight}g-{sample_index}'
            for view, distance, angle in PILOT_VIEWS:
                yield dict(shot_id=f'{sample}-{view}', role='dense', sample_id=sample,
                           nominal_weight_g=weight, target_distance_cm=distance, target_tilt_deg=angle)


def pilot_template_csv():
    rows = list(pilot_rows())
    measurements = ('filename', 'measured_g', 'distance_cm', 'tilt_deg', 'bundle_height_mm',
                    'product', 'scale_resolution_g', 'camera_and_lens', 'lineage_note', 'mask_review_note')
    result = StringIO()
    writer = csv.DictWriter(result, fieldnames=[*rows[0], *measurements])
    writer.writeheader()
    writer.writerows(rows)
    return result.getvalue()
