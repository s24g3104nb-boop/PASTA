import csv
from hashlib import sha256
from io import BytesIO, StringIO

from PIL import Image
import pytest

from pasta_weight_app_v2.experiments.capture_preflight import review_preflight, pilot_template_csv
from pasta_weight_app_v2.experiments.capture_sheet import render_sheet


def image_bytes(dense=False):
    image = render_sheet()
    if dense:
        image[240:840, 260:500] = (220, 180, 40)
    stream = BytesIO()
    Image.fromarray(image).save(stream, 'PNG')
    return stream.getvalue()


def test_printing_is_a_manual_measurement_and_projection_stays_nominal():
    content = image_bytes(dense=True)
    wrong, _ = review_preflight(content, photo_role='dense', printed_line_mm=96, flat_sheet_confirmed=True)
    assert wrong['geometry']['reference_geometry_valid'] and not wrong['within_print_line_tolerance']
    assert wrong['geometry']['projection_scale_status'] == 'nominal_print_unverified'
    checked, _ = review_preflight(content, photo_role='dense', printed_line_mm=100,
                                 flat_sheet_confirmed=True, camera_distance_cm=55, tilt_deg=15, bundle_height_mm=20)
    assert checked['original_sha256'] == sha256(content).hexdigest()
    assert checked['geometry']['sheet_plane_projection']['nominal_area_mm2'] == pytest.approx(9000, rel=.03)
    assert 0 < checked['geometry']['sheet_plane_projection']['candidate_density'] <= 1
    assert not checked['geometry']['reference']['usable_for_metric_scale']
    assert checked['geometry']['public_prediction_g'] is None
    assert not checked['geometry']['physical_scale_verified_by_software']


def test_external_ruler_check_on_sheet_has_no_foreground_projection():
    result, _ = review_preflight(image_bytes(), photo_role='empty_sheet', printed_line_mm=100,
                                 flat_sheet_confirmed=True, ruler_points_xy=[[200, 600], [600, 600]], ruler_length_mm=100)
    assert sorted(result['geometry']['reference']['detected_ids']) == [0, 1, 2, 3]
    assert result['geometry']['sheet_plane_projection'] is None
    assert result['independent_ruler']['projected_length_mm'] == pytest.approx(100, rel=.012)


def test_pilot_template_requires_actual_values_and_has_six_distinct_samples():
    rows = list(csv.DictReader(StringIO(pilot_template_csv())))
    assert len(rows) == len({r['shot_id'] for r in rows}) == 20
    weights = {r['nominal_weight_g'] for r in rows if r['role'] == 'dense'}
    assert weights == {'120', '160', '200'}
    sample_ids = {r['sample_id'] for r in rows if r['role'] == 'dense'}
    assert len(sample_ids) == 6
    for sample in sample_ids:
        assert {(r['target_distance_cm'],r['target_tilt_deg']) for r in rows if r['sample_id'] == sample} == {
            ('35','0'), ('55','0'), ('55','15')}
    assert not any(r['measured_g'] or r['filename'] or r['lineage_note'] for r in rows)
