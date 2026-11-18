"""Synthetic plane/footprint checks, not real camera or weight evaluation."""
from io import BytesIO
from pathlib import Path
import sys
import cv2
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pasta_weight_app_v2.experiments.capture_sheet import render_sheet
from pasta_weight_app_v2.experiments.pilot_geometry import analyze_pilot_geometry, ideal_height_bias


def dense_sheet():
    image = render_sheet(4)
    image[240:840, 260:500] = (220, 180, 40)
    return image


def test_dense_projection_does_not_claim_coplanarity_or_weight():
    image = dense_sheet(); before = image.copy()
    data, overlay = analyze_pilot_geometry(image, print_scale_verified=True,
                                            camera_distance_cm=50, bundle_height_mm=10, tilt_deg=0)
    assert data['reference_geometry_valid']
    assert data['sheet_plane_projection']['nominal_area_mm2'] == pytest.approx(9000, rel=.03)
    assert not data['pasta_coplanar_verified'] and not data['whole_pasta_verified']
    assert not data['reference']['coplanar_confirmed']
    assert not data['reference']['usable_for_metric_scale']
    assert data['public_prediction_g'] is None and not data['production_eligible']
    assert data['height_diagnostic']['linear_bias_fraction'] == pytest.approx(10/490)
    np.testing.assert_array_equal(image, before)
    assert not np.array_equal(image, overlay)


@pytest.mark.parametrize('transform', ['half', 'perspective'])
def test_same_dense_footprint_projection_survives_synthetic_distance_and_tilt(transform):
    image = dense_sheet()
    ref, _ = analyze_pilot_geometry(image)
    if transform == 'half':
        image = cv2.resize(image, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    else:
        source = np.array([[0,0],[839,0],[839,1187],[0,1187]], np.float32)
        target = np.array([[130,90],[900,150],[850,1320],[80,1250]], np.float32)
        image = cv2.warpPerspective(image, cv2.getPerspectiveTransform(source,target),
                                    (1000,1400), borderValue=(30,30,30))
    result, _ = analyze_pilot_geometry(image)
    assert result['reference_geometry_valid']
    assert result['sheet_plane_projection']['nominal_area_mm2'] == pytest.approx(ref['sheet_plane_projection']['nominal_area_mm2'], rel=.04)
    assert result['projection_scale_status'] == 'nominal_print_unverified'
    assert 'printed_scale_unverified' in result['qc_flags']


@pytest.mark.parametrize('defect', ['covered', 'cropped'])
def test_invalid_marker_geometry_never_returns_projection(defect):
    image = dense_sheet()
    if defect == 'covered':image[40:120,40:120] = 255
    else:image = image[20:]
    result, _ = analyze_pilot_geometry(image,print_scale_verified=True)
    assert not result['reference_geometry_valid']
    assert result['sheet_plane_projection'] is None


def test_all_markers_visible_does_not_allow_pasta_outside_sheet_or_capture():
    sheet = dense_sheet()
    image = np.full((1388,1040,3),30,np.uint8);image[100:1288,100:940] = sheet
    image[500:700,60:150] = (220,180,40)
    data, _ = analyze_pilot_geometry(image)
    assert data['reference_geometry_valid']
    assert data['candidate_foreground_outside_sheet_fraction'] > .01
    assert data['sheet_plane_projection'] is None
    image[500:700,:150] = (220,180,40)
    clipped, _ = analyze_pilot_geometry(image)
    assert 'foreground_touches_frame' in clipped['qc_flags']
    assert clipped['status'] == 'retake_capture'
    assert clipped['sheet_plane_projection'] is None


def test_padding_does_not_clear_inherited_clipping():
    image = dense_sheet()
    result, _ = analyze_pilot_geometry(image, inherited_flags=['capture_content_clipped'])
    assert result['status'] == 'retake_capture' and result['sheet_plane_projection'] is None
    assert 'capture_content_clipped' in result['qc_flags']


def test_empty_board_is_geometry_check_not_missing_pasta_failure_and_bytes_supported():
    buffer = BytesIO();Image.fromarray(render_sheet()).save(buffer,'PNG')
    result, _ = analyze_pilot_geometry(buffer.getvalue(),photo_role='empty_sheet')
    assert result['status'] == 'sheet_geometry_only'
    assert 'insufficient_foreground' not in result['qc_flags']
    assert result['sheet_plane_projection'] is None


def test_unknown_or_oblique_height_has_no_real_error_bound():
    assert ideal_height_bias()['linear_bias_fraction'] is None
    result = ideal_height_bias(50,20,15)
    assert not result['applicable_idealization'] and result['linear_bias_fraction'] is None
    assert not result['verified_real_error_bound']
    with pytest.raises(ValueError):ideal_height_bias(50,float('nan'),0)
    with pytest.raises(ValueError):analyze_pilot_geometry(dense_sheet(),print_scale_verified='yes')


@pytest.mark.parametrize('transform', ['original','half','large','perspective'])
def test_independent_ruler_segment_uses_original_coordinate_contract(transform):
    from pasta_weight_app_v2.experiments.pilot_geometry import measure_pilot_reference_segment
    image=render_sheet(4)
    points=np.array([[200,600],[600,600]],np.float64)
    if transform=='half':
        image=cv2.resize(image,None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA);points*=.5
    elif transform=='large':
        image=render_sheet(8);points*=2
    elif transform=='perspective':
        source=np.array([[0,0],[839,0],[839,1187],[0,1187]],np.float32)
        target=np.array([[130,90],[900,150],[850,1320],[80,1250]],np.float32)
        matrix=cv2.getPerspectiveTransform(source,target)
        image=cv2.warpPerspective(image,matrix,(1000,1400),borderValue=(30,30,30))
        points=cv2.perspectiveTransform(points[None,...],matrix)[0]
    metadata,_=analyze_pilot_geometry(image,photo_role='empty_sheet',print_scale_verified=True)
    measured=measure_pilot_reference_segment(metadata,points,100)
    assert measured['projected_length_mm']==pytest.approx(100,rel=.012)
    assert measured['relative_error']<.012
    assert not measured['physical_scale_verified_by_software']
    assert measured['weight_prediction_g'] is None


def test_bad_ruler_coordinates_reference_and_expected_length_are_rejected():
    from pasta_weight_app_v2.experiments.pilot_geometry import measure_pilot_reference_segment
    metadata,_=analyze_pilot_geometry(render_sheet(),photo_role='empty_sheet')
    for points in ([[1,2],[3,4],[5,6]],[[1,2],[float('nan'),4]],[[-1,2],[100,4]],[[1,2],[9999,4]],[[1,2],[1,2]]):
        with pytest.raises(ValueError):measure_pilot_reference_segment(metadata,points,100)
    for expected in (None,0,-1,float('nan'),True):
        with pytest.raises(ValueError):measure_pilot_reference_segment(metadata,[[200,600],[600,600]],expected)
    wrong=dict(metadata,reference_geometry_valid=False)
    with pytest.raises(ValueError):measure_pilot_reference_segment(wrong,[[200,600],[600,600]],100)
    image=np.full((1388,1040,3),30,np.uint8);image[100:1288,100:940]=render_sheet()
    bordered,_=analyze_pilot_geometry(image,photo_role='empty_sheet')
    with pytest.raises(ValueError):measure_pilot_reference_segment(bordered,[[50,600],[600,600]],100)


def test_empty_board_with_colored_ruler_is_not_evaluated_as_a_pasta_bundle():
    image=render_sheet();image[500:530,:700]=(220,180,40)
    result,_=analyze_pilot_geometry(image,photo_role='empty_sheet')
    assert result['reference_geometry_valid'] and result['status']=='sheet_geometry_only'
    assert result['homography_normalized_image_to_nominal_sheet_px'] is not None
    assert result['sheet_plane_projection'] is None


def test_native_border_candidate_is_retained_before_reference_downsizing():
    image=render_sheet(8);image[480:1680,520:1000]=(220,180,40)
    image[1200,0]=(220,180,40)
    result,_=analyze_pilot_geometry(image)
    assert result['reference']['input_scale_factor']<1
    assert result['source_border_candidate_fraction']>0
    assert result['status']=='retake_capture' and result['sheet_plane_projection'] is None


def test_extreme_aspect_ratio_and_invalid_inherited_flags_are_cleanly_rejected():
    with pytest.raises(ValueError):analyze_pilot_geometry(np.zeros((16,10000,3),np.uint8))
    with pytest.raises(ValueError):analyze_pilot_geometry(dense_sheet(),inherited_flags=None)
