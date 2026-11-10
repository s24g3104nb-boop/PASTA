"""Synthetic board contracts only: not physical calibration/weight evidence."""
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pasta_weight_app_v2.experiments.capture_sheet import render_sheet, build_svg
from pasta_weight_app_v2.experiments.marker_reference import analyze_marker_reference


def analyze(image, **kwargs):
    return analyze_marker_reference(image, print_scale_verified=True,
                                    coplanar_confirmed=True, **kwargs)


def test_synthetic_complete_sheet_returns_research_geometry_only():
    image = render_sheet()
    original = image.copy()
    meta, rectified, overlay = analyze(image)
    assert meta['usable_for_metric_scale'] and meta['geometry_valid']
    assert rectified.shape == (594, 420, 3)
    assert meta['sheet_pixels_per_mm'] == 2
    assert meta['weight_prediction_g'] is None
    assert meta['production_eligible'] is False
    assert meta['physical_scale_verified_by_software'] is False
    assert not np.array_equal(image, overlay)
    np.testing.assert_array_equal(image, original)
    json.dumps(meta, allow_nan=False)


@pytest.mark.parametrize('printing,coplanar', [(False, False), (False, True), (True, False)])
def test_print_and_plane_must_be_declared(printing, coplanar):
    meta, rectified, _ = analyze_marker_reference(render_sheet(), print_scale_verified=printing,
                                                coplanar_confirmed=coplanar)
    assert meta['geometry_valid'] and meta['status'] == 'confirmation_required'
    assert not meta['usable_for_metric_scale']
    assert meta['sheet_pixels_per_mm'] is None and rectified is None


@pytest.mark.parametrize('transformation', ['half', 'rotate90', 'perspective', 'jpeg', 'darker'])
def test_synthetic_supported_transformations_keep_reference(transformation):
    image = render_sheet()
    if transformation == 'half':
        image = cv2.resize(image, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    elif transformation == 'rotate90':
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif transformation == 'perspective':
        original = np.array([[0, 0], [839, 0], [839, 1187], [0, 1187]], np.float32)
        target = np.array([[130, 90], [900, 150], [850, 1320], [80, 1250]], np.float32)
        image = cv2.warpPerspective(image, cv2.getPerspectiveTransform(original, target), (1000, 1400), borderValue=(210, 210, 210))
    elif transformation == 'jpeg':
        _, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 70])
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    elif transformation == 'darker':
        image = np.round(image * .65).astype(np.uint8)
    meta, rectified, _ = analyze(image)
    assert meta['usable_for_metric_scale'], meta['reasons']
    assert rectified.shape[:2] == (594, 420)


@pytest.mark.parametrize('defect', ['covered', 'clipped', 'glare', 'tiny', 'mirror', 'blank', 'duplicate', 'moved', 'rotated_marker', 'extreme_perspective'])
def test_incomplete_ambiguous_or_deformed_board_never_guesses(defect):
    image = render_sheet()
    if defect == 'covered':
        image[38:122, 38:122] = 255
    elif defect == 'clipped':
        # All four markers survive but a paper edge is not in the frame.
        image = image[20:]
    elif defect == 'glare':
        image[60:100, 40:120] = 255
    elif defect == 'tiny':
        image = render_sheet(1)
    elif defect == 'mirror':
        image = image[:, ::-1].copy()
    elif defect == 'blank':
        image[:] = 255
    elif defect == 'duplicate':
        image[350:430, 350:430] = image[40:120, 40:120]
    elif defect == 'moved':
        image[40:120, 600:680] = image[40:120, 720:800]
        image[40:120, 720:800] = 255
    elif defect == 'rotated_marker':
        image[40:120, 40:120] = cv2.rotate(image[40:120, 40:120], cv2.ROTATE_90_CLOCKWISE)
    else:
        source = np.array([[0, 0], [839, 0], [839, 1187], [0, 1187]], np.float32)
        target = np.array([[450, 50], [650, 50], [900, 1300], [70, 1300]], np.float32)
        image = cv2.warpPerspective(image, cv2.getPerspectiveTransform(source, target), (1000, 1400), borderValue=(210, 210, 210))
    meta, rectified, _ = analyze(image)
    assert not meta['usable_for_metric_scale'], defect
    assert meta['status'] == 'retake' and meta['reasons']
    assert meta['sheet_pixels_per_mm'] is None and rectified is None
    assert meta['weight_prediction_g'] is None


def test_different_ids_are_rejected():
    image = render_sheet()
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    image[40:120, 40:120] = cv2.aruco.generateImageMarker(dictionary, 20, 80)[:, :, None]
    assert 'all_four_expected_markers_required' in analyze(image)[0]['reasons']


@pytest.mark.parametrize('image', [None, np.zeros((50, 50), np.uint8), np.zeros((50, 50, 3), float), np.zeros((4, 4, 3), np.uint8)])
def test_invalid_input_rejected(image):
    with pytest.raises(ValueError):
        analyze(image)


def test_claims_are_not_truthy_strings():
    with pytest.raises(ValueError):
        analyze_marker_reference(render_sheet(), print_scale_verified='yes')


def test_vector_sheet_dimensions_and_ruler_are_explicit():
    svg = build_svg()
    assert 'width="210mm" height="297mm" viewBox="0 0 210 297"' in svg
    assert 'M55 277 H155' in svg
    assert 'RESEARCH PROTOTYPE' in svg
