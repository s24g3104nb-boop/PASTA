"""Synthetic detector contracts; these are not real-photo accuracy evidence."""
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pasta_weight_app_v2.experiments.reference_scale import analyze_reference, VERSION
from pasta_weight_app_v2.experiments.stability_pipeline import normalize_size


def synthetic_tray():
    scene = np.full((384, 512, 3), 230, np.uint8)
    # Fully visible rounded dark rectangle, with glare and yellow foreground.
    cv2.rectangle(scene, (90, 70), (422, 314), (35, 35, 35), -1)
    cv2.rectangle(scene, (70, 90), (442, 294), (35, 35, 35), -1)
    for center in [(90, 90), (422, 90), (90, 294), (422, 294)]:
        cv2.circle(scene, center, 20, (35, 35, 35), -1)
    cv2.line(scene, (110, 100), (200, 100), (245, 245, 245), 3)
    cv2.rectangle(scene, (130, 170), (350, 205), (230, 180, 50), -1)
    return scene


def test_fully_visible_tray_has_only_unapproved_relative_geometry():
    metadata, scene, mask, overlay = analyze_reference(synthetic_tray())
    assert metadata['status'] == 'candidate_unapproved'
    assert metadata['usable_for_relative_scale']
    assert metadata['area_px'] > 80_000
    assert metadata['long_side_px'] > metadata['short_side_px'] > 200
    assert metadata['approval_status'] == 'UNAPPROVED'
    assert metadata['physical_calibration'] is False
    assert metadata['eligible_for_weight_inference'] is False
    assert metadata['same_reference_object_verified'] is False
    assert metadata['perspective_rectified'] is False
    assert metadata['version'] == VERSION
    assert len(metadata['config_sha256']) == 64
    json.dumps(metadata, allow_nan=False)
    assert mask.dtype == np.uint8 and mask.shape == scene.shape[:2]
    assert overlay.shape == scene.shape and not np.array_equal(scene, overlay)


@pytest.mark.parametrize('factor', [.5, .25])
def test_size_normalization_preserves_candidate_geometry(factor):
    scene = synthetic_tray()
    resized = cv2.resize(scene, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    before = analyze_reference(scene)[0]
    after, canonical, _, _ = analyze_reference(resized)
    assert after['usable_for_relative_scale']
    assert after['area_px'] == pytest.approx(before['area_px'], rel=.025)
    assert after['long_side_px'] == pytest.approx(before['long_side_px'], rel=.015)
    np.testing.assert_array_equal(canonical, normalize_size(resized, 512))


def test_padded_scene_scales_reference_area_without_using_image_frame():
    scene = synthetic_tray()
    small = cv2.resize(scene, None, fx=.75, fy=.75, interpolation=cv2.INTER_AREA)
    padded = np.full_like(scene, 230)
    padded[48:336, 64:448] = small
    before, after = analyze_reference(scene)[0], analyze_reference(padded)[0]
    assert after['usable_for_relative_scale']
    assert after['area_px'] / before['area_px'] == pytest.approx(.75 ** 2, rel=.025)
    assert after['long_side_px'] / before['long_side_px'] == pytest.approx(.75, rel=.015)


def test_cropped_reference_retains_candidate_but_has_no_usable_scale():
    cropped = synthetic_tray()[:, 100:]
    metadata, _, mask, _ = analyze_reference(cropped)
    assert metadata['candidate_found'] and np.any(mask)
    assert 'candidate_touches_frame' in metadata['reasons']
    assert metadata['candidate_area_px'] > 0
    assert metadata['area_px'] is None
    assert metadata['long_side_px'] is None and metadata['short_side_px'] is None
    assert not metadata['usable_for_relative_scale']


@pytest.mark.parametrize('color', [0, 255])
def test_missing_reference_never_falls_back_to_image_frame(color):
    metadata, _, _, _ = analyze_reference(np.full((384, 512, 3), color, np.uint8))
    assert not metadata['usable_for_relative_scale']
    assert metadata['area_px'] is None
    assert metadata['status'] == 'unavailable'
    assert metadata['reasons']


def test_competing_references_are_rejected():
    scene = np.full((384, 512, 3), 230, np.uint8)
    scene[70:310, 30:225] = 35
    scene[70:310, 285:480] = 35
    metadata, _, _, _ = analyze_reference(scene)
    assert metadata['candidate_count'] == 2
    assert 'ambiguous_dark_regions' in metadata['reasons']
    assert metadata['area_px'] is None


def test_reference_and_caller_array_are_not_modified():
    scene = synthetic_tray()
    original = scene.copy()
    first = analyze_reference(scene)
    second = analyze_reference(scene)
    np.testing.assert_array_equal(scene, original)
    assert first[0] == second[0]
    for lhs, rhs in zip(first[1:], second[1:]):
        np.testing.assert_array_equal(lhs, rhs)
    first[1][:] = 0
    np.testing.assert_array_equal(scene, original)


@pytest.mark.parametrize('invalid', [np.zeros((12, 12, 3), np.uint8),
                                     np.zeros((40, 40), np.uint8),
                                     np.zeros((40, 40, 3), np.float32)])
def test_invalid_inputs_are_rejected(invalid):
    with pytest.raises(ValueError):
        analyze_reference(invalid)
