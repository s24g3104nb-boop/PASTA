"""Synthetic regression fixtures for the shipped area-only estimator."""
from io import BytesIO
import json
import sys

import cv2
import numpy as np
import pytest
from PIL import Image

from pasta_weight_app_v2 import sheetless_area
from pasta_weight_app_v2.experiments.diagnose_sheetless_photo import main as diagnose_photo_file
from pasta_weight_app_v2.pasta_vision import image_to_rgb
from pasta_weight_app_v2.sheetless_area import AreaModel, Observation, estimate, estimate_checked, observe


def spread(strands):
    image = np.full((480, 640, 3), 42, np.uint8)
    for y in range(60, 60 + strands * 10, 10):
        image[y:y + 5, 64:576] = (220, 180, 40)
    return image


def test_more_visible_area_increases_weight_and_rotation_preserves_it():
    thin = estimate(spread(28))
    photo = spread(36)
    full = estimate(photo)
    rotated = estimate(np.rot90(photo))
    assert 0 < thin.weight_g < full.weight_g
    assert full.area_feature == 'area_over_length_squared'
    assert rotated.weight_g == pytest.approx(full.weight_g)


def test_low_resolution_still_converts_resolved_foreground():
    photo = cv2.resize(spread(36), (1400, 1050), interpolation=cv2.INTER_LINEAR)
    reduced = cv2.resize(photo, (700, 525), interpolation=cv2.INTER_AREA)
    assert estimate(reduced).weight_g == pytest.approx(estimate(photo).weight_g, abs=2)
    small_photo = cv2.resize(reduced, (350, 262), interpolation=cv2.INTER_AREA)
    too_small = estimate(small_photo)
    assert too_small.weight_g > 0
    assert '画像が小さすぎて' in observe(small_photo).reason
    assert not too_small.reason


@pytest.mark.parametrize('condition,reason', [
    ('empty', '領域が少なすぎます'),
    ('cropped', '画像の端'),
    ('dense', '厚い束'),
    ('bright', '暗く無地'),
    ('outside', '比較できる範囲'),
])
def test_research_checks_still_flag_capture_limits(condition, reason):
    image = spread(32)
    if condition == 'empty':
        image.fill(42)
    elif condition == 'cropped':
        image = image[70:, :]
    elif condition == 'dense':
        image[60:390, 64:576] = (220, 180, 40)
    elif condition == 'bright':
        image[image[:, :, 0] == 42] = (232, 232, 232)
    else:
        image = spread(8)
    result = estimate_checked(image)
    assert result.weight_g is None
    assert reason in result.reason


@pytest.mark.parametrize('condition', ['cropped', 'dense', 'bright'])
def test_public_estimate_displays_despite_capture_findings(condition):
    image = spread(32)
    if condition == 'cropped':
        image = image[70:, :]
    elif condition == 'dense':
        image[60:390, 64:576] = (220, 180, 40)
    else:
        image[image[:, :, 0] == 42] = (232, 232, 232)
    result = estimate(image)
    assert result.weight_g > 0 and observe(image).reason
    assert not result.reason


def test_public_estimate_reports_absent_foreground():
    result = estimate(np.full((480, 640, 3), 42, np.uint8))
    assert result.weight_g is None
    assert '候補を抽出できませんでした' in result.reason


def test_public_estimate_does_not_run_the_research_gate(monkeypatch):
    def only_for_research(*args, **kwargs):
        raise AssertionError('The public estimate must not call the capture gate')

    monkeypatch.setattr(sheetless_area, 'mask_capture_reason', only_for_research)
    assert estimate(spread(36)).weight_g > 0


def test_invalid_area_model_is_rejected():
    with pytest.raises(ValueError):
        AreaModel('image_fraction', -100, 20, .1, .2)
    with pytest.raises(ValueError):
        AreaModel('image_fraction', float('nan'), 0, .1, .2)


def test_ignores_tiny_detached_border_reflection():
    photo = spread(36)
    clean = estimate(photo)
    photo[:4, 305:315] = (220, 180, 40)
    assert estimate(photo).weight_g == pytest.approx(clean.weight_g)


def test_checks_border_fragment_near_pasta():
    photo = spread(36)[54:].copy()
    photo[:3, 305:315] = (220, 180, 40)
    result = estimate(photo)
    assert result.weight_g > 0
    assert '画像の端' in observe(photo).reason


@pytest.mark.parametrize('quarter_turns', range(4))
def test_shallow_pale_reflection_is_ignored_without_changing_weight(quarter_turns):
    clean_photo = cv2.resize(spread(36), (960, 720), interpolation=cv2.INTER_NEAREST)
    photo = clean_photo.copy()
    photo[230:280, -4:] = (175, 158, 121)
    photo[230:280, -8:-4] = (155, 147, 125)
    clean_photo = np.rot90(clean_photo, quarter_turns)
    photo = np.rot90(photo, quarter_turns)
    result = estimate(photo)
    clean = estimate(clean_photo)
    assert clean.weight_g is not None
    assert result.weight_g == clean.weight_g
    np.testing.assert_array_equal(sheetless_area.pasta_mask(photo), sheetless_area.pasta_mask(clean_photo))


@pytest.mark.parametrize('condition', [
    'yellow_streak', 'inward_tip', 'near_body', 'many_streaks', 'pale_parallel_clip', 'pale_oblique_clip',
])
def test_ambiguous_border_components_still_flag_clipping(condition):
    photo = cv2.resize(spread(36), (960, 720), interpolation=cv2.INTER_NEAREST)
    if condition == 'yellow_streak':
        photo[230:280, -4:] = (220, 180, 40)
        photo[230:280, -8:-4] = (155, 147, 125)
    elif condition == 'inward_tip':
        photo[230, -22:] = (220, 180, 40)
    elif condition == 'near_body':
        photo[228:231, -22:-12] = (220, 180, 40)
        photo[230:280, -4:] = (175, 158, 121)
        photo[230:280, -8:-4] = (155, 147, 125)
    elif condition == 'many_streaks':
        for y in range(30, 670, 100):
            photo[y:y+60, -5:] = (175, 158, 121)
            photo[y:y+60, -9:-5] = (155, 147, 125)
    elif condition == 'pale_parallel_clip':
        cv2.line(photo, (960, 230), (960, 280), (175, 158, 121), 7)
    else:
        cv2.line(photo, (965, 230), (956, 280), (175, 158, 121), 5)
    result = estimate(photo)
    assert result.weight_g > 0
    assert observe(photo).reason == sheetless_area.EDGE_CAPTURE_REASON


def test_overlay_shows_retained_foreground_ignored_streak_and_exact_contact():
    photo = cv2.resize(spread(36), (960, 720), interpolation=cv2.INTER_NEAREST)
    photo[230:280, -4:] = (175, 158, 121)
    photo[230:280, -8:-4] = (155, 147, 125)
    photo[400:404, :100] = (220, 180, 40)
    original = photo.copy()
    overlay = sheetless_area.edge_diagnostic_overlay(photo)
    np.testing.assert_array_equal(photo, original)
    np.testing.assert_array_equal(overlay[240, -1], (255, 154, 32))
    np.testing.assert_array_equal(overlay[401, 0], (255, 32, 56))
    assert not np.array_equal(overlay[401, 50], photo[401, 50])
    np.testing.assert_array_equal(overlay[0, 0], photo[0, 0])


def test_diagnostic_reports_exact_edge_runs_and_distinct_components():
    photo = cv2.resize(spread(36), (960, 720), interpolation=cv2.INTER_NEAREST)
    yellow = (220, 180, 40)
    # One component touches twice, another once.
    photo[:25, 220:224] = yellow
    photo[:25, 260:264] = yellow
    photo[24:28, 220:264] = yellow
    photo[:25, 340:344] = yellow
    photo[230:280, -4:] = (175, 158, 121)
    photo[230:280, -8:-4] = (155, 147, 125)

    diagnosis = sheetless_area.diagnose(photo)
    top = [contact for contact in diagnosis.contacts if contact.side == 'top']
    assert [(contact.start, contact.end) for contact in top] == [(220, 223), (260, 263), (340, 343)]
    assert all(contact.retained and contact.inward_px >= 25 for contact in top)
    assert top[0].component == top[1].component != top[2].component
    assert top[0].component_pixels == top[1].component_pixels
    assert diagnosis.contacts[-1].side == 'right'
    assert (diagnosis.contacts[-1].start, diagnosis.contacts[-1].end) == (230, 279)
    assert not diagnosis.contacts[-1].retained
    assert diagnosis.contacts[-1].inward_px == 4
    assert observe(photo).reason == sheetless_area.EDGE_CAPTURE_REASON


def test_diagnostic_exposes_density_even_when_the_estimator_reports_edge_first():
    photo = np.full((720, 960, 3), 42, np.uint8)
    photo[170:470, 96:900] = (220, 180, 40)
    assert observe(photo).reason == sheetless_area.THICK_CAPTURE_REASON
    photo[190:195, 900:] = (220, 180, 40)
    result = estimate(photo)
    assert result.weight_g > 0
    assert observe(photo).reason == sheetless_area.EDGE_CAPTURE_REASON

    diagnosis = sheetless_area.diagnose(photo)
    assert diagnosis.density_rejected
    assert diagnosis.thickness_ratio > sheetless_area.MAX_THICKNESS_RATIO
    right, = diagnosis.contacts
    assert (right.side, right.start, right.end, right.inward_px) == ('right', 190, 194, 864)


def test_diagnostic_has_no_density_measurement_without_foreground():
    photo = np.full((480, 640, 3), 42, np.uint8)
    diagnosis = sheetless_area.diagnose(photo)
    assert not diagnosis.contacts and not diagnosis.density_rejected
    assert diagnosis.thickness_ratio is None and diagnosis.mask_fraction == 0


def test_file_diagnostic_exports_counts_and_density_as_json(tmp_path, monkeypatch, capsys):
    photo = np.full((720, 960, 3), 42, np.uint8)
    photo[100:210, 96:] = (220, 180, 40)
    photo[:2, 450:453] = (220, 180, 40)
    path = tmp_path / 'synthetic photo.png'
    Image.fromarray(photo).save(path)
    monkeypatch.setattr(sys, 'argv', ['diagnose_sheetless_photo', str(path)])

    diagnose_photo_file()

    report = json.loads(capsys.readouterr().out)
    assert report['analysis_size'] == {'width': 960, 'height': 720}
    assert report['by_side']['right']['retained'] == {'components': 1, 'pixels': 110}
    assert report['by_side']['top']['excluded'] == {'components': 1, 'pixels': 3}
    assert report['density_rejected'] is True and report['area_weight_g'] > 0
    assert report['capture_reason'] == sheetless_area.EDGE_CAPTURE_REASON


@pytest.mark.parametrize('quality', [85, 95])
@pytest.mark.parametrize('parallel', [False, True])
def test_jpeg_color_bleed_does_not_remove_a_clipped_strand(quality, parallel):
    photo = spread(32)
    color = (175, 158, 121)
    if parallel:
        cv2.line(photo, (640, 103), (640, 137), color, thickness=5)
    else:
        cv2.line(photo, (646, 91), (638, 138), color, thickness=3)
    with BytesIO() as encoded:
        Image.fromarray(photo).save(encoded, 'JPEG', quality=quality)
        decoded = image_to_rgb(encoded)
    assert estimate(decoded).weight_g > 0
    assert observe(decoded).reason == sheetless_area.EDGE_CAPTURE_REASON


def test_reflection_retry_requires_a_stable_interior_color_mask():
    photo = cv2.resize(spread(36), (960, 720), interpolation=cv2.INTER_NEAREST)
    # Broader, near-threshold colour bands make the visible area less certain.
    for y in range(90, 630, 15):
        photo[y:y+1, 100:860] = (180, 161, 105)
    assert estimate(photo).weight_g is not None
    photo[230:280, -4:] = (175, 158, 121)
    photo[230:280, -8:-4] = (155, 147, 125)
    assert observe(photo).reason == sheetless_area.EDGE_CAPTURE_REASON


@pytest.mark.parametrize('area,expected', [(.181, 18.1), (.179, None), (.519, 51.9), (.521, None)])
def test_capture_tolerance_is_limited_at_both_ends(monkeypatch, area, expected):
    model = AreaModel('area_over_length_squared', 100, 0, .2, .5)
    monkeypatch.setattr(sheetless_area, '_load_model', lambda: model)
    monkeypatch.setattr(sheetless_area, 'observe', lambda _: Observation(.1, .2, area))
    actual = estimate_checked(None).weight_g
    if expected is None:
        assert actual is None
    else:
        assert actual == pytest.approx(expected)


@pytest.mark.parametrize('area,expected,extrapolated', [
    (.12, 2., True), (.2, 10., False), (.51, 41., True), (3., 290., True),
    (.1, None, False), (.01, None, False), (float('inf'), None, False), (float('nan'), None, False),
])
def test_public_estimate_uses_every_positive_finite_conversion(monkeypatch, area, expected, extrapolated):
    model = AreaModel('area_over_length_squared', 100, -10, .2, .5)
    monkeypatch.setattr(sheetless_area, '_load_model', lambda: model)
    monkeypatch.setattr(sheetless_area, 'observe', lambda _, **kwargs: Observation(.1, .2, area, '診断のみ'))
    result = estimate(None)
    assert result.extrapolated == extrapolated
    if expected is None:
        assert result.weight_g is None and '正の重量' in result.reason
    else:
        assert result.weight_g == pytest.approx(expected) and not result.reason
