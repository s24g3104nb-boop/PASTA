"""Synthetic regression fixtures for the shipped area-only estimator."""
import cv2
import numpy as np
import pytest

from pasta_weight_app_v2 import sheetless_area
from pasta_weight_app_v2.sheetless_area import AreaModel, Observation, estimate


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


def test_reduced_resolution_is_compared_only_when_area_is_resolvable():
    photo = cv2.resize(spread(36), (1400, 1050), interpolation=cv2.INTER_LINEAR)
    reduced = cv2.resize(photo, (700, 525), interpolation=cv2.INTER_AREA)
    assert estimate(reduced).weight_g == pytest.approx(estimate(photo).weight_g, abs=2)
    too_small = estimate(cv2.resize(reduced, (350, 262), interpolation=cv2.INTER_AREA))
    assert too_small.weight_g is None
    assert '画像が小さすぎて' in too_small.reason


@pytest.mark.parametrize('condition,reason', [
    ('empty', '領域が少なすぎます'),
    ('cropped', '画像の端'),
    ('dense', '厚い束'),
    ('bright', '暗く無地'),
    ('outside', '比較できる範囲'),
])
def test_abstains_when_the_photo_cannot_support_weight(condition, reason):
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
    result = estimate(image)
    assert result.weight_g is None
    assert reason in result.reason


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
    assert result.weight_g is None
    assert '画像の端' in result.reason


@pytest.mark.parametrize('area,expected', [(.181, 18.1), (.179, None), (.519, 51.9), (.521, None)])
def test_capture_tolerance_is_limited_at_both_ends(monkeypatch, area, expected):
    model = AreaModel('area_over_length_squared', 100, 0, .2, .5)
    monkeypatch.setattr(sheetless_area, '_load_model', lambda: model)
    monkeypatch.setattr(sheetless_area, 'observe', lambda _: Observation(.1, .2, area))
    actual = estimate(None).weight_g
    if expected is None:
        assert actual is None
    else:
        assert actual == pytest.approx(expected)
