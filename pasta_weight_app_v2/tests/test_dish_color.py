"""Synthetic scenes for the standard, model-free dish suggestion."""

import cv2
import numpy as np
import pytest

from pasta_weight_app_v2.dish_classifier import STANDARD_DISH_NAMES
from pasta_weight_app_v2.local_dish_vision import suggest_dish_from_color


def plated_scene(dish, plate, light):
    y, x = np.indices((180, 240))
    grain = 7 * np.sin(x / 3 + y / 30) + 4 * np.cos(x / 11)
    rgb = np.array([133, 86, 51], dtype=float) + grain[:, :, None]
    if plate is not None:
        plate_mask = ((x - 122) / 117)**2 + ((y - 91) / 85)**2 < 1
        rgb[plate_mask] = plate
    body = ((x - 120) / 77)**2 + ((y - 97) / 60)**2 < 1
    sauce, topping = {
        'トマトパスタ': ((214, 105, 50), (83, 117, 36)),
        'ジェノベーゼ': ((143, 143, 55), (244, 222, 166)),
        'カルボナーラ': ((232, 180, 81), (173, 67, 38)),
    }[dish]
    shade = 16 * np.sin(x / 7 + y / 5) - 14 * (x / 240)
    rgb[body] = np.array(sauce) + shade[body, None]
    # Basil, shaved cheese or bacon occupies a few patches, not the sauce.
    rgb = np.rint(np.clip(rgb, 0, 255)).astype(np.uint8)
    cv2.ellipse(rgb, (121, 72), (23, 11), 22, 0, 360, topping, -1)
    cv2.ellipse(rgb, (153, 101), (15, 8), -35, 0, 360, topping, -1)
    cv2.ellipse(rgb, (93, 111), (12, 7), 25, 0, 360, topping, -1)
    cv2.ellipse(rgb, (194, 91), (8, 3), -25, 0, 360, (242, 242, 241), -1)
    return np.rint(np.clip(rgb * light, 0, 255)).astype(np.uint8)


@pytest.mark.parametrize('dish', STANDARD_DISH_NAMES)
@pytest.mark.parametrize('plate', [(239, 237, 233), (24, 25, 27), None], ids=['white', 'black', 'wood'])
@pytest.mark.parametrize('light', [.69, 1., 1.17], ids=['dim', 'normal', 'bright'])
def test_rank_sauce_with_toppings_and_different_light(dish, plate, light):
    assert suggest_dish_from_color(plated_scene(dish, plate, light)).name == dish


@pytest.mark.parametrize('dish', STANDARD_DISH_NAMES)
def test_warm_lighting_with_a_white_plate(dish):
    photo = plated_scene(dish, (239, 237, 233), (1.04, 1., .90))
    assert suggest_dish_from_color(photo).name == dish


def test_low_colour_evidence_and_tiny_photos_still_produce_a_candidate():
    assert suggest_dish_from_color(np.full((160, 120, 3), 127, np.uint8)).name in STANDARD_DISH_NAMES
    assert suggest_dish_from_color(np.array([[[83, 117, 36]]], np.uint8)).name == 'ジェノベーゼ'
