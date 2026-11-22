"""Estimate dry spaghetti weight from its visible area without a scale reference."""

from dataclasses import dataclass, replace
from functools import lru_cache
import json
import math
from pathlib import Path

import cv2
import numpy as np

if __package__:
    from .mask_geometry import mask_geometry
else:
    from mask_geometry import mask_geometry


ESTIMATOR = 'sheetless_area_v2'
SEGMENTATION = 'opencv_hsv_lab_spread_960'
AREA_FEATURES = ('image_fraction', 'pasta_or_dark_fraction', 'area_over_length_squared')
MODEL_PATH = Path(__file__).with_name(f'{ESTIMATOR}.json')
MIN_LONG_SIDE = 640
# Small rasterization changes near a calibration endpoint should not cause
# abrupt rejection. This allowance is for capture variation, not new weights.
AREA_BOUNDARY_TOLERANCE_G = 2.0


@dataclass(frozen=True)
class Observation:
    image_fraction: float
    pasta_or_dark_fraction: float
    area_over_length_squared: float
    reason: str = ''


@dataclass(frozen=True)
class Estimate:
    weight_g: float | None
    reason: str
    observation: Observation
    area_feature: str | None = None


@dataclass(frozen=True)
class AreaModel:
    area_feature: str
    slope: float
    intercept_g: float
    area_min: float
    area_max: float

    def __post_init__(self):
        if self.area_feature not in AREA_FEATURES:
            raise ValueError('Unknown area feature')
        if not all(math.isfinite(value) for value in (self.slope, self.intercept_g, self.area_min, self.area_max)):
            raise ValueError('Non-finite area model')
        if self.slope <= 0 or not 0 < self.area_min < self.area_max or not (
                0 < self.predict(self.area_min) < self.predict(self.area_max) <= 200):
            raise ValueError('Invalid area model range')

    def predict(self, area):
        return self.slope * area + self.intercept_g


def canonical_image(rgb):
    if (not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3
            or rgb.dtype != np.uint8 or min(rgb.shape[:2]) < 2):
        raise ValueError('Expected an RGB uint8 photo')
    height, width = rgb.shape[:2]
    scale = 960 / max(height, width)
    if scale == 1:
        return rgb
    return cv2.resize(rgb, (max(1, round(width * scale)), max(1, round(height * scale))),
                      interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)


def pasta_mask(scene):
    hsv = cv2.cvtColor(scene, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(scene, cv2.COLOR_RGB2LAB)
    mask = cv2.inRange(hsv, (7, 30, 55), (45, 255, 255))
    lightness, red_green, blue_yellow = cv2.split(lab)
    mask[(blue_yellow < 145) | (red_green > 150) | (lightness < 35) | (lightness > 245)] = 0
    # Discard isolated colour noise without closing gaps between strands.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 30
    _discard_isolated_border_specks(labels, stats, keep)
    return keep[labels]


def _discard_isolated_border_specks(labels, stats, keep):
    """Keep actual clipped strands and ignore only tiny, distant edge reflections."""
    edge = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    short = np.maximum(stats[edge, cv2.CC_STAT_WIDTH], stats[edge, cv2.CC_STAT_HEIGHT]) <= 32
    specks = edge[keep[edge] & short]
    areas = stats[:, cv2.CC_STAT_AREA]
    if not specks.size or areas[specks].sum() > .005 * areas[keep].sum():
        return
    speck_pixels = np.isin(labels, specks)
    interior = keep[labels] & ~speck_pixels
    # Pixels within 10 canonical pixels of another component could be an
    # interrupted strand. Retain those components for the clipping check.
    distance = cv2.distanceTransform((~interior).astype(np.uint8), cv2.DIST_L2, 5)
    nearby = np.unique(labels[speck_pixels & (distance <= 10)])
    keep[np.setdiff1d(specks, nearby)] = False


def describe_mask(scene, mask):
    """Use the same features and capture checks for fitting and estimation."""
    if mask.dtype != np.bool_ or mask.shape != scene.shape[:2]:
        raise ValueError('Expected a binary mask aligned with the photo')
    area = int(np.count_nonzero(mask))
    hsv = cv2.cvtColor(scene, cv2.COLOR_RGB2HSV)
    dark = np.count_nonzero((hsv[:, :, 2] <= 90) & (hsv[:, :, 1] <= 210) & ~mask)
    geometry = mask_geometry(mask)
    length = geometry['quantile_length_px']
    fraction = area / mask.size
    reason = ''
    if fraction < .025 or length < 25:
        reason = '乾麺の領域が少なすぎます。麺をもう少し大きく写してください。'
    elif dark / mask.size < .09:
        reason = '暗く無地の背景に置き、反射を避けてください。'
    elif mask[[0, -1], :].any() or mask[:, [0, -1]].any():
        reason = '麺の候補が画像の端に触れています。麺全体が入るように撮影してください。'
    else:
        thickness = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        if np.quantile(thickness[mask], .75) > .018 * length or fraction > .73:
            reason = '麺が厚い束に見えます。重なりを減らして平らに広げてください。'
    return Observation(fraction, float(area / max(area + dark, 1)),
                       float(geometry['area_over_length_squared']), reason)


def observe(rgb):
    scene = canonical_image(rgb)
    observation = describe_mask(scene, pasta_mask(scene))
    if max(rgb.shape[:2]) < MIN_LONG_SIDE:
        return replace(observation, reason='画像が小さすぎて麺の面積が安定しません。元の写真を選んでください。')
    return observation


@lru_cache(maxsize=1)
def _load_model():
    document = json.loads(MODEL_PATH.read_text(encoding='utf-8'))
    if document['version'] != ESTIMATOR or document['segmentation'] != SEGMENTATION:
        raise ValueError('Incompatible area model')
    return AreaModel(**document['model'])


def estimate(rgb):
    observation = observe(rgb)
    if observation.reason:
        return Estimate(None, observation.reason, observation)
    model = _load_model()
    area = getattr(observation, model.area_feature)
    weight_g = model.predict(area)
    lower = model.predict(model.area_min) - AREA_BOUNDARY_TOLERANCE_G
    upper = model.predict(model.area_max) + AREA_BOUNDARY_TOLERANCE_G
    if not 0 < weight_g <= 200 or not lower <= weight_g <= upper:
        reason = '写真の面積が、この方法で比較できる範囲を外れています。'
        return Estimate(None, reason, observation, model.area_feature)
    return Estimate(weight_g, '', observation, model.area_feature)
