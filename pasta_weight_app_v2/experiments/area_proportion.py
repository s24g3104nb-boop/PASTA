"""Research-only 20g visible-area scaling, with no fit to higher weight labels."""

from dataclasses import dataclass
from statistics import median
import math

import cv2
import numpy as np

from ..geometry_candidate import skeletonize
from ..pasta_vision import image_quality_warnings
from ..sheetless_area import (MIN_LONG_SIDE, _load_model, canonical_image, describe_mask,
                              estimate_observation, pasta_mask)
from .sheetless_lengths import projected_reference_length

REFERENCE_G = 20
EVALUATED_MAX_G = 100
PRIMARY = 'track_area'
FEATURES = ('frame_area', 'pca_area', PRIMARY)
DESCRIPTIONS = {
    'frame_area': '面積 ÷ 画像全体の面積（距離補正なし）',
    'pca_area': '面積 ÷ 麺全体の長軸長²（20g比例換算）',
    'track_area': '面積 ÷ 長い線候補の長さ²（20g比例換算）',
}
OUTSIDE_EVALUATED_RANGE = '外挿・未検証（20〜100gの探索範囲外）'
EXPLORATORY = '探索的な概算・未検証（20g基準）'
SCALE_MISSING = '麺の長い連続線を確認できず、相対尺度を求められません。'
PATCH_TYPES = ('single', 'crossing', 'dense', 'outside_tray_pasta',
               'white_background', 'tray_reflection', 'tray_background')


def _remove_specks(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 30
    return keep[labels]


def yellow_masks(scene):
    """Compare each color condition in isolation and the current combined mask."""
    h, s, v = cv2.split(cv2.cvtColor(scene, cv2.COLOR_RGB2HSV))
    light, a, b = cv2.split(cv2.cvtColor(scene, cv2.COLOR_RGB2LAB))
    return {
        'HSV': _remove_specks((h >= 7) & (h <= 45) & (s >= 30) & (v >= 55)),
        'Lab': _remove_specks((b >= 145) & (a <= 150) & (light >= 35) & (light <= 245)),
        'HSV+Lab': pasta_mask(scene),
    }


@dataclass
class Evidence:
    scene: np.ndarray
    mask: np.ndarray
    features: dict
    area_px: int
    track_px: float
    capture_reason: str
    legacy_weight_g: float | None
    legacy_reason: str
    legacy_diagnostic_g: float | None


def analyze(rgb, *, scene=None, mask=None):
    """Measure only visible geometry. The longest track is an image-scale proxy."""
    if scene is None:
        scene = canonical_image(rgb)
    if mask is None:
        mask = pasta_mask(scene)
    observation = describe_mask(scene, mask)
    area = int(np.count_nonzero(mask))
    track = 0.
    if area and observation.area_over_length_squared > 0:
        ys, xs = np.nonzero(mask)
        crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        centerline = skeletonize(np.pad(crop, 2))
        envelope_px = math.sqrt(area / observation.area_over_length_squared)
        track = projected_reference_length(centerline, envelope_px)
    capture_reason = ' '.join(image_quality_warnings(rgb)) or observation.reason
    if max(rgb.shape[:2]) < MIN_LONG_SIDE:
        capture_reason = '画像の解像度不足'
    current = estimate_observation(observation)
    legacy = _load_model()
    raw = legacy.predict(getattr(observation, legacy.area_feature)) if area else None
    return Evidence(scene, mask,
                    {'frame_area': observation.image_fraction,
                     'pca_area': observation.area_over_length_squared,
                     PRIMARY: area / track**2 if track else None},
                    area, track, capture_reason,
                    None if capture_reason else current.weight_g, capture_reason or current.reason, raw)


def calibrate(references):
    """Three placements of one 20g sample, treated as a single reference."""
    if len(references) != 3:
        raise ValueError('同じ20g試料を配置し直した写真を3枚指定してください。')
    problems = [f'{i}枚目: {ref.capture_reason}' for i, ref in enumerate(references, 1)
                if ref.capture_reason]
    if problems:
        raise ValueError('基準写真を確認してください。' + ' '.join(problems))
    result = {}
    for key in FEATURES:
        values = [ref.features[key] for ref in references]
        if all(value is not None and math.isfinite(value) and value > 0 for value in values):
            result[key] = float(median(values))
    return result


def predict(model, evidence, method=PRIMARY):
    """Return the raw research value and the value after capture checks."""
    value = evidence.features[method]
    reference = model.get(method)
    if not reference:
        reason = '3枚の基準写真すべてでこの面積特徴を確認できませんでした。'
        return dict(diagnostic_g=None, estimated_g=None, reason=reason, status='')
    if value is None or value <= 0:
        reason = evidence.capture_reason or (SCALE_MISSING if method == PRIMARY else '乾麺の領域がありません。')
        return dict(diagnostic_g=None, estimated_g=None, reason=reason, status='')
    grams = REFERENCE_G * value / reference
    status = (OUTSIDE_EVALUATED_RANGE if grams > EVALUATED_MAX_G or grams < REFERENCE_G
              else EXPLORATORY)
    return dict(diagnostic_g=grams, estimated_g=None if evidence.capture_reason else grams,
                reason=evidence.capture_reason, status=status)


def overlay(scene, mask):
    result = scene.copy()
    color = np.array([0, 240, 175], dtype=np.float32)
    result[mask] = np.rint(.56 * scene[mask] + .44 * color).astype(np.uint8)
    return result


def patch_colors(scene, masks, sites):
    """Summarize manually marked small patches independently of mask inclusion."""
    hsv = cv2.cvtColor(scene, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(scene, cv2.COLOR_RGB2LAB)
    height, width = scene.shape[:2]
    result = []
    for site in sites:
        x, y, kind = site['x'], site['y'], site['kind']
        radius = site.get('radius', 1)
        if (kind not in PATCH_TYPES or any(type(n) is not int for n in (x, y, radius))
                or not 0 <= radius <= 10 or not radius <= x < width - radius
                or not radius <= y < height - radius):
            raise ValueError('色を確認する小領域の分類または座標が不正です。')
        patch = np.s_[y-radius:y+radius+1, x-radius:x+radius+1]
        rgb_vals = np.median(scene[patch], axis=(0, 1))
        hue, sat, val = np.median(hsv[patch], axis=(0, 1))
        light, a, b = np.median(lab[patch], axis=(0, 1))
        result.append(dict(site, rgb=rgb_vals.tolist(), H=float(2*hue), S=float(sat/255*100),
                           V=float(val/255*100), L=float(light/255*100), a=float(a-128), b=float(b-128),
                           mask_fraction={name: float(mask[patch].mean()) for name, mask in masks.items()}))
    return result


def paired_color_deltas(patches):
    """Compare patch-class medians within one image, without weight labels."""
    channels = ('H', 'S', 'V', 'L', 'a', 'b')
    single = [p for p in patches if p['kind'] == 'single']
    if not single:
        return {}
    result = {}
    for kind in ('crossing', 'dense'):
        regions = [p for p in patches if p['kind'] == kind]
        if regions:
            result[kind] = {channel: float(median(p[channel] for p in regions)
                                           - median(p[channel] for p in single)) for channel in channels}
    return result
