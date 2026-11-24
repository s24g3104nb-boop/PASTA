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
MAX_THICKNESS_RATIO = .018
MAX_MASK_FRACTION = .73
EDGE_NAMES = ('top', 'bottom', 'left', 'right')
THICK_CAPTURE_REASON = '麺が厚い束に見えます。重なりを減らして平らに広げてください。'
EDGE_CAPTURE_REASON = '麺の候補が画像の端に触れています。麺全体が入るように撮影してください。'
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
    extrapolated: bool = False


@dataclass(frozen=True)
class EdgeContact:
    """One run on a side, with inclusive zero-based coordinates along that side."""

    side: str
    retained: bool
    component: int
    start: int
    end: int
    component_pixels: int
    inward_px: int


@dataclass(frozen=True)
class EdgeDiagnostic:
    overlay: np.ndarray
    contacts: tuple[EdgeContact, ...]
    mask_fraction: float
    thickness_ratio: float | None
    density_rejected: bool


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


def _edge_sides(labels):
    return labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]


def _pasta_components(scene):
    hsv = cv2.cvtColor(scene, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(scene, cv2.COLOR_RGB2LAB)
    mask = cv2.inRange(hsv, (7, 30, 55), (45, 255, 255))
    lightness, red_green, blue_yellow = cv2.split(lab)
    mask[(blue_yellow < 145) | (red_green > 150) | (lightness < 35) | (lightness > 245)] = 0
    # Discard isolated colour noise without closing gaps between strands.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 30
    border = _edge_sides(labels)
    sides = [np.unique(side) for side in border]
    edge_count = np.bincount(np.concatenate(sides), minlength=count)
    edge_count[0] = 0
    edge = np.flatnonzero(edge_count)
    if not edge.size:
        return labels, keep, stats, []

    # Consider even sub-30px edge components: a thin, disconnected tip can
    # otherwise disappear before the clipping check.
    horizontal = np.zeros(count, dtype=bool)
    horizontal[np.union1d(sides[0], sides[1])] = True
    interior = keep & (edge_count == 0)
    edge_hits = np.bincount(np.concatenate(border), minlength=count)
    ignore = _isolated_border_noise(labels, stats, edge_count, edge_hits, horizontal,
                                    lab, interior)
    keep[edge] = True
    # An extensive group of edge detections is ambiguous even if each is small.
    areas = stats[:, cv2.CC_STAT_AREA]
    if areas[ignore].sum() > .005 * areas[keep].sum():
        ignore = []
    keep[ignore] = False
    return labels, keep, stats, ignore


def _isolated_border_noise(labels, stats, edge_count, edge_hits, horizontal, lab, interior):
    """Find small edge spots and shallow, weak-yellow streaks isolated from all colour candidates."""
    height, width = labels.shape
    clearance = 20  # canonical pixels, including near-edge tips and other specks
    ignore = []
    weak_yellow_limit = None
    blue_yellow = lab[:, :, 2]
    for component in np.flatnonzero(edge_count):
        x, y, w, h, area = stats[component]
        along, inward = (w, h) if horizontal[component] else (h, w)
        single_side = edge_count[component] == 1
        tiny = max(w, h) <= 3
        compact = (single_side and along <= 32 and inward <= min(12, max(along, 3))
                   and area >= .75 * w * h)
        # An oblique clipped strand can look pale and shallow too. Only discard
        # extended streaks whose footprint fills and follows the frame edge.
        shallow_streak = (single_side and 32 < along <= 64 and inward <= 6 and along >= 4 * inward
                          and edge_hits[component] >= .9 * along and area >= .85 * w * h)
        if not (tiny or compact or shallow_streak):
            continue
        vicinity = labels[max(0, y - clearance):min(height, y + h + clearance),
                          max(0, x - clearance):min(width, x + w + clearance)]
        if np.any((vicinity != 0) & (vicinity != component)):
            continue
        if shallow_streak:
            if weak_yellow_limit is None:
                interior_yellow = blue_yellow[interior[labels]]
                weak_yellow_limit = 0
                # Keep the edge warning if the body mask also depends heavily
                # on marginal colour, for example after JPEG chroma bleed.
                if interior_yellow.size and np.mean(interior_yellow >= 165) >= .9:
                    weak_yellow_limit = min(160, np.median(interior_yellow) - 18)
            bounds = np.s_[y:y + h, x:x + w]
            component_pixels = labels[bounds] == component
            if blue_yellow[bounds][component_pixels].max() > weak_yellow_limit:
                continue
            if not _has_reflection_halo(lab, x, y, w, h, horizontal[component]):
                continue
        ignore.append(component)
    return ignore


def _has_reflection_halo(lab, x, y, width, height, horizontal):
    # Look inward from the frame for a diffuse band just below the mask's
    # yellow threshold. A uniform pale stripe alone could be a clipped strand.
    if horizontal:
        strip = lab[:, x:x + width]
        if y:
            strip = strip[::-1]
        depth = height
    else:
        strip = lab[y:y + height, :].swapaxes(0, 1)
        if x:
            strip = strip[::-1]
        depth = width
    halo = strip[depth:depth + 4]
    # JPEG chroma bleed may tint dark pixels beside a strand. The band must
    # also be visibly lighter than a dark background.
    pale = (halo[:, :, 0] >= 80) & (halo[:, :, 2] >= 134) & (halo[:, :, 2] < 145)
    pale_columns = np.count_nonzero(pale, axis=0) >= 3
    return halo.shape[0] == 4 and np.count_nonzero(pale_columns) >= .75 * strip.shape[1]


def pasta_mask(scene):
    labels, keep, _, _ = _pasta_components(scene)
    return keep[labels]


def _contact_runs(labels, keep, stats):
    for side, edge in zip(EDGE_NAMES, _edge_sides(labels)):
        # Split adjacent labels as well as gaps so each run can be traced to
        # exactly one connected component. A corner belongs to both sides.
        bounds = np.r_[0, np.flatnonzero(edge[1:] != edge[:-1]) + 1, edge.size]
        inward = cv2.CC_STAT_HEIGHT if side in ('top', 'bottom') else cv2.CC_STAT_WIDTH
        for start, stop in zip(bounds[:-1], bounds[1:]):
            component = int(edge[start])
            if component:
                yield EdgeContact(side, bool(keep[component]), component, int(start), int(stop - 1),
                                  int(stats[component, cv2.CC_STAT_AREA]), int(stats[component, inward]))


def diagnose(rgb):
    """Report the same retained foreground used for estimation, including secondary density evidence."""
    scene = canonical_image(rgb)
    labels, keep, stats, ignored = _pasta_components(scene)
    mask = keep[labels]
    overlay = scene.copy()
    overlay[mask] = np.rint(.55 * scene[mask] + .45 * np.array([0, 235, 80])).astype(np.uint8)
    overlay[np.isin(labels, ignored)] = (255, 154, 32)
    for pixels, contacts in zip(_edge_sides(overlay), _edge_sides(mask)):
        pixels[contacts] = (255, 32, 56)
    geometry = mask_geometry(mask)
    fraction = geometry['mask_fraction']
    ratio, dense = _density_evidence(mask, geometry['quantile_length_px'], fraction)
    return EdgeDiagnostic(overlay, tuple(_contact_runs(labels, keep, stats)), fraction, ratio, dense)


def edge_diagnostic_overlay(rgb):
    """Overlay the retained mask, removed edge components, and retained edge pixels."""
    return diagnose(rgb).overlay


def describe_mask(scene, mask, *, capture_checks=True):
    """Use the same features and capture checks for fitting and estimation."""
    if mask.dtype != np.bool_ or mask.shape != scene.shape[:2]:
        raise ValueError('Expected a binary mask aligned with the photo')
    area = int(np.count_nonzero(mask))
    hsv = cv2.cvtColor(scene, cv2.COLOR_RGB2HSV)
    dark = np.count_nonzero((hsv[:, :, 2] <= 90) & (hsv[:, :, 1] <= 210) & ~mask)
    geometry = mask_geometry(mask)
    length = geometry['quantile_length_px']
    fraction = area / mask.size
    reason = mask_capture_reason(mask, length, dark_fraction=dark / mask.size) if capture_checks else ''
    area_ratio = float(geometry['area_over_length_squared']) if length > 0 else 0.
    return Observation(fraction, float(area / max(area + dark, 1)), area_ratio, reason)


def _too_little_foreground(fraction, length):
    return fraction < .025 or length < 25


def _density_evidence(mask, length, fraction):
    if _too_little_foreground(fraction, length):
        return None, False
    thickness = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    upper_quartile = float(np.quantile(thickness[mask], .75))
    rejected = upper_quartile > MAX_THICKNESS_RATIO * length or fraction > MAX_MASK_FRACTION
    return float(upper_quartile / length), bool(rejected)


def mask_capture_reason(mask, length, *, dark_fraction=None):
    """Shared framing and shape checks, with a dark-background check for the shipped model."""
    if any(side.any() for side in _edge_sides(mask)):
        return EDGE_CAPTURE_REASON
    fraction = np.count_nonzero(mask) / mask.size
    if _too_little_foreground(fraction, length):
        return '乾麺の領域が少なすぎます。麺をもう少し大きく写してください。'
    if dark_fraction is not None and dark_fraction < .09:
        return '暗く無地の背景に置き、反射を避けてください。'
    _, density_rejected = _density_evidence(mask, length, fraction)
    if density_rejected:
        return THICK_CAPTURE_REASON
    return ''


def observe(rgb, *, capture_checks=True):
    scene = canonical_image(rgb)
    observation = describe_mask(scene, pasta_mask(scene), capture_checks=capture_checks)
    if capture_checks and max(rgb.shape[:2]) < MIN_LONG_SIDE and observation.reason != EDGE_CAPTURE_REASON:
        return replace(observation, reason='画像が小さすぎて麺の面積が安定しません。元の写真を選んでください。')
    return observation


@lru_cache(maxsize=1)
def _load_model():
    document = json.loads(MODEL_PATH.read_text(encoding='utf-8'))
    if document['version'] != ESTIMATOR or document['segmentation'] != SEGMENTATION:
        raise ValueError('Incompatible area model')
    return AreaModel(**document['model'])


def estimate_observation(observation):
    """Apply the historical capture gates used by controlled experiments."""
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


def estimate_checked(rgb):
    """Keep capture and calibration gates for comparisons with the research baseline."""
    return estimate_observation(observe(rgb))


def estimate(rgb):
    """Convert visible foreground to a public estimate without applying research gates."""
    observation = observe(rgb, capture_checks=False)
    if observation.image_fraction <= 0:
        return Estimate(None, '乾麺の候補を抽出できませんでした。', observation)
    model = _load_model()
    area = getattr(observation, model.area_feature)
    weight_g = model.predict(area)
    if not math.isfinite(area) or area <= 0 or not math.isfinite(weight_g) or weight_g <= 0:
        return Estimate(None, '抽出した面積から正の重量に換算できませんでした。',
                        observation, model.area_feature)
    extrapolated = not model.area_min <= area <= model.area_max
    return Estimate(weight_g, '', observation, model.area_feature, extrapolated)
