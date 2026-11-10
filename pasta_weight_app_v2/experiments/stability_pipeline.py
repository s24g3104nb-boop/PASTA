"""Versioned offline preprocessing shared by fitting and research inference.

No production model, database or approval state is changed here. A geometry mask
is a candidate, never ground truth. Object ratios are not physical calibration.
"""
from dataclasses import asdict, dataclass
import hashlib
import json

import cv2
import numpy as np

from .direct_features import extract
from .diagnose_series import runtime_parameters

VERSION = 'direct-stability-v1'
LEGACY_KEYS = ['mask_fraction', 'long_relative', 'short_relative', 'aspect_ratio',
               'rect_density', 'perimeter_normalized', 'width_p50_relative',
               'width_cv', 'skeleton_density_512', 'foreground_L', 'foreground_b', 'texture_std']
SHAPE_KEYS = ['mask_fraction', 'quantile_length_relative', 'quantile_width_relative',
              'quantile_density', 'profile_width_cv']
OBJECT_KEYS = ['area_over_length_squared', 'width_over_length',
               'quantile_density', 'profile_width_cv']


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    target_long_side: int | None
    segmentation: str
    clean_components: bool
    feature_set: str
    version: str = VERSION
    down_interpolation: str = 'INTER_AREA'
    up_interpolation: str = 'INTER_CUBIC'
    hsv_h_min: int = 7
    hsv_h_max: int = 45
    hsv_s_min: int = 30
    hsv_v_min: int = 55
    lab_b_min: int = 145
    lab_a_max: int = 150
    lab_L_min: int = 35
    lab_L_max: int = 245
    clean_min_fraction: float = 0.00015
    support_margin_fraction: float = 0.025
    quantile_low: float = 0.01
    quantile_high: float = 0.99
    legacy_saturation_min: int = 20
    legacy_value_min: int = 75

    @property
    def keys(self):
        return {'legacy12': LEGACY_KEYS, 'shape5': SHAPE_KEYS, 'object4': OBJECT_KEYS}[self.feature_set]

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def configurations():
    opts = runtime_parameters()['DEFAULT_SPREAD']
    common = dict(legacy_saturation_min=opts['saturation_min'], legacy_value_min=opts['value_min'])
    return [PipelineConfig(*args, **common) for args in [
        ('A_hsv_legacy12', None, 'hsv', False, 'legacy12'),
        ('A_lab_legacy12', None, 'hsv_lab', False, 'legacy12'),
        ('B_normalized12', 512, 'hsv_lab', False, 'legacy12'),
        ('B_normalized5', 512, 'hsv_lab', False, 'shape5'),
        ('C_clean5', 512, 'hsv_lab', True, 'shape5'),
        ('D_object4', 512, 'hsv_lab', True, 'object4'),
    ]]


def validate_rgb(rgb):
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 16:
        raise ValueError('Expected RGB uint8 image, at least 16 px per side')


def normalize_size(rgb, long_side):
    validate_rgb(rgb)
    if long_side is None:
        return rgb.copy()
    if not isinstance(long_side, int) or long_side < 16:
        raise ValueError('Invalid canonical size')
    h, w = rgb.shape[:2]
    scale = long_side / max(h, w)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)


def raw_mask(rgb, config):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    if config.segmentation == 'hsv':
        result = cv2.inRange(hsv, np.array([7, config.legacy_saturation_min, config.legacy_value_min]), np.array([45, 255, 255]))
        return cv2.morphologyEx(result, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    if config.segmentation != 'hsv_lab':
        raise ValueError('Unknown segmentation')
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    result = cv2.inRange(hsv, np.array([config.hsv_h_min, config.hsv_s_min, config.hsv_v_min]), np.array([config.hsv_h_max, 255, 255]))
    result[(lab[:, :, 2] < config.lab_b_min) | (lab[:, :, 1] > config.lab_a_max) |
           (lab[:, :, 0] < config.lab_L_min) | (lab[:, :, 0] > config.lab_L_max)] = 0
    return result


def clean_mask(mask, config):
    """Keep substantial nearby regions and long detached strands; no hole filling.

    This can remove genuine thin noodles and cannot distinguish yellow objects.
    Removed pixels and masks must be reviewed, not automatically approved.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n < 2:
        return mask.copy()
    main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    main_points = cv2.findNonZero((labels == main).astype(np.uint8))
    support = np.zeros_like(mask)
    cv2.fillConvexPoly(support, cv2.convexHull(main_points), 255)
    radius = max(1, round(max(mask.shape) * config.support_margin_fraction))
    support = cv2.dilate(support, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1)))
    keep = np.zeros(n, np.uint8)
    cutoff = max(3, round(mask.size * config.clean_min_fraction))
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        component = labels[y:y+h, x:x+w] == i
        if area < cutoff:
            continue
        nearby = np.any(support[y:y+h, x:x+w][component] > 0)
        ys, xs = np.nonzero(component)
        short, long = sorted(cv2.minAreaRect(np.column_stack((xs, ys)).astype(np.float32))[1])
        elongated = long >= .12*max(mask.shape) and long/max(short, 1) >= 5
        if nearby or elongated:
            keep[i] = 255
    return keep[labels]


def shape_features(mask, config):
    """Robust PCA projections; no absolute-pixel predictor, skeleton or texture."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    area = len(xs)
    features = dict(mask_area_px=area, mask_fraction=area/mask.size,
                    image_pixels=int(mask.size), background_fraction=1-area/mask.size)
    length = width = density = cv = 0.
    profile = [0.] * 20
    if area > 2:
        pts = np.column_stack((xs, ys)).astype(float)
        pts -= pts.mean(axis=0)
        _, axes = np.linalg.eigh(pts.T @ pts / area)
        along = pts @ axes[:, -1]
        across = pts @ axes[:, 0]
        lo, hi = np.quantile(along, [config.quantile_low, config.quantile_high])
        low, high = np.quantile(across, [config.quantile_low, config.quantile_high])
        length, width = max(hi-lo, 1.), max(high-low, 1.)
        density = float(np.count_nonzero((along >= lo) & (along <= hi) & (across >= low) & (across <= high)) / (length*width))
        bins = np.clip(((along-lo)/length*20).astype(int), 0, 19)
        for i in range(20):
            v = across[bins == i]
            if len(v) > 2:
                profile[i] = float(np.quantile(v, .95) - np.quantile(v, .05))
        cv = float(np.std(profile) / max(np.mean(profile), 1.))
    features.update(quantile_length_px=length, quantile_width_px=width,
                    quantile_length_relative=length/max(h, w), quantile_width_relative=width/max(h, w),
                    quantile_density=density, profile_width_cv=cv,
                    area_over_length_squared=area/max(length*length, 1.), width_over_length=width/max(length, 1.),
                    projection_widths_px=profile)
    return features


def process(rgb, config):
    """The only path for both train-image and inference-image feature extraction."""
    validate_rgb(rgb)
    scene = normalize_size(rgb, config.target_long_side)
    before = raw_mask(scene, config)
    mask = clean_mask(before, config) if config.clean_components else before
    features = shape_features(mask, config)
    if config.feature_set == 'legacy12':
        old, _ = extract(scene, mask)
        features.update(old)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features['contour_perimeter_px'] = sum(cv2.arcLength(c, True) for c in contours)
    features['perimeter_over_sqrt_area'] = features['contour_perimeter_px'] / max(np.sqrt(features['mask_area_px']), 1.)
    features['components'] = cv2.connectedComponents(mask)[0] - 1
    features['removed_fraction'] = float(np.count_nonzero(before != mask) / max(1, np.count_nonzero(before)))
    gray = cv2.cvtColor(scene, cv2.COLOR_RGB2GRAY)
    features['background_mean'] = float(gray[mask == 0].mean()) if np.any(mask == 0) else 0.
    features['border_fraction'] = float((np.count_nonzero(mask[0])+np.count_nonzero(mask[-1])+np.count_nonzero(mask[:, 0])+np.count_nonzero(mask[:, -1])) / (2*sum(mask.shape)))
    flags = []
    if features['mask_fraction'] < .005: flags.append('insufficient_foreground')
    if features['border_fraction'] > .002: flags.append('foreground_touches_frame')
    if features['removed_fraction'] > .05: flags.append('substantial_mask_removal_review')
    if max(rgb.shape[:2]) < 512: flags.append('low_source_resolution')
    if cv2.Laplacian(gray, cv2.CV_32F).var() < 3: flags.append('low_detail_review')
    features['qc_flags'] = flags
    return features, scene, mask


def feature_vector(rgb, config):
    features, _, _ = process(rgb, config)
    return np.array([features[k] for k in config.keys], float)


def research_inference(rgb, artifact):
    """Optional end-to-end research adapter; never returns a public weight."""
    config = PipelineConfig(**artifact['pipeline_config'])
    if artifact['pipeline_version'] != VERSION or artifact['config_sha256'] != config.digest or artifact['feature_keys'] != config.keys:
        raise ValueError('Preprocessing/model version mismatch')
    features, _, _ = process(rgb, config)
    value = float(artifact['model'].predict(np.array([[features[k] for k in config.keys]], float))[0])
    return dict(research_prediction_g=value, predicted_weight_g=None, status='要確認',
                qc_flags=features['qc_flags'], independent_validation=False)
