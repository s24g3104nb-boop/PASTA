"""Projected spaghetti length measurements from the visible foreground.

A straight, nearly complete track supplies a relative scale. Neither the
physical strand length nor the lengths hidden below other strands are known.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from ..geometry_candidate import skeleton_measurements, skeletonize
from ..sheetless_area import MIN_LONG_SIDE, canonical_image, describe_mask, pasta_mask


@dataclass(frozen=True)
class LineFeatures:
    area_ratio: float
    skeleton_length_px: float
    width_px: float
    width_p90_over_p10: float
    reference_px: float
    branches: int
    area_length_px: float
    capture_reason: str
    scale_reason: str

    def vector(self):
        if self.scale_reason:
            return None
        return (self.area_ratio, self.skeleton_length_px / self.reference_px,
                self.area_length_px / self.reference_px, self.width_px / self.reference_px)


def projected_reference_length(centerline, envelope_px):
    """Relative scale proxy from long supported tracks, not a verified strand length."""
    # Use long supported tracks without counting them. A short fragment gives
    # an artificially small denominator and can imitate a large portion.
    minimum = max(32, .70 * envelope_px)
    tracks = cv2.HoughLinesP(centerline, rho=2, theta=np.pi / 360,
                             threshold=max(24, int(.17 * envelope_px)),
                             minLineLength=int(minimum / np.sqrt(2)),
                             maxLineGap=max(3, int(.065 * envelope_px)))
    if tracks is None:
        return 0.0
    segments = tracks[:, 0]
    lengths = np.linalg.norm(segments[:, 2:] - segments[:, :2], axis=1)
    supported = lengths[lengths >= minimum]
    return float(np.quantile(supported, .90)) if supported.size else 0.0


def measure(rgb):
    """Measure B and C using the same HSV+Lab foreground."""
    scene = canonical_image(rgb)
    mask = pasta_mask(scene)
    observation = describe_mask(scene, mask)
    area = int(np.count_nonzero(mask))
    if not area:
        return LineFeatures(0., 0., 0., 0., 0., 0, 0., observation.reason, '前景なし')
    ys, xs = np.nonzero(mask)
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    padded = np.pad(crop, 2)
    centerline = skeletonize(padded)
    # The lower quartile of non-branch widths is less affected by neighbouring
    # strands merged into the mask than the median of all centerline widths.
    length, width, width_ratio, _, branches = skeleton_measurements(padded, centerline, width_percentile=25)
    envelope = np.sqrt(area / observation.area_over_length_squared)
    reference = projected_reference_length(centerline, envelope)
    reason = ''
    if not reference:
        reason = '麺の長い連続線を確認できない'
    elif width < 1 or length <= 0:
        reason = '麺幅または合計線長を測れない'
    capture_reason = observation.reason
    if max(rgb.shape[:2]) < MIN_LONG_SIDE:
        capture_reason = '画像の解像度不足'
    return LineFeatures(observation.area_over_length_squared, length, width, width_ratio, reference, branches,
                        area / width if width > 0 else 0., capture_reason, reason)
