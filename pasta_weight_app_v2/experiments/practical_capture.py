"""Marker-free trial for displaying an estimate from one dry spaghetti photo."""

from dataclasses import dataclass
from io import BytesIO
import json
import math

import cv2
import numpy as np

from ..pasta_vision import image_to_rgb
from ..sheetless_area import THICK_CAPTURE_REASON
from .area_proportion import analyze
from .strand_scale import (AGREED, COLOR_EDGE, MAX_SIDE, canonical_width, measure_for_area,
                           reliable_width, width_outcomes)

PROFILE_FORMAT = 'pasta-lens-width-reference-v1'
# Target 20–60g with the stated development margins at both endpoints.
# These engineering bounds are used on the estimates, never on the true weight.
PILOT_MIN_G, PILOT_MAX_G = 15., 70.
MAX_METHOD_GAP = .10


@dataclass(frozen=True)
class WidthReference:
    color_edge: float
    agreed: float

    def __post_init__(self):
        for value in self.features().values():
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError('色境界と照合の正の参照係数が必要です。')

    def features(self):
        return {COLOR_EDGE: self.color_edge, AGREED: self.agreed}

    def to_json(self):
        return json.dumps(dict(format=PROFILE_FORMAT, area_features=self.features()),
                          indent=2, allow_nan=False)

    @classmethod
    def from_json(cls, content):
        if len(content) > 8 * 1024:
            raise ValueError('参照係数JSONは8KiB以下にしてください。')
        try:
            document = json.loads(content)
            features = document['area_features']
            if (set(document) == {'format', 'area_features'} and document['format'] == PROFILE_FORMAT
                    and set(features) == {COLOR_EDGE, AGREED}):
                return cls(features[COLOR_EDGE], features[AGREED])
        except (KeyError, TypeError, ValueError, OverflowError):
            pass
        raise ValueError('Pasta Lensの参照係数JSONを選んでください。')


@dataclass(frozen=True)
class VisibleSignals:
    wide_fraction: float | None = None
    mixed_direction_fraction: float | None = None
    measurable_inside_px: int = 0


@dataclass(frozen=True)
class CaptureTrial:
    estimated_g: float | None
    method: str | None
    code: str
    reason: str
    current_g: float | None
    current_reason: str
    color_diagnostic_g: float | None
    agreed_diagnostic_g: float | None
    signals: VisibleSignals


def visible_overlap_signals(evidence, scale):
    """Quantify projected contact and visible direction changes inside joined foreground."""
    width = canonical_width(scale, AGREED)
    if not width or not evidence.area_px:
        return VisibleSignals()
    distance = cv2.distanceTransform(evidence.mask.astype(np.uint8), cv2.DIST_L2, 5)
    interior = distance > .85 * width
    wide_fraction = float(interior.sum() / evidence.area_px)
    if not interior.any():
        return VisibleSignals(wide_fraction)
    chroma = cv2.cvtColor(evidence.scene, cv2.COLOR_RGB2LAB)[:, :, 2].astype(np.float32)
    gx = cv2.Sobel(chroma, cv2.CV_32F, 1, 0, ksize=3, scale=.125)
    gy = cv2.Sobel(chroma, cv2.CV_32F, 0, 1, ksize=3, scale=.125)
    sigma = max(1., width)
    # Repeated edges in one direction have one strong tensor axis. A visible
    # crossing can mix axes, as can curvature, a shadow or a reflection.
    xx, yy, xy = [cv2.GaussianBlur(v, (0, 0), sigma)[interior] for v in (gx*gx, gy*gy, gx*gy)]
    energy = xx + yy
    resolved = energy >= 1.
    count = int(resolved.sum())
    if count < 32:
        return VisibleSignals(wide_fraction, measurable_inside_px=count)
    mixed = np.hypot(xx - yy, 2*xy) < .5*energy
    return VisibleSignals(wide_fraction, float(np.mean(mixed[resolved])), count)


def decide_capture(evidence, scale, profile):
    """Preserve the current result where available, then try a checked color estimate."""
    outcomes = width_outcomes(evidence, scale, profile.features())
    signals = visible_overlap_signals(evidence, scale)
    color = outcomes[COLOR_EDGE]['diagnostic_g']
    agreed = outcomes[AGREED]['diagnostic_g']
    # Keep diagnostics out of the visible UI when this policy declines.
    def result(value, method, code, reason=''):
        return CaptureTrial(value, method, code, reason, evidence.legacy_weight_g,
                            evidence.legacy_reason, color, agreed, signals)

    if evidence.width_capture_reason == THICK_CAPTURE_REASON:
        return result(None, None, 'dense', '麺がつながって見え、隠れた麺の量を確認できません。'
                      '重なりを減らして撮り直すか、実測重量を使ってください。')
    if evidence.width_capture_reason:
        return result(None, None, 'capture', evidence.width_capture_reason)
    if evidence.legacy_weight_g is not None:
        return result(evidence.legacy_weight_g, 'legacy', 'existing')
    if (color is None or agreed is None or not math.isfinite(color) or not math.isfinite(agreed)
            or not reliable_width(scale, signals.wide_fraction)):
        return result(None, None, 'width_support', '分離して見える麺の幅を安定して測れません。')
    if not all(PILOT_MIN_G <= g <= PILOT_MAX_G for g in (color, agreed)):
        return result(None, None, 'pilot_range', 'この写真は試験中の換算範囲から外れています。')
    if abs(color - agreed) > MAX_METHOD_GAP * min(color, agreed):
        return result(None, None, 'disagreement', '色境界と複数解像度の測定結果がそろいません。')
    return result(agreed, AGREED, 'candidate')


def analyze_capture(photo, profile):
    """Application entry point. Receives image bytes and a fixed reference only."""
    evidence = analyze(image_to_rgb(BytesIO(photo)))
    scale = measure_for_area(image_to_rgb(BytesIO(photo), max_side=MAX_SIDE), evidence)
    return decide_capture(evidence, scale, profile)
