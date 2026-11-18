"""OpenCVによる乾燥スパゲッティの試験的な本数検出。

重要: 学習済みAIではない。麺が接触していたり、背景が複雑だと誤検出する。
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image, ImageOps


@dataclass
class DetectionResult:
    count: int
    overlay_rgb: np.ndarray
    mask: np.ndarray
    description: str
    warnings: List[str]
    evidence: dict | None = None


def image_to_rgb(file_or_image, max_side: int = 1400) -> np.ndarray:
    """EXIFの向きを補正してスマホ画像をRGB配列にする。"""
    if not isinstance(max_side, int) or max_side <= 0:
        raise ValueError('max_side は正の整数にしてください。')
    if isinstance(file_or_image, (str, Path)):
        size = Path(file_or_image).stat().st_size
    elif hasattr(file_or_image, 'getbuffer'):
        size = file_or_image.getbuffer().nbytes
    else:
        size = None
    if size is not None and size > 25 * 1024 * 1024:
        raise ValueError('画像は25MiB以下にしてください。')
    if isinstance(file_or_image, Image.Image):
        pil = file_or_image
    else:
        pil = Image.open(file_or_image)
    original = pil
    try:
        if pil.width * pil.height > 25_000_000:
            raise ValueError('画像は2500万画素以下にしてください。')
        pil = ImageOps.exif_transpose(pil)
        pil.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        if 'A' in pil.getbands() or 'transparency' in pil.info:
            rgba = pil.convert('RGBA')
            background = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
            pil = Image.alpha_composite(background, rgba)
        return np.asarray(pil.convert('RGB'))
    finally:
        if isinstance(file_or_image, (str, Path)):
            original.close()


def finite_nonnegative(value, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f'{name} は数値にしてください。')
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'{name} は0以上の有限値にしてください。') from None
    if not math.isfinite(number) or number < 0:
        raise ValueError(f'{name} は0以上の有限値にしてください。')
    return number


def image_quality_warnings(rgb: np.ndarray) -> list[str]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    warnings = []
    if min(rgb.shape[:2]) < 64:
        warnings.append('画像が小さすぎます。乾麺を大きく撮り直してください。')
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    dark = gray.mean() < 15
    washed_out = gray.mean() > 245 and gray.std() < 8
    blurred = laplacian.var() < 3
    if dark or washed_out or blurred:
        # A small number of sharp strands can occupy <1% of a black tray.
        # Recheck the visible colour region and its edges at the same thresholds.
        # This only allows the detector to run; it does not certify a strand count.
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        candidate = cv2.inRange(hsv, (7, 12, 60), (45, 255, 255))
        if cv2.countNonZero(candidate) >= 160:
            region = cv2.dilate(candidate, np.ones((3, 3), np.uint8)) > 0
            local = gray[region]
            if local.mean() >= 15 and local.std() >= 8 and laplacian[region].var() >= 3:
                return warnings
    if dark:
        warnings.append('暗すぎて判別できません。明るさを改善して再撮影してください。')
    elif washed_out:
        warnings.append('白飛びまたは対象がほとんど見えません。露出と背景を変えてください。')
    if blurred:
        warnings.append('輪郭を確認できません。ピントを合わせて再撮影してください。')
    return warnings


def _yellow_mask(rgb: np.ndarray, saturation_min: int, value_min: int) -> np.ndarray:
    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8
            or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 2):
        raise ValueError('2×2画素以上のRGB uint8画像が必要です。')
    if not all(0 <= finite_nonnegative(v, '色閾値') <= 255
               for v in (saturation_min, value_min)):
        raise ValueError('色閾値は0〜255にしてください。')
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # 黄色〜クリーム色の麺（赤いソース・食品には非対応）。
    low = np.array([7, saturation_min, value_min], dtype=np.uint8)
    high = np.array([45, 255, 255], dtype=np.uint8)
    return cv2.inRange(hsv, low, high)


def _close_mask(raw):
    return cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def _closing_joined_strands(component, raw, min_area):
    """Did closing join two substantial pieces that both extend along the strand?

    Two end-to-end fragments cannot both span more than half the component
    along its longer image axis. Never use the fragments to guess a new count.
    """
    before = cv2.bitwise_and(component, raw)
    if np.array_equal(before, component):
        return False
    stats = cv2.connectedComponentsWithStats(before, connectivity=8)[2][1:]
    axis = cv2.CC_STAT_WIDTH if raw.shape[1] >= raw.shape[0] else cv2.CC_STAT_HEIGHT
    long_pieces = (stats[:, cv2.CC_STAT_AREA] >= min_area) & (2 * stats[:, axis] > max(raw.shape))
    return np.count_nonzero(long_pieces) > 1


def detect_spread(
    rgb: np.ndarray,
    *,
    saturation_min: int = 20,
    value_min: int = 75,
    min_area: int = 160,
    min_elongation: float = 3.0,
) -> DetectionResult:
    """広げた麺が互いに接していないとき、独立した細長い領域を数える。"""
    if finite_nonnegative(min_area, '最小面積') <= 0 or finite_nonnegative(min_elongation, '細長さ') < 1:
        raise ValueError('最小面積は正数、細長さは1以上にしてください。')
    raw = _yellow_mask(rgb, saturation_min, value_min)
    mask = _close_mask(raw)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    overlay = rgb.copy()
    count = 0
    non_elongated = 0
    cropped = 0
    wide = 0
    merged = 0
    small_area = 0
    smallest_counted_area = None
    for index in range(1, n):
        x, y, width, height, area = (int(v) for v in stats[index])
        at_border = x == 0 or y == 0 or x + width == mask.shape[1] or y + height == mask.shape[0]
        # Discount isolated specks, while retaining long, cropped fragments
        # even if they fall below the area required for a counted strand.
        if at_border and (area >= min_area or max(width, height) >= math.sqrt(min_area)):
            cropped += 1
        if area < min_area:
            small_area += area
            continue
        # 回転した麺にも対応するため、外接矩形ではなく最小面積回転矩形を使用。
        ys, xs = np.where(labels[y:y+height, x:x+width] == index)
        if len(xs) < 5:
            continue
        pts = np.column_stack((xs + x, ys + y)).astype(np.float32)
        rect = cv2.minAreaRect(pts)
        shorter, longer = sorted(rect[1])
        if longer > 0 and shorter / longer > .04:
            wide += 1
        elongation = longer / max(1.0, shorter)
        if elongation < min_elongation:
            non_elongated += 1
            continue
        count += 1
        smallest_counted_area = area if smallest_counted_area is None else min(smallest_counted_area, area)
        component = (labels[y:y+height, x:x+width] == index).astype(np.uint8) * 255
        if _closing_joined_strands(component, raw[y:y+height, x:x+width], min_area):
            merged += 1
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (255, 76, 76), 2)
        cv2.putText(overlay, str(count), (x, max(16, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255, 240, 35), 2, cv2.LINE_AA)

    warnings = ["検出領域を1本として数える試験方式です。麺の交差・接触・影で誤差が出ます。"]
    if count == 0:
        warnings.append("細長い乾麺を検出できませんでした。暗い無地の背景で、1本ずつ離して撮影してください。")
    if non_elongated > 0:
        warnings.append(f"形が細長くない{non_elongated}個の領域は除外しました。")
    evidence = dict(mode='spread', wide_regions=wide, cropped_regions=cropped,
                    closing_merged_regions=merged, small_region_pixels=small_area,
                    smallest_counted_area=smallest_counted_area, excluded_regions=non_elongated)
    return DetectionResult(count, overlay, mask, "独立した細長い黄色領域の数", warnings, evidence)


def detect_bundle_ends(
    rgb: np.ndarray,
    *,
    min_radius: int = 5,
    max_radius: int = 22,
    min_distance: int = 17,
    hough_sensitivity: int = 15,
    saturation_min: int = 12,
    value_min: int = 60,
) -> DetectionResult:
    """端面の丸い麺をHough変換で探索する簡易方式。

    端面がすべて手前に向いている写真に限る。実画像の精度は未検証。
    """
    overlay = rgb.copy()
    if any(finite_nonnegative(v, '円検出パラメータ') <= 0 for v in
           (min_radius, max_radius, min_distance, hough_sensitivity)):
        raise ValueError('円検出パラメータは正数にしてください。')
    if min_radius >= max_radius:
        raise ValueError("min_radius は max_radius より小さくしてください")
    mask = _close_mask(_yellow_mask(rgb, saturation_min, value_min))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # 穏やかな平滑化: 断面のエッジを残す。
    blur = cv2.GaussianBlur(gray, (5, 5), 1.15)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.0,
        minDist=float(min_distance), param1=85,
        param2=float(hough_sensitivity),
        minRadius=int(min_radius), maxRadius=int(max_radius),
    )
    found = []
    if circles is not None:
        for x, y, r in np.round(circles[0]).astype(int):
            # 円中心が黄〜クリーム色の領域でない場合は除外。
            if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]):
                continue
            ry = max(1, r // 3)
            y0, y1 = max(0, y - ry), min(mask.shape[0], y + ry + 1)
            x0, x1 = max(0, x - ry), min(mask.shape[1], x + ry + 1)
            neighborhood = mask[y0:y1, x0:x1]
            if neighborhood.size == 0 or cv2.countNonZero(neighborhood) / neighborhood.size < 0.35:
                continue
            found.append((x, y, r))
            cv2.circle(overlay, (x, y), r, (255, 70, 70), 2, cv2.LINE_AA)
            cv2.circle(overlay, (x, y), 2, (50, 240, 50), -1, cv2.LINE_AA)
    warnings = ["束の先端に見える円だけを数えます。隠れた麺・傾斜・照明で大きな誤差が出る場合があります。"]
    if not found:
        warnings.append("断面を検出できませんでした。束の端面を真正面から大きく撮影し、検出設定を調整してください。")
    evidence = _endface_evidence(mask, found, min_radius)
    return DetectionResult(len(found), overlay, mask, "見つかった円形の断面数（試験方式）", warnings, evidence)


def _endface_evidence(mask, circles, min_radius):
    """Check whether the fitted discs account for the visible coloured surface."""
    evidence = dict(mode='bundle_end', cropped_regions=0, endface_coverage=0.,
                    unsupported_discs=0, unresolved_endface=False)
    if not circles:
        return evidence
    covered = np.zeros_like(mask)
    height, width = mask.shape
    for x, y, radius in circles:
        if min(x - radius, y - radius, width - 1 - x - radius, height - 1 - y - radius) <= 0:
            evidence['cropped_regions'] += 1
        # The edge detector uses a 5x5 blur. Allow its two-pixel boundary band.
        cv2.circle(covered, (x, y), radius + 2, 255, -1)
        core = max(1, radius - 2)
        left, top = max(0, x - core), max(0, y - core)
        patch = mask[top:min(height, y + core + 1), left:min(width, x + core + 1)]
        yy, xx = np.ogrid[:patch.shape[0], :patch.shape[1]]
        inside = (xx + left - x) ** 2 + (yy + top - y) ** 2 <= core ** 2
        if np.count_nonzero(patch[inside]) < .5 * np.count_nonzero(inside):
            evidence['unsupported_discs'] += 1
    residual = cv2.bitwise_and(mask, cv2.bitwise_not(covered))
    evidence['endface_coverage'] = 1 - cv2.countNonZero(residual) / max(1, cv2.countNonZero(mask))
    # A missing whole face can be small relative to a large bundle; a global
    # coverage fraction alone would accept it. Inspect local residuals too.
    if cv2.countNonZero(residual):
        distance = cv2.distanceTransform(residual, cv2.DIST_L2, 5)
        evidence['unresolved_endface'] = bool(distance.max() >= max(1, min_radius - 2))
    return evidence


def nutrition_for_weight(weight_g: float, kcal_100: float, protein_100: float,
                         fat_100: float, carb_100: float) -> dict:
    scale = finite_nonnegative(weight_g, '重量') / 100.0
    kcal_100, protein_100, fat_100, carb_100 = (
        finite_nonnegative(v, '栄養値') for v in (kcal_100, protein_100, fat_100, carb_100))
    result = {
        "kcal": round(kcal_100 * scale, 1),
        "protein_g": round(protein_100 * scale, 1),
        "fat_g": round(fat_100 * scale, 1),
        "carb_g": round(carb_100 * scale, 1),
    }
    for key, value in result.items():
        finite_nonnegative(value, key)
    return result
