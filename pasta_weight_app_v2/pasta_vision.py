"""OpenCVによる乾燥スパゲッティの試験的な本数検出。

重要: 学習済みAIではない。麺が接触していたり、背景が複雑だと誤検出する。
"""
from __future__ import annotations

from dataclasses import dataclass
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


def image_to_rgb(file_or_image, max_side: int = 1400) -> np.ndarray:
    """EXIFの向きを補正してスマホ画像をRGB配列にする。"""
    if isinstance(file_or_image, Image.Image):
        pil = file_or_image
    else:
        pil = Image.open(file_or_image)
    pil = ImageOps.exif_transpose(pil).convert('RGB')
    pil.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return np.asarray(pil)


def _yellow_mask(rgb: np.ndarray, saturation_min: int, value_min: int) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # 黄色〜クリーム色の麺（赤いソース・食品には非対応）。
    low = np.array([7, saturation_min, value_min], dtype=np.uint8)
    high = np.array([45, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, low, high)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def detect_spread(
    rgb: np.ndarray,
    *,
    saturation_min: int = 20,
    value_min: int = 75,
    min_area: int = 160,
    min_elongation: float = 3.0,
) -> DetectionResult:
    """広げた麺が互いに接していないとき、独立した細長い領域を数える。"""
    mask = _yellow_mask(rgb, saturation_min, value_min)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    overlay = rgb.copy()
    valid = []
    non_elongated = 0
    for index in range(1, n):
        x, y, width, height, area = (int(v) for v in stats[index])
        if area < min_area:
            continue
        # 回転した麺にも対応するため、外接矩形ではなく最小面積回転矩形を使用。
        ys, xs = np.where(labels[y:y+height, x:x+width] == index)
        if len(xs) < 5:
            continue
        pts = np.column_stack((xs + x, ys + y)).astype(np.float32)
        rect = cv2.minAreaRect(pts)
        shorter, longer = sorted(rect[1])
        elongation = longer / max(1.0, shorter)
        if elongation < min_elongation:
            non_elongated += 1
            continue
        valid.append((x, y, width, height, area))
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (255, 76, 76), 2)
        cv2.putText(overlay, str(len(valid)), (x, max(16, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255, 240, 35), 2, cv2.LINE_AA)

    warnings = ["検出領域を1本として数える試験方式です。麺の交差・接触・影で誤差が出ます。"]
    if len(valid) == 0:
        warnings.append("細長い乾麺を検出できませんでした。暗い無地の背景で、1本ずつ離して撮影してください。")
    if non_elongated > 0:
        warnings.append(f"形が細長くない{non_elongated}個の領域は除外しました。")
    return DetectionResult(len(valid), overlay, mask, "独立した細長い黄色領域の数", warnings)


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
    if min_radius >= max_radius:
        raise ValueError("min_radius は max_radius より小さくしてください")
    mask = _yellow_mask(rgb, saturation_min, value_min)
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
    return DetectionResult(len(found), overlay, mask, "見つかった円形の断面数（試験方式）", warnings)


def nutrition_for_weight(weight_g: float, kcal_100: float, protein_100: float,
                         fat_100: float, carb_100: float) -> dict:
    scale = max(0.0, weight_g) / 100.0
    return {
        "kcal": round(kcal_100 * scale, 1),
        "protein_g": round(protein_100 * scale, 1),
        "fat_g": round(fat_100 * scale, 1),
        "carb_g": round(carb_100 * scale, 1),
    }
