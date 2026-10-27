"""原画素を保持する仮マスク作成・手動修正。自動検品や本数の真値ではない。"""
from __future__ import annotations

import io
import math
from pathlib import Path
import warnings

import cv2
import numpy as np
from PIL import Image


MAX_PIXELS = 30_000_000
MAX_BYTES = 30_000_000


def _integer(value, label, lower, upper):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lower <= value <= upper):
        raise ValueError(f'{label}は{lower}〜{upper}の整数にしてください。')
    return int(value)


def _rgb(rgb):
    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3
            or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1
            or rgb.shape[0] * rgb.shape[1] > MAX_PIXELS):
        raise ValueError('3000万画素以下の空でないRGB uint8画像が必要です。')
    return rgb


def _mask(mask):
    if (not isinstance(mask, np.ndarray) or mask.dtype != np.uint8 or mask.ndim != 2
            or not mask.size or mask.size > MAX_PIXELS
            or not np.all((mask == 0) | (mask == 255))):
        raise ValueError('マスクは3000万画素以下、2次元uint8の0/255画像にしてください。')
    return mask


def _rect(rect, shape):
    if (not isinstance(rect, (tuple, list, np.ndarray))
            or isinstance(rect, np.ndarray) and rect.ndim != 1 or len(rect) != 4):
        raise ValueError('矩形は[x1,y1,x2,y2]で指定してください。')
    height, width = shape
    x1 = _integer(rect[0], 'x1', 0, width - 1)
    y1 = _integer(rect[1], 'y1', 0, height - 1)
    x2 = _integer(rect[2], 'x2', 1, width)
    y2 = _integer(rect[3], 'y2', 1, height)
    if x1 >= x2 or y1 >= y2:
        raise ValueError('矩形はx1<x2、y1<y2にしてください。')
    return x1, y1, x2, y2


def _rects(rects, shape):
    try:
        return [_rect(rect, shape) for rect in rects]
    except TypeError:
        raise ValueError('除外矩形は矩形のリストで指定してください。') from None


def propose_mask(rgb, roi=None, excluded_rects=(), *, hue_min=13, hue_max=36,
                 saturation_min=60, value_min=55):
    """色候補を抽出する。矩形座標は半開区間で、原画像を切り抜かず同じ寸法を返す。"""
    rgb = _rgb(rgb)
    hue_min = _integer(hue_min, '最小色相', 0, 179)
    hue_max = _integer(hue_max, '最大色相', 0, 179)
    saturation_min = _integer(saturation_min, '最小彩度', 0, 255)
    value_min = _integer(value_min, '最小明度', 0, 255)
    if hue_min > hue_max:
        raise ValueError('最小色相は最大色相以下にしてください。')
    height, width = rgb.shape[:2]
    x1, y1, x2, y2 = _rect(roi if roi is not None else (0, 0, width, height), (height, width))
    exclusions = _rects(excluded_rects, (height, width))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    candidates = cv2.inRange(hsv, (hue_min, saturation_min, value_min), (hue_max, 255, 255))
    candidate_pixels = cv2.countNonZero(candidates)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1:y2, x1:x2] = candidates[y1:y2, x1:x2]
    roi_pixels = cv2.countNonZero(mask)
    for left, top, right, bottom in exclusions:
        mask[top:bottom, left:right] = 0
    selected = cv2.countNonZero(mask)
    touches = bool(np.any(mask[y1, x1:x2]) or np.any(mask[y2 - 1, x1:x2])
                   or np.any(mask[y1:y2, x1]) or np.any(mask[y1:y2, x2 - 1]))
    component_count = cv2.connectedComponents(mask, connectivity=8)[0] - 1
    notes = ['色抽出の仮マスクです。連結領域数は麺の本数ではありません。目視で検品してください。']
    if selected == 0:
        notes.append('抽出画素がありません。色条件・向き・ROIを確認してください。')
    if touches:
        notes.append('候補がROIまたは画像の境界に接しています。麺の先端が切れていないか確認してください。')
    if selected / mask.size > .65:
        notes.append('抽出範囲が画像の65%を超えています。背景の混入を確認してください。')
    if candidate_pixels > roi_pixels:
        notes.append('ROI外の色候補を除外しました。背景だけでなく麺が除外されていないか確認してください。')
    return {
        'mask': mask, 'status': 'UNREVIEWED', 'warnings': notes,
        'stats': dict(foreground_pixels=selected, foreground_fraction=selected / mask.size,
                      candidate_components=int(component_count),
                      candidate_pixels_before_roi=candidate_pixels,
                      discarded_by_roi_pixels=candidate_pixels - roi_pixels,
                      excluded_candidate_pixels=roi_pixels - selected, touches_roi_boundary=touches),
    }


def _polygons(polygons, shape):
    height, width = shape
    validated = []
    try:
        for polygon in polygons:
            if not isinstance(polygon, (tuple, list, np.ndarray)) or not 3 <= len(polygon) <= 10_000:
                raise ValueError('修正ポリゴンには3〜10000頂点が必要です。')
            points = []
            for point in polygon:
                if not isinstance(point, (tuple, list, np.ndarray)) or len(point) != 2:
                    raise ValueError('ポリゴンの頂点は[x,y]で指定してください。')
                points.append((_integer(point[0], '頂点x', 0, width - 1),
                               _integer(point[1], '頂点y', 0, height - 1)))
            array = np.array(points, dtype=np.int32)
            if cv2.contourArea(array) <= 0:
                raise ValueError('修正ポリゴンには面積が必要です。')
            validated.append(array)
    except TypeError:
        raise ValueError('修正ポリゴンは頂点リストのリストで指定してください。') from None
    return validated


def apply_mask_edits(mask, add_polygons=(), erase_polygons=(), *, excluded_rects=()):
    """追加→消去→除外矩形の順に適用。秤の除外矩形は手動追加でも復活させない。"""
    mask = _mask(mask)
    additions = _polygons(add_polygons, mask.shape)
    erasures = _polygons(erase_polygons, mask.shape)
    exclusions = _rects(excluded_rects, mask.shape)
    result = mask.copy()
    for polygon in additions:
        cv2.fillPoly(result, [polygon], 255)
    for polygon in erasures:
        cv2.fillPoly(result, [polygon], 0)
    for left, top, right, bottom in exclusions:
        result[top:bottom, left:right] = 0
    return result


def _paired(rgb, mask):
    rgb, mask = _rgb(rgb), _mask(mask)
    if rgb.shape[:2] != mask.shape:
        raise ValueError('画像とマスクの寸法・向きを揃えてください。自動リサイズは行いません。')
    return rgb, mask


def _color(color):
    if (not isinstance(color, (tuple, list, np.ndarray))
            or isinstance(color, np.ndarray) and color.ndim != 1 or len(color) != 3):
        raise ValueError('背景色・着色はRGBの3値で指定してください。')
    return tuple(_integer(value, 'RGB値', 0, 255) for value in color)


def composite_rgb(rgb, mask, bg=(24, 24, 24)):
    """マスク内のRGB値を原画からそのままコピーする。保存時はPNGを使用する。"""
    rgb, mask = _paired(rgb, mask)
    result = np.empty_like(rgb)
    result[:] = _color(bg)
    result[mask == 255] = rgb[mask == 255]
    return result


def make_overlay(rgb, mask, alpha=.36, color=(55, 185, 55)):
    """目視検品専用。学習画像として使用しない。"""
    rgb, mask = _paired(rgb, mask)
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('着色率は0〜1の有限値にしてください。')
    color = _color(color)
    result = rgb.copy()
    selected = mask == 255
    result[selected] = np.rint(rgb[selected].astype(float) * (1 - alpha)
                               + np.asarray(color) * alpha).astype(np.uint8)
    return result


def validate_uploaded_mask(image_file, shape):
    """元画像と同じ画素座標の2値PNGを読み、向き補正・拡大縮小はしない。"""
    if not isinstance(shape, (tuple, list)) or len(shape) != 2:
        raise ValueError('必要なマスク寸法は(height,width)で指定してください。')
    expected = tuple(_integer(value, '画像寸法', 1, MAX_PIXELS) for value in shape)
    if expected[0] * expected[1] > MAX_PIXELS:
        raise ValueError('マスクは3000万画素以下にしてください。')
    if isinstance(image_file, (str, Path)):
        path = Path(image_file)
        if path.stat().st_size > MAX_BYTES:
            raise ValueError('マスクは30MB以下にしてください。')
        raw = path.read_bytes()
    elif isinstance(image_file, (bytes, bytearray, memoryview)):
        raw = bytes(image_file)
    elif hasattr(image_file, 'read'):
        position = image_file.tell() if hasattr(image_file, 'tell') else None
        if hasattr(image_file, 'seek'):
            image_file.seek(0)
        try:
            raw = image_file.read(MAX_BYTES + 1)
        finally:
            if position is not None and hasattr(image_file, 'seek'):
                image_file.seek(position)
    else:
        raise ValueError('マスクPNGのファイルまたはバイト列が必要です。')
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError('マスクは空でない30MB以下のPNGにしてください。')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as image:
                if image.format != 'PNG' or image.mode not in ('1', 'L'):
                    raise ValueError('マスクは白黒1チャンネルの2値PNGにしてください。')
                if (image.height, image.width) != expected:
                    raise ValueError('マスク寸法が一致しません。自動リサイズは行いません。')
                if image.getexif().get(274, 1) != 1:
                    raise ValueError('マスクの回転EXIFは使用できません。正規化済み画素で保存してください。')
                image.load()
                return _mask(np.asarray(image.convert('L')).copy())
    except (OSError, Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise ValueError('マスクPNGが破損しているか大きすぎます。') from exc
