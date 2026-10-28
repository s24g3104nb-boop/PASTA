"""ROI座標の明示変換と、元画素を再描画しない背景候補作成。"""
from __future__ import annotations

import math
import cv2
import numpy as np

try:
    from .photo_qc import _rect, _rgb, _mask, _integer, _rects, composite_rgb
except ImportError:
    from photo_qc import _rect, _rgb, _mask, _integer, _rects, composite_rgb


def transform_roi(roi, reference_size, target_size, *, orientation=1, roi_space='oriented'):
    """width,height基準。EXIF正規化→等比縮小の順で矩形の全4隅を変換する。"""
    if roi_space not in ('raw', 'oriented'):
        raise ValueError('ROIの基準をrawまたはorientedで明示してください。')
    def size(value):
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError('参照寸法と対象寸法を(width,height)で指定してください。')
        return tuple(_integer(v, '画像寸法', 1, 100000) for v in value)
    w, h = size(reference_size)
    tw, th = size(target_size)
    orientation = _integer(orientation, 'EXIF向き', 1, 8)
    x1, y1, x2, y2 = _rect(roi, (h, w))
    points = np.array([[x1,y1],[x2,y1],[x1,y2],[x2,y2]], dtype=float)
    if roi_space == 'raw':
        x, y = points[:,0].copy(), points[:,1].copy()
        variants = {1:(x,y), 2:(w-x,y), 3:(w-x,h-y), 4:(x,h-y),
                    5:(y,x), 6:(h-y,x), 7:(h-y,w-x), 8:(y,w-x)}
        points = np.column_stack(variants[orientation])
        if orientation >= 5:
            w, h = h, w
    sx, sy = tw / w, th / h
    if abs(sx-sy) / max(sx,sy) > .01:
        raise ValueError('縦横比が一致しません。回転・切り抜き・参照寸法を確認してください。')
    points *= (sx,sy)
    mapped = [math.floor(points[:,0].min()+1e-9), math.floor(points[:,1].min()+1e-9),
              math.ceil(points[:,0].max()-1e-9), math.ceil(points[:,1].max()-1e-9)]
    mapped = list(_rect(mapped, (th,tw)))
    return dict(roi_xyxy=mapped, reference_size=list(reference_size), target_size=[tw,th],
                orientation=orientation, roi_space=roi_space, scale_xy=[sx,sy])


def propose_conservative_mask(rgb, roi, *, excluded_rects=(), support_polygon=None,
                              min_saturation=25, min_value=35, support_margin=16):
    """細い離れた麺もアンカーに含め、支持領域の外の小物を除く未検品候補。

    凸包の拡張は色候補の選択範囲だけに適用する。最終マスクの膨張や穴埋めはしない。
    除外画素は誤除去の可能性を必ず検品する。本数・形状の正解判定ではない。
    """
    rgb = _rgb(rgb)
    height, width = rgb.shape[:2]
    x1,y1,x2,y2 = _rect(roi,(height,width))
    exclusions = _rects(excluded_rects,(height,width))
    min_saturation = _integer(min_saturation,'最小彩度',0,255)
    min_value = _integer(min_value,'最小明度',0,255)
    margin = _integer(support_margin,'支持領域の余白',0,256)
    hsv = cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    color = cv2.inRange(hsv,(7,min_saturation,min_value),(45,255,255))
    in_roi = np.zeros((height,width),np.uint8)
    in_roi[y1:y2,x1:x2] = color[y1:y2,x1:x2]
    # 支持領域は彩度の確かな画素から求め、薄い背景色がアンカーになるのを抑える。
    strong = cv2.inRange(hsv,(10,60,55),(40,255,255))
    seeds = np.zeros_like(strong)
    seeds[y1:y2,x1:x2] = strong[y1:y2,x1:x2]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(seeds)
    support = np.zeros_like(in_roi)
    anchors = []
    if count > 1:
        largest = 1 + int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
        largest_area = stats[largest,cv2.CC_STAT_AREA]
        for index in range(1,count):
            x,y,w,h,area = stats[index]
            contours,_ = cv2.findContours((labels[y:y+h,x:x+w]==index).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            pts = np.concatenate(contours).reshape(-1,2).astype(np.float32)
            short,long = sorted(cv2.minAreaRect(pts)[1])
            long_strand = long >= .12 * max(x2-x1,y2-y1) and long/max(1.,short) >= 4
            if index == largest or long_strand or (area >= .20*largest_area and long/max(1.,short) >= 2):
                pts += (x,y)
                anchors.append(pts.astype(np.int32))
        hull = cv2.convexHull(np.concatenate(anchors))
        cv2.fillConvexPoly(support,hull,255)
        if margin:
            support = cv2.dilate(support,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(margin*2+1,margin*2+1)))
    if support_polygon is not None:
        try:
            from .photo_qc import _polygons
        except ImportError:
            from photo_qc import _polygons
        polygons = _polygons([support_polygon],(height,width))
        support[:] = 0
        cv2.fillPoly(support,polygons,255)
    # 弱い色候補は確かな色候補の境界付近だけで使用する。色の薄い背景全体へ拡大しない。
    if seeds.any():
        distance = cv2.distanceTransform((seeds == 0).astype(np.uint8),cv2.DIST_L2,3)
        supported_color = np.where(distance <= 1.5,in_roi,0).astype(np.uint8)
    else:
        supported_color = np.zeros_like(in_roi)
    mask = cv2.bitwise_and(supported_color,support)
    removed = int(np.count_nonzero((in_roi>0)&(mask==0)))
    for left,top,right,bottom in exclusions:
        mask[top:bottom,left:right] = 0
    warnings = ['未検品の候補です。元画像と比較し、全ての麺・先端・細い隙間を確認してください。']
    if removed:
        warnings.append(f'背景抑制で色候補{removed}画素を除きました。麺の誤除去がないか確認が必要です。')
    if not mask.any():
        warnings.append('抽出できません。ROI・色条件・手動支持ポリゴンを見直してください。')
    if np.any(mask[y1,x1:x2]) or np.any(mask[y2-1,x1:x2]) or np.any(mask[y1:y2,x1]) or np.any(mask[y1:y2,x2-1]):
        warnings.append('ROI境界に候補があります。画角外やROIで切れた麺を検品してください。')
    return dict(mask=mask,status='UNREVIEWED',warnings=warnings,
                stats=dict(foreground_pixels=int(np.count_nonzero(mask)),
                           color_pixels_before_support=int(np.count_nonzero(in_roi)),
                           removed_color_pixels=removed,anchor_components=len(anchors)))


def audit_mask(rgb, mask, *, forbidden_rects=(), reference_mask=None):
    rgb, mask = _rgb(rgb), _mask(mask)
    if rgb.shape[:2] != mask.shape:
        raise ValueError('画像とマスクの寸法が一致しません。')
    rects = _rects(forbidden_rects,mask.shape)
    forbidden_pixels = sum(int(np.count_nonzero(mask[y1:y2,x1:x2])) for x1,y1,x2,y2 in rects)
    foreground = composite_rgb(rgb,mask)
    lost, gained = 0,0
    if reference_mask is not None:
        reference_mask = _mask(reference_mask)
        if reference_mask.shape != mask.shape:
            raise ValueError('比較マスクの寸法が一致しません。')
        lost = int(np.count_nonzero((reference_mask>0)&(mask==0)))
        gained = int(np.count_nonzero((reference_mask==0)&(mask>0)))
    return dict(status='UNREVIEWED',foreground_rgb_unchanged=bool(np.array_equal(foreground[mask>0],rgb[mask>0])),
                forbidden_foreground_pixels=forbidden_pixels,forbidden_regions_clear=forbidden_pixels==0,
                forbidden_regions_supplied=bool(rects),lost_reference_pixels=lost,gained_reference_pixels=gained,
                subject_count_verified=False, note='色候補との比較であり、本数・長さ・太さの真値ではありません。')


def background_controls(rgb,mask):
    """漏洩検証用の対照画像。元写真と同じグループでのみ評価し、精度はここでは計算しない。"""
    rgb,mask = _rgb(rgb),_mask(mask)
    if rgb.shape[:2] != mask.shape:
        raise ValueError('画像とマスクの寸法が一致しません。')
    background_only = rgb.copy()
    background_only[mask>0] = 24
    return dict(foreground_dark=composite_rgb(rgb,mask),
                foreground_gray=composite_rgb(rgb,mask,bg=(128,128,128)),background_only=background_only)


def propose_lab_mask(rgb, roi, *, excluded_rects=()):
    """Labの黄色成分による比較候補。形態学処理・最大領域だけの選別はしない。"""
    rgb=_rgb(rgb); h,w=rgb.shape[:2]
    x1,y1,x2,y2=_rect(roi,(h,w))
    lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB)
    color=cv2.inRange(lab,(25,105,143),(255,165,255))
    mask=np.zeros((h,w),np.uint8);mask[y1:y2,x1:x2]=color[y1:y2,x1:x2]
    for left,top,right,bottom in _rects(excluded_rects,(h,w)): mask[top:bottom,left:right]=0
    return dict(mask=mask,status='UNREVIEWED',warnings=['Lab色抽出の比較候補です。陰影の欠落・黄色い小物の混入を検品してください。'])


def quality_flags(rgb,mask):
    """検品の優先度を決める観測値。重なり・麺本数の正解や自動承認には使わない。"""
    rgb,mask=_rgb(rgb),_mask(mask)
    if rgb.shape[:2]!=mask.shape: raise ValueError('画像とマスクの寸法が一致しません。')
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);selected=mask>0
    flags=[]
    if not selected.any(): flags.append('empty_foreground')
    boundary=bool(mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any())
    if boundary:flags.append('subject_may_be_cut_off')
    focus=float(cv2.Laplacian(gray,cv2.CV_64F)[selected].var()) if selected.any() else None
    if focus is not None and focus<3:flags.append('possible_blur')
    bright=float((gray[selected]>245).mean()) if selected.any() else None
    if bright is not None and bright>.05:flags.append('possible_glare')
    occupancy=float(selected.mean())
    if occupancy>.65:flags.append('possible_background_contamination')
    return dict(flags=flags,foreground_fraction=occupancy,foreground_laplacian_variance=focus,
                bright_fraction=bright,touches_image_boundary=boundary,
                status='UNREVIEWED',note='重なり・撮影角度・陰影・麺の欠落は人による検品が必要です。')
