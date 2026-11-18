"""実機の失敗を再調査するための来歴と、画像から検証できる棄却条件。"""
from __future__ import annotations
import hashlib
import math
import cv2
from dataclasses import dataclass


def strand_calibration(product):
    try:
        value = product.get('g_per_strand')
        if isinstance(value, bool):
            return None
        value = float(value)
        return value if math.isfinite(value) and value > 0 else None
    except (ValueError, TypeError, OverflowError):
        return None


@dataclass(frozen=True)
class CountWeightDecision:
    weight_g: float | None
    reasons: tuple
    reason_codes: tuple
    foreground_pixels: int
    image_pixels: int
    automatic_count: int
    used_count: int
    calibration_g_per_strand: float | None


def decide_count_weight(product, result, *, mode, using_demo=False, manual_count=None):
    calibration = strand_calibration(product)
    reasons, codes = [], []

    def reject(code, message):
        codes.append(code)
        reasons.append(message)

    if not using_demo and str(product.get('name', '')).startswith('デモ用'):
        reject('demo_product', 'デモ製品の仮値は実際の写真に使用できません。実製品を登録してください。')
    if calibration is None:
        reject('uncalibrated', '製品の1本重量が未校正です。使用する製品の校正値を登録してください。')
    count = result.count if manual_count is None else manual_count
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        reject('no_count', '実際に確認した本数を1以上で入力してください。' if manual_count is not None else
               '本数を検出できませんでした。麺の全体を明るく、1本ずつ離して撮り直してください。')
    if not using_demo and manual_count is None:
        evidence = result.evidence
        if not evidence:
            reject('missing_evidence', '検出結果の確認情報が不足しています。写真から解析をやり直してください。')
        elif mode in ('広げた状態', 'spread'):
            if evidence.get('cropped_regions'):
                reject('cropped', '麺候補が画像端で切れています。全体を画面内に入れて再撮影してください。')
            if evidence.get('wide_regions') or evidence.get('closing_merged_regions'):
                reject('merged', '接触・密集・重なり、または背景混入の疑いがあります。複数本が1領域になり、校正値だけでは補正できません。麺を離して撮り直してください。')
            area = evidence.get('smallest_counted_area')
            if area and evidence.get('small_region_pixels', 0) >= area:
                reject('unresolved_regions', '数えられなかった色領域が多く、本数の不足が疑われます。反射を避け、麺の全体を撮り直してください。')
        elif mode in ('束の断面', 'bundle_end'):
            if evidence.get('cropped_regions'):
                reject('cropped', '断面候補が画像端で切れています。全体を写してください。')
            if (evidence.get('endface_coverage', 0) < .9 or evidence.get('unsupported_discs')
                    or evidence.get('unresolved_endface')):
                reject('unresolved_endfaces', '見えている麺領域の多くを断面の円として確認できません。側面や隠れた端を本数に換算できないため、全ての端面が見える向きで撮り直してください。')
        else:
            reject('unsupported_mode', 'この撮影状態では本数から重量を算出できません。')
    weight = None
    if not reasons:
        candidate = count * calibration
        if math.isfinite(candidate) and candidate > 0:
            weight = candidate
        else:
            reject('invalid_weight', '重量計算の結果が不正です。校正値と本数を確認してください。')
    return CountWeightDecision(weight, tuple(reasons), tuple(codes), cv2.countNonZero(result.mask),
                               result.mask.size, result.count, count, calibration)


def spread_rejection(mask):
    """連結成分の太さ・画像端を検査。本数・隠れた重なりの保証ではない。"""
    reasons=[]
    if mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any():
        reasons.append('麺候補が画像端で切れています。全体を画面内に入れて再撮影してください。')
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour)<100:continue
        short,long=sorted(cv2.minAreaRect(contour)[1])
        if long>0 and short/long>.04:
            reasons.append('接触・密集・重なり、または背景混入の疑いがあり、この本数方式では重さを算出できません。1本重量の校正だけでは補正できないため、麺を離して全体を再撮影してください。')
            break
    return reasons


def weight_gate(product, *, using_demo, mode, mask=None, manual=False):
    if using_demo:return []
    if product['name'].startswith('デモ用') or not math.isfinite(product['g_per_strand']) or product['g_per_strand']<=0:
        return ['実製品の1本重量が未校正です。実際の製品を登録・校正し、電子はかりで確認してください。']
    if not manual and mode=='広げた状態' and mask is not None:return spread_rejection(mask)
    return []


def trace(rgb, *, estimator, product, mode, parameters=None, model=None, raw_sha256=None):
    artifact=(model or {}).get('artifact',{})
    return dict(schema_version=1,estimator=estimator,mode=mode,product_id=product['id'],
        g_per_strand=product['g_per_strand'],parameters=parameters or {},
        inference_rgb_sha256=hashlib.sha256(str(rgb.shape).encode()+rgb.tobytes()).hexdigest(),
        inference_shape=list(rgb.shape),raw_upload_sha256=raw_sha256,
        model_id=(model or {}).get('id'),artifact_sha256=(model or {}).get('artifact_sha256'),
        feature_version=artifact.get('feature_version'),dataset_sha256=artifact.get('dataset_sha256'),
        input_contract=artifact.get('input_contract','legacy_unspecified') if model else 'opencv_raw',
        accuracy_status='unverified',note='自動テスト成功は実機精度の証拠ではありません。')
