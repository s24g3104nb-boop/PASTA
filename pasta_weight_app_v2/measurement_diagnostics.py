"""実機の失敗を再調査するための来歴と、画像から検証できる棄却条件。"""
from __future__ import annotations
import hashlib
import math
import cv2


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
            reasons.append('接触・重なり、または背景混入の疑いがあります。麺を1本ずつ離し、無地の背景で再撮影してください。')
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
