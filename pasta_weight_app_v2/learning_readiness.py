"""写真数と独立試料数を区別した学習前点検。学習・承認は実行しない。"""
from __future__ import annotations
import json
from pathlib import Path
from .photo_intake import list_intake
from .storage import DEFAULT_DB


def readiness(db_path=DEFAULT_DB):
    rows=list_intake(db_path)
    confirmed=[r for r in rows if r['weight_verified']]
    qc=[r for r in confirmed if r['qc_status']=='APPROVED' and r['leakage_checked']]
    grouped=[r for r in qc if r['group_verified'] and r['batch_id'] and r['session_id']
             and (not json.loads(r['metadata_json']).get('latest_label') or
                  json.loads(r['metadata_json'])['latest_label'].get('physical_sample_id'))]
    eligible=[r for r in grouped if r['mode'] and r['product_id']]
    # 同じsession/batch/physicalを結ぶ。候補群は独立試料とみなさない。
    components=[]
    for r in eligible:
        latest=json.loads(r['metadata_json']).get('latest_label',{})
        keys={('session',r['session_id']),('batch',r['batch_id'])}
        if latest.get('physical_sample_id'):keys.add(('physical',latest['physical_sample_id']))
        touched=[g for g in components if g & keys]
        for g in touched:keys|=g;components.remove(g)
        components.append(keys)
    reasons=[]
    for pair in {(r['mode'],r['product_id']) for r in eligible}:
        matching=[r for r in eligible if (r['mode'],r['product_id'])==pair]
        if len({r['weight_g'] for r in matching})<2:
            reasons.append(f'{pair}: 同じ撮影状態・製品で複数の確定重量が必要です。')
    if len(qc)<len(confirmed):reasons.append('確定重量の写真に未承認マスク・漏洩未検品があります。')
    if len(grouped)<len(qc) or any(not r['group_verified'] for r in confirmed):
        reasons.append('同じ麺の再使用・追加、計量、撮影セッションの対応が未確認です。')
    if len(eligible)<len(grouped):reasons.append('撮影状態または実製品の情報が不足しています。')
    if len(components)<5:reasons.append('独立訓練3群＋validation/test各1群を確保できません。製品・状態別の複数重量も必要です。')
    return dict(total=len(rows),confirmed_labels=len(confirmed),tentative_or_unconfirmed=len(rows)-len(confirmed),
                confirmed_by_weight={str(w):sum(r['weight_g']==w for r in confirmed) for w in sorted({r['weight_g'] for r in confirmed})},
                qc_approved=sum(r['qc_status']=='APPROVED' for r in rows),confirmed_qc_approved=len(qc),
                eligible_images=len(eligible),verified_independent_groups=len(components),
                training_permitted=not reasons,basic_precheck_passed=not reasons,blocking_reasons=reasons,
                notice='独立群が足りない状態で画像をランダム分割せず、精度指標は未算出とします。')
