"""承認済みマスクからの背景置換・背景のみ対照。同じ評価群でのみ実施する。"""
from __future__ import annotations
import numpy as np
from ..preprocessing import background_controls
from .evaluate import summary


def background_diagnostic(cases,predictor,*,training_group_ids):
    if not cases:raise ValueError('検品済みの独立評価画像が必要です。')
    rows=[]
    for case in cases:
        if case.get('qc_status')!='APPROVED' or not case.get('group_id') or case['group_id'] in training_group_ids:
            raise ValueError('マスク承認と訓練から独立した撮影・試料グループが必要です。')
        weight=float(case['actual_weight_g'])
        if not np.isfinite(weight) or weight<=0:raise ValueError('確認済みの有限正数の実測重量が必要です。')
        if case.get('label_confirmed') is not True:raise ValueError('候補重量は漏洩評価にも使用しません。')
        controls=background_controls(case['rgb'],case['mask'])
        results={kind:predictor(rgb) for kind,rgb in controls.items()}
        rows.append(dict(image_id=case['image_id'],group_id=case['group_id'],actual_weight_g=weight,results=results))
    scores={kind:summary([dict(actual_weight_g=r['actual_weight_g'],**r['results'][kind]) for r in rows]) for kind in controls}
    paired=[abs(r['results']['foreground_dark']['predicted_g']-r['results']['foreground_gray']['predicted_g'])
            for r in rows if all(r['results'][k]['success'] for k in ('foreground_dark','foreground_gray'))]
    bg=scores['background_only'];fg=scores['foreground_dark']
    suspect=bg['succeeded']>0 and fg['succeeded']>0 and bg['mae_g']<=fg['mae_g']
    return dict(scores=scores,rows=rows,background_shortcut_suspected=suspect,
                mean_background_change_g=float(np.mean(paired)) if paired else None,
                independent_group_count=len({r['group_id'] for r in rows}),
                notice='背景のみ対照にはマスク形状も残ります。背景学習の診断用で、単独では漏洩の証明・本番精度ではありません。')
