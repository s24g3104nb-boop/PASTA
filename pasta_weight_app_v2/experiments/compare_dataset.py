"""Offline, fixed-split comparison. Does not register or activate models.

Requires confirmed labels, reviewed images and VERIFIED physical groups before fitting.
python -m pasta_weight_app_v2.experiments.compare_dataset --csv labels.csv --calibrations products.json --output PRIVATE_NEW_DIR
"""
import argparse
from dataclasses import asdict
from itertools import combinations
import json
from pathlib import Path
import cv2
import numpy as np
from ..geometry_candidate import ProductCalibration,estimate
from ..measurement_diagnostics import weight_gate
from ..pasta_vision import detect_spread,detect_bundle_ends,image_to_rgb,image_quality_warnings
from ..pasta_ml.dataset import load_dataset,validate_splits,grouping,dataset_hash
from ..pasta_ml.evaluate import summary
from ..pasta_ml.model import features as rgb_features
from .geometry_benchmark import fit,predict,shape_features,GEOMETRY_FEATURE_VERSION

METHODS=('legacy_raw','production_guarded','geometry','rgb_ridge','geometry_ridge','hybrid')


def truth(value):return str(value).lower() in ('true','1','yes','confirmed_by_user')


def validate_research_data(samples,allow_synthetic=False):
    validate_splits(samples)
    if not samples:raise ValueError('評価データがありません。')
    sources={s.values['source'] for s in samples}
    if sources!={'real'} and not (allow_synthetic and sources=={'synthetic'}):
        raise ValueError('実写と合成は分離してください。合成は明示的な試験指定が必要です。')
    for s in samples:
        if not truth(s.values.get('weight_verified')):
            raise ValueError('未確定重量を教師または評価の正解に使用しません。')
        if not truth(s.values.get('group_verified')) or not s.values.get('physical_sample_id'):
            raise ValueError('物理試料の確認が必要です。同じ麺を追加・再撮影したものを分けないでください。')
        if s.values.get('qc_status')!='APPROVED' or not truth(s.values.get('leakage_checked')):
            raise ValueError('画像QCと秤表示等の漏洩確認が必要です。')
    for pair in {s.pair for s in samples}:
        train=[s for s in samples if s.pair==pair and s.values['split']=='train']
        if len(grouping(train))<3 or len({s.weight for s in train})<2:
            raise ValueError(pair+': 独立訓練3群以上と複数重量が必要です。')
        for split in ('val','test'):
            if not any(s.pair==pair and s.values['split']==split for s in samples):
                raise ValueError(pair+': 同じ製品・状態の独立val/testが必要です。')


def score(rows,method):
    return summary([dict(actual_weight_g=r['actual_weight_g'],**r[method]) for r in rows])


def run(csv_path,calibration_path,output,allow_synthetic=False):
    output=Path(output)
    if output.exists():raise ValueError('新しい実験ディレクトリを指定してください。')
    samples=load_dataset(csv_path)
    validate_research_data(samples,allow_synthetic)
    book=json.loads(Path(calibration_path).read_text())
    calibrations={pid:ProductCalibration(**value).validate() for pid,value in book.items()}
    if any(c.product_id!=pid for pid,c in calibrations.items()):raise ValueError('製品IDと校正表のキーが一致しません。')
    if any(s.values['product_id'] not in calibrations for s in samples):raise ValueError('対応する実測製品校正がありません。')
    rows=[]
    for s in samples:
        v=s.values;rgb=image_to_rgb(s.path);mode=v['mode'];cal=calibrations[v['product_id']]
        geo=estimate(rgb,mode,cal,single_layer=truth(v.get('single_layer')),
            perpendicular=truth(v.get('perpendicular')),all_tips_visible=truth(v.get('all_tips_visible')))
        if mode=='spread':old=detect_spread(rgb)
        elif mode=='bundle_end':old=detect_bundle_ends(rgb,min_radius=8,max_radius=27,min_distance=32,hough_sensitivity=18)
        else:old=None
        rejection=weight_gate(dict(name='verified calibration',g_per_strand=cal.g_per_strand),using_demo=False,
            mode='広げた状態' if mode=='spread' else mode,mask=old.mask if old else None)
        rejection.extend(image_quality_warnings(rgb))
        success=old is not None and old.count>0
        raw=dict(success=success,predicted_g=old.count*cal.g_per_strand if success else None)
        rows.append(dict(image_path=v['image_path'],pixel_sha256=s.pixel_hash,actual_weight_g=s.weight,pair=s.pair,
            split=v['split'],physical_sample_id=v['physical_sample_id'],mode=mode,product_id=v['product_id'],
            lighting=v.get('lighting','unknown'),background=v.get('background','unknown'),
            x_rgb=rgb_features(rgb).tolist(),x_geometry=shape_features(geo['features']).tolist(),geometry=geo,
            legacy_raw=raw,production_guarded=dict(success=success and not rejection,predicted_g=raw['predicted_g'] if success and not rejection else None,reason=' / '.join(rejection))))
    models={}
    for pair in sorted({r['pair'] for r in rows}):
        train=[r for r in rows if r['pair']==pair and r['split']=='train']
        models[pair]={method:fit([r[field] for r in train],[r['actual_weight_g'] for r in train])
                      for method,field in [('rgb_ridge','x_rgb'),('geometry_ridge','x_geometry')]}
    for r in rows:
        for method,field in [('rgb_ridge','x_rgb'),('geometry_ridge','x_geometry')]:r[method]=predict(models[r['pair']][method],np.array(r[field]))
        a,b=r['geometry'],r['geometry_ridge']
        agree=a['success'] and b['success'] and abs(a['predicted_g']-b['predicted_g'])<=max(2,.15*a['predicted_g'])
        r['hybrid']=dict(success=bool(agree),predicted_g=(a['predicted_g']+b['predicted_g'])/2 if agree else None)
    reports={}
    for split in ('train','val','test'):
        subset=[r for r in rows if r['split']==split]
        reports[split]={m:dict(overall=score(subset,m),by={f:{str(k):score([r for r in subset if r[f]==k],m) for k in sorted({r[f] for r in subset})} for f in ('actual_weight_g','mode','product_id','lighting','background')}) for m in METHODS}
        reports[split]['paired']={a+'__'+b:dict(images=len(common),a=score(common,a),b=score(common,b))
            for a,b in combinations(METHODS,2) for common in [[r for r in subset if r[a]['success'] and r[b]['success']]]}
    digest=dataset_hash(samples)
    artifact=dict(format='offline-research-only',source=samples[0].values['source'],dataset_sha256=digest,
        feature_version=GEOMETRY_FEATURE_VERSION,models=models,calibrations={k:asdict(v) for k,v in calibrations.items()},alpha=1.)
    report=dict(dataset_sha256=digest,source=samples[0].values['source'],production_eligible=False,
        notice='Not activated. Errors apply to successful predictions; compare failures and paired subsets. Test labels never fit/tune the model.',
        split_groups={split:len(grouping([s for s in samples if s.values['split']==split])) for split in ('train','val','test')},
        numpy=np.__version__,opencv=cv2.__version__,reports=reports,rows=rows)
    output.mkdir(parents=True)
    (output/'models.json').write_text(json.dumps(artifact,ensure_ascii=False,indent=2,allow_nan=False))
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--csv',required=True);p.add_argument('--calibrations',required=True);p.add_argument('--output',required=True);p.add_argument('--allow-synthetic',action='store_true')
    args=p.parse_args();r=run(args.csv,args.calibrations,args.output,args.allow_synthetic)
    print(json.dumps(r['reports']['test'],ensure_ascii=False,indent=2))
