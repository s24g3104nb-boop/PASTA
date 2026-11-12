"""Offline dense-pasta diagnostics and paired sensitivity; never independent MAE.

Inputs are a verified private receipt of 15 originals. No DB/model registry or
training approval is touched. Transformations remain attached to their originals.
"""
from dataclasses import asdict
from pathlib import Path
import argparse
import csv
import hashlib
import html
import json
import platform

import cv2
import numpy as np
from PIL import Image

from ..pasta_vision import image_to_rgb, detect_spread, image_quality_warnings
from ..measurement_diagnostics import weight_gate
from .diagnose_series import runtime_parameters
from .stability_pipeline import configurations, process, raw_mask
from .stability_experiment import transformations, TRANSFORM_SETTINGS
from .bundle_texture import extract_bundle_texture, TEXTURE_KEYS
from .metric_foreground import extract_metric_foreground
from .research_contract import propagate_capture_qc

VERSION='dense-weight-diagnostics-v1'
INTERIOR_KEYS=['interior_luminance_cv','interior_structure_coherence']


def interior_texture(rgb, mask):
    """Two dimensionless descriptors from strictly interior pixels, not pitch/count.

    Erode support to exclude the 3x3 Sobel neighborhood of background pixels.
    Constant yellow objects can share these descriptors with a real dense bundle.
    """
    inside=cv2.erode((mask>0).astype(np.uint8),np.ones((5,5),np.uint8))>0
    if inside.sum()<32:return dict.fromkeys(INTERIOR_KEYS,0.)
    light=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB)[:,:,0].astype(np.float32)
    gx=cv2.Sobel(light,cv2.CV_32F,1,0,ksize=3)[inside]
    gy=cv2.Sobel(light,cv2.CV_32F,0,1,ksize=3)[inside]
    xx,yy,xy=float(np.mean(gx*gx)),float(np.mean(gy*gy)),float(np.mean(gx*gy))
    coherence=np.sqrt((xx-yy)**2+4*xy**2)/max(xx+yy,1e-12)
    return dict(interior_luminance_cv=float(light[inside].std()/max(light[inside].mean(),1.)),
                interior_structure_coherence=float(np.clip(coherence,0,1)))


def write_csv(path, rows):
    if not rows:return
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with Path(path).open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})


def _preview(rgb, path):
    im=Image.fromarray(rgb);im.thumbnail((480,480));im.save(path)


def fit_exploratory(rows, feature_sets, output):
    """Fixed baselines only, fit ORIGINALS only, never rank for deployment."""
    import joblib, sklearn
    from .direct_models import models, metrics
    output=Path(output);output.mkdir(exist_ok=False);fits=[];predictions=[];drift=[]
    originals=[r for r in rows if r['transform']=='original']
    if len(originals)!=15 or len({r['original_sha256'] for r in originals})!=15:
        raise ValueError('Expected 15 distinct original images')
    if len({r['lineage_block'] for r in rows})!=1 or any(r['production_approved'] for r in rows):
        raise ValueError('Do not change unknown lineage or approval state')
    y=np.array([r['reported_g'] for r in originals],float)
    for name, keys in feature_sets.items():
        x=np.array([[r[k] for k in keys] for r in originals],float)
        all_x=np.array([[r[k] for k in keys] for r in rows],float)
        if not np.isfinite(all_x).all():raise ValueError('Invalid feature matrix')
        for model_name, model in models().items():
            model.fit(x,y);fitted=model.predict(x);all_pred=model.predict(all_x)
            refs={r['image_id']:float(p) for r,p in zip(originals,fitted)}
            ident=name+'__'+model_name
            # Private research artifacts cannot be loaded as a production model.
            artifact=dict(version=VERSION,model=model,feature_keys=keys,production_eligible=False,
                independent_validation=False,approval='EXPLORATORY_ONLY',fit_original_sha256=[r['original_sha256'] for r in originals],
                parameters=model.get_params(deep=False),label_status='USER_REPORTED_NOMINAL_NOT_EXACT',random_seed=42)
            artifact['source_hashes']=_source_hashes()
            artifact['pipeline_configs']=[asdict(c) for c in configurations() if c.name in ('B_normalized5','D_object4')]
            joblib.dump(artifact,output/(ident+'.joblib'))
            fit=dict(id=ident,feature_set=name,model=model_name,original_images=len(originals),
                feature_count=len(keys),design_rank=int(np.linalg.matrix_rank(np.column_stack([np.ones(len(x)),x]))),
                training_metrics=metrics(y,fitted),independent_metrics=None,
                by_weight_training={str(int(w)):metrics(y[y==w],fitted[y==w]) for w in np.unique(y)})
            fits.append(fit)
            for row,pred in zip(rows,all_pred):
                predictions.append(dict(id=ident,image_id=row['image_id'],transform=row['transform'],
                    reported_g=row['reported_g'],research_prediction_g=float(pred),
                    prediction_change_g=float(pred-refs[row['image_id']]),
                    original_sha256=row['original_sha256'],lineage_block=row['lineage_block'],
                    public_prediction_g=None,independent_test=False))
            for transform in sorted({r['transform'] for r in rows}-{'original'}):
                changes=np.array([abs(p['prediction_change_g']) for p in predictions if p['id']==ident and p['transform']==transform])
                drift.append(dict(id=ident,transform=transform,mean_change_g=float(changes.mean()),
                    p95_change_g=float(np.percentile(changes,95)),max_change_g=float(changes.max()),
                    physical_distance_experiment=False))
            standard=[r['max_change_g'] for r in drift if r['id']==ident and r['transform']!='canvas_scale75']
            fit['max_standard_transform_change_g']=max(standard)
            fit['canvas_change_g']=next(r['max_change_g'] for r in drift if r['id']==ident and r['transform']=='canvas_scale75')
    return fits,predictions,drift,dict(sklearn=sklearn.__version__,joblib=joblib.__version__)


def run(receipt_dir, output):
    root=Path(receipt_dir).resolve();out=Path(output).resolve()
    if out.exists():raise FileExistsError('Use a new private output directory')
    receipt=json.loads((root/'receipt.json').read_text())
    if receipt['independent_test_images']!=0 or receipt['images']!=15:
        raise ValueError('This diagnostic requires the unresolved 15-original receipt')
    for row in receipt['rows']:
        p=(root/'originals'/row['relative_path']).resolve()
        if not p.is_relative_to(root/'originals') or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('Changed original/path')
    out.mkdir(parents=True);cv2.setNumThreads(2)
    config_b=next(c for c in configurations() if c.name=='B_normalized5')
    config_d=next(c for c in configurations() if c.name=='D_object4')
    object_keys=['object_'+key for key in config_d.keys]
    feature_sets=dict(B5_scene=config_b.keys,D4_object=object_keys,
        E6_object_interior=object_keys+INTERIOR_KEYS,
        background_negative_control=['background_mean','background_fraction','image_aspect'])
    rows=[];baseline=[];texture=[];reference=[];cards=[]
    for record in receipt['rows']:
        ident=Path(record['filename']).stem;folder=out/ident;folder.mkdir()
        rgb=image_to_rgb(root/'originals'/record['relative_path'])
        original_flags=[]
        for transform,scene in transformations(rgb):
            b,normalized,mask_b=process(scene,config_b)
            d,object_scene,mask_d=process(scene,config_d)
            interior=interior_texture(object_scene,mask_d)
            if transform=='original':original_flags=list(b['qc_flags'])
            row={**b,**{'object_'+k:d[k] for k in config_d.keys},**interior,
                'image_id':ident,'transform':transform,'reported_g':int(record['weight_g_reported']),
                'capture_group':record['capture_group_id'],'original_sha256':record['sha256'],
                'lineage_block':'UNRESOLVED_SHARED_MATERIAL','production_approved':False,
                'image_aspect':scene.shape[1]/scene.shape[0],
                'physical_distance_cm':None,'arrangement_label':'unreviewed_dense_or_contact',
                'qc_flags':propagate_capture_qc(original_flags,b['qc_flags'])['qc_flags']}
            rows.append(row)
            raw=raw_mask(scene,config_b)
            periodic=extract_bundle_texture(scene,raw)
            texture.append({k:v for k,v in periodic.items() if k!='profiles'}|dict(image_id=ident,transform=transform))
            if transform=='original':
                result=detect_spread(scene,**runtime_parameters()['DEFAULT_SPREAD'])
                gate=weight_gate(dict(name='校正状態の診断のみ',g_per_strand=.68),using_demo=False,mode='広げた状態',mask=result.mask)
                baseline.append(dict(image_id=ident,reported_g=row['reported_g'],detected_candidates=result.count,
                    gate_reasons=gate,quality_reasons=image_quality_warnings(scene),
                    candidate_count_is_true_strands=False,public_prediction_g=None,
                    known_product_calibration=False,diagnostic_scope='GATE_WITH_POSITIVE_CALIBRATION_NOT_A_WEIGHT_ACCURACY_TEST'))
                metric=extract_metric_foreground(scene)
                reference.append(dict(image_id=ident,status=metric['status'],reference=metric['reference']))
                _preview(scene,folder/'input.jpg');_preview(result.overlay_rgb,folder/'count_overlay.jpg')
                Image.fromarray(result.mask).save(folder/'opencv_mask.png')
                Image.fromarray(mask_b).save(folder/'lab_mask.png')
                overlay=object_scene.copy();contours,_=cv2.findContours(mask_d,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay,contours,-1,(255,50,200),1);_preview(overlay,folder/'object_overlay.jpg')
                (folder/'periodic_profiles.json').write_text(json.dumps(periodic,ensure_ascii=False,allow_nan=False))
                cards.append(f'<article><h3>{html.escape(ident)} · 申告 {row["reported_g"]}g</h3><p>本数候補 {result.count} / 重量は未出力</p>'+
                    ''.join(f'<figure><img src="{ident}/{f}"><figcaption>{label}</figcaption></figure>' for f,label in [('input.jpg','元画像プレビュー'),('count_overlay.jpg','現行の本数候補'),('opencv_mask.png','現行HSV'),('lab_mask.png','HSV+Lab'),('object_overlay.jpg','形状・内部質感候補')])+
                    '<p>'+html.escape(' / '.join(gate))+'</p></article>')
            # Preserve a subset of derivative masks for direct visual inspection.
            if transform in {'original','resize50','resize25','canvas_scale75'}:
                Image.fromarray(mask_b).save(folder/(transform+'_B_mask.png'))
                Image.fromarray(mask_d).save(folder/(transform+'_D_mask.png'))
        print(ident,'diagnosed',flush=True)
    fits,predictions,drift,model_versions=fit_exploratory(rows,feature_sets,out/'research_models')
    report=dict(version=VERSION,original_images=15,total_inputs=len(rows),derived_inputs=len(rows)-15,conservative_lineage_groups=1,
        label_status='USER_REPORTED_NOMINAL',training_approved=False,independent_metrics=None,
        independent_by_weight=None,independent_by_distance=None,independent_dense_error=None,independent_failure_rate=None,
        no_independent_evaluation_reason='Physical sample lineage across weights/batches unresolved',
        feature_sets=feature_sets,pipeline_configs=[asdict(config_b),asdict(config_d)],transform_settings=TRANSFORM_SETTINGS,
        fits=fits,source_hashes=_source_hashes(),input_receipt_sha256=hashlib.sha256((root/'receipt.json').read_bytes()).hexdigest(),
        versions=dict(python=platform.python_version(),numpy=np.__version__,opencv=cv2.__version__,**model_versions),
        periodic_texture_usable_originals=sum(r['texture_available']==1 for r in texture if r['transform']=='original'),
        periodic_regression_status='NOT_FIT_INSUFFICIENT_RESOLVED_ORIGINALS_AND_WEIGHT_COVERAGE',
        metric_reference_usable_originals=sum(r['status']=='research_features_only' for r in reference),
        production_adoption=False,public_prediction_g=None)
    for name,data in [('results.json',report),('features.json',rows),('baseline.json',baseline),('texture_diagnostics.json',texture),('references.json',reference)]:
        (out/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False))
    for name,data in [('features.csv',rows),('training_fits.csv',fits),('paired_drift.csv',drift),('predictions.csv',predictions),('opencv_baseline.csv',baseline)]:write_csv(out/name,data)
    _graphs(out,rows,fits,drift)
    _write_summary(out,report,baseline)
    table=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [f['id'],round(f['training_metrics']['mae_g'],2),round(f['training_metrics']['rmse_g'],2),round(f['max_standard_transform_change_g'],2),round(f['canvas_change_g'],2)])+'</tr>' for f in fits)
    (out/'index.html').write_text('<!doctype html><html lang="ja"><meta charset="utf-8"><title>密集乾麺の重量推定診断</title><style>body{font:16px sans-serif;max-width:1300px;margin:24px auto;background:#faf8f2;color:#29352e}article{background:white;padding:16px;margin:20px 0}figure{display:inline-block;width:18%;margin:.5%}img{max-width:100%}td,th{padding:6px;border:1px solid #ddd}table{border-collapse:collapse}h1{font-size:25px}</style><h1>方式B・密集乾麺の非公開診断</h1><p>15原本・同一系列の探索。訓練内誤差と同一画像の感度であり独立した重量精度ではありません。距離別実測誤差は未算出。本番採用なし。</p><p><a href="REPORT_JA.md">日本語報告</a> · <a href="features.csv">特徴量CSV</a> · <a href="results.json">結果と再現条件</a></p><table><tr><th>候補</th><th>訓練MAE g</th><th>訓練RMSE g</th><th>通常変換最大差g</th><th>構図変換最大差g</th></tr>'+table+'</table><img src="training_and_drift.png"><img src="feature_by_weight.png">'+''.join(cards))
    return report


def _write_summary(out,report,baseline):
    lines=['# 密集乾麺の直接重量推定・探索結果','',
        '独立した物理試料の評価ではありません。原本15枚＋派生210入力の合計225入力を扱い、学習には原本15枚だけを使用しました。',
        '画像のランダム分割・重複画像による水増し・学習承認・本番モデル登録は行っていません。', '',
        '## 確認した範囲', '',
        f'- 現行OpenCVは、正の校正値を与えた診断でも {sum(bool(r["gate_reasons"]) for r in baseline)}/15枚が密集・接触等で拒否されました。これは今回の同系列写真の診断上の拒否率で、一般的な失敗率ではありません。',
        f'- 周期を解像できた画像は {report["periodic_texture_usable_originals"]}/15枚。未解像の周期を1本幅として使用しません。',
        f'- 有効な実寸基準は {report["metric_reference_usable_originals"]}/15枚。A4のない画像にmm換算は適用しません。',
        '- 未知試料MAE/RMSE・重量別誤差・実撮影距離別誤差・密集状態別誤差・独立試料の失敗率はすべて未算出（JSONではnull）。MAE5g目標を達成したとは判断できません。','',
        '## 同じ原本による比較（訓練内誤差）','',
        '|候補|訓練内MAE g|訓練内RMSE g|通常変換の最大予測差 g|余白・構図試験の最大差 g|',
        '|---|---:|---:|---:|---:|']
    for f in report['fits']:
        lines.append(f'|{f["id"]}|{f["training_metrics"]["mae_g"]:.2f}|{f["training_metrics"]["rmse_g"]:.2f}|{f["max_standard_transform_change_g"]:.2f}|{f["canvas_change_g"]:.2f}|')
    lines+=['','背景だけの負の対照が小さい訓練誤差になる場合は、背景や構図と重量ラベルの交絡を疑います。',
        'デジタル縮小・余白追加は実際のカメラ距離の試験ではありません。定数予測の変動0も精度の証拠ではありません。',
        '前景マスクは自動候補で、正解マスクとのIoUや精度は未評価です。面積や方向性は隠れた多層の量を直接表しません。','',
        '## 次の評価','',
        '別々の未使用麺で秤量し、製品・物理試料・材料系列・撮影会・距離・配置・実測重量・はかりの分解能を記録します。',
        '同じ試料の再配置・再撮影・派生画像は同じ分割に固定します。系譜とラベルを確認して分割を封印するまで、独立指標は計算しません。',
        'A4実寸特徴は印刷100mm線と全マーカー、同平面、対象全体が収まることを確認した写真でだけ比較します。',
        'weight_evaluation.pyは、承認済みの分割宣言・全テスト試行・拒否理由を検査し、試料等重みの誤差と拒否率を計算します。本番採用は自動承認しません。','']
    (out/'REPORT_JA.md').write_text('\n'.join(lines),encoding='utf-8')


def _source_hashes():
    paths=[Path(__file__)]+[Path(__file__).with_name(n) for n in (
        'bundle_texture.py','metric_foreground.py','stability_pipeline.py','stability_experiment.py',
        'direct_models.py','marker_reference.py','diagnose_series.py')]
    paths += [Path(__file__).parents[1]/n for n in ('pasta_vision.py','measurement_diagnostics.py','developer_ui.py')]
    return {str(p.relative_to(Path(__file__).parents[1])):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def _graphs(out,rows,fits,drift):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=[r for r in fits if r['model']=='ridge']
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    names=[r['feature_set'] for r in selected]
    axes[0].bar(names,[r['training_metrics']['mae_g'] for r in selected]);axes[0].set_title('TRAINING MAE ONLY (not independent)');axes[0].set_ylabel('g')
    axes[1].bar(names,[r['max_standard_transform_change_g'] for r in selected]);axes[1].set_title('Same-original maximum prediction drift');axes[1].set_ylabel('g')
    for a in axes:a.tick_params(axis='x',rotation=25)
    fig.tight_layout();fig.savefig(out/'training_and_drift.png',dpi=160);plt.close(fig)
    originals=[r for r in rows if r['transform']=='original']
    keys=['mask_fraction','object_area_over_length_squared','interior_structure_coherence','interior_luminance_cv']
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for a,key in zip(axes.ravel(),keys):
        a.scatter([r['reported_g'] for r in originals],[r[key] for r in originals]);a.set_title(key);a.set_xlabel('Reported g (same unresolved series)')
    fig.tight_layout();fig.savefig(out/'feature_by_weight.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--receipt',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();result=run(args.receipt,args.output)
    print('Independent metrics:',result['independent_metrics'],'Production:',result['production_adoption'])
