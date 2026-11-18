"""Private multi-batch diagnostic, fixed mask candidates and exploratory regression.

No DB, model activation, label-dependent image processing or independent accuracy
claims. Count-only rows are excluded from fitting. All original groups are held in
one unresolved lineage block. Transforms measure sensitivity of the same originals.
"""
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

from ..pasta_vision import image_to_rgb, detect_spread, analyze_spread_mask, image_quality_warnings
from ..measurement_diagnostics import decide_count_weight
from ..preprocessing import propose_conservative_mask
from .compare_masks import masks
from .bundle_texture import extract_bundle_texture
from .dense_weight_benchmark import interior_texture

VERSION = 'photo-batch-diagnostic-v1'
PARAMETERS = dict(DEFAULT_SPREAD=dict(saturation_min=20,value_min=75,min_area=160,min_elongation=3.),
                  DEFAULT_BUNDLE=dict(min_radius=8,max_radius=27,min_distance=32,hough_sensitivity=18))
FEATURE_GROUPS = dict(
    scene=['area_fraction','length_fraction','width_fraction','rect_density','components'],
    shape_texture=['width_over_length','rect_density','edge_density','direction_coherence',
                   'interior_luminance_cv','interior_structure_coherence'])


def image_features(rgb, mask):
    """Only image and mask enter descriptors. No physical scale is inferred."""
    h,w=mask.shape
    ys,xs=np.nonzero(mask)
    length=width=perimeter=0.
    if len(xs)>=3:
        points=np.column_stack([xs,ys]).astype(np.float32)
        width,length=sorted(cv2.minAreaRect(points)[1])
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        perimeter=sum(cv2.arcLength(c,True) for c in contours)
    area=int(cv2.countNonZero(mask))
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0);gy=cv2.Sobel(gray,cv2.CV_32F,0,1)
    inside=cv2.erode(mask,np.ones((5,5),np.uint8))>0
    a=float(np.sum(gx[inside]**2));c=float(np.sum(gy[inside]**2));b=float(np.sum(gx[inside]*gy[inside]))
    edges=cv2.Canny(rgb,60,120)>0
    result=dict(area_pixels=area,area_fraction=area/(h*w),length_px=float(length),width_px=float(width),
        length_fraction=float(length/max(h,w)),width_fraction=float(width/max(h,w)),
        width_over_length=float(width/max(length,1)),rect_density=float(area/max(area,length*width,1)),
        perimeter_px=float(perimeter),components=int(cv2.connectedComponents(mask)[0]-1),
        edge_density=float(np.count_nonzero(edges&(mask>0))/max(area,1)),
        direction_coherence=float(np.sqrt((a-c)**2+4*b*b)/max(a+c,1e-9)))
    return dict(result,**interior_texture(rgb,mask))


def regression_features(rgb):
    # One aspect-preserving resolution for fit and prediction, including transforms.
    h,w=rgb.shape[:2];size=(max(2,round(w*960/max(h,w))),max(2,round(h*960/max(h,w))))
    normalized=cv2.resize(rgb,size,interpolation=cv2.INTER_AREA if max(h,w)>960 else cv2.INTER_CUBIC)
    candidate=masks(normalized,'spread_overlap',PARAMETERS)['hsv_lab']
    return image_features(normalized,candidate)


def write_csv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()} for row in rows)


def regression_model_factories():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression,Ridge
    from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
    return dict(linear=lambda:make_pipeline(StandardScaler(),LinearRegression()),
        ridge=lambda:make_pipeline(StandardScaler(),Ridge(alpha=10.)),
        random_forest=lambda:RandomForestRegressor(n_estimators=100,min_samples_leaf=2,random_state=42,n_jobs=1),
        gradient_boosting=lambda:GradientBoostingRegressor(n_estimators=100,max_depth=2,min_samples_leaf=2,random_state=42))


def fit_models(records,images,out):
    from sklearn.metrics import mean_absolute_error,mean_squared_error
    models=regression_model_factories()
    train=[r for r in records if r['reported_g'] is not None]
    features={r['image_id']:regression_features(images[r['image_id']]) for r in records}
    variants={}
    for r in train:
        rgb=images[r['image_id']]
        variants[r['image_id']]={
            'half':regression_features(cv2.resize(rgb,None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA)),
            'rotate90':regression_features(np.rot90(rgb).copy())}
    fits=[];predictions=[]
    for group,keys in FEATURE_GROUPS.items():
        x=np.array([[features[r['image_id']][k] for k in keys] for r in train]);y=np.array([r['reported_g'] for r in train])
        for name,factory in models.items():
            model=factory();model.fit(x,y);pred=model.predict(x)
            changes=[]
            for r,estimate in zip(train,pred):
                row=dict(image_id=r['image_id'],reported_g=r['reported_g'],feature_group=group,model=name,
                         fitted_g=float(estimate),absolute_error_g=float(abs(estimate-r['reported_g'])),
                         independent_prediction=False,production_eligible=False)
                for transform,feat in variants[r['image_id']].items():
                    value=float(model.predict([[feat[k] for k in keys]])[0]);row[transform+'_g']=value
                    changes.append(abs(value-estimate))
                predictions.append(row)
            fits.append(dict(feature_group=group,model=name,original_images=len(train),
                training_mae_g=float(mean_absolute_error(y,pred)),training_rmse_g=float(np.sqrt(mean_squared_error(y,pred))),
                max_transform_delta_g=float(max(changes)),independent_mae_g=None,independent_rmse_g=None,production_eligible=False))
    write_csv(out/'regression_predictions.csv',predictions)
    write_csv(out/'regression_fits.csv',fits)
    return fits,predictions


def load_diagnostic_entries(datasets, output):
    """Load the unchanged manifests and verify each original before analysis."""
    from tools.diagnose_photo_manifest import manifest_records,photo_spec
    entries=[];seen=set()
    for dataset in datasets:
        root=Path(dataset).resolve()
        if Path(output).resolve().is_relative_to(root):raise ValueError('Output must be separate from originals')
        source=(root/'manifest.json').read_bytes()
        raw=json.loads(source.decode('utf-8-sig'))
        for record in manifest_records(raw):
            relative,weight,count=photo_spec(record);path=root.joinpath(*relative.parts)
            if not path.resolve().is_relative_to(root) or path.is_symlink():raise ValueError('Unsafe original')
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if digest!=record['sha256']:raise ValueError('Original SHA mismatch')
            if digest in seen:raise ValueError('Duplicate originals require explicit review')
            seen.add(digest)
            entries.append(dict(path=path,record=record,reported_g=weight,reported_count=count,sha256=digest,
                                dataset_root=root, manifest_sha256=hashlib.sha256(source).hexdigest()))
    return entries


def run(datasets,output):
    out=Path(output).resolve()
    if out.exists():raise FileExistsError('Use a new output folder')
    entries=load_diagnostic_entries(datasets,out)
    out.mkdir(parents=True);(out/'images').mkdir();cv2.setNumThreads(1)
    rows=[];frames={};cards=[]
    for number,entry in enumerate(entries):
        rgb=image_to_rgb(entry['path']);ident=f'photo_{number+1:02d}';frames[ident]=rgb
        Image.fromarray(rgb).save(out/'images'/f'{ident}_original.jpg')
        candidates=masks(rgb,'spread_overlap',PARAMETERS)
        h,w=rgb.shape[:2];candidates['supported']=propose_conservative_mask(rgb,[0,0,w,h])['mask']
        base=detect_spread(rgb,**PARAMETERS['DEFAULT_SPREAD'])
        methods={key:candidates[key] for key in ('hsv_baseline','hsv_no_close','hsv_lab','shadow_clahe','supported')}
        row=dict(image_id=ident,filename=entry['record']['filename'],sha256=entry['sha256'],
            reported_g=entry['reported_g'],reported_count=entry['reported_count'],
            image_quality_warnings=image_quality_warnings(rgb),physical_sample_lineage='unresolved_shared_block',
            estimated_g=None,absolute_error_g=None,production_display=False,training_approved=False,methods={})
        for name,mask in methods.items():
            result=base if name=='hsv_baseline' else analyze_spread_mask(rgb,mask,close=False)
            # Positive unit calibration is a gate-only counterfactual, never a claimed mass.
            possible=decide_count_weight(dict(name='診断用の正校正仮定',g_per_strand=1.),result,mode='spread')
            missing=decide_count_weight(dict(name='製品不明',g_per_strand=None),result,mode='spread')
            feature=image_features(rgb,mask)
            periodic=extract_bundle_texture(rgb,mask)
            row['methods'][name]=dict(count=result.count,evidence=result.evidence,features=feature,
                texture_status=periodic['texture_status'],pitch_px=periodic['pitch_px'],
                count_absolute_error=abs(result.count-entry['reported_count']) if entry['reported_count'] is not None else None,
                known_product_calibration_would_pass=possible.weight_g is not None and not row['image_quality_warnings'],
                reasons=list(missing.reasons),reason_codes=list(missing.reason_codes),
                mask_review='candidate_not_ground_truth')
            Image.fromarray(mask).save(out/'images'/f'{ident}_{name}_mask.png')
            Image.fromarray(result.overlay_rgb).save(out/'images'/f'{ident}_{name}_overlay.jpg')
        rows.append(row)
        cards.append(_card(row))
    fits,predictions=fit_models(rows,frames,out)
    report=dict(version=VERSION,originals_verified=len(rows),unique_originals=len(entries),independent_evaluation_groups=0,
        independent_mae_g=None,independent_rmse_g=None,models_activated=False,
        rows=rows,regression_fits=fits,feature_groups=FEATURE_GROUPS,parameters=PARAMETERS,
        versions=dict(python=platform.python_version(),numpy=np.__version__,opencv=cv2.__version__),
        source_sha256={str(p.relative_to(Path(__file__).resolve().parents[2])):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__).resolve(),Path(__file__).resolve().parents[1]/'pasta_vision.py',
                      Path(__file__).resolve().parents[1]/'measurement_diagnostics.py']},
        regression_settings=dict(normalized_long_side=960,ridge_alpha=10.,random_state=42,
            forest_trees=100,forest_min_leaf=2,boost_trees=100,boost_max_depth=2,boost_min_leaf=2,
            fit_scope='known-mass originals only; no independent split; count-only and transforms never fitted'))
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    flat=[dict(image_id=r['image_id'],filename=r['filename'],reported_g=r['reported_g'],reported_count=r['reported_count'],
        method=name,count=m['count'],count_absolute_error=m['count_absolute_error'],estimated_g=None,absolute_error_g=None,
        known_product_calibration_would_pass=m['known_product_calibration_would_pass'],reasons=m['reasons'],**m['features'])
        for r in rows for name,m in r['methods'].items()]
    write_csv(out/'photo_features.csv',flat)
    _graphs(out,flat,fits)
    (out/'index.html').write_text('<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
      '<title>Pasta Lens 実写真比較</title><style>body{font:16px system-ui;max-width:1200px;margin:20px auto;padding:14px;background:#faf8f2;color:#29352e}article{background:white;padding:16px;border-radius:16px;margin:20px 0}img{max-width:100%}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px}pre{white-space:pre-wrap}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ccc}</style>'
      '<h1>実写真のマスク・検出・特徴量比較</h1><p>原本SHA照合済み。製品校正が未確認のため本番の推定重量・絶対誤差は未算出。'
      '正校正の仮定は判定経路の検査です。回帰値は同じ画像に当てはめた訓練内の値で、独立精度ではありません。</p>'
      '<p><a href="report.json">全診断JSON</a> · <a href="photo_features.csv">特徴量CSV</a> · <a href="regression_fits.csv">回帰比較</a> · <a href="regression_predictions.csv">訓練内の各予測</a> · <a href="REPORT_JA.md">日本語報告</a></p>'
      '<img src="feature_trends.png"><img src="training_only.png">'+''.join(cards)+'</html>',encoding='utf-8')
    for entry in entries:
        if hashlib.sha256(entry['path'].read_bytes()).hexdigest()!=entry['sha256']:raise RuntimeError('Original changed')
    return report


def _card(row):
    ident=row['image_id'];title=html.escape(row['filename'])
    cells=[]
    for name,method in row['methods'].items():
        cells.append(f'<div><h3>{name} · {method["count"]}候補</h3><img loading="lazy" src="images/{ident}_{name}_mask.png"><img loading="lazy" src="images/{ident}_{name}_overlay.jpg"><p>'+html.escape(' '.join(method['reasons']))+'</p></div>')
    truth = f'{row["reported_g"]} g' if row['reported_g'] is not None else f'{row["reported_count"]} 本（重量不明）'
    return f'<article><h2>{title} · 申告 {truth}</h2><img loading="lazy" width="500" src="images/{ident}_original.jpg"><div class="grid">'+''.join(cells)+'</div></article>'


def _graphs(out,rows,fits):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for ax,key in zip(axes.flat,['area_fraction','width_fraction','rect_density','edge_density','direction_coherence','count']):
        for name in ('hsv_baseline','hsv_lab'):
            values=[r for r in rows if r['method']==name and r['reported_g'] is not None]
            ax.scatter([r['reported_g'] for r in values],[r[key] for r in values],label=name,alpha=.75)
        ax.set(xlabel='Reported weight (g)',ylabel=key);ax.legend()
    fig.suptitle('Dependent photos: descriptive association only');fig.tight_layout();fig.savefig(out/'feature_trends.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5));labels=[r['feature_group']+' / '+r['model'] for r in fits]
    ax.barh(labels,[r['training_mae_g'] for r in fits]);ax.set(xlabel='Training MAE (g), NOT independent accuracy');fig.tight_layout();fig.savefig(out/'training_only.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dataset',type=Path,action='append',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=run(args.dataset,args.output)
    print('Verified originals:',result['originals_verified'],'independent evaluation groups:',result['independent_evaluation_groups'])
