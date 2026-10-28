"""Reproducible synthetic comparison, explicitly NOT evidence of real-photo accuracy.
python -m pasta_weight_app_v2.experiments.geometry_benchmark --output PRIVATE_DIR
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import cv2
import numpy as np
from ..geometry_candidate import ProductCalibration, estimate
from ..pasta_vision import detect_spread
from ..measurement_diagnostics import weight_gate
from ..pasta_ml.model import features as rgb_features
from ..pasta_ml.evaluate import summary

GEOMETRY_FEATURE_VERSION='geometry-scale-free-v1'
CALIBRATION=ProductCalibration('synthetic-only',7.,280.,50,50.)


def shape_features(f):
    w=max(2*f['radius_px'] if f.get('radius_px') else f['width_px'],1.)
    return np.array([f['area_px']/w**2,f['length_px']/w,f['endpoints']/2,
        f['branches'],f['components'],f['density'],f['width_ratio'],f['mask_sensitivity']],float)


def scene(count, crossing, seed, angle=0., scale=1., shadow=False):
    rng=np.random.default_rng(seed)
    rgb=np.full((1100,1100,3),35,np.uint8)
    for i in range(count//2):
        cx=190+(i%3)*350+rng.integers(-8,9);cy=90+(i//3)*100+rng.integers(-3,4)
        for side in (-1,1):
            if crossing:
                theta=np.deg2rad(side*8);dx=140*np.cos(theta);dy=140*np.sin(theta)
                a=(round(cx-dx),round(cy-dy));b=(round(cx+dx),round(cy+dy))
            else:a=(cx-140,cy+side*13);b=(cx+140,cy+side*13)
            cv2.line(rgb,a,b,(215,175,65),5,cv2.LINE_8)
    if shadow:
        rgb[:,:550]=(rgb[:,:550]*.65).astype(np.uint8)
    if angle:
        # Pad to preserve all noodles; rotating must not secretly crop test objects.
        rgb=cv2.copyMakeBorder(rgb,250,250,250,250,cv2.BORDER_CONSTANT,value=(35,35,35))
        h,w=rgb.shape[:2]
        rgb=cv2.warpAffine(rgb,cv2.getRotationMatrix2D((w/2,h/2),angle,1),(w,h),borderValue=(35,35,35))
    if scale!=1:rgb=cv2.resize(rgb,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    return rgb


def fit(x,y,alpha=1.):
    x=np.asarray(x);y=np.asarray(y)
    mean=x.mean(0);scale=x.std(0);scale[scale<1e-8]=1
    z=(x-mean)/scale;intercept=float(y.mean())
    coef=np.linalg.solve(z.T@z+alpha*np.eye(x.shape[1]),z.T@(y-intercept))
    return dict(mean=mean.tolist(),scale=scale.tolist(),coef=coef.tolist(),intercept=intercept,
        alpha=alpha,train_min=float(y.min()),train_max=float(y.max()),
        max_distance=float(max(3,np.linalg.norm(z,axis=1).max()*1.5)))


def predict(model,x):
    z=(x-model['mean'])/model['scale']
    y=float(z@model['coef']+model['intercept'])
    valid=np.isfinite(y) and model['train_min']<=y<=model['train_max'] and np.linalg.norm(z)<=model['max_distance']
    return dict(success=bool(valid),predicted_g=y if valid else None,
        reason='experimental synthetic regression' if valid else 'outside training support')


def run(output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'report.json').exists():raise ValueError('実験結果は上書きしません。新しい出力先を指定してください。')
    rows=[]; train_x=[];train_shape=[];train_y=[]
    for split,seeds in [('train',range(10,13)),('val',range(20,22)),('test',range(30,32))]:
        for count in (20,40,60):
            for crossing in (False,True):
                for seed in seeds:
                    # Test scenarios specified in code, no test-based coefficient or threshold selection.
                    angle=0 if split=='train' else (15 if seed%2 else 0)
                    scale=1 if split=='train' else (.75 if seed%2 else 1.1)
                    rgb=scene(count,crossing,seed,angle,scale,shadow=split!='train')
                    geo=estimate(rgb,'spread',CALIBRATION,single_layer=True,perpendicular=True)
                    legacy=detect_spread(rgb)
                    xr=rgb_features(rgb);xg=shape_features(geo['features'])
                    identity=f'{split}-{count}-{int(crossing)}-{seed}'
                    record=dict(id=identity,physical_sample_id=f'synthetic-scene-{seed}',split=split,source='synthetic',actual_weight_g=count,
                        mode='spread',condition='crossing' if crossing else 'separated',angle_deg=angle,scale=scale,
                        pixel_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),geometry=geo,x_rgb=xr.tolist(),x_geometry=xg.tolist())
                    raw=dict(success=legacy.count>0,predicted_g=float(legacy.count) if legacy.count else None)
                    rejection=weight_gate(dict(name='Synthetic calibrated fixture',g_per_strand=1),using_demo=False,mode='広げた状態',mask=legacy.mask)
                    record['legacy_raw']=raw
                    record['production_guarded']=dict(success=raw['success'] and not rejection,
                        predicted_g=raw['predicted_g'] if not rejection else None,reason=' / '.join(rejection))
                    rows.append(record)
                    if split=='train':train_x.append(xr);train_shape.append(xg);train_y.append(count)
    rgb_model=fit(train_x,train_y);geometry_model=fit(train_shape,train_y)
    for r in rows:
        r['rgb_ridge']=predict(rgb_model,np.array(r['x_rgb']))
        r['geometry_ridge']=predict(geometry_model,np.array(r['x_geometry']))
        a=r['geometry'];b=r['geometry_ridge']
        agree=a['success'] and b['success'] and abs(a['predicted_g']-b['predicted_g'])<=max(2,.15*a['predicted_g'])
        r['hybrid']=dict(success=bool(agree),predicted_g=(a['predicted_g']+b['predicted_g'])/2 if agree else None,
            reason='geometry/regression agreement' if agree else 'uncertain or disagreement')
    reports={}
    for split in ('train','val','test'):
        subset=[r for r in rows if r['split']==split]
        reports[split]={}
        for method in ('legacy_raw','production_guarded','geometry','rgb_ridge','geometry_ridge','hybrid'):
            scored=[dict(actual_weight_g=r['actual_weight_g'],**r[method]) for r in subset]
            reports[split][method]=dict(overall=summary(scored),by_weight={str(w):summary([s for s in scored if s['actual_weight_g']==w]) for w in (20,40,60)},
                by_condition={c:summary([dict(actual_weight_g=r['actual_weight_g'],**r[method]) for r in subset if r['condition']==c]) for c in ('crossing','separated')})
        common=[r for r in subset if r['legacy_raw']['success'] and r['geometry']['success']]
        reports[split]['paired_legacy_geometry']=dict(images=len(common),legacy=summary([dict(actual_weight_g=r['actual_weight_g'],**r['legacy_raw']) for r in common]),geometry=summary([dict(actual_weight_g=r['actual_weight_g'],**r['geometry']) for r in common]))
    report=dict(source='synthetic',production_eligible=False,notice='Synthetic mechanism experiment, NOT real-photo accuracy. Do not activate these models.',
        feature_version=GEOMETRY_FEATURE_VERSION,python=platform.python_version(),numpy=np.__version__,opencv=cv2.__version__,
        calibration=CALIBRATION.__dict__,independent_synthetic_groups=dict(train=3,val=2,test=2),train_images=len(train_y),val_images=12,test_images=12,alpha=1.,reports=reports,rows=rows)
    for name,value in [('report',report),('rgb_model',rgb_model),('geometry_model',geometry_model)]:
        (output/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps({k:v['overall'] for k,v in reports['test'].items() if 'overall' in v},ensure_ascii=False,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    run(p.parse_args().output)
