"""Offline mask/scale diagnostics. Never trains, alters a DB, or promotes a mask/model."""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import html
import json
from pathlib import Path
import platform
import re
import subprocess
from collections import defaultdict
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from ..pasta_vision import image_to_rgb,detect_spread,detect_bundle_ends,image_quality_warnings
from ..measurement_diagnostics import spread_rejection
from ..geometry_candidate import geometry,skeletonize


def runtime_parameters():
    """Read literal UI defaults without importing Streamlit/training modules."""
    path=Path(__file__).resolve().parents[1]/'developer_ui.py'
    result={}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('DEFAULT_SPREAD','DEFAULT_BUNDLE'):
            if not isinstance(node.value,ast.Call) or not isinstance(node.value.func,ast.Name) or node.value.func.id!='dict':raise ValueError('Review changed runtime configuration')
            result[node.targets[0].id]={k.arg:ast.literal_eval(k.value) for k in node.value.keywords}
    if len(result)!=2:raise ValueError('Runtime configuration unavailable')
    return result


def descriptors(mask):
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask,connectivity=8)
    points=cv2.findNonZero(mask)
    area=int(cv2.countNonZero(mask))
    short,long=sorted(cv2.minAreaRect(points)[1]) if points is not None else (0.,0.)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    small=cv2.resize(mask,(max(1,round(mask.shape[1]*512/max(mask.shape))),max(1,round(mask.shape[0]*512/max(mask.shape)))),interpolation=cv2.INTER_NEAREST)
    geo=geometry(small)
    return dict(mask_area_px=area,mask_fraction=area/mask.size,components=n-1,
        components_area_ge160=int(np.sum(stats[1:,cv2.CC_STAT_AREA]>=160)),
        largest_component_fraction=float(stats[1:,cv2.CC_STAT_AREA].max()/area) if area else 0.,
        long_axis_px=float(long),short_axis_px=float(short),rect_density=area/max(area,long*short,1.),
        contour_perimeter_px=sum(float(cv2.arcLength(c,True)) for c in contours),
        skeleton_length_512=geo['length_px'],skeleton_width_512=geo['width_px'],
        skeleton_density_512=geo['density'],geometry_at_512=geo),labels,contours,small


def analyze(rgb,mode,params):
    if mode not in ('spread_overlap','bundle_ends_low_angle','handheld_bundle_ends'):
        raise ValueError('Unsupported diagnostic mode')
    spread=mode=='spread_overlap'
    result=detect_spread(rgb,**params['DEFAULT_SPREAD']) if spread else detect_bundle_ends(rgb,**params['DEFAULT_BUNDLE'])
    desc,labels,contours,small=descriptors(result.mask)
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    opts=params['DEFAULT_SPREAD' if spread else 'DEFAULT_BUNDLE']
    raw=cv2.inRange(hsv,np.array([7,opts.get('saturation_min',20 if spread else 12),opts.get('value_min',75 if spread else 60)]),np.array([45,255,255]))
    before=cv2.connectedComponentsWithStats(raw,connectivity=8)
    rng=np.random.default_rng(0);colors=rng.integers(40,255,(labels.max()+1,3),dtype=np.uint8);colors[0]=0
    colored=colors[labels]
    edges=rgb.copy();cv2.drawContours(edges,contours,-1,(255,0,255),2)
    sk=skeletonize(small)*255
    desc.update(detected_count=result.count,detector='connected_regions' if spread else 'hough_circles',
        count_is_verified_strand_count=False,no_closing_area_px=int(cv2.countNonZero(raw)),
        no_closing_components=int(before[0]-1),
        no_closing_largest_fraction=float(before[2][1:,cv2.CC_STAT_AREA].max()/max(1,cv2.countNonZero(raw))) if before[0]>1 else 0.,closing_added_pixels=int(np.count_nonzero((result.mask>0)&(raw==0))),
        quality_warnings=image_quality_warnings(rgb),spread_guard_reasons=spread_rejection(result.mask) if spread else [],
        measured_weight_prediction_g=None,mask_qc='NOT_VERIFIED',training_eligible=False)
    return desc,{'original':rgb,'detections':result.overlay_rgb,'mask':result.mask,'components':colored,'contours':edges,'skeleton_512':sk,'no_closing':raw}


def panel(images,path,title):
    canvas=Image.new('RGB',(1260,910),'#f5f5f0');draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18) if Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf').exists() else ImageFont.load_default()
    draw.text((12,8),title,fill='black',font=font)
    for i,key in enumerate(('original','detections','mask','components','contours','skeleton_512')):
        x=(i%3)*420;y=(i//3)*430+40
        draw.text((x+8,y),key,fill='black',font=font)
        im=Image.fromarray(images[key]).convert('RGB');im.thumbnail((410,390))
        canvas.paste(im,(x+(420-im.width)//2,y+28))
    canvas.save(path)


def correlations(rows):
    """Descriptive n=5 association within one dependent series, never accuracy."""
    groups=defaultdict(list)
    for r in rows:groups[r['capture_mode']].append(r)
    result=[]
    for mode,items in groups.items():
        weight=np.array([float(r['nominal_weight_g']) for r in items])
        for name in ('mask_area_px','mask_fraction','long_axis_px','short_axis_px','rect_density','detected_count','skeleton_length_512','skeleton_width_512','skeleton_density_512'):
            v=np.array([r[name] for r in items],float)
            result.append(dict(mode=mode,feature=name,n=len(items),pearson_r=float(np.corrcoef(weight,v)[0,1]) if np.std(v)>0 else None,
                independent_samples=1,interpretation='dependent-series description only; not validation or causal evidence'))
    return result


def run(originals,output):
    originals=Path(originals);output=Path(output)
    if output.exists():raise FileExistsError('Refusing to overwrite diagnostics')
    params=runtime_parameters();cv2.setNumThreads(2)
    with (originals/'labels.csv').open(encoding='utf-8-sig',newline='') as f:metadata=list(csv.DictReader(f))
    if len(metadata)!=15 or len({m['physical_family_group_id'] for m in metadata})!=1:raise ValueError('Expected one series of 15 images')
    for meta in metadata:
        if not re.fullmatch(r'[A-Za-z0-9_-]+',Path(meta['original_filename']).stem) or Path(meta['image_path']).name!=meta['original_filename']:raise ValueError('Invalid filename')
        if meta['capture_mode'] not in ('spread_overlap','bundle_ends_low_angle','handheld_bundle_ends'):raise ValueError('Unsupported diagnostic mode')
        if meta['nominal_weight_g'] not in ('20','40','60','80','100'):raise ValueError('Unsupported weight metadata')
    output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for meta in metadata:
        path=(originals/meta['image_path']).resolve()
        if not path.is_relative_to(originals.resolve()):raise ValueError('Invalid path')
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=meta['sha256']:raise ValueError('Received bytes changed')
        rgb=image_to_rgb(path);desc,images=analyze(rgb,meta['capture_mode'],params)
        ident=Path(meta['original_filename']).stem;folder=output/ident;folder.mkdir()
        for key,im in images.items():Image.fromarray(im).save(folder/(key+'.png'))
        panel(images,folder/'panel.jpg',f'{ident} | nominal {meta["nominal_weight_g"]}g | {meta["capture_mode"]} | detected {desc["detected_count"]}')
        # Digital perturbations expose fixed-pixel sensitivity, not actual distance/angle experiments.
        perturb=[]
        for name,other in [('resize_075',cv2.resize(rgb,None,fx=.75,fy=.75,interpolation=cv2.INTER_AREA)),
                           ('resize_050',cv2.resize(rgb,None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA)),
                           ('rotate_90',np.rot90(rgb).copy())]:
            detector=detect_spread if meta['capture_mode']=='spread_overlap' else detect_bundle_ends
            opts=params['DEFAULT_SPREAD' if meta['capture_mode']=='spread_overlap' else 'DEFAULT_BUNDLE']
            detected=detector(other,**opts)
            perturb.append(dict(transform=name,count=detected.count,mask_area_px=int(cv2.countNonZero(detected.mask)),shape=list(other.shape)))
        # Compare actual runtime with library defaults; never replace the runtime baseline.
        defaultcount=detect_bundle_ends(rgb).count if meta['capture_mode']!='spread_overlap' else None
        row=dict(meta,**desc,image_id=ident,received_sha256=digest,inference_shape=list(rgb.shape),
                 runtime_parameters=params['DEFAULT_SPREAD' if meta['capture_mode']=='spread_overlap' else 'DEFAULT_BUNDLE'],
                 library_default_count=defaultcount,digital_perturbations=perturb)
        rows.append(row);print(ident,meta['nominal_weight_g'],meta['capture_mode'],'count',desc['detected_count'],'area',desc['mask_area_px'],flush=True)
    sources=['pasta_vision.py','measurement_diagnostics.py','developer_ui.py','geometry_candidate.py','experiments/diagnose_series.py']
    base=Path(__file__).resolve().parents[1]
    report=dict(purpose='diagnosis_only',physical_families=1,training_images=0,independent_evaluation_images=0,
        qc_approved_images=0,weight_predictions_available=False,accuracy_metrics=None,
        reason='No verified product calibration, strand-count ground truth, mask ground truth or independent family.',
        source_hashes={n:hashlib.sha256((base/n).read_bytes()).hexdigest() for n in sources},
        code_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=base).decode().strip(),
        versions=dict(python=platform.python_version(),opencv=cv2.__version__,numpy=np.__version__),
        runtime_parameters=params,rows=rows,associations=correlations(rows))
    (output/'diagnostics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    columns=['image_id','nominal_weight_g','capture_mode','detected_count','mask_area_px','mask_fraction','components','components_area_ge160','largest_component_fraction','long_axis_px','short_axis_px','rect_density','contour_perimeter_px','no_closing_components','closing_added_pixels','library_default_count','no_closing_largest_fraction','skeleton_length_512','skeleton_width_512','skeleton_density_512']
    with (output/'features.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(rows)
    cards=''.join(f'<article><h2>{html.escape(r["image_id"])} / {r["nominal_weight_g"]}g / {html.escape(r["capture_mode"])}</h2><p>検出候補数 {r["detected_count"]}（本数の正解ではありません）・重量予測なし</p><a href="{r["image_id"]}/panel.jpg"><img loading="lazy" src="{r["image_id"]}/panel.jpg"></a><p>'+ ' · '.join(f'<a href="{r["image_id"]}/{k}.png">{label}</a>' for k,label in [('original','入力画像'),('mask','マスク'),('detections','検出'),('components','連結成分'),('contours','輪郭'),('skeleton_512','骨格'),('no_closing','結合処理前')])+'</p></article>' for r in rows)
    (output/'index.html').write_text('<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Pasta Lens 写真診断</title><style>body{font-family:sans-serif;background:#faf8f2;margin:24px;color:#29352e}article{background:white;border:1px solid #ddd;border-radius:16px;padding:16px;margin:20px 0}img{width:100%;max-width:1260px}h2{font-size:18px}</style><h1>15枚のOpenCV診断</h1><p>非公開・検品未承認。全15枚は同一試料系列。学習・独立精度評価は実施していません。色マスクには背景や手が含まれます。マスクの面積・骨格はパスタの真の面積・長さではありません。</p>'+cards)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--originals',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.originals,a.output)
