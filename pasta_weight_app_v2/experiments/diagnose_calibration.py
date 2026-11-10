"""Offline known-count diagnostics; no DB writes, training or calibration promotion."""
from __future__ import annotations
import argparse,csv,hashlib,html,json,math,platform
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from ..pasta_vision import image_to_rgb,detect_spread,detect_bundle_ends
from .compare_masks import masks,circles
from .diagnose_series import runtime_parameters,descriptors


def comparison_errors(count,true_count,total_g):
    if isinstance(count,bool) or not isinstance(count,int) or count<0:raise ValueError('Invalid candidate count')
    if isinstance(true_count,bool) or not isinstance(true_count,int) or true_count<=0:raise ValueError('Invalid reference count')
    if isinstance(total_g,bool) or not isinstance(total_g,(float,int)) or not math.isfinite(total_g) or total_g<=0:raise ValueError('Invalid measured mass')
    unit=total_g/true_count;weight=count*unit
    return dict(reference_count=true_count,reference_weight_g=total_g,g_per_strand=unit,
        count_difference=count-true_count,absolute_count_difference=abs(count-true_count),
        diagnostic_weight_g=weight,weight_error_g=weight-total_g,absolute_weight_error_g=abs(weight-total_g))


def spread_positions(mask,opts):
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask,8);items=[]
    for i in range(1,n):
        x,y,w,h,area=map(int,stats[i])
        if area<opts['min_area']:continue
        ys,xs=np.where(labels[y:y+h,x:x+w]==i)
        if len(xs)<5:continue
        short,long=sorted(cv2.minAreaRect(np.column_stack((xs+x,ys+y)).astype(np.float32))[1])
        ratio=long/max(1.,short)
        items.append(dict(x=x,y=y,width=w,height=h,area=area,elongation=ratio,accepted=ratio>=opts['min_elongation']))
    return items


def diagnose(rgb,mode,method,params):
    spread=mode=='overhead_bundle';mapped='spread_overlap' if spread else 'bundle_ends_low_angle'
    mask=masks(rgb,mapped,params)[method]
    f,_,contours,_=descriptors(mask);overlay=rgb.copy();cv2.drawContours(overlay,contours,-1,(200,0,220),1)
    positions=[]
    if spread:
        positions=spread_positions(mask,params['DEFAULT_SPREAD'])
        count=0
        for item in positions:
            x,y,w,h=[item[k] for k in ('x','y','width','height')]
            color=(40,255,70) if item['accepted'] else (255,150,30)
            cv2.rectangle(overlay,(x,y),(x+w,y+h),color,2)
            if item['accepted']:count+=1;cv2.putText(overlay,str(count),(x,max(18,y)),cv2.FONT_HERSHEY_SIMPLEX,.65,color,2)
    else:
        accepted,_=circles(rgb,mask,params['DEFAULT_BUNDLE'])
        raw,_=circles(rgb,np.full(mask.shape,255,np.uint8),params['DEFAULT_BUNDLE'])
        count=len(accepted);accepted_set=set(accepted);index=0
        for x,y,r in raw:
            ok=(x,y,r) in accepted_set
            positions.append(dict(x=x,y=y,radius=r,accepted=ok))
            color=(40,255,70) if ok else (255,150,30)
            cv2.circle(overlay,(x,y),r,color,2)
            if ok:
                index+=1;cv2.putText(overlay,str(index),(x,max(16,y)),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2)
        # Overlap flags are only potential duplicates, not verified duplicate strands.
        f['overlapping_circle_pairs']=sum(math.hypot(a[0]-b[0],a[1]-b[1])<a[2]+b[2] for i,a in enumerate(accepted) for b in accepted[i+1:])
    if method=='hsv_baseline':
        original=detect_spread(rgb,**params['DEFAULT_SPREAD']) if spread else detect_bundle_ends(rgb,**params['DEFAULT_BUNDLE'])
        if count!=original.count or not np.array_equal(mask,original.mask):raise AssertionError('Baseline drift')
    f.update(candidate_count=count,positions=positions,training_approved=False,mask_qc='UNAPPROVED')
    return f,mask,overlay


def run(originals,output):
    originals=Path(originals).resolve();output=Path(output)
    if output.exists():raise FileExistsError('No overwrites')
    rows=list(csv.DictReader((originals/'labels.csv').open(encoding='utf-8-sig')))
    modes={'overhead_bundle','low_angle_ends','closeup_ends'}
    if len(rows)!=3 or {r['capture_mode'] for r in rows}!=modes or len({r['sample_group_id'] for r in rows})!=1 or not rows[0]['sample_group_id'].strip():raise ValueError('Expected three views of one sample')
    names=set()
    for r in rows:
        name=r['filename'];p=(originals/'photos'/name).resolve()
        if Path(name).name!=name or not p.is_relative_to(originals) or name in names:raise ValueError('Unsafe or duplicate image')
        if hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:raise ValueError('Changed original')
        if (r['strand_count_confirmed'],r['measured_total_g'],r['g_per_strand_calculated'])!=('50','34','0.68'):raise ValueError('Review changed reference labels')
        names.add(name)
    output.mkdir(parents=True);params=runtime_parameters();cv2.setNumThreads(2);results=[];cards=[]
    for r in rows:
        ident=Path(r['filename']).stem;folder=output/ident;folder.mkdir();rgb=image_to_rgb(originals/'photos'/r['filename'])
        Image.fromarray(rgb).save(folder/'original.jpg');panels=[]
        for scale in (1.,.75,.5):
            scene=rgb if scale==1 else cv2.resize(rgb,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
            for method in ('hsv_baseline','hsv_lab'):
                f,mask,overlay=diagnose(scene,r['capture_mode'],method,params)
                f.update(image_id=ident,mode=r['capture_mode'],method=method,scale=scale,input_width=scene.shape[1],input_height=scene.shape[0],**comparison_errors(f['candidate_count'],50,34))
                results.append(f)
                if scale==1:
                    Image.fromarray(mask).save(folder/(method+'_mask.png'));Image.fromarray(overlay).save(folder/(method+'_overlay.jpg'))
                    panel=Image.new('RGB',(1440,540),'#faf8f2');d=ImageDraw.Draw(panel)
                    for i,arr in enumerate((scene,mask,overlay)):
                        im=Image.fromarray(arr).convert('RGB');im.thumbnail((470,490));panel.paste(im,(i*480,40))
                    d.text((10,10),f'{ident} {method}: {f["candidate_count"]} candidates vs 50; diagnostic {f["diagnostic_weight_g"]:.2f}g vs 34g',fill='black');panel.save(folder/(method+'_panel.jpg'))
                    panels.append(f'<h3>{method}: {f["candidate_count"]}候補 / 換算{f["diagnostic_weight_g"]:.2f}g</h3><a href="{ident}/{method}_overlay.jpg"><img src="{ident}/{method}_panel.jpg"></a><p><a href="{ident}/{method}_mask.png">原寸マスク</a> · <a href="{ident}/{method}_overlay.jpg">検出位置</a></p>')
        cards.append(f'<article><h2>{ident} {html.escape(r["capture_mode"])}</h2>'+''.join(panels)+'</article>')
        print(ident,[(x['method'],x['candidate_count']) for x in results if x['image_id']==ident and x['scale']==1],flush=True)
    base=Path(__file__).resolve().parents[1]
    report=dict(purpose='known_count_offline_diagnosis',sample_groups=1,independence_from_previous='unknown',training_images=0,production_calibration_approved=False,
        reference=dict(count=50,total_g=34,g_per_strand=.68,product_name=None,diameter_mm=None,length_mm=None,scale_resolution_g=None),
        notes=['Same sample used to derive .68g and compare counts; not independent mass prediction validation.', 'Count differences are net discrepancies, not verified false-negative/false-positive counts.','Digital resizing is not a controlled camera-distance experiment.'],
        parameters=params,versions=dict(python=platform.python_version(),opencv=cv2.__version__,numpy=np.__version__),
        label_sha256=hashlib.sha256((originals/'labels.csv').read_bytes()).hexdigest(),image_sha256={r['filename']:r['sha256'] for r in rows},
        source_hashes={str(p.relative_to(base)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),base/'pasta_vision.py',base/'developer_ui.py',base/'experiments/compare_masks.py',base/'experiments/diagnose_series.py')},rows=results)
    (output/'diagnostics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    columns=['image_id','mode','method','scale','input_width','input_height','candidate_count','count_difference','diagnostic_weight_g','weight_error_g','absolute_weight_error_g','mask_area_px','mask_fraction','components','long_axis_px','short_axis_px']
    with (output/'results.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(results)
    (output/'index.html').write_text('<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>50本・34g 診断</title><style>body{font-family:sans-serif;background:#faf8f2;margin:20px}article{background:white;padding:16px;margin:20px 0;border-radius:12px}img{max-width:100%}</style><h1>同じ50本・34gの3撮影条件</h1><p>非公開・学習未承認・本番未適用。0.68g/本はこの試料自身から算出した検証用の値です。独立評価ではありません。</p><p>左=入力／中央=マスク／右=輪郭と候補。緑=採用候補、橙=色または形条件で除外した候補、紫=マスク輪郭。色は正解・不正解を表しません。候補の番号は麺の個体IDではありません。</p><a href="results.csv">18条件CSV</a> · <a href="diagnostics.json">詳細・候補座標JSON</a>'+''.join(cards))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--originals',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.originals,a.output)
