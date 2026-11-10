"""Label-blind candidate features for offline direct-weight exploration."""
import argparse,csv,hashlib,html,json,platform
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from ..pasta_vision import image_to_rgb,detect_spread
from .compare_masks import masks
from .diagnose_series import runtime_parameters,descriptors


def extract(rgb,mask):
    f,_,contours,_=descriptors(mask);h,w=mask.shape;ys,xs=np.nonzero(mask)
    flags=[]
    border=int(np.count_nonzero(mask[0])+np.count_nonzero(mask[-1])+np.count_nonzero(mask[:,0])+np.count_nonzero(mask[:,-1]))
    if border>8:flags.append('foreground_at_image_border')
    if f['mask_fraction']<.005:flags.append('too_little_foreground')
    if f['largest_component_fraction']>.8:flags.append('merged_region_possible_overlap')
    profile=[0.]*20;angle=0.
    if len(xs)>2:
        pts=np.column_stack((xs,ys)).astype(float);centered=pts-pts.mean(axis=0);_,vec=np.linalg.eigh(centered.T@centered/len(pts));axis=vec[:,-1]
        # Fix arbitrary eigenvector sign to stabilize profiles across repeated runs.
        if axis[np.argmax(np.abs(axis))]<0:axis=-axis
        angle=float(np.arctan2(axis[1],axis[0]));normal=np.array([-axis[1],axis[0]])
        along=centered@axis;across=centered@normal;bins=np.minimum(19,((along-along.min())/max(1.,np.ptp(along))*20).astype(int))
        for b in range(20):
            v=across[bins==b]
            profile[b]=float(np.ptp(v)+1) if len(v) else 0.
    length=max(f['long_axis_px'],1);gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB)
    fg=mask>0;bg=~fg;lap=cv2.Laplacian(gray,cv2.CV_32F)
    f.update(long_relative=f['long_axis_px']/max(h,w),short_relative=f['short_axis_px']/max(h,w),aspect_ratio=f['long_axis_px']/max(1,f['short_axis_px']),
        width_p50_relative=float(np.median(profile))/length,width_cv=float(np.std(profile)/max(1,np.mean(profile))),
        orientation_deg=float(np.degrees(angle)),orientation_cos2=float(np.cos(2*angle)),orientation_sin2=float(np.sin(2*angle)),image_aspect=w/h,
        perimeter_normalized=f['contour_perimeter_px']/max(1,np.sqrt(f['mask_area_px'])),
        foreground_L=float(lab[:,:,0][fg].mean()) if fg.any() else 0.,foreground_b=float(lab[:,:,2][fg].mean()) if fg.any() else 0.,
        texture_std=float(gray[fg].std()) if fg.any() else 0.,texture_laplacian=float(lap[fg].var()) if fg.any() else 0.,
        background_mean=float(gray[bg].mean()) if bg.any() else 0.,background_std=float(gray[bg].std()) if bg.any() else 0.,background_bright_fraction=float(np.mean(gray[bg]>220)) if bg.any() else 0.,
        projection_widths_px=profile,border_pixels=border,qc_flags=flags,mask_approved=False,public_prediction_g=None,public_status='要確認：未検証の実験用マスク・モデル')
    return f,contours


def run(receipt,output):
    root=Path(receipt);out=Path(output)
    if out.exists():raise FileExistsError('No overwrites')
    ledger=json.loads((root/'receipt.json').read_text());params=runtime_parameters();cv2.setNumThreads(2)
    for r in ledger['rows']:
        p=(root/'originals'/r['relative_path']).resolve()
        if not p.is_relative_to((root/'originals').resolve()) or hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:raise ValueError('Original changed')
    out.mkdir(parents=True);rows=[];cards=[]
    for r in ledger['rows']:
        ident=Path(r['filename']).stem;folder=out/ident;folder.mkdir();rgb=image_to_rgb(root/'originals'/r['relative_path']);Image.fromarray(rgb).save(folder/'original.jpg');parts=[]
        for transform,scene in [('original',rgb),('resize_half',cv2.resize(rgb,None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA)),('rotate90',np.rot90(rgb).copy())]:
            candidates=masks(scene,'spread_overlap',params)
            for method in ('hsv_baseline','hsv_lab'):
                mask=candidates[method];f,contours=extract(scene,mask)
                f.update(image_id=ident,reported_g=int(r['weight_g_reported']),capture_group=r['capture_group_id'],lineage_block=r['conservative_lineage_block'],method=method,transform=transform,source_sha256=r['sha256'])
                f['legacy_candidate_count']=detect_spread(scene,**params['DEFAULT_SPREAD']).count if method=='hsv_baseline' else None
                rows.append(f)
                if transform=='original':
                    overlay=scene.copy();cv2.drawContours(overlay,contours,-1,(255,0,255),2)
                    Image.fromarray(mask).save(folder/(method+'_mask.png'));Image.fromarray(overlay).save(folder/(method+'_overlay.jpg'))
                    panel=Image.new('RGB',(1200,450),'#faf8f2');d=ImageDraw.Draw(panel)
                    d.text((8,8),f'{ident} {method} area={f["mask_area_px"]} border={f["border_pixels"]} [UNAPPROVED]',fill='black')
                    for i,arr in enumerate((scene,mask,overlay)):
                        im=Image.fromarray(arr).convert('RGB');im.thumbnail((390,410));panel.paste(im,(i*400,30))
                    panel.save(folder/(method+'_panel.jpg'))
                    parts.append(f'<h3>{method}</h3><a href="{ident}/{method}_overlay.jpg"><img loading="lazy" src="{ident}/{method}_panel.jpg"></a><p>面積 {f["mask_area_px"]} / 占有率 {f["mask_fraction"]:.3f} / 長短辺 {f["long_axis_px"]:.1f}, {f["short_axis_px"]:.1f}px / QC {html.escape(str(f["qc_flags"]))}</p><a href="{ident}/{method}_mask.png">原寸マスク</a>')
        cards.append(f'<article><h2>{ident} 申告{r["weight_g_reported"]}g</h2>'+''.join(parts)+'</article>');print(ident,flush=True)
    report=dict(purpose='exploratory_nominal_labels_only',images=15,capture_groups=5,independent_test_images=0,supervised_approval=False,rows=rows,
        runtime_parameters=params,feature_version=1,versions=dict(python=platform.python_version(),opencv=cv2.__version__,numpy=np.__version__),
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('compare_masks.py'),Path(__file__).with_name('diagnose_series.py')]},receipt_sha256=hashlib.sha256((root/'receipt.json').read_bytes()).hexdigest())
    (out/'features.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    keys=[k for k,v in rows[0].items() if not isinstance(v,(dict,list))]
    with (out/'features.csv').open('w',newline='') as stream:
        wr=csv.DictWriter(stream,fieldnames=keys,extrasaction='ignore');wr.writeheader();wr.writerows(rows)
    (out/'index.html').write_text('<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>方式B・候補マスク検品</title><style>body{background:#faf8f2;font-family:sans-serif;margin:20px}article{background:white;padding:16px;margin:20px 0}img{max-width:100%}</style><h1>方式B：15枚の候補マスク</h1><p>原本とマスクを別保存。生成マスク未承認。QCフラグは誤りを網羅せず、表示なしでも検品合格ではありません。各パネルは元画像・マスク・輪郭です。</p><a href="features.csv">90条件の特徴量CSV</a>'+''.join(cards))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--receipt',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.receipt,a.output)
