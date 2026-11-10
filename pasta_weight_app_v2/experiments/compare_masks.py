"""Offline, unapproved segmentation candidates. No fitting, DB writes or g prediction."""
from __future__ import annotations
import argparse,csv,hashlib,html,json,platform,re
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from ..pasta_vision import image_to_rgb,detect_spread
from .diagnose_series import runtime_parameters,descriptors

METHODS=('hsv_baseline','hsv_no_close','lab_only','hsv_lab','shadow_clahe','opening','watershed','manual_roi')
PARAMETERS=dict(lab_b_min=145,lab_a_max=150,lab_L_min=35,lab_L_max=245,
 hsv_h_min=7,hsv_h_max=45,hsv_s_min=30,hsv_v_min=55,
 clahe_clip=2.,clahe_tiles=[8,8],opening_kernel=[3,3],
 watershed_max_side=512,watershed_peak_distance_fraction=.25,
 local_radius_at1400=[2,16],local_distance_at1400=6,local_param2=12,
 local_mask_occupancy=.65)


def roi_mask(shape,annotation):
    pts=annotation['roi_polygon_normalized']
    result=np.zeros(shape[:2],np.uint8)
    if pts:
        arr=np.array(pts,float)
        if arr.ndim!=2 or arr.shape[1]!=2 or len(arr)<3 or not np.isfinite(arr).all() or (arr<0).any() or (arr>1).any():raise ValueError('Invalid ROI')
        xy=np.rint(arr*np.array([shape[1]-1,shape[0]-1])).astype(np.int32)
        cv2.fillPoly(result,[xy],255)
    return result


def masks(rgb,mode,params):
    if mode not in ('spread_overlap','bundle_ends_low_angle','handheld_bundle_ends'):raise ValueError('Unknown capture state')
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB)
    spread=mode=='spread_overlap';opts=params['DEFAULT_SPREAD' if spread else 'DEFAULT_BUNDLE']
    raw=cv2.inRange(hsv,np.array([7,opts.get('saturation_min',20 if spread else 12),opts.get('value_min',75 if spread else 60)]),np.array([45,255,255]))
    labmask=((lab[:,:,2]>=145)&(lab[:,:,1]<=150)&(lab[:,:,0]>=35)&(lab[:,:,0]<=245)).astype(np.uint8)*255
    chroma=cv2.inRange(hsv,np.array([7,30,55]),np.array([45,255,255]))
    combined=cv2.bitwise_and(labmask,chroma)
    corrected=lab.copy();corrected[:,:,0]=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8)).apply(lab[:,:,0])
    ch=cv2.cvtColor(cv2.cvtColor(corrected,cv2.COLOR_LAB2RGB),cv2.COLOR_RGB2HSV)
    shadow=cv2.bitwise_and(labmask,cv2.inRange(ch,np.array([7,30,45]),np.array([45,255,255])))
    return dict(hsv_baseline=cv2.morphologyEx(raw,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8)),
        hsv_no_close=raw,lab_only=labmask,hsv_lab=combined,shadow_clahe=shadow,
        opening=cv2.morphologyEx(combined,cv2.MORPH_OPEN,np.ones((3,3),np.uint8)),watershed=combined.copy())


def split_regions(mask,rgb):
    """Distance-marker watershed can split blobs, never recovers known strand identities."""
    size=(max(1,round(mask.shape[1]*512/max(mask.shape))),max(1,round(mask.shape[0]*512/max(mask.shape))))
    small=cv2.resize(mask,size,interpolation=cv2.INTER_NEAREST)
    dist=cv2.distanceTransform(small,cv2.DIST_L2,5)
    if not dist.max():return np.zeros(small.shape,np.int32),0
    peaks=(dist>=cv2.dilate(dist,np.ones((9,9),np.uint8))-.001)&(dist>.25*dist.max())
    n,markers=cv2.connectedComponents(peaks.astype(np.uint8))
    markers=markers.astype(np.int32)+1;markers[(small>0)&~peaks]=0
    scene=cv2.resize(rgb,size,interpolation=cv2.INTER_AREA)
    labels=cv2.watershed(scene,markers)
    labels[small==0]=0
    count=len([x for x in np.unique(labels) if x>1])
    return labels,count


def circles(rgb,mask,opts,local=False,scale=1.):
    gray=cv2.GaussianBlur(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),(5,5),1.15)
    if local:
        ys,xs=np.nonzero(mask)
        if not len(xs):return [],rgb.copy()
        x0,y0,x1,y1=max(0,xs.min()-3),max(0,ys.min()-3),min(mask.shape[1],xs.max()+4),min(mask.shape[0],ys.max()+4)
        image=gray[y0:y1,x0:x1]
        radius0=max(1,round(2*scale));radius1=max(radius0+1,round(16*scale));distance=max(2,6*scale);sensitivity=12
    else:
        x0=y0=0;image=gray;radius0=opts['min_radius'];radius1=opts['max_radius'];distance=opts['min_distance'];sensitivity=opts['hough_sensitivity']
    if min(image.shape)<4:return [],rgb.copy()
    cand=cv2.HoughCircles(image,cv2.HOUGH_GRADIENT,dp=1.,minDist=float(distance),param1=85,param2=float(sensitivity),minRadius=radius0,maxRadius=radius1)
    found=[];overlay=rgb.copy()
    if cand is not None:
        for x,y,r in np.round(cand[0]).astype(int):
            x+=x0;y+=y0
            if not (0<=x<mask.shape[1] and 0<=y<mask.shape[0]):continue
            ry=max(1,r//3);patch=mask[max(0,y-ry):min(mask.shape[0],y+ry+1),max(0,x-ry):min(mask.shape[1],x+ry+1)]
            if patch.size and np.count_nonzero(patch)/patch.size >= (.65 if local else .35):
                found.append((int(x),int(y),int(r)));cv2.circle(overlay,(x,y),r,(255,70,70),2)
    return found,overlay


def probe_stats(mask,annotation):
    """Sparse human-picked pixels are diagnostic references, not whole-mask truth."""
    result={}
    for kind in ('positive','negative'):
        points=annotation[kind+'_points']
        values=[]
        for x,y in points:
            if not 0<=x<=1 or not 0<=y<=1:raise ValueError('Invalid probe')
            ix=round(x*(mask.shape[1]-1));iy=round(y*(mask.shape[0]-1))
            values.append(bool(mask[iy,ix]))
        result[kind+'_probe_hits']=sum(values);result[kind+'_probe_count']=len(values)
    return result


def feature_row(mask,rgb,method,mode,annotation,params,scale):
    f,labels,contours,small=descriptors(mask)
    # All pixels incl hand/background remain candidates; masks are never ground truth.
    f.update(probe_stats(mask,annotation))
    f['roi_fraction']=float(np.count_nonzero(mask & roi_mask(rgb.shape,annotation))/max(1,np.count_nonzero(mask))) if annotation['roi_polygon_normalized'] else None
    f['circle_candidates']=None;f['local_circle_candidates']=None
    overlay=rgb.copy();cv2.drawContours(overlay,contours,-1,(255,0,255),1)
    if mode!='spread_overlap':
        found,overlay=circles(rgb,mask,params['DEFAULT_BUNDLE']);f['circle_candidates']=len(found)
        if method=='manual_roi' and annotation['roi_polygon_normalized']:
            found,overlay=circles(rgb,mask,params['DEFAULT_BUNDLE'],local=True,scale=scale);f['local_circle_candidates']=len(found)
        cv2.drawContours(overlay,contours,-1,(255,0,255),1)
    if mode=='spread_overlap' and method=='hsv_baseline':f['legacy_detected_count']=detect_spread(rgb,**params['DEFAULT_SPREAD']).count
    else:f['legacy_detected_count']=None
    f['watershed_regions']=None;colored=None
    if method=='watershed':
        split,count=split_regions(mask,rgb);f['watershed_regions']=count
        palette=np.random.default_rng(42).integers(45,255,(max(2,int(split.max())+1),3),dtype=np.uint8);palette[0:2]=0
        colored=palette[np.maximum(split,0)];colored[split<0]=255
    f['method']=method;f['capture_mode']=mode;f['scale']=scale
    f['mask_qc']='NOT_VERIFIED';f['training_eligible']=False;f['predicted_weight_g']=None
    return f,overlay,colored


def source_annotations(rgb,a):
    overlay=rgb.copy();roi=roi_mask(rgb.shape,a)
    contour,_=cv2.findContours(roi,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(overlay,contour,-1,(0,180,255),3)
    for kind,color in [('positive',(0,255,80)),('negative',(255,50,50))]:
        for x,y in a[kind+'_points']:cv2.circle(overlay,(round(x*(rgb.shape[1]-1)),round(y*(rgb.shape[0]-1))),7,color,-1)
    return overlay


def validate_records(records,draft,originals):
    """Validate the complete intake before creating any experiment output."""
    modes=('spread_overlap','bundle_ends_low_angle','handheld_bundle_ends')
    expected={(str(w),m) for w in (20,40,60,80,100) for m in modes}
    families={r['physical_family_group_id'].strip() for r in records}
    cells={(r['nominal_weight_g'],r['capture_mode']) for r in records}
    if len(records)!=15 or len(families)!=1 or '' in families or cells!=expected:
        raise ValueError('Expected complete 5-weight by 3-view series with one nonempty family')
    identifiers=set();paths=set()
    for r in records:
        ident=Path(r['original_filename']).stem
        if not re.fullmatch(r'[A-Za-z0-9_-]+',ident) or ident not in draft or ident in identifiers:
            raise ValueError('Invalid, duplicate image id or missing review draft')
        p=(originals/r['image_path']).resolve()
        if not p.is_relative_to(originals) or p in paths or p.name!=r['original_filename']:
            raise ValueError('Unsafe or duplicate original path')
        if hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:
            raise ValueError('Changed original')
        roi_mask((10,10,3),draft[ident]);probe_stats(np.zeros((10,10),np.uint8),draft[ident])
        if not isinstance(draft[ident].get('observed_view'),str):
            raise ValueError('Missing observed view')
        identifiers.add(ident);paths.add(p)


def run(originals,annotations,output):
    originals=Path(originals).resolve();output=Path(output)
    if output.exists():raise FileExistsError('No overwrites')
    annotation_bytes=Path(annotations).read_bytes();draft=json.loads(annotation_bytes)
    with (originals/'labels.csv').open(encoding='utf-8-sig',newline='') as f:records=list(csv.DictReader(f))
    validate_records(records,draft,originals)
    output.mkdir(parents=True,exist_ok=False);params=runtime_parameters();cv2.setNumThreads(2)
    results=[];cards=[]
    for r in records:
        ident=Path(r['original_filename']).stem;a=draft[ident];rgb=image_to_rgb(originals/r['image_path']);folder=output/ident;folder.mkdir()
        Image.fromarray(source_annotations(rgb,a)).save(folder/'annotation.jpg')
        main_images=[]
        for scale in (1.,.75,.5):
            scene=rgb if scale==1 else cv2.resize(rgb,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
            candidates=masks(scene,r['capture_mode'],params)
            candidates['manual_roi']=cv2.bitwise_and(candidates['hsv_lab'],roi_mask(scene.shape,a))
            baseline=candidates['hsv_baseline']
            for method,mask in candidates.items():
                f,overlay,segments=feature_row(mask,scene,method,r['capture_mode'],a,params,scale)
                f.update(image_id=ident,nominal_weight_g=int(r['nominal_weight_g']),observed_view=a['observed_view'])
                f['added_vs_baseline_px']=int(np.count_nonzero((mask>0)&(baseline==0)))
                f['removed_vs_baseline_px']=int(np.count_nonzero((mask==0)&(baseline>0)))
                # Cropping away a region can remove real noodles; this is not an error metric.
                f['manual_roi_available']=bool(a['roi_polygon_normalized'])
                if method=='manual_roi' and not a['roi_polygon_normalized']:
                    # Missing endface ROI is unavailable, not a measured zero.
                    for key in list(f):
                        if key not in ('method','capture_mode','scale','image_id','nominal_weight_g','observed_view','mask_qc','training_eligible','predicted_weight_g','manual_roi_available'):
                            f[key]=None
                    f['status']='UNSUPPORTED_SIDE_VIEW'
                else:f['status']='DIAGNOSTIC_ONLY'
                results.append(f)
                if scale==1:
                    Image.fromarray(mask).save(folder/(method+'_mask.png'))
                    # Contour and circle overlay, plus comparison colors vs baseline.
                    Image.fromarray(overlay).save(folder/(method+'_overlay.jpg'))
                    diff=rgb.copy();diff[(mask>0)&(baseline==0)]=[0,230,100];diff[(mask==0)&(baseline>0)]=[255,70,70]
                    Image.fromarray(diff).save(folder/(method+'_delta.jpg'))
                    if segments is not None:Image.fromarray(segments).save(folder/'watershed_segments.png')
                    thumb=Image.new('RGB',(520,400),'#f7f7f2');draw=ImageDraw.Draw(thumb);draw.text((10,6),f'{method} area={f["mask_area_px"]} cc={f["components"]}',fill='black')
                    for i,arr in enumerate([scene,mask,overlay,diff]):
                        im=Image.fromarray(arr).convert('RGB');im.thumbnail((250,175));thumb.paste(im,((i%2)*260,(i//2)*190+24))
                    thumb.save(folder/(method+'_panel.jpg'))
                    segment_link=f' · <a href="{ident}/watershed_segments.png">分割領域</a>' if segments is not None else ''
                    main_images.append(f'<section><h3>{method}</h3><img loading="lazy" src="{ident}/{method}_panel.jpg"><p>{f["status"]} / 円候補 {f["circle_candidates"]} / 局所円 {f["local_circle_candidates"]} / 分割領域 {f["watershed_regions"]}</p><a href="{ident}/{method}_mask.png">マスク原寸</a> · <a href="{ident}/{method}_overlay.jpg">輪郭・円</a> · <a href="{ident}/{method}_delta.jpg">基準との差分</a>{segment_link}</section>')
        cards.append(f'<article data-mode="{html.escape(r["capture_mode"])}"><h2>{ident} / 名目{html.escape(r["nominal_weight_g"])}g / {html.escape(r["capture_mode"])}</h2><p>{html.escape(a["observed_view"])}</p><details><summary>手動ROI・参照点（未承認）</summary><img loading="lazy" src="{ident}/annotation.jpg"><p>水色枠=粗い手動ROI、緑=麺参照点、赤=背景・手参照点。完全な正解マスクではありません。</p></details><div class="grid">'+''.join(main_images)+'</div></article>')
        print(ident,'done',flush=True)
    base=Path(__file__).resolve().parents[1]
    report=dict(purpose='candidate_mask_comparison_only',families=1,images=15,training_images=0,independent_test_images=0,
        calibration_applied=False,accuracy_metrics=None,methods=list(METHODS),runtime_parameters=params,candidate_parameters=PARAMETERS,
        annotation_sha256=hashlib.sha256(annotation_bytes).hexdigest(),annotation_status='DRAFT_NOT_APPROVED',
        source_hashes={str(p.relative_to(base)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),base/'pasta_vision.py',base/'geometry_candidate.py',base/'experiments/diagnose_series.py']},
        label_sha256=hashlib.sha256((originals/'labels.csv').read_bytes()).hexdigest(),
        image_sha256={r['original_filename']:r['sha256'] for r in records},
        versions=dict(python=platform.python_version(),opencv=cv2.__version__,numpy=np.__version__),rows=results)
    (output/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    keys=[k for k in results[0] if k!='geometry_at_512']
    with (output/'features.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in results)
    page='''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>マスク方式比較（非公開）</title><style>body{font-family:sans-serif;background:#faf8f2;color:#29352e;margin:20px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}section,article{background:white;border:1px solid #ddd;border-radius:12px;padding:12px;margin:10px 0}img{max-width:100%}details img{max-height:600px}select{font-size:18px;padding:8px}</style><h1>15枚・8方式のマスク比較</h1><p>非公開／全15枚は1物理試料系列／学習未承認／重量推定・独立精度評価は未実施。</p><p>各パネル：左上=元画像、右上=マスク、左下=輪郭・円、右下=現行との差分。緑=追加、赤=除去であり、正解・不正解の色分けではありません。</p><p>mask面積には手・背景が含まれ得ます。局所ROIは人手で指定した比較条件で、撮影だけの自動推定ではありません。分割数・円数は正解本数ではありません。60g手持ち側面の局所断面は未対応です。</p><label>撮影状態 <select id="mode"><option value="all">すべて</option><option>spread_overlap</option><option>bundle_ends_low_angle</option><option>handheld_bundle_ends</option></select></label><p><a href="feature_trends.png">特徴量の推移</a> · <a href="scale_sensitivity.png">縮小変化</a> · <a href="features.csv">全特徴量CSV</a></p>'''+''.join(cards)+'''<script>document.querySelector('#mode').onchange=e=>document.querySelectorAll('article').forEach(a=>a.hidden=e.target.value!=='all'&&a.dataset.mode!==e.target.value)</script></html>'''
    (output/'index.html').write_text(page)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--originals',required=True);p.add_argument('--annotations',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.originals,a.annotations,a.output)
