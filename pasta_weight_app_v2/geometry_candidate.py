"""Experimental geometry estimator. Never activated by the production app.

Visible foreground is not total mass: dense piles and hidden strands are rejected.
Thresholds are engineering guards, not calibrated probabilities of accuracy.
"""
from dataclasses import dataclass, asdict
import math
import cv2
import numpy as np

VERSION = 'geometry-visible-strands-v1'


@dataclass(frozen=True)
class ProductCalibration:
    product_id: str
    diameter_mm: float
    length_mm: float
    weighed_strands: int
    weighed_mass_g: float

    def validate(self):
        if not self.product_id or isinstance(self.weighed_strands, bool) or not isinstance(self.weighed_strands, int) or self.weighed_strands < 10:
            raise ValueError('校正には実製品IDと10本以上の実測本数が必要です。')
        for value in (self.diameter_mm, self.length_mm, self.weighed_mass_g):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError('麺径・長さ・実測重量は有限の正数が必要です。')
        if self.length_mm / self.diameter_mm < 20:
            raise ValueError('長い乾麺用の校正です。製品の長さと直径を確認してください。')
        return self

    @property
    def g_per_strand(self):
        self.validate()
        return self.weighed_mass_g / self.weighed_strands


def candidate_mask(rgb, *, strict=False):
    """HSV AND Lab chroma suppress neutral shadows/white reflections; no closing across strands."""
    if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or min(rgb.shape[:2])<16:
        raise ValueError('16px以上のRGB uint8画像が必要です。')
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB)
    mask=cv2.inRange(hsv,np.array([7,35 if strict else 20,55 if strict else 35]),np.array([45,255,255]))
    mask[lab[:,:,2] < (145 if strict else 138)] = 0
    # Small specks only. No global dilation or opening that erases thin noodles.
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
    cutoff=max(3,round(mask.size*0.000008))
    keep=np.zeros(n,np.uint8)
    keep[1:]=(stats[1:,cv2.CC_STAT_AREA]>=cutoff)*255
    return keep[labels]


def structural_mask(mask, mode='spread'):
    """Optional shape prior; removes short round background objects, not a reviewed segmentation."""
    if mode not in ('spread','bundle_side'):
        return mask.copy()
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
    keep=np.zeros(n,np.uint8)
    for i in range(1,n):
        x,y,w,h,area=stats[i]
        if area<8 or max(w,h)<max(mask.shape)*.12:continue
        ys,xs=np.nonzero(labels[y:y+h,x:x+w]==i)
        short,long=sorted(cv2.minAreaRect(np.column_stack((xs,ys)).astype(np.float32))[1])
        if long/max(1.,short)>=2.5:keep[i]=255
    return keep[labels]


def skeletonize(mask):
    """Zhang–Suen thinning with a zero border and bounded convergence."""
    p=np.pad(mask>0,1).astype(np.uint8)
    for _ in range(max(mask.shape)):
        changed=False
        for step in (0,1):
            c=p[1:-1,1:-1]
            ns=[p[:-2,1:-1],p[:-2,2:],p[1:-1,2:],p[2:,2:],p[2:,1:-1],p[2:,:-2],p[1:-1,:-2],p[:-2,:-2]]
            total=sum(ns)
            transitions=sum(((ns[i]==0)&(ns[(i+1)%8]==1)).astype(np.uint8) for i in range(8))
            if step==0:guard=(ns[0]*ns[2]*ns[4]==0)&(ns[2]*ns[4]*ns[6]==0)
            else:guard=(ns[0]*ns[2]*ns[6]==0)&(ns[0]*ns[4]*ns[6]==0)
            remove=(c==1)&(total>=2)&(total<=6)&(transitions==1)&guard
            if remove.any():c[remove]=0;changed=True
        if not changed:return p[1:-1,1:-1]
    raise ValueError('骨格化が収束しません。画像を確認してください。')


def geometry(mask):
    ys,xs=np.nonzero(mask)
    if not len(xs):return dict(area_px=0.,length_px=0.,width_px=0.,width_ratio=0.,endpoints=0,branches=0,density=0.,border=False,components=0)
    cropped=mask[ys.min():ys.max()+1,xs.min():xs.max()+1]
    padded=np.pad(cropped,2)
    sk=skeletonize(padded)
    distance=cv2.distanceTransform((padded>0).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    neighbors=cv2.filter2D(sk,-1,np.ones((3,3),np.uint8))-sk
    core=(sk>0)&(neighbors==2)
    widths=2*distance[core]-1
    widths=widths[widths>0]
    width=float(np.median(widths)) if len(widths) else 0.
    ratio=float(np.percentile(widths,90)/max(.1,np.percentile(widths,10))) if len(widths) else 0.
    horizontal=np.count_nonzero(sk[:,:-1]&sk[:,1:]);vertical=np.count_nonzero(sk[:-1]&sk[1:])
    diagonal1=np.count_nonzero(sk[:-1,:-1]&sk[1:,1:]&~sk[:-1,1:]&~sk[1:,:-1])
    diagonal2=np.count_nonzero(sk[:-1,1:]&sk[1:,:-1]&~sk[:-1,:-1]&~sk[1:,1:])
    ends=int(np.count_nonzero((sk>0)&(neighbors==1)))
    length=float(horizontal+vertical+math.sqrt(2)*(diagonal1+diagonal2)+ends/2)
    hull=cv2.convexHull(np.column_stack((xs,ys)).astype(np.int32))
    return dict(area_px=float(len(xs)),length_px=length,width_px=width,width_ratio=ratio,
        endpoints=ends,branches=int(np.count_nonzero((sk>0)&(neighbors>=3))),
        density=float(len(xs)/max(len(xs),cv2.contourArea(hull))),
        border=bool(mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any()),
        components=int(cv2.connectedComponents(cropped,8)[0]-1))


def inspect_image(rgb):
    # Uniform longest-side normalization; never stretch aspect ratio into a square.
    scale=min(1.,1000/max(rgb.shape[:2]))
    image=cv2.resize(rgb,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA) if scale<1 else rgb
    mask=candidate_mask(image)
    features=geometry(mask)
    strict=candidate_mask(image,strict=True)
    features['mask_sensitivity']=float(np.count_nonzero(mask!=strict)/max(1,np.count_nonzero(mask)))
    features['working_shape']=list(image.shape[:2])
    return features,mask


def rectify_plane(rgb, corners, width_mm, height_mm, pixels_per_mm=2.):
    """Explicit reference-plane homography; no inferred plate size or invented camera pose.

    corners: TL, TR, BR, BL on the same physical plane as the noodles.
    """
    src=np.asarray(corners,np.float32)
    if src.shape!=(4,2) or not np.isfinite(src).all() or not cv2.isContourConvex(src):
        raise ValueError('同一平面の凸四角形の基準点4つを順番に指定してください。')
    h,w=rgb.shape[:2]
    if (src<0).any() or (src[:,0]>=w).any() or (src[:,1]>=h).any() or cv2.contourArea(src)<100:
        raise ValueError('基準点は画像内の十分な面積を持つ四角形にしてください。')
    if any(not math.isfinite(v) or v<=0 for v in (width_mm,height_mm,pixels_per_mm)):
        raise ValueError('基準寸法は有限の正数です。')
    ow,oh=round(width_mm*pixels_per_mm),round(height_mm*pixels_per_mm)
    if min(ow,oh)<16 or max(ow,oh)>2000:raise ValueError('補正後の画像寸法は16〜2000pxにしてください。')
    dst=np.float32([[0,0],[ow-1,0],[ow-1,oh-1],[0,oh-1]])
    transform=cv2.getPerspectiveTransform(src,dst)
    return cv2.warpPerspective(rgb,transform,(ow,oh)),transform.tolist()


def estimate(rgb, mode, calibration=None, *, single_layer=False, perpendicular=False, all_tips_visible=False):
    f,mask=inspect_image(rgb)
    reasons=[]
    if calibration is None:reasons.append('製品の麺径・長さ・本数と総重量による校正がありません。')
    else:calibration.validate()
    if f['area_px']==0:reasons.append('乾麺の候補領域がありません。')
    if f['border']:reasons.append('麺候補が画像端で切れています。')
    if f['mask_sensitivity']>.15:reasons.append('照明・影・反射によって抽出領域が不安定です。')
    if not perpendicular:reasons.append('撮影面への垂直撮影または同一平面の基準点補正を確認してください。')
    if mode=='bundle_side':reasons.append('束の側面では隠れた層の量を画像だけで決められません。')
    elif mode=='spread':
        if not single_layer:reasons.append('単層配置を確認してください。隠れた麺は計測できません。')
        if f['width_px']<3:reasons.append('麺幅が不足しています。近づいてピントを合わせてください。')
        if f['width_ratio']>1.8:reasons.append('麺幅が不均一です。密集・背景混入・遠近差を確認してください。')
    elif mode=='bundle_end':
        if not all_tips_visible:reasons.append('全端面が正面に見える配置を確認してください。')
    else:reasons.append('撮影状態が未対応です。')
    count=None
    if not reasons and mode=='spread':
        expected_length=f['width_px']*calibration.length_mm/calibration.diameter_mm
        area_count=f['area_px']/(f['width_px']*expected_length)
        length_count=f['length_px']/expected_length
        endpoint_count=f['endpoints']/2
        f.update(area_count=area_count,length_count=length_count,endpoint_count=endpoint_count)
        estimates=[area_count,length_count,endpoint_count]
        if min(estimates)<=0 or max(estimates)/min(estimates)>1.35:
            reasons.append('面積・総延長・端点数の推定が一致しません。密集を解いて再撮影してください。')
        else:count=(area_count+length_count)/2
    elif not reasons and mode=='bundle_end':
        dist=cv2.distanceTransform(mask,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        local=(dist>=cv2.dilate(dist,np.ones((3,3),np.uint8))-1e-5)&(dist>=2)
        radii=dist[local]
        if not len(radii):reasons.append('端面の半径が小さすぎます。')
        else:
            radius=float(np.median(radii)); kernel=max(3,2*int(radius*.7)+1)
            peaks=(dist>=cv2.dilate(dist,np.ones((kernel,kernel),np.uint8))-1e-5)&(dist>=radius*.7)
            count=float(cv2.connectedComponents(peaks.astype(np.uint8),8)[0]-1)
            area_count=f['area_px']/(math.pi*radius**2)
            f.update(radius_px=radius,peak_count=count,area_count=area_count)
            if count<=0 or abs(area_count-count)/max(1,count)>.25:
                reasons.append('端面の面積と中心数が一致しません。側面や背景の混入を確認してください。')
    return dict(success=not reasons,predicted_g=count*calibration.g_per_strand if not reasons else None,
        reason=' / '.join(reasons) if reasons else '実験候補。実物精度未検証。',features=f,
        estimator=VERSION,calibration=asdict(calibration) if calibration else None,experimental=True)
