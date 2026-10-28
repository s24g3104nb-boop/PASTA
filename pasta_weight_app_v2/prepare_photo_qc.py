"""私有写真の検品候補を再作成する。重量の登録・学習・自動承認は行わない。"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .photo_intake import list_intake, load_original, load_preview
from .photo_qc import propose_mask, make_overlay
from .preprocessing import transform_roi, propose_conservative_mask, audit_mask, background_controls,propose_lab_mask,quality_flags
from .storage import DEFAULT_DB


def prepare(root, *, db_path=DEFAULT_DB, inferred_max_side=None):
    root = Path(root).resolve()
    output = root/'dataset/preview_qc_v2'
    for name in ('masks','foreground','overlay','lost_candidates','controls','lab_masks'):
        (output/name).mkdir(parents=True,exist_ok=True)
    references_path = root/'dataset/metadata/roi_reference_sizes.json'
    references = json.loads(references_path.read_text()) if references_path.exists() else {}
    manifest = {}
    for row in list_intake(db_path):
        if Path(row['package_root']).resolve() != root:
            continue
        name = row['filename']; stem = Path(name).stem
        try:
            started=time.perf_counter()
            try:
                rgb = load_original(row); variant='original'
            except (ValueError,OSError):
                rgb = load_preview(row); variant='preview'
            h,w = rgb.shape[:2]
            metadata=json.loads(row['metadata_json'])
            raw_roi=metadata.get('roi_xyxy','')
            old_roi = [int(v) for v in raw_roi.split(',')] if raw_roi else [0,0,w,h]
            reference = references.get(name)
            if reference is None:
                if 0 <= old_roi[0] < old_roi[2] <= w and 0 <= old_roi[1] < old_roi[3] <= h:
                    reference = dict(reference_size=[w,h],roi_space='oriented',orientation=1,
                                     basis='provided ROI already in preview bounds',confirmed=False)
                elif inferred_max_side is not None:
                    scale = inferred_max_side/max(w,h)
                    reference = dict(reference_size=[round(w*scale),round(h*scale)],roi_space='oriented',
                                     orientation=1,basis='inferred size; original dimensions unavailable',confirmed=False)
                else:
                    raise ValueError('ROI基準寸法が不明です。基準寸法または明示した推定最大辺が必要です。')
                references[name] = reference
            transform = transform_roi(old_roi,reference['reference_size'],(w,h),
                                      orientation=reference.get('orientation',1),roi_space=reference.get('roi_space','oriented'))
            roi = transform['roi_xyxy']
            result = propose_conservative_mask(rgb,roi)
            baseline = propose_mask(rgb,roi=roi)['mask']
            mask = result['mask']
            controls = background_controls(rgb,mask)
            audit = audit_mask(rgb,mask,reference_mask=baseline)
            lab=propose_lab_mask(rgb,roi)['mask']
            removed = rgb.copy(); removed[(baseline>0)&(mask==0)] = [255,0,90]
            paths = {}
            for folder,array in (('masks',mask),('foreground',controls['foreground_dark']),
                                 ('overlay',make_overlay(rgb,mask)),('lost_candidates',removed),('lab_masks',lab)):
                path = output/folder/(stem+'.png')
                Image.fromarray(array).save(path,'PNG')
                paths[folder] = str(path.relative_to(root))
            for kind in ('foreground_gray','background_only'):
                Image.fromarray(controls[kind]).save(output/'controls'/(stem+'_'+kind+'.png'),'PNG')
            manifest[name] = dict(status='UNREVIEWED',source_variant=variant,
                capture_group_id=row['capture_group_id'],batch_id=row['batch_id'],session_id=row['session_id'],
                source_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),shape=[h,w],
                original_roi=old_roi,transform=transform,reference=reference,paths=paths,
                mask_sha256=hashlib.sha256((root/paths['masks']).read_bytes()).hexdigest(),
                audit=audit,stats=result['stats'],warnings=result['warnings'],quality=quality_flags(rgb,mask),
                method_comparison=dict(hsv_pixels=int(np.count_nonzero(baseline)),
                    conservative_pixels=int(np.count_nonzero(mask)),lab_pixels=int(np.count_nonzero(lab)),
                    lab_disagreement_pixels=int(np.count_nonzero(lab!=mask)),
                    truth_available=False),processing_seconds=time.perf_counter()-started,
                leakage_reviewed=False,weight_verified=bool(row['weight_verified']))
        except (OSError,ValueError,KeyError) as exc:
            manifest[name] = dict(status='FAILED_UNAPPROVED',error=str(exc))
    references_path.write_text(json.dumps(references,ensure_ascii=False,indent=2))
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    return manifest


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('--db',default=str(DEFAULT_DB))
    parser.add_argument('--infer-reference-max-side',type=int,
                        help='元寸法が欠ける場合の推定値。推定座標は承認されず記録されます。')
    args=parser.parse_args()
    if args.infer_reference_max_side is not None and not 1 <= args.infer_reference_max_side <= 100000:
        parser.error('推定最大辺は1〜100000にしてください。')
    result=prepare(args.root,db_path=args.db,inferred_max_side=args.infer_reference_max_side)
    print(json.dumps(dict(total=len(result),candidates=sum(r['status']=='UNREVIEWED' for r in result.values()),
                          failed=sum(r['status']!='UNREVIEWED' for r in result.values()),approved=0),ensure_ascii=False))
