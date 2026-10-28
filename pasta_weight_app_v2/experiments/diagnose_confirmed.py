"""Read-only diagnosis of confirmed intake photos; no fit, promotion or independent accuracy claim."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import cv2
import numpy as np
from ..geometry_candidate import estimate,inspect_image,structural_mask,geometry
from ..pasta_vision import image_to_rgb, detect_spread
from ..pasta_ml.model import features
from ..learning_readiness import readiness
from ..storage import DEFAULT_DB


def run(photo_root, output, db_path=DEFAULT_DB):
    root=Path(photo_root).resolve();output=Path(output)
    if output.exists():raise ValueError('診断結果は上書きしません。')
    with sqlite3.connect(f'file:{Path(db_path).resolve()}?mode=ro',uri=True) as con:
        con.row_factory=sqlite3.Row
        records=[dict(r) for r in con.execute('SELECT * FROM photo_intake WHERE weight_verified=1 ORDER BY id')]
        excluded=con.execute('SELECT COUNT(*) FROM photo_intake WHERE weight_verified=0').fetchone()[0]
        products=[dict(r) for r in con.execute('SELECT id,name,g_per_strand FROM products')]
        model_count=con.execute('SELECT COUNT(*) FROM training_models').fetchone()[0] if con.execute("SELECT 1 FROM sqlite_master WHERE name='training_models'").fetchone() else None
    rows=[];transforms=[]
    selected=Counter()
    for row in records:
        metadata=json.loads(row['metadata_json']);name=metadata.get('canonical_filename',row['filename'])
        path=(root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file():raise ValueError('診断用画像が見つかりません。')
        rgb=image_to_rgb(path)
        raw=detect_spread(rgb)
        geo=estimate(rgb,'spread')  # No guessed calibration or view confirmations.
        raw_features,raw_mask=inspect_image(rgb)
        structure_features=geometry(structural_mask(raw_mask))
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(dict(image_id=metadata.get('canonical_id',name),weight_g=row['weight_g'],
            label_status='confirmed',received_sha256=digest,reference_sha256=row['expected_sha256'],
            matches_reference=digest==row['expected_sha256'],group_verified=bool(row['group_verified']),
            mode_status='UNVERIFIED_spread_diagnostic_only',legacy_connected_count=raw.count,
            candidate=geo,structural_foreground_diagnostic=structure_features))
        if selected[row['weight_g']]<2:
            selected[row['weight_g']]+=1
            original=features(rgb);fg=geo['features']
            for transform,image in [('rotate90',np.rot90(rgb).copy()),('resize75',cv2.resize(rgb,None,fx=.75,fy=.75,interpolation=cv2.INTER_AREA))]:
                newg,_=inspect_image(image)
                transforms.append(dict(image_id=metadata.get('canonical_id',name),transform=transform,
                    rgb_feature_l2=float(np.linalg.norm(features(image)-original)),
                    area_over_width_squared_before=fg['area_px']/max(1.,fg['width_px'])**2,
                    area_over_width_squared_after=newg['area_px']/max(1.,newg['width_px'])**2,
                    endpoints_before=fg['endpoints'],endpoints_after=newg['endpoints']))
    report=dict(purpose='diagnosis_only_not_independent_evaluation',readiness=readiness(db_path),
        confirmed_images=len(rows),unconfirmed_images_excluded=excluded,products=products,local_model_count=model_count,
        rejection_reasons=dict(Counter(reason for r in rows for reason in r['candidate']['reason'].split(' / '))),
        accuracy=dict(mae_g=None,rmse_g=None,max_absolute_error_g=None,failure_rate=None,
            reason='製品校正・撮影状態・QC・独立物理試料が未確認。校正値を仮定した重量誤差は報告しない。'),
        unsafe_legacy_36g_reproductions=[dict(image_id=r['image_id'],confirmed_weight_g=r['weight_g'],connected_regions=r['legacy_connected_count'],hypothetical_demo_g_per_strand=1.,hypothetical_predicted_g=4.,absolute_error_g=36.,notice='Unsafe old demo-calibration configuration only; not proof of the reported phone incident and not a valid accuracy evaluation.') for r in rows if r['weight_g']==40 and r['legacy_connected_count']==4],
        rows=rows,transform_diagnostics=transforms,
        notice='Confirmed weights are retained only as metadata. No training labels were used to fit or tune. Receipt/reference hash differences remain explicit.')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','transform_diagnostics','products')},ensure_ascii=False,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--photos',required=True);p.add_argument('--output',required=True);p.add_argument('--db',default=str(DEFAULT_DB))
    args=p.parse_args();run(args.photos,args.output,args.db)
