"""Synthetic end-face scale experiment; no evidence of real-photo accuracy."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from ..geometry_candidate import estimate
from ..pasta_vision import detect_bundle_ends
from ..pasta_ml.evaluate import summary
from .geometry_benchmark import CALIBRATION


def run(output):
    path=Path(output)
    if path.exists():raise ValueError('結果は上書きしません。')
    rows=[]
    for count in (12,20,28):
        for radius in (4,8,16):
            rgb=np.full((560,320,3),35,np.uint8)
            for i in range(count):cv2.circle(rgb,(50+i%4*60,50+i//4*70),radius,(215,175,65),-1)
            old=detect_bundle_ends(rgb,min_radius=8,max_radius=27,min_distance=32,hough_sensitivity=18)
            new=estimate(rgb,'bundle_end',CALIBRATION,perpendicular=True,all_tips_visible=True)
            rows.append(dict(actual_weight_g=count,radius_px=radius,
                legacy=dict(success=old.count>0,predicted_g=float(old.count) if old.count else None),candidate=new))
    report=dict(source='synthetic',production_eligible=False,rows=rows,
        summaries={method:summary([dict(actual_weight_g=r['actual_weight_g'],**r[method]) for r in rows]) for method in ('legacy','candidate')})
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False))
    print(json.dumps(report['summaries'],indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)
