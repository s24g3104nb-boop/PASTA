"""Synthetic-only diagnostic contracts; no learning or accuracy claims."""
import csv
from pathlib import Path
import sys
import cv2
import numpy as np
import pytest
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from pasta_weight_app_v2.experiments.diagnose_series import runtime_parameters,analyze,run
from pasta_weight_app_v2.pasta_vision import detect_spread


def test_uses_ui_defaults_not_library_endface_defaults():
    p=runtime_parameters()
    assert p['DEFAULT_BUNDLE']==dict(min_radius=8,max_radius=27,min_distance=32,hough_sensitivity=18)
    rgb=np.zeros((100,160,3),np.uint8)
    cv2.line(rgb,(20,30),(140,30),(240,190,60),5)
    old=rgb.copy();d,images=analyze(rgb,'spread_overlap',p)
    assert np.array_equal(rgb,old) and d['detected_count']==detect_spread(rgb,**p['DEFAULT_SPREAD']).count
    assert np.array_equal(images['mask'],detect_spread(rgb,**p['DEFAULT_SPREAD']).mask)
    assert d['measured_weight_prediction_g'] is None and not d['training_eligible']
    assert d['mask_qc']=='NOT_VERIFIED'


def test_unknown_modes_are_rejected():
    with pytest.raises(ValueError):analyze(np.zeros((80,80,3),np.uint8),'unknown',runtime_parameters())


def test_series_report_has_no_supervised_or_independent_evaluation(tmp_path):
    import hashlib
    originals=tmp_path/'originals';originals.mkdir();rows=[]
    for weight in (20,40,60,80,100):
        for mode in ('spread_overlap','bundle_ends_low_angle','handheld_bundle_ends'):
            name=f'{weight}_{mode}.jpeg';path=originals/name
            Image.new('RGB',(96,72),'black').save(path)
            rows.append(dict(image_path=name,original_filename=name,nominal_weight_g=str(weight),capture_mode=mode,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),physical_family_group_id='ONE'))
    with (originals/'labels.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report=run(originals,tmp_path/'output')
    assert len(report['rows'])==15 and report['physical_families']==1
    assert report['training_images']==report['independent_evaluation_images']==report['qc_approved_images']==0
    assert report['accuracy_metrics'] is None and not report['weight_predictions_available']
    assert all(r['received_sha256']==m['sha256'] for r,m in zip(report['rows'],rows))
    assert (tmp_path/'output/index.html').is_file()
    with pytest.raises(FileExistsError):run(originals,tmp_path/'output')
