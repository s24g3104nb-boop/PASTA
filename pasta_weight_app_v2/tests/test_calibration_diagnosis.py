"""Synthetic contracts for diagnostic known-count mass conversion, not accuracy."""
import sys
from pathlib import Path
import cv2
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from pasta_weight_app_v2.experiments.diagnose_calibration import comparison_errors,diagnose,run
from pasta_weight_app_v2.experiments.diagnose_series import runtime_parameters


def test_mass_conversion_is_same_sample_arithmetic_and_signed_error():
    exact=comparison_errors(50,50,34)
    assert exact['diagnostic_weight_g']==34 and exact['absolute_weight_error_g']==0
    under=comparison_errors(8,50,34)
    assert under['g_per_strand']==.68
    assert under['weight_error_g']==pytest.approx(-28.56)
    assert comparison_errors(60,50,34)['weight_error_g']==pytest.approx(6.8)


@pytest.mark.parametrize('args',[(True,50,34),(-1,50,34),(2,0,34),(2,50,float('nan')),(2,50,0)])
def test_invalid_reference_or_counts_rejected(args):
    with pytest.raises(ValueError):comparison_errors(*args)


def test_baseline_equivalence_and_input_is_immutable():
    rgb=np.zeros((160,240,3),np.uint8)
    for x in (40,100,160):cv2.circle(rgb,(x,80),10,(240,190,60),-1)
    before=rgb.copy()
    for mode in ('overhead_bundle','low_angle_ends'):
        f,mask,overlay=diagnose(rgb,mode,'hsv_baseline',runtime_parameters())
        assert not f['training_approved'] and f['mask_qc']=='UNAPPROVED'
    assert np.array_equal(rgb,before)


def test_output_overwrite_rejected_before_input_reads(tmp_path):
    with pytest.raises(FileExistsError):run(tmp_path/'missing',tmp_path)
