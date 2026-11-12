"""Synthetic sheet measurements only, never real weight validation."""
import sys
from pathlib import Path
import cv2
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from pasta_weight_app_v2.experiments.metric_foreground import extract_metric_foreground
from pasta_weight_app_v2.experiments.capture_sheet import render_sheet


def fixture():
    image=render_sheet(4)
    image[180:580,100:260]=(220,180,40)
    return image


def test_metric_projection_is_scale_normalized_only_with_declared_reference():
    rgb=fixture();unchanged=rgb.copy()
    a=extract_metric_foreground(rgb,print_scale_verified=True,coplanar_confirmed=True)
    b=extract_metric_foreground(cv2.resize(rgb,None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA),print_scale_verified=True,coplanar_confirmed=True)
    assert a['status']==b['status']=='research_features_only'
    assert a['features']['visible_area_mm2']==pytest.approx(4000,rel=.03)
    assert b['features']['visible_area_mm2']==pytest.approx(a['features']['visible_area_mm2'],rel=.03)
    assert a['public_prediction_g'] is None and not a['production_eligible']
    np.testing.assert_array_equal(rgb,unchanged)


def test_unverified_print_or_missing_marker_never_returns_metric_features():
    rgb=fixture()
    a=extract_metric_foreground(rgb)
    assert a['features'] is None and a['status']=='reference_unavailable'
    rgb[40:120,40:120]=255
    b=extract_metric_foreground(rgb,print_scale_verified=True,coplanar_confirmed=True)
    assert b['features'] is None


def test_source_clipping_survives_rectification():
    rgb=fixture();rgb[300:500,:30]=(220,180,40)
    result=extract_metric_foreground(rgb,print_scale_verified=True,coplanar_confirmed=True)
    assert result['features'] is None
    assert 'foreground_touches_frame' in result['original_qc_flags']
