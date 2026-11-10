"""Reporting contracts: missing data and scale changes must not imply accuracy."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pasta_weight_app_v2.experiments.summarize_masks import association, scale_changes


def test_association_excludes_unsupported_and_missing_but_keeps_real_zero():
    rows = [
        dict(nominal_weight_g=20, area=0, status='DIAGNOSTIC_ONLY'),
        dict(nominal_weight_g=40, area=10, status='DIAGNOSTIC_ONLY'),
        dict(nominal_weight_g=60, area=0, status='UNSUPPORTED_SIDE_VIEW'),
        dict(nominal_weight_g=80, area=None, status='DIAGNOSTIC_ONLY'),
    ]
    result = association(rows, 'area')
    assert result['n_images'] == 2
    assert result['pearson_r'] == pytest.approx(1)
    assert result['independent_series'] == 1
    assert 'descriptive' in result['interpretation']


def test_constant_features_have_no_correlation_value():
    rows = [dict(nominal_weight_g=w, area=5) for w in (20, 40, 60)]
    assert association(rows, 'area')['pearson_r'] is None


def test_resize_uses_fraction_not_raw_area_and_preserves_absent_counts():
    common = dict(image_id='synthetic', method='hsv_baseline',
                  capture_mode='spread_overlap', nominal_weight_g=20,
                  mask_fraction=.25, circle_candidates=None)
    rows = [dict(common, scale=1, mask_area_px=400, components=2),
            dict(common, scale=.5, mask_area_px=100, components=1)]
    changes = scale_changes(rows)
    assert changes[1]['foreground_fraction_relative_change_pct'] == 0
    assert changes[1]['components_change'] == -1
    assert changes[1]['circle_candidates_change'] is None
    assert changes[1]['circle_candidates_unchanged'] is None


def test_unsupported_scale_rows_never_enter_comparisons():
    row = dict(image_id='synthetic', method='manual_roi', scale=1,
               status='UNSUPPORTED_SIDE_VIEW')
    assert scale_changes([row]) == []
