import cv2
import numpy as np
import pytest

from pasta_weight_app_v2.experiments.spread_count_candidate import compare_spread_counts
from pasta_weight_app_v2.measurement_diagnostics import decide_count_weight


def strands():
    rgb = np.full((480,640,3), 32, np.uint8)
    for i in range(6):
        rgb[85+i*47:90+i*47, 96:540] = (220, 180, 60)
    return rgb


def test_detached_warm_border_is_removed_and_calibration_remains_required():
    rgb = strands()
    rgb[-36:] = (230, 216, 204)
    before = rgb.copy()
    hsv, candidate = compare_spread_counts(rgb)
    assert hsv.evidence['cropped_regions'] == 1
    assert candidate.count == 6 and candidate.evidence['excluded_source_border_regions'] == 1
    assert candidate.evidence['cropped_regions'] == 0
    decision = decide_count_weight(dict(name='test', g_per_strand=.7), candidate, mode='spread')
    assert decision.weight_g == pytest.approx(4.2)
    assert decide_count_weight(dict(name='test'), candidate, mode='spread').weight_g is None
    np.testing.assert_array_equal(rgb, before)


def test_pale_strand_connected_to_source_edge_still_stops_weight():
    rgb = strands()
    rgb[85:90, :96] = (230, 216, 204)
    _, lab = compare_spread_counts(rgb)
    assert lab.count == 6 and lab.evidence['linked_source_border_regions'] == 1
    assert not lab.mask[:, 0].any()
    decision = decide_count_weight(dict(name='test', g_per_strand=.7), lab, mode='spread')
    assert decision.weight_g is None and 'cropped' in decision.reason_codes


@pytest.mark.parametrize('transform', ['half', 'brighter', 'rotate90'])
def test_separate_synthetic_layout_with_neutral_reflection(transform):
    rgb = strands()
    # Specular white over the tray, clear of the individual pasta strands.
    cv2.ellipse(rgb, (310, 68), (53, 11), 0, 0, 360, (245, 245, 245), -1)
    if transform == 'half':
        rgb = cv2.resize(rgb, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    elif transform == 'brighter':
        rgb = np.clip(rgb.astype(float)*1.1, 0, 255).astype(np.uint8)
    else:
        rgb = np.rot90(rgb).copy()
    _, candidate = compare_spread_counts(rgb)
    assert candidate.count == 6
    assert decide_count_weight(dict(name='test', g_per_strand=.7), candidate, mode='spread').weight_g is not None
