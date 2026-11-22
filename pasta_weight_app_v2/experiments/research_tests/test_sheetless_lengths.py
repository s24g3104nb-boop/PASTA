import cv2
import numpy as np
import pytest

from pasta_weight_app_v2.experiments.compare_sheetless_lengths import fit, report_weight
from pasta_weight_app_v2.experiments.sheetless_lengths import measure


def strands(*, staggered=False, touching=False):
    photo = np.full((720, 960, 3), 40, dtype=np.uint8)
    for i in range(10):
        y = 240 + i * (4 if touching else 20)
        x = 140 + (120 if staggered and i % 2 else 0)
        photo[y:y + 5, x:x + 530] = (218, 183, 48)
    return photo


def test_staggering_changes_envelope_instead_of_visible_length():
    aligned = measure(strands())
    staggered = measure(strands(staggered=True))
    assert abs(staggered.area_ratio / aligned.area_ratio - 1) > .15
    for photo in (aligned, staggered):
        assert not photo.capture_reason and not photo.scale_reason
        _, skeleton_equiv, area_equiv, _ = photo.vector()
        assert skeleton_equiv == pytest.approx(10, rel=.02)
        assert area_equiv == pytest.approx(10, rel=.02)
        assert photo.width_px == pytest.approx(5)


def test_length_and_width_survive_right_angle_rotation():
    horizontal = strands(staggered=True)
    vertical = cv2.rotate(horizontal, cv2.ROTATE_90_CLOCKWISE)
    assert measure(horizontal).vector() == pytest.approx(measure(vertical).vector())


@pytest.mark.parametrize('method', ['B', 'C', 'D'])
def test_calibrated_method_returns_grams_within_its_support(method):
    obs = measure(strands())
    vec = np.array(obs.vector())
    model = fit(method, [vec*.8, vec, vec*1.2], [24., 30., 36.])
    result, reason = report_weight(model, method, obs)
    assert result == pytest.approx(30)
    assert not reason
    result, reason = report_weight(model, method, measure(strands(touching=True)))
    assert result is None
    assert '厚い束' in reason


def test_empty_foreground_has_no_relative_scale():
    result = measure(np.full((720, 960, 3), 40, dtype=np.uint8))
    assert result.vector() is None
    assert result.capture_reason
