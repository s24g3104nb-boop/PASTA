from dataclasses import replace
from io import BytesIO
import json

import numpy as np
from PIL import Image
import pytest

from pasta_weight_app_v2.experiments.area_proportion import analyze
from pasta_weight_app_v2.experiments.practical_capture import (
    WidthReference, analyze_capture, decide_capture, visible_overlap_signals,
)
from pasta_weight_app_v2.experiments.strand_scale import AGREED, COLOR_EDGE, calibrate_widths, measure_for_area
from pasta_weight_app_v2.sheetless_area import THICK_CAPTURE_REASON


@pytest.fixture
def capture_geometry():
    rgb = np.full((720, 960, 3), 40, np.uint8)
    for y in range(110, 630, 23):
        rgb[y:y + 6, 100:840] = (224, 187, 82)
    evidence = analyze(rgb)
    scale = measure_for_area(rgb, evidence)
    ref = calibrate_widths([(evidence, scale)]*3)
    return rgb, evidence, scale, WidthReference(ref[COLOR_EDGE], ref[AGREED])


def test_reference_file_contains_only_validated_coefficients():
    profile = WidthReference(123.4, 167.)
    assert WidthReference.from_json(profile.to_json()) == profile
    invalid = json.loads(profile.to_json())
    invalid['weight_g'] = 100
    with pytest.raises(ValueError):
        WidthReference.from_json(json.dumps(invalid))
    with pytest.raises(ValueError):
        WidthReference.from_json(b'{}')
    for value in (True, 0, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            WidthReference(value, 1.)


def test_switching_after_range_stop_uses_the_image_and_a_fixed_profile(capture_geometry):
    rgb, evidence, scale, profile = capture_geometry
    stopped = replace(evidence, legacy_weight_g=None, legacy_reason='換算範囲外')
    # The measured area is four times this reference, yielding 80g.
    outside = WidthReference(profile.color_edge/4, profile.agreed/4)
    result = decide_capture(stopped, scale, outside)
    assert result.estimated_g is None and result.code == 'pilot_range'

    mismatch = WidthReference(profile.color_edge, profile.agreed/2)
    result = decide_capture(stopped, scale, mismatch)
    assert result.estimated_g is None and result.code == 'disagreement'

    # This reference yields an estimated 60g from the geometry. The shot and
    # profile contain no per-photo truth weight.
    inside = WidthReference(profile.color_edge/3, profile.agreed/3)
    result = decide_capture(stopped, scale, inside)
    assert result.code == 'candidate' and result.method == AGREED
    assert result.estimated_g == pytest.approx(60.)

    existing = replace(stopped, legacy_weight_g=44.3, legacy_reason='')
    preserved = decide_capture(existing, scale, inside)
    assert preserved.method == 'legacy' and preserved.estimated_g == 44.3

    file = BytesIO()
    Image.fromarray(rgb).save(file, format='PNG')
    actual = analyze_capture(file.getvalue(), inside)
    assert actual.estimated_g == (actual.current_g if actual.current_g is not None else pytest.approx(60.))


def test_density_and_clipping_veto_even_close_weight_diagnostics(capture_geometry):
    rgb, evidence, scale, profile = capture_geometry
    thick = replace(evidence, width_capture_reason=THICK_CAPTURE_REASON, legacy_weight_g=None)
    result = decide_capture(thick, scale, profile)
    assert result.color_diagnostic_g == pytest.approx(result.agreed_diagnostic_g)
    assert result.estimated_g is None and result.code == 'dense'
    assert '隠れた麺の量' in result.reason

    clipped = rgb.copy()
    clipped[110:116, :100] = rgb[111, 110]
    clipped_evidence = analyze(clipped)
    result = decide_capture(clipped_evidence, measure_for_area(clipped, clipped_evidence), profile)
    assert result.estimated_g is None and result.code == 'capture'
    assert '画像の端' in result.reason


def test_dense_features_measure_visible_directions_at_fixed_projected_area(capture_geometry):
    _, evidence, scale, _ = capture_geometry
    height, width = 144, 192
    y, x = np.ogrid[:height, :width]
    mask = (y >= 20) & (y < 124) & (x >= 20) & (x < 172)
    base = np.full((height, width, 3), (224, 185, 95), np.uint8)
    parallel, crossed = base.copy(), base.copy()
    parallel[:, :, 2] = np.rint(95 + 22*np.sin(y/2))
    crossed[:, :, 2] = np.rint(95 + 22*np.sin(y/2) + 22*np.sin(x/2))
    evidence = replace(evidence, mask=mask, area_px=int(mask.sum()))
    along = visible_overlap_signals(replace(evidence, scene=parallel), scale)
    mixed = visible_overlap_signals(replace(evidence, scene=crossed), scale)
    assert mixed.wide_fraction == along.wide_fraction
    assert mixed.mixed_direction_fraction > along.mixed_direction_fraction
