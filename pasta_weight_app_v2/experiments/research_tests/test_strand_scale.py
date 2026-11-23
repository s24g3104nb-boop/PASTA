from dataclasses import replace

import numpy as np
import pytest

from pasta_weight_app_v2.experiments.area_proportion import PRIMARY, analyze, calibrate
from pasta_weight_app_v2.experiments.compare_area_proportion import scaled_photo
from pasta_weight_app_v2.experiments.strand_scale import (
    AGREED, COLOR_EDGE, NATIVE_MASK, calibrate_widths, canonical_width,
    choose_estimator, measure_for_area, measure_strand_scale, thickness_diagnostic, width_outcomes,
)
from pasta_weight_app_v2.sheetless_area import THICK_CAPTURE_REASON


def _slanted_strands():
    """Pixel coverage from 8px ribbons with inclined, subpixel boundaries."""
    y, x = np.ogrid[:960, :1280]
    position = (y - .125*x) / np.hypot(1, .125)
    distance = np.abs((position - 191.3 + 17/2) % 17 - 17/2)
    covered = np.clip(4.5 - distance, 0, 1)
    foreground = (x >= 150) & (x < 1100) & (position > 180) & (position < 480)
    weight = (covered * foreground)[..., None]
    return np.rint(42 + weight * (np.array([221, 185, 74]) - 42)).astype(np.uint8)


def test_native_color_edges_keep_the_same_width_on_known_slanted_strands():
    rgb = _slanted_strands()
    first = measure_strand_scale(rgb)
    assert first.long_side_px == 1280 and first.check_long_side_px == 960
    assert first.widths[NATIVE_MASK].width_px is not None
    assert first.widths[COLOR_EDGE].width_px == pytest.approx(8, abs=.6)
    assert first.widths[AGREED].tiles >= 3

    ev = analyze(rgb)
    ref = calibrate_widths([(ev, first)]*3)
    for factor in (75, 60):
        resized = scaled_photo(rgb, f'smaller_in_frame_{factor}')
        after = analyze(resized)
        scale = measure_for_area(resized, after)
        assert canonical_width(scale, AGREED) == pytest.approx(canonical_width(first, AGREED) * factor/100,
                                                               rel=.08)
        assert width_outcomes(after, scale, ref)[AGREED]['diagnostic_g'] == pytest.approx(20., rel=.13)


def test_choice_falls_back_to_verified_track_when_width_is_unavailable():
    rgb = _slanted_strands()
    evidence = analyze(rgb)
    assert not evidence.width_capture_reason and evidence.track_px > 0
    scale = measure_for_area(rgb, evidence)
    result = width_outcomes(evidence, scale, calibrate_widths([(evidence, scale)]*3))
    assert choose_estimator(evidence, scale, result, calibrate([evidence]*3))['method'] == AGREED
    missing_width = replace(scale, widths=measure_strand_scale(np.full_like(rgb, 32)).widths,
                            track_960_px=evidence.track_px)
    fallback = choose_estimator(evidence, missing_width, result, calibrate([evidence]*3))
    assert fallback['method'] == PRIMARY
    assert fallback['estimated_g'] == pytest.approx(20.)


def test_top_projection_cannot_separate_lateral_contact_from_a_hidden_layer():
    def render(layers):
        rgb = np.full((640, 960, 3), 40, np.uint8)
        for y, x in layers:
            rgb[y, x] = [220, 185, 83]
        return rgb

    contacting = [(slice(y, y + 7), slice(100, 850)) for y in range(200, 326, 7)]
    singles = [(slice(y, y + 6), slice(100, 850)) for y in (80, 105, 130)]
    flat = render(singles + contacting)
    hidden = render(singles + contacting + contacting)
    np.testing.assert_array_equal(flat, hidden)

    evidence = analyze(flat)
    scale = measure_for_area(flat, evidence)
    assert evidence.width_capture_reason == THICK_CAPTURE_REASON
    # Use an isolated synthetic reference only to verify veto handling.
    reference_rgb = _slanted_strands()
    ref_evidence = analyze(reference_rgb)
    ref_scale = measure_for_area(reference_rgb, ref_evidence)
    outcome = width_outcomes(evidence, scale, calibrate_widths([(ref_evidence, ref_scale)]*3))
    assert thickness_diagnostic(evidence, outcome) is not None
    assert outcome[AGREED]['estimated_g'] is None
    assert choose_estimator(evidence, scale, outcome, {})['estimated_g'] is None
