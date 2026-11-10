"""Synthetic contracts for preprocessing reuse, image safety and lineage retention."""
from dataclasses import asdict, replace
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pasta_weight_app_v2.experiments.stability_pipeline import (
    configurations, normalize_size, clean_mask, shape_features, process,
    research_inference, VERSION)
from pasta_weight_app_v2.experiments.stability_experiment import transformations
from pasta_weight_app_v2.experiments.direct_features import extract
from pasta_weight_app_v2.experiments.compare_masks import masks
from pasta_weight_app_v2.experiments.diagnose_series import runtime_parameters


def scene():
    rgb = np.zeros((240, 320, 3), np.uint8)
    rgb[70:155, 35:285] = (230, 180, 50)
    rgb[90:95, 35:285] = (60, 60, 60)
    return rgb


@pytest.mark.parametrize('index,method', [(0, 'hsv_baseline'), (1, 'hsv_lab')])
def test_baselines_preserve_previous_masks_and_feature_values(index, method):
    rgb = scene(); config = configurations()[index]
    old_mask = masks(rgb, 'spread_overlap', runtime_parameters())[method]
    old, _ = extract(rgb, old_mask)
    new, _, new_mask = process(rgb, config)
    np.testing.assert_array_equal(old_mask, new_mask)
    np.testing.assert_allclose([old[k] for k in config.keys], [new[k] for k in config.keys])


def test_canonical_size_preserves_aspect_and_source():
    rgb = scene(); before = rgb.copy()
    assert normalize_size(rgb, 512).shape == (384, 512, 3)
    np.testing.assert_array_equal(rgb, before)
    with pytest.raises(ValueError): normalize_size(rgb.astype(float), 512)
    with pytest.raises(ValueError): normalize_size(rgb, 0)


def test_cleaning_removes_tiny_remote_speck_but_keeps_detached_long_strand():
    mask = np.zeros((300, 400), np.uint8); mask[100:160, 50:340] = 255
    mask[175:177, 50:340] = 255; mask[5, 5] = 255
    before = mask.copy(); config = configurations()[4]
    cleaned = clean_mask(mask, config)
    assert cleaned[5, 5] == 0
    assert np.all(cleaned[175:177, 50:340] == 255)
    np.testing.assert_array_equal(mask, before)
    assert np.count_nonzero(cleaned) == np.count_nonzero(mask)-1


def test_object_normalization_can_erase_frame_scale_cues():
    config = configurations()[5]
    full = np.zeros((400, 400), np.uint8); full[100:220, 50:350] = 255
    small = cv2.resize(full, None, fx=.5, fy=.5, interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros_like(full); canvas[100:300, 100:300] = small
    a, b = shape_features(full, config), shape_features(canvas, config)
    assert b['mask_fraction'] == pytest.approx(a['mask_fraction']/4)
    assert b['area_over_length_squared'] == pytest.approx(a['area_over_length_squared'], rel=.03)
    assert b['quantile_length_relative'] == pytest.approx(a['quantile_length_relative']/2, rel=.03)


def test_transformations_are_deterministic_and_do_not_mutate_original():
    rgb = scene(); before = rgb.copy()
    first, second = dict(transformations(rgb)), dict(transformations(rgb))
    assert len(first) == 15
    assert first['resize25'].shape == (60, 80, 3)
    assert first['roundtrip25'].shape == rgb.shape
    assert first['rotate3'].shape[0] > rgb.shape[0]
    for key in first: np.testing.assert_array_equal(first[key], second[key])
    np.testing.assert_array_equal(rgb, before)


def test_inference_shares_feature_extraction_and_never_releases_public_weight():
    config = configurations()[4]; rgb = scene()
    expected, _, _ = process(rgb, config)
    class Model:
        def predict(self, matrix):
            np.testing.assert_allclose(matrix[0], [expected[k] for k in config.keys])
            return [52.]
    artifact = dict(model=Model(), pipeline_version=VERSION, pipeline_config=asdict(config),
                    config_sha256=config.digest, feature_keys=config.keys)
    result = research_inference(rgb, artifact)
    assert result['research_prediction_g'] == 52 and result['predicted_weight_g'] is None
    artifact['pipeline_config']['target_long_side'] = 256
    with pytest.raises(ValueError, match='mismatch'): research_inference(rgb, artifact)


def test_missing_foreground_yields_review_flag_and_finite_features():
    config = configurations()[4]
    f, _, mask = process(np.zeros((100, 100, 3), np.uint8), config)
    assert 'insufficient_foreground' in f['qc_flags']
    assert not mask.any() and np.isfinite([f[k] for k in config.keys]).all()
