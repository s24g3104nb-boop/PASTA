import cv2
import numpy as np
import pytest

from pasta_weight_app_v2.experiments.weight_robustness import feature_vectors, compare_models


def fixture():
    rgb = np.full((720, 960, 3), 38, np.uint8)
    rgb[270:390, 180:780] = (224, 180, 45)
    return rgb


def test_clean_object_ratios_ignore_a_change_in_frame_fill_and_background():
    rgb = fixture()
    original = feature_vectors(rgb)
    crop = cv2.resize(rgb, None, fx=.75, fy=.75, interpolation=cv2.INTER_AREA)
    padded = np.full_like(rgb, 75)
    padded[90:630, 120:840] = crop
    changed = feature_vectors(padded)
    assert changed['scene']['area_fraction'] < .65*original['scene']['area_fraction']
    for key in ('area_over_length_squared', 'width_over_length', 'quantile_density'):
        assert changed['object_geometry'][key] == pytest.approx(original['object_geometry'][key], rel=.03)
    assert changed['background_control']['background_light_mean'] > original['background_control']['background_light_mean']


def test_exploratory_metrics_leave_validation_and_test_empty_and_exclude_count_only():
    features = feature_vectors(fixture())
    samples = [dict(sha256=str(i), vectors=features, reported_g=grams, transform=variant)
               for variant in ('original', 'resize50') for i, grams in enumerate((120, 140, 160, None))]
    fits, predictions = compare_models(samples)
    assert len(fits) == 20
    assert {p['original_sha256'] for p in predictions} == {'0', '1', '2'}
    for model in fits:
        assert model['fitted_originals'] == 3 and model['validation'] is None and model['independent_test'] is None
        assert model['paired_stress']['resize50']['pairs'] == 3
        assert model['paired_stress']['resize50']['max_absolute_delta_g'] == 0
