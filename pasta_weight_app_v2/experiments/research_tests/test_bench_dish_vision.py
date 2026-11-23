import numpy as np

from pasta_weight_app_v2.experiments.bench_dish_vision import ABSTAIN, CONDITIONS, summarize, transform


def test_transform_uses_fixed_parameters_and_preserves_input():
    original = np.full((180, 360, 3), 100, dtype=np.uint8)
    for condition in CONDITIONS:
        transformed = transform(original, condition)
        assert transformed.dtype == np.uint8 and transformed.ndim == 3 and transformed.shape[2] == 3
    assert (original == 100).all()
    np.testing.assert_array_equal(transform(original, 'warm_cast')[0, 0], [112, 100, 86])
    np.testing.assert_array_equal(transform(original, 'brightness_60')[0, 0], [60, 60, 60])
    assert transform(original, 'long_edge_112').shape == (56, 112, 3)
    light, dark = (transform(original, f'{tone}_frame') for tone in ('light', 'dark'))
    assert light.shape == dark.shape
    np.testing.assert_array_equal(light[22:-22, 43:-43], original)
    assert light[0, 0, 0] == 235 and dark[0, 0, 0] == 32


def test_summary_keeps_wrong_candidates_and_abstentions_separate():
    tomato = 'トマトパスタ'
    rows = [
        dict(truth=tomato, prediction=tomato, original=ABSTAIN),
        dict(truth=tomato, prediction=ABSTAIN, original=tomato),
        dict(truth=tomato, prediction='カルボナーラ', original=tomato),
    ]
    assert summarize(rows) == dict(photos=3, correct=1, abstained=1, wrong_candidate=1,
                                   changed_from_original=3, new_abstentions=1, recovered_from_abstention=1)
