import csv
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest

from pasta_weight_app_v2.experiments import bench_dish_vision as bench
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
                                   false_reject_rate=1/3,
                                   changed_from_original=3, new_abstentions=1, recovered_from_abstention=1)


def test_compare_original_scope_rates_and_protect_the_denominators(tmp_path, monkeypatch):
    photos = {'supported1.png': b'correct', 'supported2.png': b'abstain',
              'outside1.png': b'wrong', 'outside2.png': b'abstain-control'}
    for name, pixels in photos.items():
        (tmp_path/name).write_bytes(pixels)
    baseline = tmp_path/'labels.csv'
    with baseline.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['image_path', 'class_label', 'training_image_sha256'])
        writer.writerows((name, 'tomato_pasta', sha256(photos[name]).hexdigest())
                         for name in ('supported1.png', 'supported2.png'))
    controls = tmp_path/'controls.csv'
    with controls.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['file', 'kind', 'sha256'])
        writer.writerows((name, 'other dish', sha256(photos[name]).hexdigest())
                         for name in ('outside1.png', 'outside2.png'))

    monkeypatch.setattr(bench, 'audit_receipt', lambda root: ({'photos': 2}, []))
    monkeypatch.setattr(bench, 'image_to_rgb', lambda path: path.read_bytes())
    model = SimpleNamespace(predict=lambda rgb: SimpleNamespace(
        name='トマトパスタ' if rgb in {b'correct', b'wrong'} else None))

    results = bench.evaluate_dataset(model, tmp_path, original_only=True)
    assert results['all_controls']['false_accept_rate'] is None
    assert results['all_controls']['rejection_rate'] is None
    assert results['frame_disagreements'] is None

    results = bench.evaluate_dataset(model, tmp_path, controls, tmp_path, original_only=True)
    assert list(results['by_condition']) == ['original']
    original = results['by_condition']['original']
    assert original['photos'] == 2 and original['false_reject_rate'] == .5
    assert original['by_class']['カルボナーラ']['false_reject_rate'] is None
    assert results['controls']['other dish'] == results['all_controls'] == dict(
        photos=2, abstained=1, false_accepts=1, false_accept_rate=.5, rejection_rate=.5,
        candidates={'トマトパスタ': 1, ABSTAIN: 1})

    (tmp_path/'outside1.png').write_bytes(photos['supported1.png'])
    text = controls.read_text(encoding='utf-8')
    controls.write_text(text.replace(sha256(photos['outside1.png']).hexdigest(),
                                     sha256(photos['supported1.png']).hexdigest()), encoding='utf-8')
    with pytest.raises(ValueError, match='duplicate'):
        bench.evaluate_dataset(model, tmp_path, controls, tmp_path, original_only=True)
