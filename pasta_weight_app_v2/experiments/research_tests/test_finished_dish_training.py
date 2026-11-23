import csv
from hashlib import sha256
import json

import numpy as np
import pytest
from PIL import Image

from pasta_weight_app_v2.dish_classifier import EMBEDDING_SIZE, LinearDishHead, STANDARD_DISH_NAMES
from pasta_weight_app_v2.experiments import train_finished_dish as train
from pasta_weight_app_v2.local_dish_vision import Suggestion


def dataset(tmp_path):
    manifest = tmp_path / 'labels.csv'
    rows = []
    for split in train.SPLITS:
        for label in STANDARD_DISH_NAMES:
            n = len(rows)
            name = f'{n}.png'
            path = tmp_path / name
            Image.new('RGB', (24, 24), (n, 80, 120)).save(path)
            rows.append(dict(file=name, label=label, session_id=f'session-{n}', split=split,
                             sha256=sha256(path.read_bytes()).hexdigest()))

    def write():
        with manifest.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write()
    return manifest, rows, write


def test_manifest_requires_disjoint_sessions_and_original_hashes(tmp_path):
    manifest, rows, write = dataset(tmp_path)
    assert [len(examples) for examples in train.read_splits(manifest, tmp_path).values()] == [3, 3, 3]
    rows[3]['session_id'] = rows[0]['session_id']
    write()
    with pytest.raises(ValueError, match='同一の調理回'):
        train.read_splits(manifest, tmp_path)
    rows[3]['session_id'] = 'fixed'
    rows[0]['sha256'] = '0' * 64
    write()
    with pytest.raises(ValueError, match='一致しません'):
        train.read_splits(manifest, tmp_path)


def test_metrics_count_wrong_predictions_and_abstentions(tmp_path):
    manifest, _, _ = dataset(tmp_path)
    rows = train.read_splits(manifest, tmp_path)['test']
    rows.append(train.Example(tmp_path / 'out.png', train.OUT_OF_SCOPE, 'another-session', 'test'))
    carbonara, tomato, genovese = STANDARD_DISH_NAMES
    report = train.evaluate(rows, [carbonara, carbonara, None, None])
    assert report['coverage'] == .5 and report['overall_accuracy'] == .5
    assert report['out_of_scope_rejection'] == 1
    assert report['per_class'][carbonara]['precision'] == .5
    assert report['per_class'][carbonara]['recall'] == 1
    assert report['per_class'][carbonara]['f1'] == pytest.approx(2 / 3)
    assert report['per_class'][tomato]['recall'] == 0
    assert report['per_class'][genovese]['recall'] == 0
    assert report['confusion_matrix'] == [[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1]]


def test_emit_validated_head_with_no_private_metadata(tmp_path, monkeypatch):
    manifest, _, _ = dataset(tmp_path)
    head = LinearDishHead(np.eye(3, EMBEDDING_SIZE) * 8, np.zeros(3))

    class PixelEncoder:
        def image_evidence(self, rgb):
            expected = int(rgb[0, 0, 0]) % 3
            return np.eye(3, EMBEDDING_SIZE)[expected], Suggestion(STANDARD_DISH_NAMES[expected], (), None, 'test')

    monkeypatch.setattr(train, 'LocalDishClassifier', lambda *args: PixelEncoder())

    def train_only(features, labels):
        assert len(features) == 3 and tuple(labels) == STANDARD_DISH_NAMES
        return head

    monkeypatch.setattr(train, 'fit_head', train_only)
    results = train.run(manifest, tmp_path, 'mocked-encoder', tmp_path/'out')
    assert results['test']['linear_head']['overall_accuracy'] == 1
    artifact = tmp_path/'out'/'head.json'
    assert 'session-' not in artifact.read_text(encoding='utf-8')
    restored = LinearDishHead.load(artifact, train.CHECKPOINT_SHA256)
    image = np.eye(3, EMBEDDING_SIZE)[0]
    assert restored.select(image, STANDARD_DISH_NAMES[0]) == STANDARD_DISH_NAMES[0]
    assert restored.select(image, None) is None
    assert restored.select(image, STANDARD_DISH_NAMES[1]) == STANDARD_DISH_NAMES[0]
    assert restored.select(np.zeros(EMBEDDING_SIZE), STANDARD_DISH_NAMES[0]) is None
    with pytest.raises(ValueError, match='does not match'):
        LinearDishHead.load(artifact, 'wrong-checkpoint')
    assert json.loads((tmp_path/'out'/'metrics.json').read_text()) == results
    artifact.write_text(json.dumps({**head.artifact(train.CHECKPOINT_SHA256), 'weights': {}}))
    with pytest.raises(ValueError, match='Invalid dish classifier coefficients'):
        LinearDishHead.load(artifact, train.CHECKPOINT_SHA256)
