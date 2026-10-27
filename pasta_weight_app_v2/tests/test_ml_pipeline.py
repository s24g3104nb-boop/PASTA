from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pasta_weight_app_v2.pasta_ml.dataset import (
    dataset_hash, grouping, image_rgb, import_dataset, load_dataset,
    split_dataset, validate_splits, write_csv,
)
from pasta_weight_app_v2.pasta_ml.evaluate import evaluate, load_calibrations, summary
from pasta_weight_app_v2.pasta_ml.model import FEATURE_VERSION, MODEL_VERSION, features, load_model, predict, train
from pasta_weight_app_v2.generate_demo import generate_images


@pytest.fixture
def dataset(tmp_path):
    rows = []
    for i in range(15):
        rgb = np.random.default_rng(i).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(rgb).save(tmp_path / f'{i}.png')
        rows.append(dict(image_path=f'{i}.png', batch_id=f'b{i}', session_id=f's{i}',
                         product_id='test', mode='spread', actual_weight_g=20 + i % 5 * 20,
                         source='synthetic', lighting='fixture', background='fixture'))
    path = tmp_path / 'labels.csv'
    save_rows(path, rows)
    return path, rows


def save_rows(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_dataset_import_exif_strip_and_repeat_safety(dataset, tmp_path):
    path, _ = dataset
    original = load_dataset(path)
    dest = tmp_path / 'managed'
    imported_path = import_dataset(original, dest)
    imported = load_dataset(imported_path)
    assert len(imported) == 15
    assert [s.pixel_hash for s in original] == [s.pixel_hash for s in imported]
    with Image.open(imported[0].path) as image:
        assert not image.getexif()
    with pytest.raises(ValueError, match='存在しない'):
        import_dataset(original, dest)
    assert path.is_file()


@pytest.mark.parametrize('weight', ['0', '-1', 'nan', 'inf', 'oops'])
def test_invalid_weight_rejected(dataset, weight):
    path, rows = dataset
    rows[0]['actual_weight_g'] = weight
    save_rows(path, rows)
    with pytest.raises(ValueError, match='CSV 2行目'):
        load_dataset(path)


@pytest.mark.parametrize('bad_path', ['../outside.png', '/tmp/outside.png', 'C:\\photos\\image.png'])
def test_path_escape_rejected(dataset, bad_path):
    path, rows = dataset
    rows[0]['image_path'] = bad_path
    save_rows(path, rows)
    with pytest.raises(ValueError, match='相対パス'):
        load_dataset(path)


def test_empty_corrupt_and_unknown_mode_rejected(dataset):
    path, rows = dataset
    path.write_text(','.join(rows[0]) + '\n')
    with pytest.raises(ValueError, match='データがありません'):
        load_dataset(path)
    save_rows(path, rows)
    rows[0]['mode'] = 'cooked'
    save_rows(path, rows)
    with pytest.raises(ValueError, match='modeは'):
        load_dataset(path)
    rows[0]['mode'] = 'bundle'
    save_rows(path, rows)
    assert load_dataset(path)[0].values['mode'] == 'bundle_end'
    (path.parent / '0.png').write_bytes(b'not an image')
    with pytest.raises(ValueError, match='CSV 2行目'):
        load_dataset(path)


def test_duplicate_and_transitive_groups(dataset):
    path, rows = dataset
    samples = load_dataset(path)
    assert len(grouping(samples)) == 15
    samples[1].values['batch_id'] = samples[0].values['batch_id']
    samples[2].values['session_id'] = samples[1].values['session_id']
    assert any(set(group) == {0, 1, 2} for group in grouping(samples))
    # A reencoded/brightness changed image is conservatively grouped by dHash.
    samples[3].perceptual_hash = samples[2].perceptual_hash ^ 1
    assert any({0, 1, 2, 3} <= set(group) for group in grouping(samples))
    rows[1]['image_path'] = rows[0]['image_path']
    save_rows(path, rows)
    with pytest.raises(ValueError, match='矛盾'):
        load_dataset(path)


def test_split_reproducibility_no_leakage_and_no_overwrite(dataset, tmp_path):
    path, _ = dataset
    first = split_dataset(load_dataset(path), seed=17)
    second = split_dataset(load_dataset(path), seed=17)
    assert [s.values['split'] for s in first] == [s.values['split'] for s in second]
    assert {s.values['split'] for s in first} == {'train', 'val', 'test'}
    validate_splits(first)
    output = tmp_path / 'split.csv'
    write_csv(first, output)
    validate_splits(load_dataset(output))
    with pytest.raises(ValueError, match='既に存在'):
        write_csv(first, output)
    with pytest.raises(ValueError, match='作り直しません'):
        split_dataset(first)
    a = next(s for s in first if s.values['split'] == 'train')
    b = next(s for s in first if s.values['split'] == 'test')
    b.values['session_id'] = a.values['session_id']
    with pytest.raises(ValueError, match='リーク'):
        validate_splits(first)


def test_few_independent_groups_refuses_split(dataset):
    path, _ = dataset
    samples = load_dataset(path)
    for s in samples:
        s.values['session_id'] = 'one-session'
    with pytest.raises(ValueError, match='3つ以上'):
        split_dataset(samples)


def test_training_explicit_synthetic_and_safe_model_roundtrip(dataset, tmp_path):
    path, _ = dataset
    samples = split_dataset(load_dataset(path))
    output = tmp_path / 'model.json'
    with pytest.raises(ValueError, match='synthetic'):
        train(samples, output)
    model = train(samples, output, allow_synthetic=True)
    loaded = load_model(output)
    assert loaded['model_version'] == model['model_version']
    assert loaded['source'] == 'synthetic'
    assert loaded['training_data_hash'] == dataset_hash([s for s in samples if s.values['split'] == 'train'])
    assert model['validation_report']['overall']['total'] > 0
    assert features(image_rgb(samples[0].path)).shape == (36,)
    unknown = predict(loaded, image_rgb(samples[0].path), 'spread', 'unknown')
    assert unknown['success'] is False and unknown['predicted_g'] is None
    with pytest.raises(ValueError, match='上書き'):
        train(samples, output, allow_synthetic=True)
    loaded['entries']['spread:test']['coef'][0] = 'nan'
    output.write_text(json.dumps(loaded))
    with pytest.raises(ValueError, match='係数'):
        load_model(output)


@pytest.mark.parametrize('payload', [[], None, {'entries': []},
                         {'model_version': MODEL_VERSION, 'feature_version': FEATURE_VERSION,
                          'source': 'real', 'entries': [{}]}])
def test_invalid_model_schema_is_user_error(tmp_path, payload):
    path = tmp_path / 'malformed.json'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_model(path)


def test_test_labels_not_used_for_training(dataset, tmp_path):
    path, _ = dataset
    samples = split_dataset(load_dataset(path))
    first = train(samples, tmp_path / 'a.json', allow_synthetic=True)
    for s in samples:
        if s.values['split'] == 'test':
            s.values['actual_weight_g'] = '999'
    second = train(samples, tmp_path / 'b.json', allow_synthetic=True)
    assert first['entries'] == second['entries']
    assert first['training_data_hash'] == second['training_data_hash']
    assert first['dataset_sha256'] != second['dataset_sha256']
    with pytest.raises(ValueError, match='一致しません'):
        evaluate(samples, 'test', model=first, allow_synthetic=True)


def test_regression_output_is_continuous_for_unseen_synthetic_image(dataset, tmp_path):
    path, _ = dataset
    samples = split_dataset(load_dataset(path))
    model = train(samples, tmp_path / 'continuous.json', allow_synthetic=True)
    assert {s.weight for s in samples if s.values['split'] == 'train'} <= {20, 40, 60, 80, 100}
    predictions = [predict(model, image_rgb(s.path), s.values['mode'], s.values['product_id'])
                   for s in samples if s.values['split'] != 'train']
    succeeded = [p['predicted_g'] for p in predictions if p['success']]
    # This checks the output interface, not real-world accuracy at intermediate weights.
    assert succeeded
    assert any(not np.isclose(weight / 20, round(weight / 20)) for weight in succeeded)


def test_failure_accounting_and_metrics():
    result = summary([dict(success=True, actual_weight_g=100, predicted_g=110),
                      dict(success=True, actual_weight_g=50, predicted_g=40),
                      dict(success=False, actual_weight_g=20, predicted_g=None)])
    assert result['total'] == 3 and result['failed'] == 1
    assert result['mae_g'] == result['rmse_g'] == 10
    assert result['mape_percent'] == pytest.approx(15)
    assert result['bias_g'] == 0
    assert summary([dict(success=False)])['mae_g'] is None


def test_baseline_missing_calibration_counts_failure(dataset, tmp_path):
    path, _ = dataset
    samples = split_dataset(load_dataset(path))
    report = evaluate(samples, 'val', calibrations={}, allow_synthetic=True)
    assert report['overall']['failed'] == report['overall']['total']
    assert report['overall']['mae_g'] is None
    assert 'spread' in report['by']['mode']
    calibration = tmp_path / 'calibration.csv'
    calibration.write_text('product_id,g_per_strand\ntest,nan\n')
    with pytest.raises(ValueError, match='校正値'):
        load_calibrations(calibration)


def test_both_modes_have_independent_regression_entries(dataset, tmp_path):
    path, rows = dataset
    extra = []
    for i in range(15, 30):
        rgb = np.random.default_rng(i).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(rgb).save(tmp_path / f'{i}.png')
        extra.append(dict(rows[i - 15], image_path=f'{i}.png', batch_id=f'b{i}',
                          session_id=f's{i}', mode='bundle_end'))
    save_rows(path, rows + extra)
    samples = split_dataset(load_dataset(path))
    model = train(samples, tmp_path / 'both.json', allow_synthetic=True)
    assert set(model['entries']) == {'spread:test', 'bundle_end:test'}
    image = image_rgb(samples[-1].path)
    assert predict(model, image, 'bundle', 'test') == predict(model, image, 'bundle_end', 'test')


def test_opencv_baseline_executes_real_detector_for_each_mode(dataset, tmp_path):
    path, rows = dataset
    generate_images(tmp_path)
    rows = [dict(rows[0], split='train'),
            dict(rows[1], image_path='demo_spread_12.png', actual_weight_g=12, split='val'),
            dict(rows[2], image_path='demo_bundle_16.png', actual_weight_g=16, mode='bundle_end', split='test')]
    save_rows(path, rows)
    samples = load_dataset(path)
    for split in ('val', 'test'):
        report = evaluate(samples, split, calibrations={'test': 1.}, allow_synthetic=True)
        assert report['overall']['succeeded'] == 1
        assert report['overall']['mae_g'] == 0


def test_cli_end_to_end(dataset, tmp_path):
    path, _ = dataset

    def run(*args, expected=0):
        result = subprocess.run([sys.executable, '-m', 'pasta_weight_app_v2.pasta_ml', *map(str, args)],
                                cwd=ROOT, capture_output=True, text=True)
        assert result.returncode == expected, result.stdout + result.stderr
        return result

    run('validate', path)
    managed = tmp_path / 'imported'
    run('import', path, '--destination', managed)
    split = managed / 'split.csv'
    run('split', managed / 'dataset.csv', '--output', split)
    model = tmp_path / 'cli-model.json'
    run('train', split, '--output', model, '--allow-synthetic')
    run('evaluate', split, '--model', model, '--split', 'test', '--output', tmp_path / 'no.json',
        '--allow-synthetic', expected=2)
    report = tmp_path / 'report.json'
    run('evaluate', split, '--model', model, '--split', 'test', '--final-test', '--output', report,
        '--allow-synthetic')
    assert json.loads(report.read_text())['source'] == 'synthetic'
    assert json.loads(report.read_text())['overall']['total'] > 0
    run('evaluate', split, '--model', model, '--split', 'test', '--final-test', '--output', tmp_path / 'again.json',
        '--allow-synthetic', expected=2)
    run('predict', managed / 'images' / next((managed / 'images').iterdir()).name,
        '--model', model, '--mode', 'spread', '--product-id', 'test')
