import io
import json
from pathlib import Path
import sqlite3
import sys
import time
from uuid import uuid4

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'pasta_weight_app_v2'))
from pasta_weight_app_v2.storage import init_db, save_product
from pasta_weight_app_v2.training_data import list_samples, register_sample
from pasta_weight_app_v2.training_manager import (
    activate_model, assign_persistent_splits, get_active_model, init_training_manager,
    list_jobs, list_models, run_job, start_training,
)
from pasta_weight_app_v2.pasta_ml.dataset import load_dataset
from pasta_weight_app_v2.training_data import export_dataset


def add_samples(db, start=0, stop=15, source='synthetic'):
    ids = []
    for i in range(start, stop):
        rgb = np.random.default_rng(i).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        stream = io.BytesIO()
        Image.fromarray(rgb).save(stream, 'PNG')
        ids.append(register_sample(stream.getvalue(), dict(
            actual_weight_g=20 + (i % 5) * 20, mode='spread', source=source,
            product_id='2', batch_id=f'b{i}', session_id=f's{i}',
            thickness_mm=1.6, captured_at='2026-10-28T12:00:00+09:00'), db_path=db))
    return ids


def queued_job(db, allow_synthetic=True):
    init_training_manager(db)
    ident = uuid4().hex
    with sqlite3.connect(db) as con:
        con.execute('''INSERT INTO training_jobs
            (id,status,progress,message,created_at,alpha,allow_synthetic)
            VALUES (?,'queued',0,'test','2026-10-28T00:00:00+00:00',1,?)''',
            (ident, int(allow_synthetic)))
    return ident


def wait_finished(db, ident, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = next(item for item in list_jobs(db) if item['id'] == ident)
        if job['status'] in {'succeeded', 'failed'}:
            return job
        time.sleep(.05)
    raise AssertionError('worker did not finish')


def test_background_worker_failure_visible_without_data(tmp_path):
    db = tmp_path / 'test.sqlite3'
    init_db(db)
    started = time.monotonic()
    ident = start_training(db_path=db)
    assert time.monotonic() - started < 2
    job = wait_finished(db, ident)
    assert job['status'] == 'failed'
    assert '学習画像がありません' in job['error']
    assert list_models(db) == []
    assert get_active_model(db) is None


def test_worker_immutable_snapshot_report_and_synthetic_activation_refused(tmp_path):
    db = tmp_path / 'test.sqlite3'
    init_db(db)
    add_samples(db)
    ident = start_training(db_path=db, allow_synthetic=True)
    job = wait_finished(db, ident)
    assert job['status'] == 'succeeded', job
    assert job['progress'] == 100 and job['sample_count'] == 15
    report = job['report']
    assert set(report['split_counts']) == {'train', 'val', 'test'}
    assert report['regression']['split'] == report['opencv']['split'] == 'val'
    assert report['regression']['dataset_sha256'] == report['opencv']['dataset_sha256']
    assert report['regression']['overall']['total'] == report['opencv']['overall']['total']
    assert report['opencv']['overall']['failed'] == report['opencv']['overall']['total']
    model = list_models(db)[0]
    assert model['source'] == 'synthetic' and not model['active']
    assert Path(model['model_path']).is_file()
    with pytest.raises(ValueError, match='合成画像'):
        activate_model(model['id'], db)
    assert get_active_model(db) is None


def test_default_worker_excludes_synthetic_and_requires_real(tmp_path):
    db = tmp_path / 'test.sqlite3'
    init_db(db)
    add_samples(db, stop=3)
    ident = queued_job(db, allow_synthetic=False)
    run_job(ident, db)
    job = next(item for item in list_jobs(db) if item['id'] == ident)
    assert job['status'] == 'failed' and '実写学習画像がありません' in job['error']
    assert list_models(db) == []


def test_persistent_split_addition_retains_all_existing_assignments(tmp_path):
    db = tmp_path / 'test.sqlite3'
    init_db(db)
    init_training_manager(db)
    add_samples(db)
    first = load_dataset(export_dataset(tmp_path / 'first', db_path=db))
    assign_persistent_splits(first, db)
    initial = {s.pixel_hash: s.values['split'] for s in first}
    add_samples(db, start=15, stop=20)
    second = load_dataset(export_dataset(tmp_path / 'second', db_path=db))
    assign_persistent_splits(second, db)
    assert all(s.values['split'] == initial[s.pixel_hash] for s in second if s.pixel_hash in initial)
    # Editing session IDs cannot join previously separated holdout partitions.
    train = next(s for s in second if s.values['split'] == 'train')
    test = next(s for s in second if s.values['split'] == 'test')
    test.values['session_id'] = train.values['session_id']
    with pytest.raises(ValueError, match='結び付け'):
        assign_persistent_splits(second, db)


def test_stopped_job_recovery_and_missing_model(tmp_path):
    db = tmp_path / 'test.sqlite3'
    init_training_manager(db)
    ident = queued_job(db)
    with sqlite3.connect(db) as con:
        con.execute('UPDATE training_jobs SET worker_pid=? WHERE id=?', (999999999, ident))
    assert list_jobs(db)[0]['status'] == 'failed'
    with pytest.raises(ValueError, match='存在しません'):
        activate_model('missing', db)


def test_model_integrity_and_active_loading(tmp_path):
    db = tmp_path / 'test.sqlite3'
    init_db(db)
    save_product('test product', .9, 350, 12, 2, 70, db_path=db)
    # Pixels are controlled synthetic fixtures, deliberately labeled real only to
    # exercise the real-only API branch; this test asserts no real-world accuracy.
    add_samples(db, source='real')
    ident = queued_job(db, allow_synthetic=False)
    model_id = run_job(ident, db)
    assert model_id is not None, list_jobs(db)
    activate_model(model_id, db)
    active = get_active_model(db)
    assert active['id'] == model_id and active['artifact']['source'] == 'real'
    assert active['report']['calibrations'] == {'2': .9}
    path = Path(active['model_path'])
    Path(str(path) + '.test-used.json').write_text('{}')
    opened = get_active_model(db)
    assert opened['test_opened']
    assert '開封済み' in opened['report']['notice']
    path.write_text(path.read_text() + ' ')
    with pytest.raises(ValueError, match='変更されています'):
        get_active_model(db)


@pytest.mark.parametrize('alpha', [0, -1, float('nan'), float('inf'), True])
def test_invalid_job_configuration(alpha, tmp_path):
    with pytest.raises(ValueError):
        start_training(db_path=tmp_path / 'test.sqlite3', alpha=alpha)
