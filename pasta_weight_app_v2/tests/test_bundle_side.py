"""側面を断面に誤分類せず、旧DBの全記録を保持する。写真精度の試験ではない。"""
import csv
import io
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pasta_weight_app_v2.pasta_ml.dataset import load_dataset
from pasta_weight_app_v2.pasta_ml.evaluate import evaluate
from pasta_weight_app_v2.pasta_ml.model import predict, train
from pasta_weight_app_v2.training_data import (
    export_dataset, init_training_db, list_samples, register_sample, update_sample,
)


def photo(seed=0):
    stream = io.BytesIO()
    Image.fromarray(np.random.default_rng(seed).integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(stream, 'PNG')
    return stream.getvalue()


def metadata(mode='bundle_side', index=0):
    return dict(actual_weight_g=20 + index % 5 * 20, mode=mode, source='synthetic',
                product_id='test', batch_id=f'b{index}', session_id=f's{index}')


def legacy_database(tmp_path):
    current = tmp_path / 'definition.sqlite3'
    init_training_db(current)
    with sqlite3.connect(current) as con:
        sample_sql = con.execute("SELECT sql FROM sqlite_master WHERE name='training_samples'").fetchone()[0]
        revision_sql = con.execute("SELECT sql FROM sqlite_master WHERE name='training_sample_revisions'").fetchone()[0]
    old = tmp_path / 'legacy.sqlite3'
    with sqlite3.connect(old) as con:
        con.row_factory = sqlite3.Row
        con.execute(sample_sql.replace("'spread', 'bundle_end', 'bundle_side'", "'spread', 'bundle_end'"))
        con.execute(revision_sql)
        con.execute("ALTER TABLE training_samples ADD COLUMN capture_note TEXT NOT NULL DEFAULT 'bundle_side'")
        con.execute('ALTER TABLE training_samples ADD COLUMN doubled_weight REAL GENERATED ALWAYS AS(actual_weight_g * 2) VIRTUAL')
        con.execute('CREATE INDEX preserve_capture_note ON training_samples(capture_note)')
        con.execute('CREATE TABLE audit_updates(sample_id INTEGER)')
        con.execute('''CREATE TRIGGER preserve_update_trigger AFTER UPDATE ON training_samples
            BEGIN INSERT INTO audit_updates(sample_id) VALUES (NEW.id); END''')
        con.execute('CREATE VIEW preserve_training_view AS SELECT id,capture_note FROM training_samples')
        for ident in (10, 11, 12):
            con.execute('''INSERT INTO training_samples
                (id,pixel_hash,perceptual_hash,image_filename,actual_weight_g,mode,source,product_id,
                 batch_id,session_id,parent_sample_id,registered_at,updated_at,deleted_at,capture_note)
                VALUES (?,?,?,?,20,'bundle_end','synthetic','test','b1','s1',?,?,?,?,?)''',
                (ident, f'hash-{ident}', '1', f'old-{ident}.png', 10 if ident == 11 else None,
                 'old-created', 'old-updated', 'old-deleted' if ident == 12 else None, f'custom-{ident}'))
            row = dict(con.execute('SELECT * FROM training_samples WHERE id=?', (ident,)).fetchone())
            con.execute('''INSERT INTO training_sample_revisions
                (sample_id,version,changed_at,action,snapshot_json) VALUES (?,1,'old','register',?)''',
                (ident, json.dumps(row)))
        con.execute("UPDATE sqlite_sequence SET seq=99 WHERE name='training_samples'")
    return old


def test_fresh_side_mode_and_legacy_alias_export(tmp_path):
    db = tmp_path / 'training.sqlite3'
    side_id = register_sample(photo(), metadata(), db_path=db)
    end_id = register_sample(photo(1), metadata('bundle', 1), db_path=db)
    samples = {row['id']: row for row in list_samples(db)}
    assert samples[side_id]['mode'] == 'bundle_side'
    assert samples[end_id]['mode'] == 'bundle_end'
    dataset = load_dataset(export_dataset(tmp_path / 'snapshot', db_path=db))
    assert {sample.values['mode'] for sample in dataset} == {'bundle_side', 'bundle_end'}


def test_legacy_migration_keeps_columns_rows_history_fk_sequence_indexes_and_view(tmp_path):
    db = legacy_database(tmp_path)
    with sqlite3.connect(db) as con:
        original_rows = con.execute('SELECT * FROM training_samples ORDER BY id').fetchall()
        original_revisions = con.execute('SELECT * FROM training_sample_revisions ORDER BY id').fetchall()
    init_training_db(db)
    init_training_db(db)
    with sqlite3.connect(db) as con:
        assert con.execute('SELECT * FROM training_samples ORDER BY id').fetchall() == original_rows
        assert con.execute('SELECT * FROM training_sample_revisions ORDER BY id').fetchall() == original_revisions
        assert con.execute('PRAGMA foreign_key_check').fetchall() == []
        assert con.execute("SELECT seq FROM sqlite_sequence WHERE name='training_samples'").fetchone()[0] == 99
        assert con.execute('SELECT count(*) FROM audit_updates').fetchone()[0] == 0
        assert con.execute('SELECT capture_note FROM preserve_training_view WHERE id=11').fetchone()[0] == 'custom-11'
        assert con.execute("SELECT name FROM sqlite_master WHERE name='preserve_capture_note'").fetchone()
    backups = list((tmp_path / 'backups').glob('legacy.before-capture-modes-*.sqlite3'))
    assert len(backups) == 1
    with sqlite3.connect(backups[0]) as con:
        assert con.execute('SELECT count(*) FROM training_samples').fetchone()[0] == 3
        assert "'bundle_side'" not in con.execute("SELECT sql FROM sqlite_master WHERE name='training_samples'").fetchone()[0].split('CHECK', 1)[1].split(')', 1)[0]
    added = register_sample(photo(7), metadata(index=7), db_path=db)
    assert added == 100
    update_sample(added, {'actual_weight_g': 75}, db_path=db)
    with sqlite3.connect(db) as con:
        assert con.execute('SELECT sample_id FROM audit_updates').fetchall() == [(100,)]
        assert con.execute('SELECT doubled_weight FROM training_samples WHERE id=100').fetchone()[0] == 150


def test_corrupt_parent_rolls_back_migration_without_losing_rows(tmp_path):
    db = legacy_database(tmp_path)
    with sqlite3.connect(db) as con:
        con.execute('UPDATE training_samples SET parent_sample_id=999 WHERE id=11')
        original = con.execute('SELECT * FROM training_samples ORDER BY id').fetchall()
        original_schema = con.execute("SELECT sql FROM sqlite_master WHERE name='training_samples'").fetchone()[0]
    with pytest.raises(ValueError, match='参照整合性'):
        init_training_db(db)
    with sqlite3.connect(db) as con:
        assert con.execute('SELECT * FROM training_samples ORDER BY id').fetchall() == original
        assert con.execute("SELECT sql FROM sqlite_master WHERE name='training_samples'").fetchone()[0] == original_schema
        assert con.execute("SELECT name FROM sqlite_master WHERE name='_training_samples_capture_modes_v2'").fetchone() is None


def test_unknown_schema_stops_without_rebuild(tmp_path):
    db = tmp_path / 'unknown.sqlite3'
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE training_samples(id INTEGER PRIMARY KEY, mode TEXT CHECK(mode='spread' OR mode='bundle_end'))")
        con.execute("INSERT INTO training_samples VALUES(5,'spread')")
    with pytest.raises(ValueError, match='未知のスキーマ'):
        init_training_db(db)
    with sqlite3.connect(db) as con:
        assert con.execute('SELECT * FROM training_samples').fetchall() == [(5, 'spread')]
        assert con.execute("SELECT 1 FROM sqlite_master WHERE name='training_sample_revisions'").fetchone() is None


def test_side_regression_is_independent_and_opencv_refuses_side(tmp_path, monkeypatch):
    db = tmp_path / 'side.sqlite3'
    for index in range(15):
        register_sample(photo(index), metadata(index=index), db_path=db)
    path = export_dataset(tmp_path / 'experiment', db_path=db)
    samples = load_dataset(path)
    for i, sample in enumerate(samples):
        sample.values['split'] = 'train' if i < 11 else ('val' if i < 13 else 'test')
    model = train(samples, tmp_path / 'side-model.json', allow_synthetic=True)
    assert set(model['entries']) == {'bundle_side:test'}
    rgb = np.asarray(Image.open(samples[0].path))
    assert predict(model, rgb, 'bundle_end', 'test')['success'] is False
    import pasta_weight_app_v2.pasta_ml.evaluate as evaluation_module
    def forbidden(*args, **kwargs):
        raise AssertionError('Hough end-face detector must never receive bundle_side images')
    monkeypatch.setattr(evaluation_module, 'detect_bundle_ends', forbidden)
    report = evaluate(samples, 'val', calibrations={'test': 1}, allow_synthetic=True)
    assert report['overall']['total'] == report['overall']['failed'] == 2
    assert report['overall']['mae_g'] is None
    assert all('適用外' in row['reason'] for row in report['rows'])
