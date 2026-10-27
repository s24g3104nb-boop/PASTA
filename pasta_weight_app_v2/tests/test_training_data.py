import csv
import io
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_data import (delete_sample, export_dataset, init_training_db,
                           list_samples, register_sample, update_sample)
from pasta_ml.dataset import load_dataset


def photo(seed=1, exif=False):
    image = Image.fromarray(np.random.default_rng(seed).integers(0, 255, (40, 50, 3), dtype=np.uint8))
    stream = io.BytesIO()
    metadata = image.getexif()
    if exif:
        metadata[270] = 'private image description'
    image.save(stream, 'PNG', exif=metadata)
    stream.seek(0)
    return stream


def labels(**changes):
    return dict(actual_weight_g=35, mode='spread', source='real', product_id='product-a',
                batch_id='weighing-1', session_id='session-1', thickness_mm=1.6,
                captured_at='2026-10-28T13:00:00+09:00', **changes)


def test_register_multiple_photos_and_normalized_duplicate(tmp_path):
    db = tmp_path / 'app.sqlite3'
    original = photo(exif=True)
    ident = register_sample(original, labels(), db_path=db)
    register_sample(photo(2), labels(), db_path=db)
    rows = list_samples(db)
    assert len(rows) == 2
    first = rows[0]
    assert first['id'] == ident and first['actual_weight_g'] == 35
    assert first['thickness_mm'] == 1.6 and first['version'] == 1
    assert first['registered_at'] and first['image_path'].endswith('.png')
    with Image.open(first['image_path']) as managed:
        assert not managed.getexif()
    with pytest.raises(ValueError, match='同じ画像'):
        register_sample(photo(), labels(), db_path=db)
    assert len(list_samples(db)) == 2


@pytest.mark.parametrize('key,value', [
    ('actual_weight_g', None), ('actual_weight_g', 0), ('actual_weight_g', -1),
    ('actual_weight_g', float('nan')), ('actual_weight_g', float('inf')),
    ('actual_weight_g', True), ('batch_id', ''), ('session_id', ''),
    ('product_id', ''), ('source', 'unknown'), ('mode', 'cooked'),
    ('thickness_mm', -1), ('thickness_mm', float('nan')), ('captured_at', 'yesterday'),
    ('individual_strand_count', 2.5),
])
def test_invalid_labels_do_not_register(tmp_path, key, value):
    db = tmp_path / 'app.sqlite3'
    metadata = labels()
    metadata[key] = value
    with pytest.raises(ValueError):
        register_sample(photo(), metadata, db_path=db)
    assert list_samples(db) == []
    assert not list((tmp_path / 'training' / 'images').glob('*.png'))


def test_missing_weight_and_invalid_image(tmp_path):
    db = tmp_path / 'app.sqlite3'
    metadata = labels()
    del metadata['actual_weight_g']
    with pytest.raises(ValueError, match='実測重量'):
        register_sample(photo(), metadata, db_path=db)
    with pytest.raises(ValueError, match='画像'):
        register_sample(b'not an image', labels(), db_path=db)
    assert list_samples(db) == []


def test_metadata_edit_version_and_soft_delete_preserve_photos(tmp_path):
    db = tmp_path / 'app.sqlite3'
    ident = register_sample(photo(), labels(), db_path=db)
    before = list_samples(db)[0]
    updated = update_sample(ident, {'actual_weight_g': 75, 'mode': 'bundle'},
                            db_path=db, expected_version=1)
    assert updated['actual_weight_g'] == 75 and updated['mode'] == 'bundle_end'
    assert updated['version'] == 2 and updated['pixel_hash'] == before['pixel_hash']
    with pytest.raises(ValueError, match='他の操作'):
        update_sample(ident, {'actual_weight_g': 60}, db_path=db, expected_version=1)
    with pytest.raises(ValueError, match='他の操作'):
        delete_sample(ident, db_path=db, expected_version=1)
    delete_sample(ident, db_path=db, expected_version=2)
    assert list_samples(db) == []
    archived = list_samples(db, include_deleted=True)[0]
    assert archived['deleted_at'] and archived['version'] == 3
    assert Path(archived['image_path']).is_file()
    with sqlite3.connect(db) as con:
        revisions = con.execute('SELECT version,action,snapshot_json FROM training_sample_revisions ORDER BY id').fetchall()
    assert [(row[0], row[1]) for row in revisions] == [(1, 'register'), (2, 'update'), (3, 'delete')]
    assert json.loads(revisions[0][2])['actual_weight_g'] == 35
    with pytest.raises(ValueError, match='同じ画像'):
        register_sample(photo(), labels(), db_path=db)


def test_export_snapshot_stays_immutable_after_edit_delete(tmp_path):
    db = tmp_path / 'app.sqlite3'
    ident = register_sample(photo(), labels(), db_path=db)
    exported = export_dataset(tmp_path / 'experiment_1', db_path=db)
    snapshot = load_dataset(exported)
    assert snapshot[0].weight == 35
    assert snapshot[0].values['training_sample_id'] == str(ident)
    assert snapshot[0].values['sample_version'] == '1'
    assert snapshot[0].values['thickness_mm'] == '1.6'
    image_bytes = snapshot[0].path.read_bytes()
    update_sample(ident, {'actual_weight_g': 75}, db_path=db)
    delete_sample(ident, db_path=db)
    assert load_dataset(exported)[0].weight == 35
    assert snapshot[0].path.read_bytes() == image_bytes
    with pytest.raises(ValueError, match='登録されていません'):
        export_dataset(tmp_path / 'experiment_2', db_path=db)


def test_snapshot_refuses_overwrite_and_modified_image(tmp_path):
    db = tmp_path / 'app.sqlite3'
    register_sample(photo(), labels(), db_path=db)
    exported = export_dataset(tmp_path / 'experiment', db_path=db)
    with pytest.raises(ValueError, match='新しいディレクトリ'):
        export_dataset(exported.parent, db_path=db)
    row = list_samples(db)[0]
    Path(row['image_path']).write_bytes(photo(99).getvalue())
    target = tmp_path / 'broken'
    with pytest.raises(ValueError, match='変更されています'):
        export_dataset(target, db_path=db)
    assert not target.exists()


def test_additive_tables_preserve_legacy_user_data(tmp_path):
    db = tmp_path / 'legacy.sqlite3'
    with sqlite3.connect(db) as con:
        con.execute('CREATE TABLE measurements (id INTEGER PRIMARY KEY, original TEXT)')
        con.execute("INSERT INTO measurements VALUES (4,'existing user data')")
    init_training_db(db)
    register_sample(photo(), labels(), db_path=db)
    with sqlite3.connect(db) as con:
        assert con.execute('SELECT * FROM measurements').fetchall() == [(4, 'existing user data')]


def test_augmented_children_inherit_group_labels_and_prevent_parent_drift(tmp_path):
    db = tmp_path / 'app.sqlite3'
    parent = register_sample(photo(), labels(), db_path=db)
    metadata = labels()
    metadata['parent_sample_id'] = parent
    child = register_sample(photo(2), metadata, db_path=db)
    assert list_samples(db)[1]['parent_sample_id'] == parent
    with pytest.raises(ValueError, match='元画像と同じ'):
        update_sample(child, {'batch_id': 'unrelated'}, db_path=db)
    with pytest.raises(ValueError, match='拡張画像がある'):
        update_sample(parent, {'actual_weight_g': 70}, db_path=db)
    with pytest.raises(ValueError, match='先に拡張画像'):
        delete_sample(parent, db_path=db)
    delete_sample(child, db_path=db)
    update_sample(parent, {'actual_weight_g': 70}, db_path=db)
    delete_sample(parent, db_path=db)
    assert not list_samples(db)


def test_export_selected_samples(tmp_path):
    db = tmp_path / 'app.sqlite3'
    first = register_sample(photo(), labels(), db_path=db)
    second = register_sample(photo(2), labels(), db_path=db)
    exported = export_dataset(tmp_path / 'selection', db_path=db, sample_ids=[second])
    samples = load_dataset(exported)
    assert len(samples) == 1 and samples[0].values['training_sample_id'] == str(second)
    with pytest.raises(ValueError, match='存在しない'):
        export_dataset(tmp_path / 'missing', db_path=db, sample_ids=[first, 999])
