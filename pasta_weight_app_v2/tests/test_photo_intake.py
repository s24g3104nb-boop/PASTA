"""すべて人工画像を使い、原写真・重量未確認の学習登録を防ぐ。"""
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from photo_intake import (add_original, confirm_group, list_intake, load_manifest,
                          load_original, merge_zip, package_inventory, promote_photo,
                          reassign_photo, save_review, set_photo_mode)
from storage import init_db, save_product
from training_data import list_samples


@pytest.fixture
def intake(tmp_path):
    root = tmp_path / 'Pasta_Lens_42_Photos'
    metadata = root / 'dataset/metadata'
    metadata.mkdir(parents=True)
    photos, rows = [], []
    for i in range(2):
        stream = io.BytesIO()
        rgb = np.random.default_rng(i).integers(0, 255, (80, 80, 3), dtype=np.uint8)
        Image.fromarray(rgb).save(stream, 'JPEG')
        data = stream.getvalue()
        photos.append(data)
        rows.append(dict(filename=f'photo_{i}.jpeg', original_sha256=hashlib.sha256(data).hexdigest(),
                         capture_group_id='A_pending', session_id='unconfirmed_session',
                         weight_g='', weight_verified='NO', mask_quality='UNREVIEWED', diameter_mm=''))
    with (metadata / 'labels_NEED_CONFIRMATION.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (metadata / 'originals_SHA256.txt').write_text(''.join(
        r['original_sha256'] + '  ' + r['filename'] + '\n' for r in rows))
    db = tmp_path / 'measurements.sqlite3'
    init_db(db)
    save_product('fixture product', .9, 350, 12, 2, 70, db_path=db)
    load_manifest(root, db, expected_count=2)
    return root, db, photos


def confirm_fixture(root, db):
    confirm_group(root, 'A_pending', 35, batch_id='fixture_weighing', session_id='fixture_session',
                  product_id='2', confirmed_same_batch=True, confirmed_by='fixture user', db_path=db)


def test_metadata_only_stays_pending_and_no_training_samples(intake):
    root, db, _ = intake
    rows = list_intake(db)
    assert len(rows) == 2
    assert all(r['weight_g'] is None and not r['weight_verified'] for r in rows)
    assert package_inventory(root)['missing'] == ['photo_0.jpeg', 'photo_1.jpeg']
    assert list_samples(db) == []
    with pytest.raises(ValueError, match='未確認'):
        promote_photo(rows[0]['id'], db)


def test_exact_original_match_and_full_decode(intake):
    root, db, photos = intake
    assert add_original(photos[0], root) == 'photo_0.jpeg'
    assert add_original(photos[0], root) == 'photo_0.jpeg'
    assert load_original(list_intake(db)[0]).shape == (80, 80, 3)
    report = package_inventory(root)
    assert report['received'] == ['photo_0.jpeg']
    assert not report['hash_mismatch'] and not report['damaged']
    with pytest.raises(ValueError, match='一致しません'):
        add_original(b'not the original', root)
    path = root / 'dataset/originals/photo_0.jpeg'
    path.write_bytes(b'corrupt')
    report = package_inventory(root)
    assert report['hash_mismatch'] == ['photo_0.jpeg'] and len(report['damaged']) == 1
    with pytest.raises(ValueError, match='SHA-256'):
        load_original(list_intake(db)[0])


@pytest.mark.parametrize('weight', [None, 0, -1, float('inf'), float('nan')])
def test_bad_weight_cannot_confirm(intake, weight):
    root, db, _ = intake
    with pytest.raises(ValueError):
        confirm_group(root, 'A_pending', weight, batch_id='b', session_id='s', product_id='2',
                      confirmed_same_batch=True, confirmed_by='fixture', db_path=db)
    assert not any(row['weight_verified'] for row in list_intake(db))


def test_same_batch_and_real_product_required(intake):
    root, db, _ = intake
    kwargs = dict(batch_id='b', session_id='s', confirmed_by='fixture', db_path=db)
    with pytest.raises(ValueError, match='同じ実測'):
        confirm_group(root, 'A_pending', 35, product_id='2', confirmed_same_batch=False, **kwargs)
    with pytest.raises(ValueError, match='実際の製品'):
        confirm_group(root, 'A_pending', 35, product_id='1', confirmed_same_batch=True, **kwargs)


def test_all_gates_and_lossless_training_promotion(intake):
    root, db, photos = intake
    add_original(photos[0], root)
    row = list_intake(db)[0]
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[10:70, 15:65] = 255
    save_review(row['id'], mask, reviewer='fixture', shape_confirmed=True,
                leakage_checked=True, db_path=db)
    with pytest.raises(ValueError, match='未確認'):
        promote_photo(row['id'], db)
    confirm_fixture(root, db)
    with pytest.raises(ValueError, match='撮影状態'):
        promote_photo(row['id'], db)
    set_photo_mode(row['id'], 'spread', db)
    ident = promote_photo(row['id'], db)
    assert promote_photo(row['id'], db) == ident
    samples = list_samples(db)
    assert len(samples) == 1 and samples[0]['actual_weight_g'] == 35
    assert samples[0]['batch_id'] == 'fixture_weighing'
    assert samples[0]['session_id'] == 'fixture_session'
    processed = np.asarray(Image.open(samples[0]['image_path']))
    rgb = load_original(row)
    assert np.array_equal(processed[mask > 0], rgb[mask > 0])
    assert np.all(processed[mask == 0] == 24)
    with pytest.raises(ValueError, match='学習登録済み'):
        reassign_photo(row['id'], 'another', db)


def test_qc_pending_leakage_and_modified_mask_cannot_promote(intake):
    root, db, photos = intake
    add_original(photos[0], root)
    confirm_fixture(root, db)
    row = list_intake(db)[0]
    set_photo_mode(row['id'], 'spread', db)
    with pytest.raises(ValueError, match='未完了'):
        promote_photo(row['id'], db)
    mask = np.full((80, 80), 255, dtype=np.uint8)
    with pytest.raises(ValueError, match='漏洩領域'):
        save_review(row['id'], mask, reviewer='fixture', shape_confirmed=True,
                    leakage_checked=False, db_path=db)
    with pytest.raises(ValueError, match='秤除外領域'):
        save_review(row['id'], mask, excluded_rects=[(0, 0, 10, 10)], reviewer='fixture',
                    shape_confirmed=True, leakage_checked=True, db_path=db)
    save_review(row['id'], mask, reviewer='fixture', shape_confirmed=True,
                leakage_checked=True, db_path=db)
    revised = list_intake(db)[0]
    Path(revised['mask_path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='変更されています'):
        promote_photo(row['id'], db)
    assert list_samples(db) == []


def test_reassign_resets_confirmation_and_reload_preserves_confirmed_state(intake):
    root, db, _ = intake
    confirm_fixture(root, db)
    load_manifest(root, db, expected_count=2)
    assert all(r['weight_g'] == 35 for r in list_intake(db))
    reassign_photo(list_intake(db)[0]['id'], 'new_pending_group', db)
    row = list_intake(db)[0]
    assert row['weight_g'] is None and not row['weight_verified'] and not row['group_verified']


def archive_bytes(mapping):
    result = io.BytesIO()
    with zipfile.ZipFile(result, 'w') as archive:
        for name, data in mapping.items():
            archive.writestr(name, data)
    return result.getvalue()


def test_independent_archives_merge_without_overwrite_or_code_escape(intake, tmp_path):
    root, _, photos = intake
    archive = archive_bytes({'Pasta_Lens_42_Photos/dataset/originals/photo_0.jpeg': photos[0]})
    assert merge_zip(archive, root) == ['dataset/originals/photo_0.jpeg']
    assert merge_zip(archive, root) == []
    with pytest.raises(ValueError, match='上書き'):
        merge_zip(archive_bytes({'Pasta_Lens_42_Photos/dataset/originals/photo_0.jpeg': b'other'}), root)
    with pytest.raises(ValueError):
        merge_zip(archive_bytes({'Pasta_Lens_42_Photos/../app.py': b'wrong'}), root)
    assert not (tmp_path / 'app.py').exists()


def test_conflicting_manifest_does_not_overwrite_ledger(intake):
    root, db, _ = intake
    metadata = root / 'dataset/metadata'
    content = (metadata / 'originals_SHA256.txt').read_text()
    (metadata / 'originals_SHA256.txt').write_text(content.replace('photo_0', 'different'))
    with pytest.raises(ValueError, match='一致しません'):
        load_manifest(root, db, expected_count=2)
    assert len(list_intake(db)) == 2


def test_preview_zip_is_separate_from_verified_original_and_never_auto_ready(intake):
    root, db, photos = intake
    source = Image.open(io.BytesIO(photos[0]))
    preview = io.BytesIO()
    source.save(preview, 'JPEG', quality=65)
    archive = archive_bytes({'A_CANDIDATE/photo_0.jpeg': preview.getvalue()})
    assert merge_zip(archive, root) == ['dataset/previews/photo_0.jpeg']
    from photo_intake import load_preview
    row = list_intake(db)[0]
    assert load_preview(row).shape == (80,80,3)
    result = package_inventory(root)
    assert result['received'] == [] and result['preview_received'] == ['photo_0.jpeg']
    assert not result['preview_damaged']
    confirm_fixture(root, db)
    set_photo_mode(row['id'], 'spread', db)
    with pytest.raises(ValueError, match='元JPEG'):
        save_review(row['id'], np.full((80,80),255,dtype=np.uint8), reviewer='fixture',
                    shape_confirmed=True, leakage_checked=True, db_path=db)
    assert list_samples(db) == []
