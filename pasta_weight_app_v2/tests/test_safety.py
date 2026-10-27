import io
import sqlite3
from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pasta_vision import image_to_rgb, nutrition_for_weight, detect_bundle_ends, detect_spread
from storage import init_db, list_measurements, save_measurement, save_product
from pasta_vision import image_quality_warnings


@pytest.mark.parametrize('value', [-1, float('nan'), float('inf')])
def test_invalid_nutrition_and_product(value, tmp_path):
    with pytest.raises(ValueError):
        nutrition_for_weight(value, 350, 12, 2, 70)
    with pytest.raises(ValueError):
        nutrition_for_weight(100, value, 12, 2, 70)
    db = tmp_path / 'test.sqlite3'
    init_db(db)
    with pytest.raises(ValueError):
        save_product('bad', value, 350, 12, 2, 70, db_path=db)


def test_image_formats_exif_and_limits():
    image = Image.new('RGBA', (12, 10), (255, 0, 0, 0))
    assert np.all(image_to_rgb(image) == 255)
    assert image_to_rgb(Image.new('L', (8, 6))).shape == (6, 8, 3)
    image = Image.new('RGB', (30, 10))
    exif = image.getexif()
    exif[274] = 6
    stream = io.BytesIO()
    image.save(stream, 'JPEG', exif=exif)
    stream.seek(0)
    assert image_to_rgb(stream).shape == (30, 10, 3)
    with pytest.raises(ValueError, match='2500万'):
        image_to_rgb(Image.new('1', (5001, 5000)))
    with pytest.raises((ValueError, OSError)):
        image_to_rgb(io.BytesIO(b'not an image'))


def test_detector_invalid_parameters_and_empty_image():
    rgb = np.zeros((50, 50, 3), dtype=np.uint8)
    assert detect_spread(rgb).count == 0
    with pytest.raises(ValueError):
        detect_spread(rgb, saturation_min=300)
    with pytest.raises(ValueError):
        detect_bundle_ends(rgb, min_distance=0)
    with pytest.raises(ValueError):
        detect_bundle_ends(rgb, min_radius=22, max_radius=5)


def test_quality_gate_for_dark_overexposed_small_and_normal_sample(tmp_path):
    assert any('暗すぎ' in s for s in image_quality_warnings(np.zeros((100, 100, 3), dtype=np.uint8)))
    assert any('白飛び' in s for s in image_quality_warnings(np.full((100, 100, 3), 255, dtype=np.uint8)))
    assert any('小さすぎ' in s for s in image_quality_warnings(np.zeros((32, 32, 3), dtype=np.uint8)))
    from generate_demo import generate_images
    generate_images(tmp_path)
    assert image_quality_warnings(image_to_rgb(tmp_path / 'demo_spread_12.png')) == []


def test_training_and_measurement_share_transparent_preprocessing(tmp_path):
    from pasta_weight_app_v2.pasta_ml.dataset import image_rgb
    path = tmp_path / 'transparent.png'
    Image.new('RGBA', (80, 80), (200, 0, 0, 0)).save(path)
    assert np.array_equal(image_rgb(path), image_to_rgb(path))


def test_additive_migration_preserves_legacy_record(tmp_path):
    db = tmp_path / 'legacy.sqlite3'
    with sqlite3.connect(db) as con:
        con.execute('''CREATE TABLE measurements (
            id INTEGER PRIMARY KEY, created_at TEXT, product_name TEXT, mode TEXT,
            detected_count INTEGER, estimated_g REAL, actual_g REAL, kcal REAL,
            protein_g REAL, fat_g REAL, carb_g REAL, image_filename TEXT)''')
        con.execute("INSERT INTO measurements VALUES (17,'old','original','spread',3,3,NULL,10,1,1,1,NULL)")
    init_db(db)
    init_db(db)
    record = list_measurements(db)[0]
    assert record['id'] == 17 and record['product_name'] == 'original'
    assert record['source_kind'] == 'unknown' and record['kcal_100'] is None


def measurement_args(db):
    return dict(product_name='test', mode='spread', detected_count=2, estimated_g=2,
                actual_g=None, nutrition=nutrition_for_weight(2, 350, 12, 2, 70),
                image_rgb=np.zeros((30, 30, 3), dtype=np.uint8), db_path=db)


def test_measurement_provenance_and_snapshot(tmp_path):
    db = tmp_path / 'test.sqlite3'
    init_db(db)
    ident = save_measurement(**measurement_args(db), source_kind='synthetic',
                             estimator='manual_count', product_snapshot=dict(
                                 g_per_strand=1, kcal_100=350, protein_100=12, fat_100=2, carb_100=70))
    row = list_measurements(db)[0]
    assert row['id'] == ident and row['source_kind'] == 'synthetic'
    assert row['kcal_100'] == 350 and row['estimator'] == 'manual_count'


def test_failed_db_write_removes_photo(tmp_path):
    db = tmp_path / 'uninitialized.sqlite3'
    with pytest.raises(sqlite3.OperationalError):
        save_measurement(**measurement_args(db))
    assert not list((tmp_path / 'photos').glob('*.jpg'))


def test_bad_measurement_does_not_write(tmp_path):
    db = tmp_path / 'test.sqlite3'
    init_db(db)
    args = measurement_args(db)
    args['estimated_g'] = float('nan')
    with pytest.raises(ValueError):
        save_measurement(**args)
    assert list_measurements(db) == []
    assert not (tmp_path / 'photos').exists()
