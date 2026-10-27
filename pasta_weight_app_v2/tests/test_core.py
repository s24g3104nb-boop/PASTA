from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate_demo import generate_images
from pasta_vision import detect_bundle_ends, detect_spread, image_to_rgb, nutrition_for_weight
from storage import init_db, list_measurements, list_products, save_measurement, save_product


def test_sample_spread(tmp_path):
    generate_images(tmp_path)
    rgb = image_to_rgb(tmp_path / 'demo_spread_12.png')
    result = detect_spread(rgb)
    assert result.count == 12, f'12 strands expected, got {result.count}'


def test_sample_bundle(tmp_path):
    generate_images(tmp_path)
    rgb = image_to_rgb(tmp_path / 'demo_bundle_16.png')
    result = detect_bundle_ends(rgb, min_radius=8, max_radius=27,
                                min_distance=32, hough_sensitivity=18)
    assert result.count == 16, f'16 ends expected, got {result.count}'


def test_nutrition():
    out = nutrition_for_weight(50, 350, 12, 2, 70)
    assert out == {'kcal': 175, 'protein_g': 6, 'fat_g': 1, 'carb_g': 35}


def test_db_and_history(tmp_path):
    db_file = tmp_path / 'test.sqlite3'
    init_db(db_file)
    assert len(list_products(db_file)) == 1
    save_product('テスト乾麺', 0.9, 355, 10, 1, 73, db_path=db_file)
    assert len(list_products(db_file)) == 2
    save_measurement(product_name='テスト乾麺', mode='広げた状態',
                     detected_count=12, estimated_g=10.8, actual_g=11.0,
                     nutrition=nutrition_for_weight(10.8, 355, 10, 1, 73),
                     image_rgb=np.zeros((30, 30, 3), dtype=np.uint8),
                     db_path=db_file)
    records = list_measurements(db_file)
    assert len(records) == 1
    assert records[0]['detected_count'] == 12
    assert (tmp_path / 'photos' / records[0]['image_filename']).is_file()
