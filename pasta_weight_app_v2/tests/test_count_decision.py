"""Synthetic regression checks for display failures, not real-photo accuracy."""
import cv2
import numpy as np
import pytest
from pasta_weight_app_v2.pasta_vision import detect_spread, detect_bundle_ends, image_quality_warnings
from pasta_weight_app_v2.measurement_diagnostics import decide_count_weight, strand_calibration
from pasta_weight_app_v2.storage import init_db, save_product, get_product, list_products

PRODUCT = dict(id=2, name='検証製品', g_per_strand=.68)


def scene(background=42):
    rgb = np.full((480,640,3), background, np.uint8)
    for i in range(6):
        rgb[90+i*40:94+i*40,100:530] = (220,180,40)
    return rgb


def decision(rgb, product=PRODUCT):
    return decide_count_weight(product, detect_spread(rgb), mode='広げた状態')


def test_separated_and_black_background_are_measurable():
    for level in (42, 3):
        rgb = scene(level)
        assert image_quality_warnings(rgb) == []
        value = decision(rgb)
        assert value.automatic_count == 6
        assert value.weight_g == pytest.approx(4.08)


def test_small_border_speck_does_not_hide_valid_strands_but_crop_does():
    rgb = scene()
    rgb[0,0] = (220,180,40)
    assert decision(rgb).weight_g == pytest.approx(4.08)
    rgb[90:94,:100] = (220,180,40)
    assert 'cropped' in decision(rgb).reason_codes
    assert decision(rgb).weight_g is None


def test_dense_region_and_closing_merger_cannot_be_counted_as_one():
    rgb = scene()
    rgb[90:295,100:530] = (220,180,40)
    assert 'merged' in decision(rgb).reason_codes
    rgb = np.full((480,640,3),42,np.uint8)
    rgb[120:124,100:530] = (220,180,40)
    rgb[125:129,100:530] = (220,180,40)
    assert detect_spread(rgb).count == 1
    assert 'merged' in decision(rgb).reason_codes


@pytest.mark.parametrize('calibration', [None, 0, -1, float('nan'), float('inf'), True, 'bad'])
def test_bad_calibration_has_no_weight(calibration):
    product = dict(PRODUCT, g_per_strand=calibration)
    assert strand_calibration(product) is None
    value = decision(scene(), product)
    assert value.weight_g is None and 'uncalibrated' in value.reason_codes


def test_demo_real_image_no_pasta_and_explicit_manual_count():
    result = detect_spread(scene())
    demo = dict(PRODUCT, name='デモ用（仮）', g_per_strand=1.)
    assert decide_count_weight(demo,result,mode='広げた状態').weight_g is None
    assert decide_count_weight(demo,result,mode='広げた状態',using_demo=True).weight_g == 6.
    empty = np.full((480,640,3),42,np.uint8)
    assert 'no_count' in decision(empty).reason_codes
    assert decide_count_weight(PRODUCT,result,mode='広げた状態',manual_count=0).weight_g is None
    assert decide_count_weight(PRODUCT,result,mode='広げた状態',manual_count=8).weight_g == pytest.approx(5.44)


def test_end_face_mode_cannot_bypass_dense_rejection():
    rgb = scene()
    rgb[90:295,100:530] = (220,180,40)
    result = detect_bundle_ends(rgb,min_radius=8,max_radius=27,min_distance=32,hough_sensitivity=18)
    assert decide_count_weight(PRODUCT,result,mode='束の断面').weight_g is None


def test_saved_product_row_and_optimistic_update_protect_data(tmp_path):
    db = tmp_path/'test.sqlite3'
    init_db(db)
    saved = save_product('検証製品',.68,350,12,2,70,db_path=db)
    assert get_product(saved['id'],db) == saved
    updated = save_product('検証製品',.75,350,12,2,70,product_id=saved['id'],expected_product=saved,db_path=db)
    with pytest.raises(ValueError,match='変更'):
        save_product('検証製品',.88,350,12,2,70,product_id=saved['id'],expected_product=saved,db_path=db)
    assert get_product(saved['id'],db) == updated
    demo = list_products(db)[0]
    with pytest.raises(ValueError):
        save_product('実製品に見せたデモ',.68,350,12,2,70,product_id=demo['id'],db_path=db)
    assert get_product(demo['id'],db) == demo
