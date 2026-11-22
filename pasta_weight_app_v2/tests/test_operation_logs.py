"""Prespecified cases in docs/42_OPERATION_LOGS.md, using isolated data only."""
from datetime import date, datetime
from io import BytesIO
import logging
from pathlib import Path
import sqlite3
import sys

import cv2
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import capture_ui
import cooking
import meal_diary
from operation_logs import NUTRITION, RECORDS, WEIGHT, configure_operation_log, log_error, log_info
from pasta_vision import nutrition_for_weight
from storage import init_db, list_measurements, save_measurement


PRIVATE = 'PRIVATE_TEST_VALUE_DO_NOT_LOG'
COUNT_SUCCESS = '検出本数による重量計算 完了'
COUNT_SKIP = '検出結果・校正条件を確認して重量推定を見送り'


@pytest.fixture
def log_path(tmp_path):
    logger = logging.getLogger('pasta_lens.operations')
    previous = list(logger.handlers)
    for handler in previous:
        logger.removeHandler(handler)
    logger.addHandler(logging.NullHandler())
    configure_operation_log(tmp_path)
    try:
        yield tmp_path / 'logs' / 'pasta_lens.log'
    finally:
        for handler in logger.handlers.copy():
            logger.removeHandler(handler)
            handler.close()
        for handler in previous:
            logger.addHandler(handler)


def events(log_path):
    entries = []
    text = log_path.read_text(encoding='utf-8')
    assert PRIVATE not in text and 'Traceback' not in text
    for line in text.splitlines():
        stamp, level, feature, message = line.split(' | ')
        assert stamp.endswith('Z') and datetime.fromisoformat(stamp).utcoffset().total_seconds() == 0
        assert feature in (WEIGHT, NUTRITION, RECORDS)
        assert level in ('INFO', 'ERROR')
        entries.append((level, feature, message))
    return entries


def errors(log_path):
    return [row for row in events(log_path) if row[0] == 'ERROR']


def sample_rgb(*, dense=False):
    rgb = np.full((420, 680, 3), 28, dtype=np.uint8)
    yellow = (230, 180, 65)
    if dense:
        cv2.rectangle(rgb, (220, 55), (275, 365), yellow, -1)
    else:
        for x in range(170, 470, 60):
            cv2.line(rgb, (x, 55), (x, 365), yellow, 6)
    return rgb


def run_count_app(monkeypatch, *, calibration=.68, dense=False):
    product = dict(id=31, name=PRIVATE, g_per_strand=calibration,
                   kcal_100=350, protein_100=12, fat_100=2, carb_100=70)
    monkeypatch.setattr(capture_ui, 'get_product', lambda product_id: product)
    app = AppTest.from_string('''
import streamlit as st
from capture_ui import render_results
render_results(st.session_state['result_request'])
''', default_timeout=15)
    app.session_state['_app_admin'] = True
    app.session_state['result_request'] = dict(
        rgb=sample_rgb(dense=dense), product=product, mode='広げた状態',
        using_demo=False, using_regression=False, parameters={})
    app.run()
    assert not app.exception
    return app


def shown_weights(app):
    return [metric.value for metric in app.metric if metric.label == '推定重量']


def test_w01_calibrated_separated_strands_show_weight(log_path, monkeypatch):
    app = run_count_app(monkeypatch)
    assert shown_weights(app) == ['3.4 g']
    assert events(log_path).count(('INFO', WEIGHT, COUNT_SUCCESS)) == 1
    assert not errors(log_path)


def test_w02_missing_calibration_is_expected(log_path, monkeypatch):
    app = run_count_app(monkeypatch, calibration=0)
    assert not shown_weights(app)
    assert any('未校正' in notice.value for notice in app.warning)
    assert app.button(key='manual_weight_save')
    assert ('INFO', WEIGHT, COUNT_SKIP) in events(log_path)
    assert not errors(log_path)


def test_w03_merged_strands_are_expected_rejection(log_path, monkeypatch):
    app = run_count_app(monkeypatch, dense=True)
    assert not shown_weights(app)
    assert any('重なり' in notice.value for notice in app.warning)
    assert app.button(key='manual_weight_save')
    assert ('INFO', WEIGHT, COUNT_SKIP) in events(log_path)
    assert not errors(log_path)


def test_w04_mask_rerender_preserves_result_without_duplicate_info(log_path, monkeypatch):
    app = run_count_app(monkeypatch)
    app.checkbox(key='show_capture_mask').set_value(True).run()
    assert not app.exception and shown_weights(app) == ['3.4 g']
    assert events(log_path).count(('INFO', WEIGHT, COUNT_SUCCESS)) == 1


def test_w05_detector_failure_is_sanitized_and_recoverable(log_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError(PRIVATE)

    monkeypatch.setattr(capture_ui, 'detect_spread', fail)
    app = run_count_app(monkeypatch)
    assert not shown_weights(app)
    assert any('本数の検出を完了できませんでした' in notice.value for notice in app.error)
    assert app.button(key='manual_weight_save')
    assert errors(log_path) == [('ERROR', WEIGHT, '乾麺の本数検出 失敗')]


def test_w06_unreadable_upload_is_expected(log_path):
    app = AppTest.from_string('from capture_ui import render_capture\nrender_capture()')
    app.session_state['capture_source'] = '写真をアップロード'
    app.session_state['capture_photo'] = BytesIO(PRIVATE.encode())
    app.run()
    assert not app.exception and any('写真を読み込めませんでした' in note.value for note in app.error)
    assert app.button(key='decode_retake')
    assert events(log_path) == [('INFO', WEIGHT, '写真入力を見送り')]


def test_n01_dry_nutrition_scales_all_four_values(log_path):
    assert nutrition_for_weight(50, 350, 12, 2, 70) == dict(kcal=175, protein_g=6, fat_g=1, carb_g=35)
    assert events(log_path) == [('INFO', NUTRITION, '乾麺の栄養計算 完了')]


def test_n02_zero_is_valid_and_negative_is_rejected(log_path):
    assert set(nutrition_for_weight(0, 350, 12, 2, 70).values()) == {0}
    with pytest.raises(ValueError):
        nutrition_for_weight(-1, 350, 12, 2, 70)
    assert events(log_path) == [('INFO', NUTRITION, '乾麺の栄養計算 完了')]


def recipe_data():
    dry = dict(nutrition=dict(kcal=350., protein_g=12., fat_g=2., carb_g=70.))
    return dry, [cooking.Ingredient(PRIVATE, 50, 80, 4, 2, 10)]


def test_n03_ingredients_and_equal_servings(log_path):
    dry, ingredients = recipe_data()
    result = cooking.calculate_meal(dry, ingredients, servings=2)
    assert result['total']['kcal'] == 390
    assert result['portion'] == dict(kcal=195, protein_g=7, fat_g=1.5, carb_g=37.5)
    assert events(log_path) == [('INFO', NUTRITION, '材料と取り分けの栄養計算 完了')]


def test_n04_missing_nutrition_remains_unknown_in_saved_record(log_path, tmp_path):
    db = tmp_path / 'test.sqlite3'
    init_db(db)
    app = AppTest.from_string('''
import streamlit as st
from manual_weight_ui import render_manual_weight
render_manual_weight(st.session_state['product'], db_path=st.session_state['test_db'])
''', default_timeout=15)
    app.session_state['test_db'] = db
    app.session_state['product'] = dict(id=31, name=PRIVATE, g_per_strand=0,
                                       kcal_100=0, protein_100=0, fat_100=0, carb_100=0)
    app.run()
    app.number_input[0].set_value(100.).run()
    assert any('栄養は未計算です' in notice.value for notice in app.info)
    app.button(key='manual_weight_save').click().run()
    assert not app.exception
    record, = list_measurements(db)
    assert record['actual_g'] == 100
    assert all(record[name] is None for name in ('kcal', 'protein_g', 'fat_g', 'carb_g'))
    assert events(log_path) == [('INFO', RECORDS, '測定履歴の保存 完了')]


def test_n05_repeated_fraction_calculation_is_stable(log_path):
    dry, ingredients = recipe_data()
    first = cooking.calculate_meal(dry, ingredients, consumed_fraction=.25)
    second = cooking.calculate_meal(dry, ingredients, consumed_fraction=.25)
    assert first == second and first['portion']['kcal'] == 97.5
    assert dry['nutrition']['kcal'] == 350 and ingredients[0].amount_g == 50
    assert events(log_path) == [('INFO', NUTRITION, '材料と取り分けの栄養計算 完了')] * 2


def test_n06_unexpected_calculation_failure_does_not_leak_details(log_path, monkeypatch):
    def fail(values):
        raise RuntimeError(PRIVATE)

    monkeypatch.setattr(cooking.math, 'fsum', fail)
    dry, ingredients = recipe_data()
    with pytest.raises(RuntimeError, match=PRIVATE):
        cooking.calculate_meal(dry, ingredients, servings=2)
    assert events(log_path) == [('ERROR', NUTRITION, '材料と取り分けの栄養計算 失敗')]


def store_manual(db):
    return save_measurement(product_name=PRIVATE, mode='manual', detected_count=0,
                            estimated_g=None, actual_g=100, nutrition=None, image_rgb=None,
                            db_path=db, estimator='manual_weight')


def store_photo(db, token):
    return save_measurement(product_name=PRIVATE, mode='spread', detected_count=5,
                            estimated_g=3.4, actual_g=None,
                            nutrition=dict(kcal=12, protein_g=.4, fat_g=.1, carb_g=2.4),
                            image_rgb=sample_rgb(), db_path=db, submission_token=token)


def test_r01_manual_weight_persists_without_photo_or_estimate(log_path, tmp_path):
    db = tmp_path / 'test.sqlite3'
    init_db(db)
    ident = store_manual(db)
    saved, = list_measurements(db)
    assert saved['id'] == ident and saved['actual_g'] == 100
    assert saved['estimated_g'] is None and saved['image_filename'] is None
    assert saved['estimator'] == 'manual_weight'
    assert events(log_path) == [('INFO', RECORDS, '測定履歴の保存 完了')]


def test_r02_blank_meal_name_is_not_an_operational_failure(log_path, tmp_path):
    db = tmp_path / 'test.sqlite3'
    with pytest.raises(ValueError, match='料理名'):
        meal_diary.save('local-test', date(2026, 11, 18), None, ' ', db=db)
    assert meal_diary.entries(db) == []
    assert events(log_path) == []


def test_r03_empty_lists_are_valid(log_path, tmp_path):
    db = tmp_path / 'test.sqlite3'
    init_db(db)
    assert list_measurements(db) == [] and meal_diary.entries(db) == []
    assert not errors(log_path)


def test_r04_retry_keeps_one_measurement_and_one_photo(log_path, tmp_path):
    db = tmp_path / 'test.sqlite3'
    init_db(db)
    ident = store_photo(db, 'same-receipt')
    assert store_photo(db, 'same-receipt') == ident
    row, = list_measurements(db)
    assert row['id'] == ident and len(list((tmp_path / 'photos').glob('*.jpg'))) == 1
    assert not errors(log_path)


def test_r05_meal_add_edit_delete_are_recorded(log_path, tmp_path):
    db = tmp_path / 'test.sqlite3'
    day = date(2026, 11, 18)
    original = meal_diary.save('local-test', day, None, PRIVATE, db=db)
    updated = meal_diary.save('local-test', day, None, '変更後', memo=PRIVATE, edit=True,
                              expected_updated_at=original['updated_at'], db=db)
    assert meal_diary.get('local-test', db)['dish_name'] == '変更後'
    meal_diary.delete('local-test', db, expected_updated_at=updated['updated_at'])
    assert meal_diary.entries(db) == []
    assert events(log_path) == [('INFO', RECORDS, '食事記録の保存・編集 完了')] * 2 + [
        ('INFO', RECORDS, '食事記録の削除処理 完了')]


def test_r06_db_failure_preserves_old_row_and_photo(log_path, tmp_path):
    db = tmp_path / 'test.sqlite3'
    init_db(db)
    old_id = store_photo(db, 'older-receipt')
    before = {path.name: path.read_bytes() for path in (tmp_path / 'photos').iterdir()}
    with sqlite3.connect(db) as connection:
        connection.execute(f"""CREATE TRIGGER fail_save BEFORE INSERT ON measurements
                            BEGIN SELECT RAISE(ABORT, '{PRIVATE}'); END""")
    with pytest.raises(sqlite3.IntegrityError, match=PRIVATE):
        store_photo(db, 'new-receipt')
    record, = list_measurements(db)
    assert record['id'] == old_id
    assert {path.name: path.read_bytes() for path in (tmp_path / 'photos').iterdir()} == before
    assert errors(log_path) == [('ERROR', RECORDS, '測定履歴の保存 失敗')]


def test_l01_common_format_and_single_handler_across_configuration(log_path, tmp_path):
    configure_operation_log(tmp_path)
    configure_operation_log(tmp_path)
    log_info(WEIGHT, '乾麺の重量計算 完了')
    log_info(NUTRITION, '乾麺の栄養計算 完了')
    log_error(RECORDS, '測定履歴の保存 失敗')
    assert events(log_path) == [('INFO', WEIGHT, '乾麺の重量計算 完了'),
                                ('INFO', NUTRITION, '乾麺の栄養計算 完了'),
                                ('ERROR', RECORDS, '測定履歴の保存 失敗')]


def test_l02_log_file_failures_do_not_change_calculation_or_commit(log_path, tmp_path, capsys, monkeypatch):
    db = tmp_path / 'test.sqlite3'
    init_db(db)
    unavailable = tmp_path / PRIVATE
    unavailable.write_text('occupied', encoding='utf-8')
    configure_operation_log(unavailable)
    assert nutrition_for_weight(50, 350, 12, 2, 70)['kcal'] == 175

    configure_operation_log(tmp_path)
    logger = logging.getLogger('pasta_lens.operations')
    handler = next(handler for handler in logger.handlers if isinstance(handler, logging.FileHandler))

    def fail(record):
        raise OSError(PRIVATE)

    monkeypatch.setattr(handler, 'shouldRollover', fail)
    ident = store_manual(db)
    assert list_measurements(db)[0]['id'] == ident
    stderr = capsys.readouterr().err
    assert stderr.count('操作ログの保存先を利用できません') == 2
    assert PRIVATE not in stderr and 'Traceback' not in stderr
