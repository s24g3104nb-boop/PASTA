from pathlib import Path
import sqlite3
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cooking import Ingredient, calculate_meal, dry_snapshot, template_ingredients
from cooking_storage import (delete_recipe_template, init_cooking_db, list_cooking_sessions,
                             list_recipe_templates, save_cooking_session, save_recipe_template)
from dish_classifier import UntrainedDishClassifier
from pasta_vision import nutrition_for_weight
from storage import init_db, list_measurements, list_products, save_measurement


def dry_record(**changes):
    row = {'id': 1, 'product_name': 'テスト製品', 'estimated_g': 100, 'actual_g': 100,
           'kcal': 350, 'protein_g': 12, 'fat_g': 2, 'carb_g': 70}
    row.update(changes)
    return row


def ingredients():
    # 計算試験用の架空値。食材の実際の栄養値ではない。
    return [Ingredient('卵', 50, 140, 10, 8, 0),
            Ingredient('ベーコン', 25, 400, 20, 30, 0),
            Ingredient('チーズ', 20, 400, 30, 30, 0)]


def test_meal_acceptance_600_kcal_and_portion():
    result = calculate_meal(dry_snapshot(dry_record()), ingredients(), servings=2)
    assert result['dry']['kcal'] == 350
    assert result['additional']['kcal'] == 250
    assert result['total']['kcal'] == 600
    assert result['portion']['kcal'] == 300
    assert result['total']['protein_g'] == 28
    changed = ingredients()
    changed[0] = Ingredient('卵', 100, 140, 10, 8, 0)
    assert calculate_meal(dry_snapshot(dry_record()), changed,
                          consumed_fraction=0.25)['portion']['kcal'] == 167.5


@pytest.mark.parametrize('invalid', [None, '', float('nan'), float('inf'), -1, True])
def test_missing_or_invalid_ingredient_amount_blocks_total(invalid):
    with pytest.raises(ValueError):
        calculate_meal(dry_snapshot(dry_record()),
                       [Ingredient('油', invalid, 900, 0, 100, 0)], servings=1)


@pytest.mark.parametrize('invalid', [None, float('nan'), -5])
def test_unknown_nutrient_not_treated_as_zero(invalid):
    with pytest.raises(ValueError):
        Ingredient('ソース', 10, invalid, 0, 0, 0).nutrition()


@pytest.mark.parametrize('kwargs', [{}, {'servings': 0}, {'servings': 1.5},
                                   {'consumed_fraction': 0}, {'consumed_fraction': 1.5},
                                   {'servings': 2, 'consumed_fraction': 0.5}])
def test_invalid_portion(kwargs):
    with pytest.raises(ValueError):
        calculate_meal(dry_snapshot(dry_record()), [], **kwargs)


def test_actual_weight_and_product_snapshot_preferred():
    record = dry_record(estimated_g=50, actual_g=100, kcal=175,
                        kcal_100=355, protein_100=13, fat_100=2, carb_100=71,
                        source_kind='real')
    result = dry_snapshot(record)
    assert result['dry_weight_g'] == 100
    assert result['weight_source'] == 'actual'
    assert result['nutrition']['kcal'] == 355
    assert result['source_kind'] == 'real'


def test_legacy_snapshot_rounding_and_provenance():
    result = dry_snapshot(dry_record(estimated_g=50, actual_g=100, kcal=175))
    assert result['nutrition']['kcal'] == 350
    assert result['source_kind'] == 'unknown'
    assert '丸め誤差' in result['nutrition_source']
    assert dry_snapshot(dry_record(actual_g=None))['weight_source'] == 'estimated'
    with pytest.raises(ValueError):
        dry_snapshot(dry_record(estimated_g=0))


def test_untrained_classifier_never_invents_prediction():
    result = UntrainedDishClassifier().predict(np.zeros((10, 10, 3)))
    assert result.status == 'untrained'
    assert result.label is None
    assert result.score is None


def setup_db(tmp_path):
    db = tmp_path / 'legacy.sqlite3'
    init_db(db)
    save_measurement(product_name='保存済み製品', mode='広げた状態', detected_count=100,
                     estimated_g=100, actual_g=100,
                     nutrition=nutrition_for_weight(100, 350, 12, 2, 70),
                     image_rgb=np.zeros((10, 10, 3), dtype=np.uint8), db_path=db)
    return db, list_measurements(db)[0]['id']


def test_additive_migration_preserves_legacy_and_saved_snapshot(tmp_path):
    db, measurement_id = setup_db(tmp_path)
    original_records = list_measurements(db)
    original_products = list_products(db)
    init_cooking_db(db)
    init_cooking_db(db)
    saved_id = save_cooking_session(measurement_id=measurement_id, selected_dish='カルボナーラ',
                                   confirmed_same_batch=True, ingredients=ingredients(),
                                   servings=2, db_path=db)
    assert list_measurements(db) == original_records
    assert list_products(db) == original_products
    with sqlite3.connect(db) as con:
        con.execute('UPDATE measurements SET kcal=999 WHERE id=?', (measurement_id,))
        assert con.execute('SELECT COUNT(*) FROM cooking_ingredients').fetchone()[0] == 3
    session = list_cooking_sessions(db)[0]
    assert session['id'] == saved_id
    assert session['measurement_id'] == measurement_id
    assert session['classifier_status'] == 'untrained/manual'
    assert session['calculation']['total']['kcal'] == 600


def test_batch_confirmation_and_missing_record_required(tmp_path):
    db, measurement_id = setup_db(tmp_path)
    common = dict(selected_dish='その他／不明', ingredients=[], servings=1, db_path=db)
    with pytest.raises(ValueError, match='同一調理バッチ'):
        save_cooking_session(measurement_id=measurement_id,
                             confirmed_same_batch=False, **common)
    with pytest.raises(ValueError, match='ありません'):
        save_cooking_session(measurement_id=999, confirmed_same_batch=True, **common)
    assert list_cooking_sessions(db) == []


def test_original_schema_without_new_provenance_columns(tmp_path):
    db = tmp_path / 'original.sqlite3'
    with sqlite3.connect(db) as con:
        con.execute('''CREATE TABLE measurements (
            id INTEGER PRIMARY KEY, product_name TEXT, estimated_g REAL, actual_g REAL,
            kcal REAL, protein_g REAL, fat_g REAL, carb_g REAL)''')
        con.execute("INSERT INTO measurements VALUES(1, '旧製品', 50, 100, 175, 6, 1, 35)")
        before = con.execute('PRAGMA table_info(measurements)').fetchall()
    save_cooking_session(measurement_id=1, selected_dish='その他／不明',
                         confirmed_same_batch=True, ingredients=[], servings=1, db_path=db)
    with sqlite3.connect(db) as con:
        assert con.execute('PRAGMA table_info(measurements)').fetchall() == before
    saved = list_cooking_sessions(db)[0]
    assert saved['calculation']['total']['kcal'] == 350
    assert saved['dry_snapshot']['source_kind'] == 'unknown'


def test_invalid_material_does_not_save_partial_session(tmp_path):
    db, measurement_id = setup_db(tmp_path)
    with pytest.raises(ValueError):
        save_cooking_session(measurement_id=measurement_id, selected_dish='カルボナーラ',
                             confirmed_same_batch=True,
                             ingredients=[Ingredient('油', None, 900, 0, 100, 0)],
                             servings=1, db_path=db)
    assert list_cooking_sessions(db) == []


def register_template(db, **changes):
    values = dict(name='利用者レシピ', selected_dish='カルボナーラ', ingredients=ingredients(),
                  reference_dry_g=100, default_servings=2, source='試験用製品ラベルと利用者分量', db_path=db)
    values.update(changes)
    return save_recipe_template(**values)


def test_recipe_template_reuse_scale_snapshot_and_delete(tmp_path):
    db, measurement_id = setup_db(tmp_path)
    template_id = register_template(db)
    template = list_recipe_templates(db)[0]
    assert template['id'] == template_id
    assert template['reference_dry_g'] == 100
    assert template_ingredients(template, 200)[0].amount_g == 100
    save_cooking_session(measurement_id=measurement_id, selected_dish='カルボナーラ',
                         confirmed_same_batch=True, ingredients=template_ingredients(template, 100),
                         servings=2, recipe_template_id=template_id, db_path=db)
    saved = list_cooking_sessions(db)[0]['calculation']
    assert saved['total']['kcal'] == 600
    assert saved['portion']['kcal'] == 300
    assert saved['nutrition_basis'] == 'template_estimate'
    assert saved['recipe_template']['source'] == '試験用製品ラベルと利用者分量'
    delete_recipe_template(template_id, db_path=db)
    assert list_recipe_templates(db) == []
    assert list_cooking_sessions(db)[0]['calculation'] == saved


@pytest.mark.parametrize('changes', [{'name': ''}, {'source': ''}, {'reference_dry_g': 0},
                                    {'default_servings': 0}, {'ingredients': []},
                                    {'ingredients': [Ingredient('油', None, 900, 0, 100, 0)]}])
def test_recipe_template_requires_known_inputs(tmp_path, changes):
    with pytest.raises(ValueError):
        register_template(tmp_path / 'template.sqlite3', **changes)


def test_template_missing_or_mismatched_dish_refused(tmp_path):
    db, measurement_id = setup_db(tmp_path)
    template_id = register_template(db)
    for selected_dish, recipe_id in [('ボロネーゼ', template_id), ('カルボナーラ', 999)]:
        with pytest.raises(ValueError):
            save_cooking_session(measurement_id=measurement_id, selected_dish=selected_dish,
                                 confirmed_same_batch=True, ingredients=ingredients(), servings=1,
                                 recipe_template_id=recipe_id, db_path=db)
    assert list_cooking_sessions(db) == []


def test_same_measurement_reuse_refused_and_old_duplicates_preserved(tmp_path):
    db, measurement_id = setup_db(tmp_path)
    values = dict(measurement_id=measurement_id, selected_dish='カルボナーラ',
                  confirmed_same_batch=True, ingredients=ingredients(), servings=2, db_path=db)
    save_cooking_session(**values)
    with pytest.raises(ValueError, match='二重計上'):
        save_cooking_session(**values)
    with sqlite3.connect(db) as con:
        con.execute('''INSERT INTO cooking_sessions
            (measurement_id, created_at, selected_dish, same_batch_confirmed, classifier_status,
             dry_snapshot_json, calculation_json)
            SELECT measurement_id, created_at, selected_dish, same_batch_confirmed, classifier_status,
                   dry_snapshot_json, calculation_json FROM cooking_sessions''')
    rows = list_cooking_sessions(db)
    assert len(rows) == 2
    assert all(row['duplicate_measurement_count'] == 2 for row in rows)
    with pytest.raises(ValueError, match='二重計上'):
        save_cooking_session(**values)
    assert len(list_cooking_sessions(db)) == 2


def test_concurrent_saves_do_not_double_count(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    db, measurement_id = setup_db(tmp_path)
    init_cooking_db(db)
    def save():
        try:
            return save_cooking_session(measurement_id=measurement_id, selected_dish='カルボナーラ',
                                        confirmed_same_batch=True, ingredients=ingredients(),
                                        servings=2, db_path=db)
        except ValueError as exc:
            return str(exc)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: save(), range(2)))
    assert sum(isinstance(result, int) for result in results) == 1
    assert len(list_cooking_sessions(db)) == 1
