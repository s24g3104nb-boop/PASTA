"""Streamlit上で材料編集から料理保存までの契約を確認する。"""
import json
from pathlib import Path
import sqlite3

from streamlit.testing.v1 import AppTest


def test_material_entry_and_600_kcal_portion_saved_in_ui(tmp_path):
    app_dir = str(Path(__file__).resolve().parents[1])
    db = str(tmp_path / 'cooking_ui.sqlite3')
    script = f'''
import sys
sys.path.insert(0, {app_dir!r})
import numpy as np
from storage import init_db, list_measurements, save_measurement
from cooking_ui import render_cooking_tab
db = {db!r}
init_db(db)
if not list_measurements(db):
    save_measurement(product_name='デモ用UIテスト', mode='spread', detected_count=100,
        estimated_g=100, actual_g=100,
        nutrition={{'kcal':350,'protein_g':12,'fat_g':2,'carb_g':70}},
        image_rgb=np.zeros((10,10,3), dtype=np.uint8), db_path=db)
render_cooking_tab(db)
'''
    at = AppTest.from_string(script).run(timeout=15)
    assert not at.exception
    assert at.selectbox[0].value is None
    at.selectbox[0].set_value(1).run(timeout=15)
    assert not at.exception
    rows = [
        {'name': '卵', 'amount_g': 50, 'kcal_100': 140, 'protein_100': 10, 'fat_100': 8, 'carb_100': 0},
        {'name': 'ベーコン', 'amount_g': 25, 'kcal_100': 400, 'protein_100': 20, 'fat_100': 30, 'carb_100': 0},
        {'name': 'チーズ', 'amount_g': 20, 'kcal_100': 400, 'protein_100': 30, 'fat_100': 30, 'carb_100': 0},
    ]
    at.selectbox(key='ingredient_input_style').set_value('表でまとめて入力').run(timeout=15)
    # AppTestはdata_editorの直接編集APIを持たないため、同じウィジェット編集状態を与える。
    at.session_state['cooking_ingredients_1'] = {
        'edited_rows': {}, 'added_rows': rows, 'deleted_rows': []}
    at.checkbox[0].check()
    at.checkbox[1].check()
    at.number_input[0].set_value(2)
    at.button[0].click().run(timeout=15)
    assert not at.exception
    assert not at.error
    assert any('保存しました' in item.value for item in at.success)
    with sqlite3.connect(db) as con:
        serialized = con.execute('SELECT calculation_json FROM cooking_sessions').fetchone()[0]
    result = json.loads(serialized)
    assert result['total']['kcal'] == 600
    assert result['portion']['kcal'] == 300
    assert any('保存結果はCG／デモ製品' in item.value for item in at.warning)
    assert any('実測重量' in item.value and '100.00g' in item.value for item in at.caption)


def test_recipe_template_prefill_confirmation_and_approximation_history(tmp_path):
    app_dir = str(Path(__file__).resolve().parents[1])
    db = str(tmp_path / 'template_ui.sqlite3')
    script = f'''
import sys
sys.path.insert(0, {app_dir!r})
import numpy as np
from storage import init_db, list_measurements, save_measurement
from cooking import Ingredient
from cooking_storage import list_recipe_templates, save_recipe_template
from cooking_ui import render_cooking_tab
db = {db!r}
init_db(db)
if not list_measurements(db):
    save_measurement(product_name='試験製品', mode='spread', detected_count=100,
        estimated_g=100, actual_g=100,
        nutrition={{'kcal':350,'protein_g':12,'fat_g':2,'carb_g':70}},
        image_rgb=np.zeros((10,10,3), dtype=np.uint8), db_path=db)
if not list_recipe_templates(db):
    save_recipe_template(name='試験レシピ', selected_dish='カルボナーラ',
        ingredients=[Ingredient('卵', 50, 140, 10, 8, 0),
                     Ingredient('ベーコン', 25, 400, 20, 30, 0),
                     Ingredient('チーズ', 20, 400, 30, 30, 0)],
        reference_dry_g=100, default_servings=2, source='試験値・基準100g分', db_path=db)
render_cooking_tab(db)
'''
    at = AppTest.from_string(script).run(timeout=15)
    at.selectbox[0].set_value(1).run(timeout=15)
    at.selectbox[2].set_value(1).run(timeout=15)
    assert not at.exception
    at.selectbox(key='ingredient_input_style').set_value('表でまとめて入力').run(timeout=15)
    assert at.number_input[0].value == 2
    at.checkbox[0].check()
    at.checkbox[1].check()
    at.button[0].click().run(timeout=15)
    assert any('仮分量を確認' in item.value for item in at.error)
    at.checkbox[2].check()
    at.button[0].click().run(timeout=15)
    assert not at.exception
    assert not at.error
    with sqlite3.connect(db) as con:
        result = json.loads(con.execute('SELECT calculation_json FROM cooking_sessions').fetchone()[0])
    assert result['total']['kcal'] == 600
    assert result['portion']['kcal'] == 300
    assert result['nutrition_basis'] == 'template_estimate'
    assert result['recipe_template']['name'] == '試験レシピ'
    assert any('利用者登録レシピに基づく概算' in item.value for item in at.warning)
