"""Keep cooked recipe portions and weight origins clear while editing a draft."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_weight_origins_and_material_portion_updates_survive_navigation():
    app_dir = str(Path(__file__).resolve().parents[1])
    app = AppTest.from_string(f'''
import sys
sys.path.insert(0, {app_dir!r})
import streamlit as st
from finished_recipe_ui import begin_recipe, render_finished_recipe

measurements = {{
    1: dict(actual_g=92, estimated_g=90),
    2: dict(actual_g=None, estimated_g=65),
}}
draft = st.session_state.setdefault('draft', dict(id='meal', measurement_id=None))
if st.radio('画面', ['料理', 'ホーム'], key='page') == '料理':
    begin_recipe(draft, 'カルボナーラ', measurements)
    render_finished_recipe(draft, measurements)
''').run()

    def caption_contains(text):
        return any(text in caption.value for caption in app.caption)

    def nutrition_text(label):
        return next(item.value for item in app.markdown if f'**{label}**' in item.value)

    assert not app.exception and caption_contains('初期値の80 gは標準レシピの目安')
    dry = 'finished_meal_1_dry'
    assert app.selectbox(key=dry).options == [
        '標準レシピの参考重量を入力', '実測 92 g · #1', '写真からの推定 65 g · #2',
    ]
    app.number_input(key='finished_meal_1_reference').set_value(100).run()
    app.number_input(key='finished_meal_1_portion').set_value(50).run()
    assert '638.3 kcal' in nutrition_text('料理全体')
    assert '319.' in nutrition_text('食べた分')

    app.button(key='finished_meal_1_ingredients_edit_0').click().run()
    next(field for field in app.number_input if field.label == '使った量（g）').set_value(40)
    next(button for button in app.button if button.label == 'この食材を保存').click().run()
    assert '624.0 kcal' in nutrition_text('料理全体')
    assert '312.0 kcal' in nutrition_text('食べた分')
    assert caption_contains('たんぱく質') and caption_contains('脂質') and caption_contains('炭水化物')

    app.radio(key='page').set_value('ホーム').run()
    app.radio(key='page').set_value('料理').run()
    assert not app.exception
    assert app.number_input(key='finished_meal_1_reference').value == 100
    assert app.number_input(key='finished_meal_1_portion').value == 50
    assert app.session_state['draft']['finished_recipe']['ingredients'][0]['amount_g'] == 40
    assert '312.0 kcal' in nutrition_text('食べた分')

    app.selectbox(key=dry).set_value(1).run()
    assert caption_contains('実測重量 92 g')
    app.selectbox(key=dry).set_value(2).run()
    assert caption_contains('写真による推定重量 65 g')
