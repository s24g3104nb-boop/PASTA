"""Standard-recipe editing in the existing meal diary draft."""
from copy import deepcopy

import streamlit as st

from finished_recipe import REFERENCE_DRY_G, calculate_finished_recipe, recipe_dry_basis, standard_rows
from ingredient_ui import render_ingredients
from ui_shell import dry_weight_label

DRY_LABELS = {'reference': '参考重量', 'actual': '実測重量', 'estimated': '記録した概算重量'}


def begin_recipe(draft, dish, measurements):
    if draft.get('finished_recipe', {}).get('dish') == dish:
        return
    weight = draft.get('finished_recipe', {}).get('reference_dry_g') or REFERENCE_DRY_G
    reference = {'reference_dry_g': weight}
    dry = recipe_dry_basis(reference, measurements.get(draft['measurement_id']))
    draft['finished_recipe'] = dict(dish=dish, reference_dry_g=weight,
                                    ingredients=standard_rows(dish, dry['dry_weight_g']),
                                    consumed_percent=100, ingredients_confirmed=False)
    draft['recipe_weight'] = dry['dry_weight_g']
    draft['cooking_session_id'] = None
    draft['recipe_revision'] = draft.get('recipe_revision', 0) + 1


def show_nutrients(label, values):
    st.markdown(f"**{label}**  \n**{values['kcal']:.1f} kcal**")
    st.caption(f"たんぱく質 {values['protein_g']:.1f} g  \n"
               f"脂質 {values['fat_g']:.1f} g  \n"
               f"炭水化物 {values['carb_g']:.1f} g")


def _sync_reference_rows(recipe, before, weight):
    if before != weight:
        previous = standard_rows(recipe['dish'], before)
        updated = standard_rows(recipe['dish'], weight)
        recipe['ingredients'][:] = [updated[previous.index(row)] if row in previous else row
                                  for row in recipe['ingredients']]


def _dry_option(ident, measurements):
    if ident is None:
        return '標準レシピの参考重量を入力'
    measurement = measurements.get(ident)
    if measurement is None:
        return f'保存済みの乾麺 #{ident}'
    return f'{dry_weight_label(measurement)} · #{ident}'


def render_finished_recipe(draft, measurements):
    recipe = draft['finished_recipe']
    key = f"finished_{draft['id']}_{draft.get('recipe_revision', 0)}"
    with st.container(key='finished_recipe_editor'):
        st.caption(f"{recipe['dish']}の標準レシピです。材料と分量は変更できます。")
        selected = draft.get('measurement_id')
        options = [None, *measurements]
        if selected not in options:
            options.append(selected)
        st.selectbox('計算に使う乾麺', options, index=options.index(selected), key=key+'_dry',
                     format_func=lambda ident: _dry_option(ident, measurements))
        draft['measurement_id'] = st.session_state[key+'_dry']
        measurement = measurements.get(draft['measurement_id'])
        if draft['measurement_id'] is None:
            recipe['reference_dry_g'] = st.number_input(
                '乾麺の参考重量（g）', min_value=0.1, value=recipe['reference_dry_g'] or REFERENCE_DRY_G,
                step=1.0, key=key+'_reference')
            st.caption(f'初期値の{REFERENCE_DRY_G:g} gは標準レシピの目安です。'
                       '実測値やAI推定値ではありません。')
        elif measurement is None:
            st.error('乾麺の記録を開けません。参考重量か、別の記録を選んでください。')
            return False
        try:
            dry = recipe_dry_basis(recipe, measurement)
        except ValueError as exc:
            st.error(str(exc))
            return False
        weight = dry['dry_weight_g']
        if measurement:
            st.caption(f"保存した乾麺の{DRY_LABELS[dry['weight_source']]} {weight:g} g")
        before = draft.get('recipe_weight', weight)
        _sync_reference_rows(recipe, before, weight)
        draft['recipe_weight'] = weight
        # Keep the edit buttons reachable during a reference-weight update.
        render_ingredients(standard_rows(recipe['dish'], weight), key+'_ingredients',
                           rows=recipe['ingredients'])
        if draft.get('recipe_confirmed_basis') != (draft['measurement_id'], weight, recipe['ingredients']):
            st.session_state.pop(key+'_confirmed', None)
            recipe['ingredients_confirmed'] = False
        recipe['ingredients_confirmed'] = st.checkbox('実際に使った材料と分量を確認しました',
                                                       value=recipe['ingredients_confirmed'], key=key+'_confirmed')
        draft['recipe_confirmed_basis'] = (draft['measurement_id'], weight, deepcopy(recipe['ingredients']))
        recipe['consumed_percent'] = st.number_input('食べた割合（%）', min_value=1.0, max_value=100.0,
                                                     step=1.0, value=float(recipe['consumed_percent']), key=key+'_portion')
        try:
            result = calculate_finished_recipe(recipe, measurement)
        except ValueError as exc:
            st.error(str(exc))
            return False
        with st.container(key='finished_nutrition_comparison'):
            total, portion = st.columns(2)
            with total, st.container(border=True, key='finished_nutrition_total'):
                show_nutrients('料理全体', result['total'])
            with portion, st.container(border=True, key='finished_nutrition_portion'):
                show_nutrients('食べた分', result['portion'])
        return True
