"""完成料理タブ。画像分類、材料入力、計算・保存を分離する。"""
from __future__ import annotations

import sqlite3

import pandas as pd
import streamlit as st

from cooking import Ingredient, calculate_meal, dry_snapshot, template_ingredients
from cooking_storage import (delete_recipe_template, list_cooking_sessions, list_recipe_templates,
                             save_cooking_session, save_recipe_template)
from dish_classifier import DISH_NAMES, INGREDIENT_HINTS
from standard_recipes import recipe_choices
from storage import DEFAULT_DB, list_measurements
from ingredient_ui import render_ingredients
from ui_shell import dry_weight_label, start_capture


def _diary_link(session, key):
    from diary_ui import start_diary
    st.button('この料理を食事日記に記録する', key=key,
              on_click=start_diary,
              kwargs={'measurement_id': session['measurement_id'],
                      'cooking_session_id': session['id'], 'confirmed_dish': session.get('selected_dish'),
                      'from_cooking': True}, width='stretch')


def _restore_controls():
    # Widget values disappear when another page is rendered. Keep a separate
    # draft so visiting a record or settings does not discard cooking input.
    for key, value in st.session_state.get('cooking_control_draft', {}).items():
        if key not in st.session_state:
            st.session_state[key] = value
    incoming = st.session_state.pop('cooking_source_measurement_id', None)
    if incoming is not None:
        st.session_state['cooking_measurement'] = incoming
    dish = st.session_state.pop('cooking_incoming_dish', None)
    if dish in DISH_NAMES:
        st.session_state['cooking_dish'] = dish


def _remember_controls():
    prefixes = ('cooking_batch_', 'cooking_template_', 'cooking_servings_',
                'cooking_complete_', 'cooking_division_', 'cooking_fraction_',
                'cooking_recipe_name_', 'cooking_recipe_source_',
                'cooking_reference_g_', 'cooking_reference_servings_')
    singles = {'cooking_measurement', 'cooking_dish', 'ingredient_input_style'}
    st.session_state['cooking_control_draft'] = {
        key: value for key, value in st.session_state.items()
        if key in singles or key.startswith(prefixes)
    }


def _number_input(label, *, key, initial, **options):
    # Initialize the draft only once, including when the widget is restored.
    st.session_state.setdefault(key, initial)
    return st.number_input(label, key=key, **options)


def render_cooking_tab(db_path=DEFAULT_DB):
    _restore_controls()
    st.subheader('まず、乾麺の記録を選ぼう')
    measurements = [row for row in list_measurements(db_path)
                    if st.session_state.get('_app_admin') or row['source_kind'] != 'synthetic']
    if not measurements:
        st.caption('使った乾麺の重さを保存すると、栄養を計算できます。')
        st.button('乾麺を撮影する', type='primary', width='stretch', key='cooking_capture',
                  on_click=start_capture, args=('写真をアップロード',))
        return
    mapping = {item['id']: item for item in measurements}
    if st.session_state.get('cooking_measurement') not in mapping:
        st.session_state.pop('cooking_measurement', None)
    measurement_id = st.selectbox(
        'この料理に使った乾麺の記録', [None] + list(mapping),
        format_func=lambda key: '履歴を選択してください' if key is None else
        f"ID {key} | {mapping[key]['created_at']} | {mapping[key]['product_name']} | {dry_weight_label(mapping[key])}",
        key='cooking_measurement')
    if measurement_id is None:
        _remember_controls()
        return
    record = mapping[measurement_id]
    try:
        snapshot = dry_snapshot(record)
    except ValueError as exc:
        st.error(str(exc))
        _diary_link({'measurement_id': measurement_id, 'id': None}, 'cooking_unknown_to_diary')
        return
    kind = snapshot['source_kind']
    if kind == 'synthetic' or record['product_name'].startswith('デモ用'):
        st.warning('この履歴はCGまたはデモ製品の値です。料理計算もデモで、実際の食事管理には使えません。')
    elif kind == 'unknown':
        st.warning('旧履歴のため写真の来歴が不明です。実写・製品情報を確認してください。')
    st.write(f'使用する乾麺　{dry_weight_label(record)}')
    st.caption(snapshot['nutrition_source'])
    sessions = [r for r in list_cooking_sessions(db_path) if st.session_state.get('_app_admin') or r['dry_snapshot'].get('source_kind') != 'synthetic']
    used_sessions = [item for item in sessions if item['measurement_id'] == measurement_id]
    if used_sessions:
        st.warning('この乾麺履歴は料理ID ' + '、'.join(str(item['id']) for item in used_sessions) +
                   ' に使用済みです。乾麺の二重計上を防ぐため、新しい料理として保存できません。')
        _diary_link(used_sessions[0], 'cooking_used_to_diary')
    same_batch = st.checkbox('この記録の乾麺を今回の料理に使いました',
                             key=f'cooking_batch_{measurement_id}')
    selected_dish = st.selectbox('料理名（変更できます）', DISH_NAMES, key='cooking_dish')
    hints = INGREDIENT_HINTS.get(selected_dish, ())
    if hints:
        st.caption('材料確認のヒント: ' + '、'.join(hints) + '。実際に使ったものだけを入力してください。')
    st.caption('写真から材料の分量は分かりません。実際に使った量（g）を入力してください。')
    templates = {item['id']: item for item in list_recipe_templates(db_path)
                 if item['selected_dish'] == selected_dish}
    all_templates = {**templates, **recipe_choices(selected_dish)}
    template_id = st.selectbox('標準レシピ・登録済みレシピ（任意）', [None] + list(all_templates),
                               format_func=lambda key: '使用しない：今回の材料を入力' if key is None
                               else all_templates[key]['name'], key=f'cooking_template_{selected_dish}')
    template = all_templates.get(template_id)
    if template:
        st.warning('レシピテンプレートを参考にした概算です。材料量を乾麺gの比率で仮入力しています。'
                   '今回実際に使った材料・量を確認・修正してください。')
        st.caption(f"出典・前提: {template['source']} / 基準乾麺 {template['reference_dry_g']:g}g / "
                   f"基準 {template['default_servings']}人分")
        initial_rows = [vars(item) for item in template_ingredients(template, snapshot['dry_weight_g'])]
        editor_key = f'cooking_ingredients_{measurement_id}_template_{template_id}'
    else:
        initial_rows = []
        editor_key = f'cooking_ingredients_{measurement_id}'
    initial_frame = pd.DataFrame(initial_rows) if initial_rows else pd.DataFrame({
        'name': pd.Series(dtype='str'), **{key: pd.Series(dtype='float64')
        for key in ('amount_g', 'kcal_100', 'protein_100', 'fat_100', 'carb_100')}})
    st.subheader('使った食材と、食べる分量')
    input_style=st.selectbox('食材の入力方法',['食材ごとに入力','表でまとめて入力'],key='ingredient_input_style')
    if input_style=='食材ごとに入力':
        frame=render_ingredients(initial_rows,editor_key)
    # Individual changes reach the session immediately, including before a
    # navigation click; the explicit button still controls database writes.
    with st.container():
        if input_style=='表でまとめて入力':
            frame = st.data_editor(
                initial_frame, num_rows='dynamic', hide_index=True, key=editor_key,
                column_config={
                    'name': st.column_config.TextColumn('材料名', required=True),
                    'amount_g': st.column_config.NumberColumn('使用量(g)', min_value=0.0, required=True),
                    'kcal_100': st.column_config.NumberColumn('kcal/100g', min_value=0.0, required=True),
                    'protein_100': st.column_config.NumberColumn('P(g)/100g', min_value=0.0, required=True),
                    'fat_100': st.column_config.NumberColumn('F(g)/100g', min_value=0.0, required=True),
                    'carb_100': st.column_config.NumberColumn('C(g)/100g', min_value=0.0, required=True),
                })
        complete = st.checkbox('使った追加材料・ソース・油・調味料をすべて入力しました（未入力分は合計に含まれません）',
                               key=f'cooking_complete_{measurement_id}')
        template_confirmed = True
        if template:
            template_confirmed = st.checkbox('テンプレートの仮分量を確認・修正しました（概算として保存）',
                                              key=f'cooking_template_confirm_{measurement_id}_{template_id}')
        division = st.radio('取り分け方法', ['等分人数', '自分が食べる割合'], horizontal=True,
                            key=f'cooking_division_{measurement_id}')
        servings_input = _number_input('等分する人数', min_value=1, step=1,
                                       initial=template['default_servings'] if template else 1,
                                       key=f'cooking_servings_{measurement_id}_{template_id}')
        fraction_input = _number_input('食べる割合（1=全部、0.5=半分）',
                                       min_value=0.001, max_value=1.0, initial=1.0, step=0.05,
                                       key=f'cooking_fraction_{measurement_id}')
        preview_click = st.button('栄養をプレビュー', key=f'cooking_preview_{measurement_id}')
        save = st.button('入力を確認して計算・保存', disabled=bool(used_sessions),
                         key=f'cooking_save_{measurement_id}')
        st.caption('この材料入力を再利用したい場合は、分量と出典をレシピテンプレートとして登録できます。')
        with st.expander('この材料をレシピとして登録（任意）'):
            template_name = st.text_input('登録するテンプレート名', key=f'cooking_recipe_name_{measurement_id}')
            template_source = st.text_input('材料・栄養値の出典、分量の前提', key=f'cooking_recipe_source_{measurement_id}')
            reference_dry_g = _number_input('この入力材料に対応する基準乾麺g', min_value=0.1, step=0.1,
                                            initial=float(snapshot['dry_weight_g']),
                                            key=f'cooking_reference_g_{measurement_id}')
            reference_servings = _number_input('テンプレートの基準人数', min_value=1, step=1,
                                               initial=template['default_servings'] if template else 1,
                                               key=f'cooking_reference_servings_{measurement_id}')
            register_template = st.button('入力材料をレシピテンプレートとして登録',
                                          key=f'cooking_register_template_{measurement_id}')
    if register_template:
        try:
            save_recipe_template(name=template_name, selected_dish=selected_dish,
                                 ingredients=[Ingredient(**row) for row in frame.to_dict('records')],
                                 reference_dry_g=float(reference_dry_g),
                                 default_servings=int(reference_servings), source=template_source, db_path=db_path)
            st.success('利用者レシピテンプレートを登録しました。次回の選択肢に表示されます。')
        except (ValueError, TypeError, OverflowError, sqlite3.Error) as exc:
            st.error(f'テンプレートを登録できませんでした: {exc}')
    if preview_click or save:
        try:
            if save and not same_batch:
                raise ValueError('同一調理バッチの確認が必要です。')
            if not complete:
                raise ValueError('追加材料の入力が完了してから計算してください。')
            if save and not template_confirmed:
                raise ValueError('テンプレートの仮分量を確認・修正してください。')
            ingredients = [Ingredient(**row) for row in frame.to_dict('records')]
            servings = int(servings_input) if division == '等分人数' else None
            fraction = float(fraction_input) if division != '等分人数' else None
            result = calculate_meal(snapshot, ingredients, servings=servings,
                                    consumed_fraction=fraction,
                                    nutrition_basis='template_estimate' if template else 'confirmed_amounts')
            if template:
                result['recipe_template'] = template
            if preview_click:
                _show_result(result)
            else:
                session_id = save_cooking_session(
                    measurement_id=int(measurement_id), selected_dish=selected_dish,
                    confirmed_same_batch=same_batch, ingredients=ingredients,
                    servings=servings, consumed_fraction=fraction, recipe_template_id=template_id,
                    classifier_status='user_confirmed_vision' if selected_dish == st.session_state.get('cooking_classified_dish')
                    else 'untrained/manual', db_path=db_path)
                st.success(f'料理の計算条件を保存しました（ID {session_id}）。')
                _show_result(result)
                _diary_link({'id': session_id, 'measurement_id': measurement_id, 'selected_dish': selected_dish},
                            'cooking_saved_to_diary')
        except (ValueError, TypeError, OverflowError) as exc:
            st.error(f'計算・保存できませんでした: {exc}')
        except sqlite3.Error:
            st.error('料理を保存できませんでした。入力内容を保持しています。保存先を確認し、もう一度お試しください。')
    if templates:
        with st.expander('利用者レシピテンプレートの削除'):
            delete_id = st.selectbox('削除するテンプレート', list(templates),
                                     format_func=lambda key: templates[key]['name'], key='cooking_delete_template')
            confirm_delete = st.checkbox('テンプレートのみを削除する（保存済み料理の計算条件は残ります）')
            if st.button('テンプレートを削除', disabled=not confirm_delete):
                delete_recipe_template(delete_id, db_path=db_path)
                st.rerun()
    _remember_controls()


def _show_result(result):
    with st.container(border=True):
        st.metric('食べる分のカロリー', f"{result['portion']['kcal']:.0f} kcal")
        columns=st.columns(3)
        for column,label,key in zip(columns,['たんぱく質','脂質','炭水化物'],['protein_g','fat_g','carb_g']):
            column.metric(label,f"{result['portion'][key]:.1f} g")
    if result.get('nutrition_basis') == 'template_estimate':
        st.warning('参考レシピに基づく概算です。今回の材料使用量を完全に実測した値ではありません。')
        template = result.get('recipe_template', {})
        if template:
            st.caption(f"参考レシピ: {template['name']} / 出典・前提: {template['source']} / "
                       f"基準乾麺 {template['reference_dry_g']:g}g、{template['default_servings']}人分")
    else:
        st.caption('利用者が確認した材料使用量に基づく計算です。乾麺重量や栄養表示自体の誤差は含まれます。')
    labels = {'dry': '乾麺分', 'additional': '追加材料分',
              'total': '料理全体', 'portion': '選択した取り分け分'}
    st.dataframe(pd.DataFrame([
        {'内訳': label, 'kcal': round(result[key]['kcal'], 1),
         'P(g)': round(result[key]['protein_g'], 1), 'F(g)': round(result[key]['fat_g'], 1),
         'C(g)': round(result[key]['carb_g'], 1)} for key, label in labels.items()
    ]), hide_index=True)
    st.caption(f"食べる割合: {result['consumed_fraction']:.3f}。{result['assumptions']}")


def render_cooking_page(db_path=DEFAULT_DB):
    section=st.selectbox('料理メニュー',['料理を作る','保存した料理・レシピ'],key='cooking_section')
    if section=='料理を作る':
        with st.container(key='cooking_create'):
            render_cooking_tab(db_path)
    else:
        with st.container(key='cooking_saved'):
            _render_saved_cooking(db_path)


def _render_saved_cooking(db_path):
    from ui_shell import empty_state
    sessions = [r for r in list_cooking_sessions(db_path) if st.session_state.get('_app_admin') or r['dry_snapshot'].get('source_kind') != 'synthetic']
    templates=list_recipe_templates(db_path)
    if not sessions and not templates:
        empty_state('いつものひと皿を保存しよう','乾麺と食材を入力して、料理やレシピを保存できます。')
    for session in sessions:
        with st.container(border=True):
            st.subheader(session['selected_dish'])
            st.caption(session['created_at'])
            snapshot = session['dry_snapshot']
            if snapshot.get('source_kind') == 'synthetic' or snapshot['product_name'].startswith('デモ用'):
                st.warning('この保存結果はCG／デモ製品に基づく計算です。')
            elif snapshot.get('source_kind') != 'real':
                st.caption('元の写真の来歴は未確認です。')
            if session['duplicate_measurement_count'] > 1:
                st.warning('同じ乾麺の記録を使った料理が複数あります。')
            source = '実測重量' if snapshot['weight_source'] == 'actual' else '推定重量'
            st.caption(f"乾麺 {snapshot['dry_weight_g']:g} g · {source} · {snapshot['nutrition_source']}")
            with st.expander('栄養と計算の内訳'):_show_result(session['calculation'])
            _diary_link(session, f"cooking_card_to_diary_{session['id']}")
    for template in templates:
        with st.container(border=True):
            st.subheader(template['name'])
            st.caption(f"{template['selected_dish']} · 基準乾麺 {template['reference_dry_g']:g} g")
            st.caption(template['source'])
            with st.expander('材料を確認'):
                for item in template['ingredients']:st.write(f"{item['name']} · {item['amount_g']:g} g")
