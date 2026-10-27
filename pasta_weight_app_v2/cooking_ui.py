"""完成料理タブ。画像分類、材料入力、計算・保存を分離する。"""
from __future__ import annotations

import sqlite3

import pandas as pd
import streamlit as st

from cooking import Ingredient, calculate_meal, dry_snapshot
from cooking_storage import list_cooking_sessions, save_cooking_session
from dish_classifier import DISH_NAMES, INGREDIENT_HINTS, UntrainedDishClassifier
from storage import DEFAULT_DB, list_measurements


def render_cooking_tab(db_path=DEFAULT_DB):
    st.subheader('完成料理の材料・カロリー')
    st.warning('写真から材料の使用量や正確な総カロリーは分かりません。'
               '使った乾麺の履歴と、ソース・油・具材・調味料の実際の使用量を確認してください。')
    st.caption(UntrainedDishClassifier().predict().message)
    st.caption('完成料理の重量を予測する機能ではありません。料理名の選択だけでは材料量は決まりません。')
    measurements = list_measurements(db_path)
    if not measurements:
        st.info('先に「重量を推定」タブで、この料理に使用する乾麺の測定結果を保存してください。')
        return
    mapping = {item['id']: item for item in measurements}
    measurement_id = st.selectbox(
        'この料理に使用した乾麺の計測履歴ID', [None] + list(mapping),
        format_func=lambda key: '履歴を選択してください' if key is None else
        f"ID {key} | {mapping[key]['created_at']} | {mapping[key]['product_name']} | "
        f"実測 {mapping[key]['actual_g']}g / 推定 {mapping[key]['estimated_g']}g",
        key='cooking_measurement')
    if measurement_id is None:
        return
    record = mapping[measurement_id]
    try:
        snapshot = dry_snapshot(record)
    except ValueError as exc:
        st.error(str(exc))
        return
    kind = snapshot['source_kind']
    if kind == 'synthetic' or record['product_name'].startswith('デモ用'):
        st.warning('この履歴はCGまたはデモ製品の値です。料理計算もデモで、実際の食事管理には使えません。')
    elif kind == 'unknown':
        st.warning('旧履歴のため写真の来歴が不明です。実写・製品情報を確認してください。')
    weight_source = '実測重量' if snapshot['weight_source'] == 'actual' else '推定重量（実測なし）'
    st.write(f"使用する乾麺: {snapshot['dry_weight_g']:.2f}g / {weight_source}")
    st.caption(snapshot['nutrition_source'])
    same_batch = st.checkbox('この履歴の乾麺を今回の料理に使いました（同一調理バッチ）',
                             key=f'cooking_batch_{measurement_id}')
    selected_dish = st.selectbox('料理名（手動選択）', DISH_NAMES, key='cooking_dish')
    hints = INGREDIENT_HINTS.get(selected_dish, ())
    if hints:
        st.caption('材料確認のヒント: ' + '、'.join(hints) + '。実際に使ったものだけを入力してください。')
    st.caption('数値の標準レシピは自動入力しません。商品ラベル等の100gあたり栄養値と使用gを入力してください。'
               '「卵1個」等は食べる部分のgに換算し、値が不明な行を0で埋めないでください。')
    with st.form(f'cooking_editor_{measurement_id}'):
        frame = st.data_editor(
            pd.DataFrame({'name': pd.Series(dtype='str'), **{
                key: pd.Series(dtype='float64') for key in
                ('amount_g', 'kcal_100', 'protein_100', 'fat_100', 'carb_100')}}),
            num_rows='dynamic', hide_index=True, key=f'cooking_ingredients_{measurement_id}',
            column_config={
                'name': st.column_config.TextColumn('材料名', required=True),
                'amount_g': st.column_config.NumberColumn('使用量(g)', min_value=0.0, required=True),
                'kcal_100': st.column_config.NumberColumn('kcal/100g', min_value=0.0, required=True),
                'protein_100': st.column_config.NumberColumn('P(g)/100g', min_value=0.0, required=True),
                'fat_100': st.column_config.NumberColumn('F(g)/100g', min_value=0.0, required=True),
                'carb_100': st.column_config.NumberColumn('C(g)/100g', min_value=0.0, required=True),
            })
        complete = st.checkbox('使った追加材料・ソース・油・調味料をすべて入力しました（未入力分は合計に含まれません）')
        division = st.radio('取り分け方法', ['等分人数', '自分が食べる割合'], horizontal=True)
        servings_input = st.number_input('等分する人数', min_value=1, value=1, step=1)
        fraction_input = st.number_input('食べる割合（1=全部、0.5=半分）',
                                         min_value=0.001, max_value=1.0, value=1.0, step=0.05)
        save = st.form_submit_button('入力を確認して計算・保存')
    if save:
        try:
            if not same_batch:
                raise ValueError('同一調理バッチの確認が必要です。')
            if not complete:
                raise ValueError('追加材料の入力が完了してから計算してください。')
            ingredients = [Ingredient(**row) for row in frame.to_dict('records')]
            servings = int(servings_input) if division == '等分人数' else None
            fraction = float(fraction_input) if division != '等分人数' else None
            result = calculate_meal(snapshot, ingredients, servings=servings,
                                    consumed_fraction=fraction)
            session_id = save_cooking_session(
                measurement_id=int(measurement_id), selected_dish=selected_dish,
                confirmed_same_batch=same_batch, ingredients=ingredients,
                servings=servings, consumed_fraction=fraction, db_path=db_path)
            st.success(f'料理の計算条件を保存しました（ID {session_id}）。')
            _show_result(result)
        except (ValueError, TypeError, OverflowError, sqlite3.Error) as exc:
            st.error(f'計算・保存できませんでした: {exc}')
    sessions = list_cooking_sessions(db_path)
    if sessions:
        with st.expander('保存した料理の計算履歴'):
            for session in sessions[:20]:
                st.write(f"ID {session['id']} / {session['selected_dish']} / "
                         f"乾麺履歴ID {session['measurement_id']} / {session['created_at']}")
                past = session['dry_snapshot']
                if past.get('source_kind') == 'synthetic' or past['product_name'].startswith('デモ用'):
                    st.warning('この保存結果はCG／デモ製品に基づくデモ計算です。実際の料理の栄養値ではありません。')
                elif past.get('source_kind', 'unknown') == 'unknown':
                    st.warning('この保存結果の元画像は来歴不明です。実写か確認されていません。')
                past_source = '実測重量' if past['weight_source'] == 'actual' else '推定重量（実測なし）'
                st.caption(f"乾麺 {past['dry_weight_g']:.2f}g / {past_source} / {past['nutrition_source']}")
                _show_result(session['calculation'])


def _show_result(result):
    labels = {'dry': '乾麺分', 'additional': '追加材料分',
              'total': '料理全体', 'portion': '選択した取り分け分'}
    st.dataframe(pd.DataFrame([
        {'内訳': label, 'kcal': round(result[key]['kcal'], 1),
         'P(g)': round(result[key]['protein_g'], 1), 'F(g)': round(result[key]['fat_g'], 1),
         'C(g)': round(result[key]['carb_g'], 1)} for key, label in labels.items()
    ]), hide_index=True)
    st.caption(f"食べる割合: {result['consumed_fraction']:.3f}。{result['assumptions']}")
