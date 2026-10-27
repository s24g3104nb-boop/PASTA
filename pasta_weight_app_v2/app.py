"""Pasta Lens: 乾麺推定、学習管理と材料ベース栄養計算。"""
from __future__ import annotations

from pathlib import Path
import sqlite3
import hashlib

import pandas as pd
from PIL import Image
import streamlit as st

from generate_demo import generate_images
from pasta_vision import detect_bundle_ends, detect_spread, image_to_rgb, nutrition_for_weight
from storage import (DATA_DIR, init_db, list_measurements, list_products,
                     save_measurement, save_product)
from cooking_ui import render_cooking_tab
from training_ui import render_training_tab
from training_manager import get_active_model
from measurement_ui import render_weight_result, render_regression_measurement
from pasta_vision import image_quality_warnings

BASE_DIR = Path(__file__).resolve().parent
st.set_page_config(page_title='Pasta Lens | 乾麺量推定', page_icon='🍝', layout='centered')
init_db()
if not all((BASE_DIR / 'sample_images' / name).is_file()
           for name in ('demo_spread_12.png', 'demo_bundle_16.png')):
    generate_images()

st.markdown('''<style>
.block-container {padding-top: 1.5rem; padding-bottom: 3rem; max-width: 850px;}
h1 {letter-spacing: -0.035em;}
div[data-testid="stMetric"] {border: 1px solid #d8d8d8; padding: 12px; border-radius: 12px;}
</style>''', unsafe_allow_html=True)

st.title('🍝 Pasta Lens')
st.caption('乾麺の写真から重量と栄養を推定するプロトタイプ')
st.info('OpenCV本数検出と実験用の重量回帰を比較できます。AI学習管理で実写真を登録・学習し、'
        'モデルを選択できます。実物写真での精度はまだ未検証です。', icon='ℹ️')

measure_tab, history_tab, product_tab, cooking_tab, training_tab, guide_tab = st.tabs(
    ['📷 重量を推定', '🕒 測定履歴', '📦 パスタデータ', '🍳 料理と栄養', '🧪 AI学習管理', '❓ 使い方'])

with measure_tab:
    st.subheader('1. 撮影状態を選択')
    mode = st.radio('乾麺の置き方', ['広げた状態', '束の断面', '束の側面・上面（回帰用）'], horizontal=False,
                    help='OpenCVの円検出は正面の断面のみが対象です。束の側面・上面は別の回帰モデルで扱います。')
    st.subheader('2. 画像を選択')
    source_kind = st.radio('画像の入力方法',
                           ['サンプル画像（今すぐ試す）', '写真をアップロード', 'スマホで撮影'],
                           horizontal=False)
    photo = None
    using_demo = source_kind.startswith('サンプル')
    if using_demo and mode == '束の側面・上面（回帰用）':
        st.info('この状態のCGサンプルはありません。対応する実写真を使用してください。')
    elif using_demo:
        path = BASE_DIR / 'sample_images' / ('demo_spread_12.png' if mode == '広げた状態'
                                             else 'demo_bundle_16.png')
        photo = Image.open(path)
        st.caption('表示されるのは**CGのテスト画像**です。実際の乾麺写真ではありません。')
    elif source_kind == '写真をアップロード':
        photo = st.file_uploader('JPEG / PNG形式の乾麺写真', type=['jpg', 'jpeg', 'png'],
                                 key='upload_pasta')
    else:
        photo = st.camera_input('乾麺を撮影', key='camera_pasta')
        st.caption('スマートフォンのブラウザでカメラを使用するには、HTTPSなどの安全な接続が必要です。'
                   'カメラが開けない場合は標準カメラで撮影後、アップロードしてください。')

    st.subheader('3. 製品を選択')
    products = list_products()
    product_ids = {p['id']: p for p in products}
    p_id = st.selectbox('重量・栄養計算に使う製品',
                        options=list(product_ids),
                        format_func=lambda n: product_ids[n]['name'], key='which_product')
    product = product_ids[p_id]
    if product['name'].startswith('デモ用'):
        st.warning('現在の製品データは**仮の数値**です。実際の重量・カロリーではありません。'
                   '後から「パスタデータ」タブで自分の製品を登録してください。')
    elif product['g_per_strand'] <= 0:
        st.warning('この製品は1本あたりの重量が未設定です。まずパスタデータから校正値を登録してください。')

    st.subheader('4. 写真を解析')
    method = st.radio('推定方式', ['OpenCV本数検出', '学習モデル（実験用回帰）'], key='estimate_method')
    using_regression = method.startswith('学習')
    active_model = None
    model_error = None
    if using_regression:
        try:
            active_model = get_active_model()
        except (OSError, ValueError, sqlite3.Error) as exc:
            model_error = str(exc)

    with st.expander('検出パラメータを調整（うまく数えられない場合）'):
        if mode == '広げた状態':
            sat = st.slider('色の濃さ（最小彩度）', 0, 120, 20, 5)
            val = st.slider('明るさ（最小輝度）', 0, 180, 75, 5)
            min_area = st.slider('領域の最小面積（画素）', 30, 2500, 160, 10)
            elongation = st.slider('細長さの基準', 1.5, 8.0, 3.0, 0.5)
        elif mode == '束の断面':
            min_radius = st.slider('円の最小半径（画素）', 2, 35, 8)
            max_radius = st.slider('円の最大半径（画素）', 9, 65, 27)
            min_distance = st.slider('検出円の最小間隔（画素）', 5, 90, 32)
            sensitivity = st.slider('円検出の厳しさ（大きいほど厳しい）', 5, 45, 18)
        else:
            st.caption('束の側面・上面はHough円検出の対象外です。対応する回帰モデルが必要です。')

    if photo is None:
        st.info('サンプル画像を選択するか、乾麺写真をアップロードしてください。')
    else:
        try:
            rgb = image_to_rgb(photo)
            input_key = (hashlib.sha256(rgb.tobytes()).hexdigest() + mode + str(p_id) + source_kind
                         + method + (active_model['id'] if active_model else ''))
            if st.session_state.get('measurement_input') != input_key:
                for key in ('override_count', 'manual_count', 'actual_known'):
                    st.session_state.pop(key, None)
                st.session_state['measurement_input'] = input_key
            if not using_demo:
                problems = image_quality_warnings(rgb)
                if problems:
                    raise ValueError(' '.join(problems))
            if using_regression:
                result = None
            elif mode == '広げた状態':
                # expander は閉じていてもStreamlitウィジェットが作られる。
                result = detect_spread(rgb, saturation_min=sat, value_min=val,
                                       min_area=min_area, min_elongation=elongation)
            elif mode == '束の断面':
                result = detect_bundle_ends(rgb, min_radius=min_radius,
                                            max_radius=max_radius,
                                            min_distance=min_distance,
                                            hough_sensitivity=sensitivity)
            else:
                raise ValueError('束の側面・上面はOpenCV断面検出の対象外です。学習モデル方式を選択してください。')
        except Exception as exc:
            st.error(f'画像を解析できませんでした。画像形式または設定を確認してください。詳細: {exc}')
        else:
            if using_regression:
                if model_error:
                    st.error(f'モデルを読み込めません: {model_error}')
                elif active_model is None:
                    st.warning('有効なモデルがありません。AI学習管理で実写真のモデルを学習・選択してください。')
                else:
                    render_regression_measurement(active_model, rgb=rgb, product=product,
                                                  mode=mode, using_demo=using_demo)
            else:
                st.image(result.overlay_rgb, caption='検出箇所：赤枠 / 赤丸',
                         width='stretch')
                with st.expander('色抽出マスクを表示'):
                    st.image(result.mask, caption='白い部分が検出対象', width='stretch')
                st.metric(result.description, f'{result.count} 個')
                for msg in result.warnings:
                    st.caption('⚠️ ' + msg)

                override = st.checkbox('誤検出があるので、本数を手動修正する',
                                       key='override_count')
                if override:
                    detected_count = st.number_input('実際に確認した本数', min_value=0,
                                                      value=result.count, step=1,
                                                      key='manual_count')
                    st.caption('この値は手動入力です。AIが認識した結果ではありません。')
                else:
                    detected_count = result.count

                if detected_count <= 0:
                    st.warning('本数が0のため推定できません。画像やパラメータを変更してください。')
                elif product['g_per_strand'] <= 0:
                    st.warning('選択した製品の「1本あたりの重量」を登録すると推定できます。')
                else:
                    weight = detected_count * product['g_per_strand']
                    render_weight_result(weight=weight, product=product, rgb=rgb, mode=mode,
                                         using_demo=using_demo, detected_count=detected_count,
                                         estimator='manual_count' if override else 'opencv',
                                         explanation=f'計算：{detected_count}本 × {product["g_per_strand"]:.4f}g/本 = {weight:.2f}g。')

with history_tab:
    st.subheader('測定履歴')
    rows = list_measurements()
    if not rows:
        st.info('まだ履歴がありません。「重量を推定」タブで結果を保存してください。')
    else:
        frame = pd.DataFrame(rows)
        frame['誤差(g)'] = frame.apply(
            lambda row: round(abs(row['estimated_g'] - row['actual_g']), 2)
            if pd.notna(row['actual_g']) else None, axis=1)
        shown = frame.rename(columns={
            'id': 'ID', 'created_at': '日時', 'product_name': '製品',
            'mode': '撮影状態', 'detected_count': '本数',
            'estimated_g': '推定g', 'actual_g': '実測g', 'kcal': 'kcal', 'estimator': '方式', 'source_kind': '画像種別',
        })
        shown.loc[shown['方式'].str.startswith('ridge:'), '本数'] = None
        st.dataframe(shown[['方式', '画像種別', 'ID', '日時', '製品', '撮影状態', '本数', '推定g', '実測g', '誤差(g)', 'kcal']],
                     hide_index=True, width='stretch')
        measured = frame[(frame['source_kind'] == 'real') & (frame['actual_g'] > 0)].copy()
        if len(measured):
            measured['absolute_error_g'] = (measured['estimated_g'] - measured['actual_g']).abs()
            measured['absolute_percentage_error'] = measured['absolute_error_g'] / measured['actual_g'] * 100
            st.metric('実写・実測値がある履歴のMAE', f"{measured['absolute_error_g'].mean():.2f} g")
            st.dataframe(measured.groupby(['mode', 'estimator']).agg(
                件数=('id', 'count'), MAE_g=('absolute_error_g', 'mean'),
                MAPE_percent=('absolute_percentage_error', 'mean')).reset_index(), hide_index=True)
            st.caption('保存された推定成功分の参考値です。学習モデルの評価は独立した写真で行ってください。')
        chosen = st.selectbox('写真を確認する履歴ID', options=frame['id'].tolist())
        entry = next(r for r in rows if r['id'] == chosen)
        path = DATA_DIR / 'photos' / (entry['image_filename'] or '')
        if path.is_file():
            st.image(str(path), caption=f"{entry['created_at']} / {entry['mode']}",
                     width='stretch')

with product_tab:
    st.subheader('製品データの登録・更新')
    st.write('1本あたりの重量は、**同じ製品・長さの乾麺を例えば50本数え、'
             '電子はかりで測った総重量を50で割る**と求められます。'
             '栄養値は商品パッケージの「乾麺100gあたり」を入力してください。')
    products_for_edit = list_products()
    mapping = {p['id']: p for p in products_for_edit}
    selected = st.selectbox('編集する製品', options=[0] + list(mapping),
                            format_func=lambda p: '＋ 新しい製品を登録' if p == 0
                            else mapping[p]['name'], key='product_edit_mode')
    current = mapping.get(selected, {})
    calibrate = st.checkbox('数えた本数と総重量から1本の重量を計算する', value=False)
    with st.form('product_editor'):
        name = st.text_input('製品名', value=current.get('name', ''),
                             placeholder='例：〇〇スパゲッティ 1.6mm')
        if calibrate:
            calib_n = st.number_input('校正で数えた本数', min_value=1, value=50, step=1)
            calib_w = st.number_input('その本数の実測重量（g）', min_value=0.0, value=0.0, step=0.1)
            g_per_strand = calib_w / calib_n
            st.caption(f'1本あたりの重量：{g_per_strand:.4f} g')
        else:
            g_per_strand = st.number_input('1本あたりの重量（g）', min_value=0.0,
                                           max_value=20.0, value=float(current.get('g_per_strand', 0.0)),
                                           step=0.001, format='%.4f')
        kc = st.number_input('乾麺100gあたり kcal', min_value=0.0,
                             value=float(current.get('kcal_100', 0.0)), step=1.0)
        pr = st.number_input('乾麺100gあたり たんぱく質（g）', min_value=0.0,
                             value=float(current.get('protein_100', 0.0)), step=0.1)
        fa = st.number_input('乾麺100gあたり 脂質（g）', min_value=0.0,
                             value=float(current.get('fat_100', 0.0)), step=0.1)
        ca = st.number_input('乾麺100gあたり 炭水化物（g）', min_value=0.0,
                             value=float(current.get('carb_100', 0.0)), step=0.1)
        submitted = st.form_submit_button('製品データを保存', type='primary')
    if submitted:
        try:
            if calibrate and calib_w <= 0:
                raise ValueError('校正用の実測重量は0gより大きい値を入力してください。')
            save_product(name, g_per_strand, kc, pr, fa, ca,
                         product_id=(selected or None))
            st.success('製品データを保存しました。')
            st.rerun()
        except (sqlite3.IntegrityError, ValueError) as err:
            st.error(f'保存できませんでした（商品名の重複や入力値をご確認ください）：{err}')

with cooking_tab:
    render_cooking_tab()

with training_tab:
    render_training_tab()

with guide_tab:
    st.subheader('準備するもの')
    st.markdown('''
    - パソコン（Pythonを動かす）とスマートフォン
    - 乾燥スパゲッティ（最初は同じ銘柄、同じ長さを使用）
    - 電子はかり（実測重量の校正に使用）

    **広げた状態：** 暗く無地の台の上に、麺が交差しないように広げて真上から撮影。

    **束の断面：** 束の先端を真正面から撮影。円が重なったり、端面が隠れたりすると誤差が増えます。

    **基準物：** 定規や硬貨は不要。ただし、製品ごとの1本当たりの重量は事前の実測が必要です。

    **実装状況：** 写真入力、OpenCV検出、重量・栄養計算、商品登録、履歴保存。
    束はHough変換で円を検出する試験段階。乾麺の学習・評価はAI学習管理画面とCLIで実行できます。
    料理名の画像分類器は未学習です。料理タブでは手動選択と材料入力で栄養を計算します。

    **データの取扱い：** 保存した写真・履歴は、このPythonアプリを動かしているPCの
    `data/`フォルダに置かれます。認証機能がないため、インターネットに一般公開しないでください。
    ''')
