import sqlite3
import streamlit as st
from storage import list_products, save_product

def render_products():
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


def render_guide():
    st.subheader('準備するもの')
    st.markdown('''
    - パソコン（Pythonを動かす）とスマートフォン
    - 乾燥スパゲッティ（最初は同じ銘柄、同じ長さを使用）
    - 電子はかり（実測重量の校正に使用）

    **広げた状態：** 暗く無地の台の上に、麺が交差しないように広げて真上から撮影。

    **束の断面：** 束の先端を真正面から撮影。円が重なったり、端面が隠れたりすると誤差が増えます。

    **基準物：** 定規や硬貨は不要。ただし、製品ごとの1本当たりの重量は事前の実測が必要です。

    **食事の記録：** ホームの「食べたパスタを記録する」から、料理名と食べた日を保存できます。
    写真と時刻は任意です。記録画面のカレンダーや一覧で振り返れます。

    **栄養計算：** 乾麺の結果は乾麺分のみです。料理全体は、使った食材と分量を入力して計算します。
    完成料理の写真AIによるカロリー推定は準備中です。計算しなくても食事日記を保存できます。

    **データの取扱い：** 保存した写真・履歴は、このPythonアプリを動かしているPCの
    `data/`フォルダに置かれます。認証機能がないため、インターネットに一般公開しないでください。
    ''')
