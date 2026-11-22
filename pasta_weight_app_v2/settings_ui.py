import sqlite3
import streamlit as st
from storage import list_products, save_product
from ui_shell import resume_capture
from operation_logs import RECORDS, WEIGHT, log_error, log_info


def select_measurement_product(products, label, key, *, using_demo=False):
    if not products:
        return None
    options = list(products)
    last_key = 'last_demo_product_id' if using_demo else 'last_product_id'
    preferred = st.session_state.get(last_key)
    if preferred not in products:
        preferred = options[0]
        real_ids = [ident for ident, row in products.items() if not row['name'].startswith('デモ用')]
        if real_ids and not using_demo:
            preferred = real_ids[0] if len(real_ids) == 1 else None
    if st.session_state.get(key) not in products:
        st.session_state.pop(key, None)
    ident = st.selectbox(label, options, index=options.index(preferred) if preferred is not None else None,
                         placeholder='使った乾麺の製品を選択', key=key,
                         format_func=lambda value: products[value]['name'])
    if ident is not None:
        st.session_state[last_key] = ident
    return products.get(ident)


def open_calibration(product=None):
    """Choose the product from the photo, or an empty form for a demo replacement."""
    ident = product['id'] if product else None
    editor_id = ident if product and not product['name'].startswith('デモ用') else 0
    st.session_state.update(
        main_nav='設定', settings_section='製品・校正', next_product_editor_id=editor_id,
        calibration_origin=(st.session_state.get('capture_generation'), ident),
    )


def _resume_after_calibration():
    st.session_state.pop('calibration_origin', None)
    resume_capture()


def render_products():
    notice = st.session_state.pop('product_saved_notice', None)
    if notice:
        st.success(f'製品データを保存しました。{notice}' if isinstance(notice, str) else '製品データを保存しました。')
    if st.session_state.get('result_request') or st.session_state.get('capture_photo') is not None:
        st.button('測定画面に戻る', key='calibration_return', on_click=_resume_after_calibration)
    st.subheader('製品データの登録・更新')
    st.write('1本あたりの重量は、**同じ製品・長さの乾麺を例えば50本数え、'
             '電子はかりで測った総重量を50で割る**と求められます。'
             '栄養値は商品パッケージの「乾麺100gあたり」を入力してください。')
    try:
        products_for_edit = list_products()
    except (OSError, sqlite3.Error):
        log_error(WEIGHT, '製品の校正情報の読み込み 失敗')
        st.error('製品情報を開けませんでした。ほかの画面で保存中の場合は少し待って、読み直してください。')
        if st.button('製品情報を読み直す', key='editor_products_retry'):
            st.rerun()
        return
    mapping = {p['id']: p for p in products_for_edit
               if st.session_state.get('_app_admin') or not p['name'].startswith('デモ用')}
    next_editor = st.session_state.pop('next_product_editor_id', None)
    if next_editor is not None:
        st.session_state['product_edit_mode'] = next_editor if next_editor in mapping else 0
    if st.session_state.get('product_edit_mode') not in {0, *mapping}:
        st.session_state['product_edit_mode'] = 0
    selected = st.selectbox('編集する製品', options=[0] + list(mapping),
                            format_func=lambda p: '＋ 新しい製品を登録' if p == 0
                            else mapping[p]['name'], key='product_edit_mode')
    current = mapping.get(selected, {})
    if current.get('name', '').startswith('デモ用'):
        st.info('これはサンプルの仮データです。「＋ 新しい製品を登録」を選んで、お使いの製品の校正値を入力してください。')
        return
    previous = st.session_state.get('product_editor_displayed')
    st.session_state['product_editor_displayed'] = current
    calibrate = st.checkbox('数えた本数と総重量から1本の重量を計算する', key=f'calibrate_from_units_{selected}')
    with st.form(f'product_editor_{selected}'):
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
            saved = save_product(name, g_per_strand, kc, pr, fa, ca,
                                 product_id=selected or None, expected_product=previous)
            log_info(RECORDS, '製品・校正情報の保存 完了')
            st.session_state['last_product_id'] = saved['id']
            st.session_state['next_product_editor_id'] = saved['id']
            notice = f'次の測定では「{saved["name"]}」を選択します。'
            request = st.session_state.get('result_request')
            origin = st.session_state.pop('calibration_origin', None)
            if request and 'saved_product' not in request:
                original_id = request['product']['id']
                if selected == original_id or origin == (st.session_state.get('capture_generation'), original_id):
                    # Only the product changes; keep the original photo and its counted regions.
                    request['product'] = saved
                    request.pop('_logged_estimate_outcome', None)
                    notice = f'測定画面に戻ると「{saved["name"]}」の登録値で重さを確認できます。'
            st.session_state['product_saved_notice'] = notice
            st.rerun()
        except ValueError as err:
            st.error(str(err))
        except sqlite3.IntegrityError:
            st.error('同じ名前の製品が登録されています。既存の製品を選んで更新するか、別の名前にしてください。')
        except (OSError,sqlite3.Error):
            log_error(RECORDS, '製品・校正情報の保存 失敗')
            st.error('保存できませんでした。入力内容は残っています。空き容量やファイルの使用状況を確認して、もう一度保存してください。')


def render_guide():
    from capture_guide_ui import render_sheet_guide
    with st.expander('A4撮影補助シートの使い方'):
        render_sheet_guide()
    st.markdown('''乾麺は重ねずに広げ、全体が入るよう真上から撮影します。

アプリでカメラが使えないときは、スマートフォンの標準カメラで撮影して写真を選びます。

推定できないときは、電子はかりで測った重さを保存できます。完成した料理を撮った後、乾麺の記録と使った材料を選ぶと、食べた分の栄養を計算できます。''')
