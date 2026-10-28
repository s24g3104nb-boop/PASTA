"""測定方式に依存しない結果表示と履歴保存。"""
import sqlite3
import streamlit as st
from pasta_vision import nutrition_for_weight
from storage import DEFAULT_DB, save_measurement
from pasta_ml.model import predict
from measurement_diagnostics import trace


def render_weight_result(*, weight, product, rgb, mode, using_demo, detected_count,
                         estimator, explanation, db_path=DEFAULT_DB, diagnostics=None):
    nutrit = nutrition_for_weight(weight, product['kcal_100'],
                                  product['protein_100'],
                                  product['fat_100'], product['carb_100'])
    st.subheader('あなたのパスタ')
    with st.container(key='weight_hero'):
        st.metric('推定重量', f'{weight:.1f} g')
    st.metric('カロリー', f'{nutrit["kcal"]:.1f} kcal')
    with st.container(key='nutrition_cards'):
        c1,c2,c3=st.columns(3)
        c1.metric('たんぱく質', f'{nutrit["protein_g"]:.1f} g')
        c2.metric('脂質', f'{nutrit["fat_g"]:.1f} g')
        c3.metric('炭水化物', f'{nutrit["carb_g"]:.1f} g')
    st.caption(f'使用方式: {estimator}')
    st.caption(explanation + ' 栄養値は乾麺のみで、ソースは含みません。')
    st.caption('推定の信頼性: CGテスト用' if using_demo else
               '推定の信頼性: 実物での精度保証・予測区間は未検証です。重なり・斜め撮影・反射・製品差で誤差が出ます。')
    has_actual = st.checkbox('重量を修正・実測値を登録する',
                             key='actual_known')
    actual = None
    if has_actual:
        actual = st.number_input('実測重量（g）', min_value=0.1,
                                 value=50.0, step=0.1)
        st.write(f'絶対誤差：**{abs(weight - actual):.1f} g**'
                 f' ／ 誤差率：**{abs(weight - actual) / actual * 100:.1f}%**')
    if st.button('履歴に保存', type='primary', width='stretch'):
        try:
            save_measurement(product_name=product['name'], mode=mode,
                             detected_count=detected_count, estimated_g=weight,
                             actual_g=actual, nutrition=nutrit, image_rgb=rgb,
                             source_kind='synthetic' if using_demo else 'real',
                             estimator=estimator,
                             product_snapshot=product, db_path=db_path,diagnostics=diagnostics)
            st.success('測定履歴に保存しました。下の「履歴」から確認できます。')
        except (OSError, ValueError, sqlite3.Error) as exc:
            st.error(f'保存できませんでした。保存先と入力を確認してください: {exc}')
    if 'capture_source' in st.session_state:
        from ui_shell import start_capture
        st.button('もう一度撮影', on_click=start_capture, args=('スマホで撮影',), key='result_retake', width='stretch')
    st.warning('表示値は試験的な推定値です。実際の食事管理に使う場合は'
               '電子はかりで確認してください。')


def render_regression_measurement(active, *, rgb, product, mode, using_demo, db_path=DEFAULT_DB, show_image=True):
    if using_demo or active['artifact']['source'] != 'real':
        st.warning('実写モデルの測定には実物の写真を使用してください。CGは精度評価に使えません。')
        return
    modes = {'広げた状態': 'spread', '束の断面': 'bundle_end',
             '束の側面・上面（回帰用）': 'bundle_side',
             'spread': 'spread', 'bundle_end': 'bundle_end', 'bundle_side': 'bundle_side'}
    canonical_mode = modes.get(mode)
    if canonical_mode is None:
        st.warning('撮影状態を確認してください。未知の状態を断面として扱いません。')
        return
    if active['artifact'].get('input_contract')=='reviewed_foreground':
        st.warning('このモデルは検品済み前景専用です。未処理の撮影画像には適用できません。AI学習管理で前処理を確認してください。')
        return
    prediction = predict(active['artifact'], rgb, canonical_mode, str(product['id']))
    if show_image:
        st.image(rgb, caption='回帰モデルへの入力画像', width='stretch')
    if not prediction['success']:
        st.warning(prediction['reason'] + ' 再撮影するか、対応する製品・モードの学習データを増やしてください。')
        return
    render_weight_result(weight=prediction['predicted_g'], product=product, rgb=rgb, mode=mode,
                         using_demo=False, detected_count=0, estimator=f"ridge:{active['id']}",
                         explanation=f"モデルID {active['id']} による連続重量の推定です。本数は計数していません。",
                         db_path=db_path,diagnostics=trace(rgb,estimator=f"ridge:{active['id']}",product=product,mode=mode,model=active))
