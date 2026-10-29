"""Pasta Lens: 乾麺推定、学習管理と材料ベース栄養計算。"""
from __future__ import annotations

from pathlib import Path
import sqlite3
import hashlib

from PIL import Image
import streamlit as st

from pasta_vision import detect_bundle_ends, detect_spread, image_to_rgb
from storage import list_products
from training_manager import get_active_model
from measurement_ui import render_weight_result, render_regression_measurement
from pasta_vision import image_quality_warnings
from measurement_diagnostics import weight_gate,trace

BASE_DIR = Path(__file__).resolve().parent
from ui_shell import start_capture

def render_capture():
    if st.session_state.get('result_request'):
        render_results(st.session_state['result_request'])
        return
    source_kind=st.session_state['capture_source']
    generation=st.session_state.get('capture_generation',0)
    st.button('ホームに戻る', icon=':material/arrow_back:', on_click=lambda: st.session_state.pop('capture_source',None))
    st.subheader('パスタを撮影' if source_kind=='スマホで撮影' else '写真からはじめる')
    st.button('撮り直す・写真を選び直す', on_click=start_capture, args=(source_kind,), key='retake_photo')
    mode=st.selectbox('乾麺の置き方', ['広げた状態','束の断面','束の側面・上面（回帰用）'],key='capture_mode',
        format_func=lambda x: {'広げた状態':'広げている','束の断面':'束の先端を写している','束の側面・上面（回帰用）':'束を横・上から写している（現在は未対応）'}[x],
        help='自動判定ではありません。束の側面・上面は対応する学習モデルが必要です。')
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
                                 key=f'upload_pasta_{generation}')
    else:
        photo = st.camera_input('乾麺を撮影', key=f'camera_pasta_{generation}')
        st.caption('スマートフォンのブラウザでカメラを使用するには、HTTPSなどの安全な接続が必要です。'
                   'カメラが開けない場合は標準カメラで撮影後、アップロードしてください。')

    if photo is None and st.session_state.get('capture_photo') is not None:
        photo=st.session_state['capture_photo']
    if photo is None and st.session_state.get('editing_photo') is not None:
        photo=st.session_state['editing_photo']
    if source_kind == 'スマホで撮影':
        st.button('カメラが開かない場合は写真から選ぶ',on_click=start_capture,args=('写真をアップロード',),key='camera_fallback')
    if photo is None:
        st.caption('写真は解析のためにこのアプリのサーバーへ送信されます。保存ボタンを押すと履歴に残ります。')
        return
    try:
        preview = image_to_rgb(photo)
        st.session_state['capture_photo']=photo
    except Exception as exc:
        st.error(f'写真を読み込めませんでした。JPEGまたはPNGを選び直してください。')
        return
    st.image(preview, caption='撮影した写真 · 向きは自動補正済み', width='stretch')
    st.caption('麺の全体が写っているか確認してください。撮影状態は手動で選択します。')
    st.subheader('写真の確認')
    products = list_products()
    product_ids = {p['id']: p for p in products}
    if 'which_product' not in st.session_state and st.session_state.get('last_product_id') in product_ids:
        st.session_state['which_product']=st.session_state['last_product_id']
    p_id = st.selectbox('重量・栄養計算に使う製品',
                        options=list(product_ids),
                        format_func=lambda n: product_ids[n]['name'], key='which_product')
    product = product_ids[p_id]
    st.session_state['last_product_id']=p_id
    if product['name'].startswith('デモ用'):
        st.warning('現在の製品データは**仮の数値**です。実際の重量・カロリーではありません。'
                   '設定の「製品・校正」で自分の製品を登録してください。')
    elif product['g_per_strand'] <= 0:
        st.warning('この製品は1本あたりの重量が未設定です。まずパスタデータから校正値を登録してください。')

    from developer_ui import DEFAULT_SPREAD, DEFAULT_BUNDLE
    developer = st.session_state.get('dev_tools_enabled', False)
    method = st.session_state.get('research_method', 'OpenCV本数検出') if developer else 'OpenCV本数検出'
    using_regression = method.startswith('学習')
    active_model = None
    model_error = None
    if using_regression:
        try:
            active_model = get_active_model()
        except (OSError, ValueError, sqlite3.Error):
            model_error = '読み込みに失敗しました。開発・検証ツールで確認してください。'
    parameters = (dict(st.session_state.get('research_spread', DEFAULT_SPREAD)) if developer else dict(DEFAULT_SPREAD)) if mode == '広げた状態' else ((dict(st.session_state.get('research_bundle', DEFAULT_BUNDLE)) if developer else dict(DEFAULT_BUNDLE)) if mode == '束の断面' else {})

    if mode == '束の側面・上面（回帰用）' and not using_regression:
        st.info('この置き方は現在の通常測定では対応していません。麺を離して広げるか、束の先端が見えるように撮り直してください。')
        return
    if st.button('この写真で解析する', type='primary', width='stretch', key='analyze_photo'):
        for key in ('override_count','manual_count','actual_known'):
            st.session_state.pop(key,None)
        st.session_state['result_request']=dict(rgb=preview,photo=photo,product=product,mode=mode,
            using_demo=using_demo,using_regression=using_regression,active_model=active_model,
            model_error=model_error,parameters=parameters,
            raw_sha256=hashlib.sha256(photo.getvalue()).hexdigest() if hasattr(photo,'getvalue') else None)
        st.rerun()


def edit_capture():
    request=st.session_state.pop('result_request',None)
    if request:
        st.session_state['editing_photo']=request['photo']
        st.session_state['capture_mode']=request['mode']
        st.session_state['which_product']=request['product']['id']
        st.session_state['estimate_method']='学習モデル（実験用回帰）' if request['using_regression'] else 'OpenCV本数検出'


def render_results(request):
    rgb=request['rgb']; product=request['product']; mode=request['mode']
    using_demo=request['using_demo']; using_regression=request['using_regression']
    st.button('写真・設定を変更',icon=':material/arrow_back:',on_click=edit_capture,key='edit_capture')
    st.image(rgb,caption='解析した写真' + (' · CGサンプル' if using_demo else ''),width='stretch')
    if not using_demo:
        problems=image_quality_warnings(rgb)
        if problems:
            st.warning(' '.join(problems))
            st.button('撮り直す',on_click=start_capture,args=('スマホで撮影',),key='failed_retake')
            return
    if using_regression:
        if request['model_error']:
            st.error(f"モデルを読み込めません: {request['model_error']}")
        elif request['active_model'] is None:
            st.warning('有効なモデルがありません。設定のAI学習管理でモデルを学習・選択してください。')
        else:
            with st.spinner('画像を解析しています…'):
                render_regression_measurement(request['active_model'],rgb=rgb,product=product,mode=mode,using_demo=using_demo,show_image=False)
        return
    try:
        with st.spinner('画像を解析しています…'):
            if mode=='広げた状態':result=detect_spread(rgb,**request['parameters'])
            elif mode=='束の断面':result=detect_bundle_ends(rgb,**request['parameters'])
            else:raise ValueError('束の側面・上面はOpenCV断面検出の対象外です。学習モデル方式を選択してください。')
    except Exception as exc:
        st.error(f'画像を解析できませんでした。写真・設定を変更してください。')
        if st.button('再試行する'):st.rerun()
        return
    st.caption('✓ 解析が完了しました')
    st.session_state['dev_last_detection']=dict(overlay=result.overlay_rgb,mask=result.mask,count=result.count,parameters=request['parameters'])
    override=False
    detected_count=result.count
    if st.session_state.get('dev_tools_enabled'):
        with st.expander('検出の詳細・本数を修正'):
            st.image(result.overlay_rgb,caption='検出箇所：赤枠 / 赤丸',width='stretch')
            st.image(result.mask,caption='白い部分が検出対象',width='stretch')
            st.metric(result.description,f'{result.count} 個')
            for msg in result.warnings:st.caption('⚠️ '+msg)
            override=st.checkbox('誤検出があるので、本数を手動修正する',key='override_count')
            detected_count=st.number_input('実際に確認した本数',min_value=0,value=result.count,step=1,key='manual_count') if override else result.count
            if override:st.caption('この値は手動入力です。AIが認識した結果ではありません。')
    rejection=weight_gate(product,using_demo=using_demo,mode=mode,mask=result.mask,manual=override)
    if rejection:
        st.warning(' '.join(rejection))
        st.button('もう一度撮影',on_click=start_capture,args=('スマホで撮影',),key='rejection_retake',width='stretch')
    elif product['g_per_strand']<=0:
        st.warning('この製品の1本重量を設定の製品・校正で登録してください。')
    elif detected_count<=0:
        st.warning('本数を確認できませんでした。写真・設定を変更してください。')
    else:
        weight=detected_count*product['g_per_strand']
        estimator='manual_count' if override else 'opencv'
        render_weight_result(weight=weight,product=product,rgb=rgb,mode=mode,using_demo=using_demo,
              detected_count=detected_count,estimator=estimator,
            diagnostics=trace(rgb,estimator=estimator,product=product,mode=mode,raw_sha256=request['raw_sha256'],
                parameters=dict(automatic_count=result.count,used_count=detected_count,detection=request['parameters'])),
            explanation=f'計算：{detected_count}本 × {product["g_per_strand"]:.4f}g/本 = {weight:.2f}g。')
