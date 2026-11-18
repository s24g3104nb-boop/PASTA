"""Pasta Lens: 乾麺推定、学習管理と材料ベース栄養計算。"""
from __future__ import annotations

from pathlib import Path
import sqlite3
import hashlib
import logging

from PIL import Image
import streamlit as st

from pasta_vision import detect_bundle_ends, detect_spread, image_to_rgb
from storage import get_product, list_products
from training_manager import get_active_model
from measurement_ui import render_weight_result, render_regression_measurement
from settings_ui import open_calibration, select_measurement_product
from pasta_vision import image_quality_warnings
from measurement_diagnostics import decide_count_weight, strand_calibration, trace

BASE_DIR = Path(__file__).resolve().parent
from ui_shell import start_capture, start_manual_weight


def _remember_mode():
    st.session_state['capture_last_mode']=st.session_state['capture_mode']

def render_capture():
    if st.session_state.get('result_request'):
        render_results(st.session_state['result_request'])
        return
    source_kind=st.session_state['capture_source']
    generation=st.session_state.get('capture_generation',0)
    st.button('ホームに戻る', icon=':material/arrow_back:', on_click=lambda: st.session_state.pop('capture_source',None))
    st.subheader('パスタを撮影' if source_kind=='スマホで撮影' else '写真からはじめる')
    st.caption('写真を確認 → 重さを確認 → 保存。推定精度は検証中です。はかりの数値でも記録できます。')
    st.button('撮り直す・写真を選び直す', on_click=start_capture, args=(source_kind,), key='retake_photo')
    if 'capture_mode' not in st.session_state:
        st.session_state['capture_mode']=st.session_state.get('capture_last_mode','広げた状態')
    mode=st.selectbox('乾麺の置き方', ['広げた状態','束の断面','束の側面・上面（回帰用）'],key='capture_mode',
        format_func=lambda x: {'広げた状態':'広げている','束の断面':'束の先端を写している','束の側面・上面（回帰用）':'束を横・上から写している（現在は未対応）'}[x],
        help='写真に合った置き方を選んでください。束の側面・上面は通常の測定では未対応です。',on_change=_remember_mode)
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
    except Exception:
        logging.exception('Unable to decode capture image')
        st.error('写真を読み込めませんでした。JPEGまたはPNG（25MB・2500万画素以下）を選び直してください。iPhoneのHEIC写真はJPEGに変換してください。')
        st.button('写真を選び直す',key='decode_retake',on_click=start_capture,args=('写真をアップロード',),width='stretch')
        st.button('はかりで測った重さを記録する',key='decode_manual',on_click=start_manual_weight,width='stretch')
        return
    st.image(preview, caption='撮影した写真 · 向きは自動補正済み', width='stretch')
    st.caption('麺の全体が写っているか確認してください。撮影状態は手動で選択します。')
    st.subheader('写真の確認')
    try:
        products = list_products()
    except (OSError,sqlite3.Error):
        logging.exception('Unable to load capture products')
        st.error('製品情報を開けませんでした。写真は保持しています。少し待って、もう一度お試しください。')
        if st.button('製品情報を読み直す',key='products_retry'):st.rerun()
        return
    product_ids = {p['id']: p for p in products}
    if not product_ids:
        st.info('使用する製品を登録してください。写真は保持しています。')
        st.button('製品を登録する', key='capture_add_product', on_click=open_calibration)
        return
    product = select_measurement_product(product_ids, '重量・栄養計算に使う製品', 'which_product', using_demo=using_demo)
    if product is None:
        st.info('使った製品を選ぶと、この写真を解析できます。')
        return
    if product['name'].startswith('デモ用'):
        st.warning('現在の製品データは**仮の数値**です。実際の重量・カロリーではありません。'
                   '設定の「製品・校正」で自分の製品を登録してください。')
    elif strand_calibration(product) is None:
        st.info('この製品は重量の校正が未設定です。推定できない場合も、はかりで測った重さを入力して保存できます。')
    if not using_demo and (product['name'].startswith('デモ用') or strand_calibration(product) is None):
        st.button('製品を登録・校正する', key='capture_calibration', on_click=open_calibration, args=(product,))

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
        from manual_weight_ui import recovery
        recovery(product,preview,synthetic=using_demo)
        return
    if st.button('この写真で解析する', type='primary', width='stretch', key='analyze_photo'):
        for key in ('override_count','manual_count','actual_known','dev_last_detection'):
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
        last_key = 'last_demo_product_id' if request['using_demo'] else 'last_product_id'
        st.session_state[last_key] = request['product']['id']
        st.session_state['estimate_method']='学習モデル（実験用回帰）' if request['using_regression'] else 'OpenCV本数検出'


def _diagnostic_product(product):
    st.caption(f'使用製品：{product["name"]}')
    calibration = strand_calibration(product)
    if calibration is None:
        st.caption('1本あたり：校正値が未設定です。')
    elif product['name'].startswith('デモ用'):
        st.caption(f'1本あたり：{calibration:.4f} g（デモ用の仮値）')
    else:
        st.caption(f'1本あたり：{calibration:.4f} g（製品に登録した校正値）')


def _unfinished_count_diagnostic(product, reason):
    with st.expander('写真と重量の診断'):
        st.caption('写真：読み込み済み')
        _diagnostic_product(product)
        st.write('パスタの色の候補・自動検出：結果なし')
        st.write(f'重量計算：未実行。{reason}')


def _count_diagnostic(result, product, mode, using_demo):
    """Use one decision for the displayed diagnosis, weight and save action."""
    manual_count = None
    with st.expander('写真と重量の診断'):
        st.caption('写真：読み込み済み · OpenCVの本数検出を実行')
        _diagnostic_product(product)
        if st.session_state.get('dev_tools_enabled'):
            st.metric(result.description, f'{result.count} 個')
            if st.checkbox('誤検出があるので、本数を手動修正する', key='override_count'):
                entered = st.number_input('実際に確認した本数', min_value=1,
                                          value=None, step=1, key='manual_count')
                # An empty correction means zero confirmed strands, never reuse the rejected count.
                manual_count = entered if entered is not None else 0
                st.caption('重量計算には手入力した本数を使います。')
        decision = decide_count_weight(product, result, mode=mode, using_demo=using_demo,
                                       manual_count=manual_count)
        st.write(f'パスタの色の候補：{decision.foreground_pixels:,} / {decision.image_pixels:,} 画素')
        st.write(f'自動検出：{decision.automatic_count} 本の候補')
        if decision.weight_g is None:
            st.write('重量計算：算出できませんでした。')
            for reason in decision.reasons:
                st.caption(reason)
        else:
            source = '入力した本数' if manual_count is not None else '検出候補'
            st.write(f'重量計算：{source} {decision.used_count} 本 × '
                     f'{decision.calibration_g_per_strand:.4f} g/本 = {decision.weight_g:.2f} g')
        if st.checkbox('検出位置とマスクを見る', key='show_capture_mask'):
            st.image(result.overlay_rgb, caption='赤枠・赤丸が数えた候補です。', width='stretch')
            st.image(result.mask, caption='白は麺に近い色の領域です。背景の一部も含まれることがあります。',
                     width='stretch')
    return decision, manual_count is not None


def render_results(request):
    from manual_weight_ui import recovery
    rgb=request['rgb']; product=request['product']; mode=request['mode']
    using_demo=request['using_demo']; using_regression=request['using_regression']
    st.button('写真・設定を変更',icon=':material/arrow_back:',on_click=edit_capture,key='edit_capture')
    st.image(rgb,caption='解析した写真' + (' · CGサンプル' if using_demo else ''),width='stretch')
    if 'saved_product' in request:
        product = request['saved_product']
    else:
        try:
            product = get_product(product['id'])
        except (OSError, sqlite3.Error):
            logging.exception('Unable to refresh product calibration for this measurement')
            st.error('選んだ製品の校正値を読み込めません。写真は保持しています。少し待って製品情報を読み直してください。')
            if st.button('製品情報を読み直す', key='result_product_retry'):
                st.rerun()
            return
        if product is None:
            st.warning('選んだ製品が見つかりません。「写真・設定を変更」からお使いの製品を選び直してください。')
            return
        if product != request['product']:
            request['product'] = product
            st.session_state['capture_product_refreshed'] = True
            # Discard an in-flight save click intended for the previous calibration.
            st.rerun()
    if st.session_state.pop('capture_product_refreshed', False):
        st.info('製品情報が更新されました。新しい校正値による結果を確認してください。')
    if not using_demo:
        problems=image_quality_warnings(rgb)
        if problems:
            st.warning(' '.join(problems))
            if not using_regression:
                _unfinished_count_diagnostic(product, '画像の明るさ・大きさ・ピントを確認してください。')
            recovery(product,rgb,synthetic=using_demo)
            return
    if using_regression:
        succeeded = False
        if request['model_error']:
            st.error(f"モデルを読み込めません: {request['model_error']}")
        elif request['active_model'] is None:
            st.warning('有効なモデルがありません。設定のAI学習管理でモデルを学習・選択してください。')
        else:
            with st.spinner('画像を解析しています…'):
                try:
                    succeeded = render_regression_measurement(request['active_model'],rgb=rgb,product=product,mode=mode,using_demo=using_demo,show_image=False)
                except Exception:
                    logging.exception('Unable to run the selected regression model')
                    st.warning('重量を推定できませんでした。実測入力で記録するか、写真を選び直してください。')
        if not succeeded:recovery(product,rgb,synthetic=using_demo)
        return
    try:
        result=request.get('_detection_result')
        if result is None:
            with st.spinner('画像を解析しています…'):
                if mode=='広げた状態':result=detect_spread(rgb,**request['parameters'])
                elif mode=='束の断面':result=detect_bundle_ends(rgb,**request['parameters'])
                else:raise ValueError('この撮影状態は現在の測定に対応していません。')
            # A request is replaced when its photo, placement or settings change.
            # Keep this private result in this session only, not a shared cache.
            request['_detection_result']=result
    except Exception:
        logging.exception('Unable to detect strands in the capture image')
        st.error('本数の検出を完了できませんでした。再試行するか、写真・設定を変更してください。')
        _unfinished_count_diagnostic(product, '本数の検出処理でエラーが発生しました。')
        if st.button('再試行する'):st.rerun()
        recovery(product,rgb,synthetic=using_demo)
        return
    decision, manual = _count_diagnostic(result, product, mode, using_demo)
    st.session_state['dev_last_detection'] = dict(
        overlay=result.overlay_rgb, mask=result.mask, count=result.count, parameters=request['parameters'],
        product_name=product['name'], calibration=decision.calibration_g_per_strand,
        reasons=decision.reasons, weight_g=decision.weight_g, manual=manual, using_demo=using_demo)
    if decision.weight_g is None:
        st.warning('重量を計算できません。' + ' '.join(decision.reasons))
        recovery(product, rgb, synthetic=using_demo)
        return

    if manual:
        st.success('入力した本数から重量を計算しました。')
    elif using_demo:
        st.success('CGサンプルの重量計算ができました。')
    else:
        st.success('写真の検出本数と製品の校正値から重量を推定しました。')
    estimator = 'manual_count' if manual else 'opencv'
    diagnostics = trace(rgb, estimator=estimator, product=product, mode=mode,
                        raw_sha256=request.get('raw_sha256'), parameters=dict(
                            automatic_count=decision.automatic_count, used_count=decision.used_count,
                            foreground_pixels=decision.foreground_pixels, image_pixels=decision.image_pixels,
                            detection=request['parameters']))
    render_weight_result(weight=decision.weight_g, product=product, rgb=rgb, mode=mode,
                         using_demo=using_demo, detected_count=decision.used_count, estimator=estimator,
                         diagnostics=diagnostics,
                         explanation=f'計算：{decision.used_count}本 × '
                                     f'{decision.calibration_g_per_strand:.4f}g/本 = {decision.weight_g:.2f}g。')
