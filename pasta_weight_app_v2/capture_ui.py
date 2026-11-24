"""Pasta Lens: 乾麺推定、学習管理と材料ベース栄養計算。"""
from __future__ import annotations

from pathlib import Path
import sqlite3
import hashlib

from PIL import Image, UnidentifiedImageError
import streamlit as st

from pasta_vision import detect_bundle_ends, detect_spread, image_to_rgb
from storage import get_product, list_products
from measurement_ui import render_weight_result, render_regression_measurement
from settings_ui import open_calibration, select_measurement_product
from pasta_vision import image_quality_warnings
from mask_geometry import mask_geometry
from measurement_defaults import DEFAULT_SPREAD, DEFAULT_BUNDLE
from measurement_diagnostics import decide_count_weight, strand_calibration, trace
from nutrition_state import STANDARD_DRY_PASTA
from operation_logs import WEIGHT, log_error, log_info
from sheetless_area import (AREA_BOUNDARY_TOLERANCE_G, EDGE_CAPTURE_REASON, ESTIMATOR as AREA_ESTIMATOR,
                            MAX_MASK_FRACTION, MAX_THICKNESS_RATIO, THICK_CAPTURE_REASON, diagnose)
from sheetless_area import estimate as estimate_area

BASE_DIR = Path(__file__).resolve().parent
from ui_shell import start_capture, start_manual_weight

_EDGE_SIDE_LABELS = {'top': '上端', 'bottom': '下端', 'left': '左端', 'right': '右端'}


def _remember_mode():
    st.session_state['capture_last_mode']=st.session_state['capture_mode']

def _clear_previous_capture():
    for key in ('capture_photo', 'editing_photo', 'a4_area_result'):
        st.session_state.pop(key, None)


def render_capture():
    if st.session_state.get('result_request'):
        render_results(st.session_state['result_request'])
        return
    source_kind=st.session_state['capture_source']
    generation=st.session_state.get('capture_generation',0)
    st.button('ホームに戻る', icon=':material/arrow_back:', on_click=lambda: st.session_state.pop('capture_source',None))
    st.subheader('パスタを撮影' if source_kind=='スマホで撮影' else '写真からはじめる')
    st.caption('乾麺をできるだけ平らに広げ、重なりを少なくして、真上から撮影してください。')
    st.caption('暗く無地で反射の少ない背景を使い、麺に似た黄色い物を避けて、両端が写るようにしてください。')
    st.button('撮り直す・写真を選び直す', on_click=start_capture, args=(source_kind,), key='retake_photo')
    modes = ['広げた状態', '束の断面']
    admin = st.session_state.get('_app_admin', False)
    mode = modes[0]
    if admin:
        modes.append('束の側面・上面（回帰用）')
        if st.session_state.get('capture_mode') not in modes:
            st.session_state['capture_mode'] = st.session_state.get('capture_last_mode', modes[0])
            if st.session_state['capture_mode'] not in modes:
                st.session_state['capture_mode'] = modes[0]
        mode=st.selectbox('乾麺の置き方', modes, key='capture_mode',
            format_func=lambda x: {'広げた状態':'広げている','束の断面':'束の先端を写している','束の側面・上面（回帰用）':'束を横・上から写している（現在は未対応）'}[x],
            help='写真に合った置き方を選んでください。',on_change=_remember_mode)
    photo = None
    using_demo = admin and source_kind.startswith('サンプル')
    if using_demo and mode == '束の側面・上面（回帰用）':
        st.info('この状態のCGサンプルはありません。対応する実写真を使用してください。')
    elif using_demo:
        path = BASE_DIR / 'sample_images' / ('demo_spread_12.png' if mode == '広げた状態'
                                             else 'demo_bundle_16.png')
        photo = Image.open(path)
        st.caption('表示されるのは**CGのテスト画像**です。実際の乾麺写真ではありません。')
    elif source_kind == '写真をアップロード':
        photo = st.file_uploader('JPEG / PNG形式の乾麺写真', type=['jpg', 'jpeg', 'png'],
                                 key=f'upload_pasta_{generation}', on_change=_clear_previous_capture)
    else:
        photo = st.camera_input('乾麺を撮影', key=f'camera_pasta_{generation}', on_change=_clear_previous_capture)
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
        if isinstance(exc, (UnidentifiedImageError, Image.DecompressionBombError, ValueError)):
            log_info(WEIGHT, '写真入力を見送り')
        else:
            log_error(WEIGHT, '写真の読み込み 失敗')
        st.error('写真を読み込めませんでした。JPEGまたはPNG（25MB・2500万画素以下）を選び直してください。iPhoneのHEIC写真はJPEGに変換してください。')
        st.button('写真を選び直す',key='decode_retake',on_click=start_capture,args=('写真をアップロード',),width='stretch')
        st.button('はかりで測った重さを記録する',key='decode_manual',on_click=start_manual_weight,width='stretch')
        return
    if not admin:
        st.session_state['result_request'] = dict(
            rgb=preview, photo=photo, product=STANDARD_DRY_PASTA, mode=mode,
            using_demo=False, using_regression=False, parameters=dict(DEFAULT_SPREAD),
            raw_sha256=hashlib.sha256(photo.getvalue()).hexdigest() if hasattr(photo, 'getvalue') else None,
        )
        st.rerun()
    st.image(preview, caption='撮影した写真 · 向きは自動補正済み', width='stretch')
    st.caption('麺の全体が写っているか確認してください。撮影状態は手動で選択します。')
    st.subheader('写真の確認')
    try:
        products = list_products()
    except (OSError,sqlite3.Error):
        log_error(WEIGHT, '製品情報の読み込み 失敗')
        st.error('製品情報を開けませんでした。写真は保持しています。少し待って、もう一度お試しください。')
        if st.button('製品情報を読み直す',key='products_retry'):st.rerun()
        return
    product_ids = {p['id']: p for p in products
                   if st.session_state.get('_app_admin') or not p['name'].startswith('デモ用')}
    if not product_ids:
        st.info('使用する製品を登録してください。写真は保持しています。')
        st.button('製品を登録する', key='capture_add_product', on_click=open_calibration)
        return
    product = select_measurement_product(product_ids, '重量・栄養計算に使う製品', 'which_product', using_demo=using_demo)
    if product is None:
        st.info('使った製品を選ぶと、この写真を解析できます。')
        return
    if not using_demo and mode == '広げた状態' and hasattr(photo, 'getvalue'):
        from area_ui import render_area_candidate
        if render_area_candidate(preview, product, photo.getvalue()):
            return
    if product['name'].startswith('デモ用'):
        st.warning('現在の製品データは**仮の数値**です。実際の重量・カロリーではありません。'
                   '設定の「製品・校正」で自分の製品を登録してください。')
    elif strand_calibration(product) is None:
        st.info('この製品は重量の校正が未設定です。推定できない場合も、はかりで測った重さを入力して保存できます。')
    if not using_demo and (product['name'].startswith('デモ用') or strand_calibration(product) is None):
        st.button('製品を登録・校正する', key='capture_calibration', on_click=open_calibration, args=(product,))

    developer = st.session_state.get('dev_tools_enabled', False)
    method = st.session_state.get('research_method', 'OpenCV本数検出') if developer else 'OpenCV本数検出'
    using_regression = method.startswith('学習')
    active_model = None
    model_error = None
    if using_regression:
        from training_manager import get_active_model
        try:
            active_model = get_active_model()
        except (OSError, ValueError, sqlite3.Error):
            log_error(WEIGHT, '選択したモデルの読み込み 失敗')
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
    if not st.session_state.get('_app_admin'):
        return
    with st.expander('写真と重量の診断'):
        st.caption('写真：読み込み済み')
        _diagnostic_product(product)
        st.write('パスタの色の候補・自動検出：結果なし')
        st.write(f'重量計算：未実行。{reason}')


def _count_diagnostic(result, product, mode, using_demo):
    """Use one decision for the displayed diagnosis, weight and save action."""
    manual_count = None
    if not st.session_state.get('_app_admin'):
        return decide_count_weight(product, result, mode=mode, using_demo=using_demo), False
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


def _log_estimate_outcome(request, outcome):
    if request.get('_logged_estimate_outcome') != outcome:
        log_info(WEIGHT, outcome)
        request['_logged_estimate_outcome'] = outcome


def _change_placement(request):
    mode = st.session_state['result_capture_mode']
    request['mode'] = mode
    request['parameters'] = dict(DEFAULT_SPREAD if mode == '広げた状態' else DEFAULT_BUNDLE)
    for key in ('_detection_result', '_geometry', '_logged_estimate_outcome'):
        request.pop(key, None)


def _optional_result_product(request):
    try:
        products = {p['id']: p for p in list_products() if not p['name'].startswith('デモ用')}
    except (OSError, sqlite3.Error):
        log_error(WEIGHT, '製品情報の読み込み 失敗')
        st.warning('登録済みの製品を読み込めませんでした。写真は保持しています。一般的な乾麺として実測重量を入力できます。')
        if st.button('製品情報を読み直す', key='result_product_retry'):
            st.rerun()
        return STANDARD_DRY_PASTA
    if not products:
        request['product'] = STANDARD_DRY_PASTA
        return STANDARD_DRY_PASTA
    preferred = (request['product']['id'] if request.get('product_initialized')
                 else st.session_state.get('last_product_id'))
    from settings_ui import select_optional_product
    product = select_optional_product(products, '使った乾麺の製品（任意）', 'which_product', preferred)
    previous = request['product']
    if product != previous:
        request['product'] = product
        request.pop('_logged_estimate_outcome', None)
        if request.get('product_initialized') and product['id'] == previous['id']:
            # Refresh a changed calibration before accepting a pending save.
            st.session_state['capture_product_refreshed'] = True
            st.rerun()
    request['product_initialized'] = True
    return product


def _edge_contact_tables(diagnostic):
    summary = []
    for side, label in _EDGE_SIDE_LABELS.items():
        row = {'画像の辺': label}
        for retained, prefix in ((True, '残した'), (False, '除いた')):
            contacts = [contact for contact in diagnostic.contacts
                        if contact.side == side and contact.retained == retained]
            row[f'{prefix}成分数'] = len({contact.component for contact in contacts})
            row[f'{prefix}接触画素数'] = sum(contact.end - contact.start + 1 for contact in contacts)
        summary.append(row)
    st.dataframe(summary, hide_index=True, width='stretch')
    if not diagnostic.contacts:
        st.caption('画像の端に触れた候補はありません。')
        return

    height, width = diagnostic.overlay.shape[:2]
    st.caption(f'解析画像は幅 {width} × 高さ {height} px です。左上を (0, 0) とし、x は右へ、y は下へ増えます。'
               '接触区間は両端を含みます。角の画素や複数の辺に触れた成分は、触れた辺ごとに数えます。')
    runs = []
    for contact in diagnostic.contacts:
        if contact.side in ('top', 'bottom'):
            y = 0 if contact.side == 'top' else height - 1
            start, end = (contact.start, y), (contact.end, y)
        else:
            x = 0 if contact.side == 'left' else width - 1
            start, end = (x, contact.start), (x, contact.end)
        runs.append({
            '画像の辺': _EDGE_SIDE_LABELS[contact.side],
            '判定': '残した' if contact.retained else '除いた',
            '接触区間 (x, y)': f'{start}～{end}',
            '成分 ID': contact.component,
            '接触画素数': contact.end - contact.start + 1,
            '成分全体の画素数': contact.component_pixels,
            '端から内側への長さ (px)': contact.inward_px,
        })
    st.dataframe(runs, hide_index=True, width='stretch', height=240)


def _show_density_diagnostic(diagnostic):
    ratio = diagnostic.thickness_ratio
    if ratio is None:
        st.caption('候補領域が小さいため、麺の重なりや密集を判定できませんでした。')
        return
    st.caption(f'密集の判定に使った値は、画像全体に対する候補面積が {diagnostic.mask_fraction:.2%}'
               f'（上限 {MAX_MASK_FRACTION:.0%}）、候補の長さに対する厚みの指標が {ratio:.2%}'
               f'（上限 {MAX_THICKNESS_RATIO:.1%}）です。')
    if diagnostic.density_rejected:
        limits = []
        if diagnostic.mask_fraction > MAX_MASK_FRACTION:
            limits.append('候補面積')
        if ratio > MAX_THICKNESS_RATIO:
            limits.append('厚みの指標')
        detail = f"{'と'.join(limits)}が上限を超えました。" if limits else ''
        st.warning(detail + THICK_CAPTURE_REASON)


def _show_area_diagnostic(request):
    with st.expander('乾麺の抽出結果を見る'):
        if '_area_diagnostic' not in request:
            try:
                request['_area_diagnostic'] = diagnose(request['rgb'])
            except Exception:
                log_error(WEIGHT, '乾麺の抽出結果の可視化 失敗')
                request['_area_diagnostic'] = None
        diagnostic = request['_area_diagnostic']
        if diagnostic is None:
            st.caption('この写真の抽出結果を表示できませんでした。')
            return None
        st.image(diagnostic.overlay, caption='解析に使った写真に抽出結果を重ねています。',
                 width='stretch', output_format='PNG')
        st.caption('緑は乾麺の候補、オレンジは判定から除いた画像端の候補、赤は見切れの可能性がある画像端の候補です。'
                   '画像端の赤い画素が縮小表示で見えにくい場合は、表の接触区間で位置を確認できます。')
        _edge_contact_tables(diagnostic)
        _show_density_diagnostic(diagnostic)
        return diagnostic


def render_results(request):
    from manual_weight_ui import recovery
    rgb=request['rgb']; product=request['product']; mode=request['mode']
    using_demo=request['using_demo']; using_regression=request['using_regression']
    admin = st.session_state.get('_app_admin', False)
    if admin:
        st.button('写真・設定を変更',icon=':material/arrow_back:',on_click=edit_capture,key='edit_capture')
    st.image(rgb,caption='乾麺の写真' + (' · CGサンプル' if using_demo else ''),width='stretch')
    if not admin and 'saved_product' not in request:
        with st.expander('束の先端を撮影した場合'):
            st.selectbox('乾麺の置き方', ['広げた状態', '束の断面'],
                         index=0 if mode == '広げた状態' else 1,
                         key='result_capture_mode', on_change=_change_placement, args=(request,))
    if 'saved_product' in request:
        product = request['saved_product']
    elif admin:
        try:
            product = get_product(product['id'])
        except (OSError, sqlite3.Error):
            log_error(WEIGHT, '校正情報の読み直し 失敗')
            st.error('選んだ製品の校正値を読み込めません。写真は保持しています。少し待って製品情報を読み直してください。')
            if st.button('製品情報を読み直す', key='result_product_retry'):
                st.rerun()
            return
        if product is None:
            st.warning('選んだ製品が見つかりません。「写真・設定を変更」からお使いの製品を選び直してください。')
            return
        if product != request['product']:
            request['product'] = product
            request.pop('_logged_estimate_outcome', None)
            st.session_state['capture_product_refreshed'] = True
            # Discard an in-flight save click intended for the previous calibration.
            st.rerun()
    if st.session_state.pop('capture_product_refreshed', False):
        st.info('製品情報が更新されました。新しい校正値による結果を確認してください。')
    use_area = not admin and not using_demo and not using_regression and mode == '広げた状態'
    area_diagnostic = _show_area_diagnostic(request) if use_area else None
    if not using_demo:
        if '_quality_warnings' not in request:
            request['_quality_warnings'] = image_quality_warnings(rgb)
        problems = request['_quality_warnings']
        if problems:
            _log_estimate_outcome(request, '画質を確認して重量推定を見送り')
            st.warning(' '.join(problems))
            if not using_regression:
                _unfinished_count_diagnostic(product, '画像の明るさ・大きさ・ピントを確認してください。')
            if not admin and 'saved_product' not in request:
                product = _optional_result_product(request)
            recovery(product,rgb,synthetic=using_demo)
            return
    area_reason = None
    if use_area:
        if '_area_estimate' not in request:
            try:
                with st.spinner('写真から乾麺の面積を調べています…'):
                    request['_area_estimate'] = estimate_area(rgb)
            except Exception:
                log_error(WEIGHT, '面積による重量推定 失敗')
                request['_area_estimate'] = None
        area = request['_area_estimate']
        if area is not None and area.weight_g is not None:
            if 'saved_product' not in request:
                product = _optional_result_product(request)
            _log_estimate_outcome(request, '面積による重量推定 完了')
            diagnostics = trace(rgb, estimator=AREA_ESTIMATOR, product=product, mode=mode,
                                raw_sha256=request.get('raw_sha256'),
                                parameters=dict(feature=area.area_feature,
                                                area_ratio=getattr(area.observation, area.area_feature),
                                                isolated_border_speck_filter=True,
                                                boundary_tolerance_g=AREA_BOUNDARY_TOLERANCE_G))
            render_weight_result(weight=area.weight_g, product=product, rgb=rgb, mode=mode,
                                 using_demo=False, detected_count=0, estimator=AREA_ESTIMATOR,
                                 diagnostics=diagnostics, explanation='乾麺の可視面積による概算')
            return
        area_reason = area.reason if area is not None else '面積の解析を完了できませんでした。'
        if area_reason in (EDGE_CAPTURE_REASON, THICK_CAPTURE_REASON):
            if area_reason == EDGE_CAPTURE_REASON:
                _log_estimate_outcome(request, '写真の見切れを確認して重量推定を見送り')
                message = '写真の端で麺が切れている可能性があるため、重さを推定できませんでした。'
                if area_diagnostic is not None:
                    touched = {contact.side for contact in area_diagnostic.contacts if contact.retained}
                    sides = '・'.join(label for side, label in _EDGE_SIDE_LABELS.items() if side in touched)
                    if sides:
                        message += f'{sides}に触れた乾麺の候補があります。'
                    if area_diagnostic.density_rejected:
                        message += '麺の候補が密集していて、重なりを判断できません。隙間を作って平らに広げてください。'
                message += '麺全体が入るように撮り直すか、下に実測重量を入力してください。'
            else:
                _log_estimate_outcome(request, '麺の重なり・密集を確認して重量推定を見送り')
                message = THICK_CAPTURE_REASON + '撮り直すか、下に実測重量を入力してください。'
            st.warning(message)
            if 'saved_product' not in request:
                product = _optional_result_product(request)
            recovery(product, rgb, synthetic=using_demo)
            return
    if using_regression:
        succeeded = False
        if request['model_error']:
            st.error(f"モデルを読み込めません: {request['model_error']}")
        elif request['active_model'] is None:
            _log_estimate_outcome(request, '有効なモデルがなく重量推定を見送り')
            st.warning('有効なモデルがありません。設定のAI学習管理でモデルを学習・選択してください。')
        else:
            with st.spinner('画像を解析しています…'):
                try:
                    succeeded = render_regression_measurement(request['active_model'],rgb=rgb,product=product,mode=mode,using_demo=using_demo,show_image=False)
                except Exception:
                    log_error(WEIGHT, '回帰による重量推定 失敗')
                    st.warning('重量を推定できませんでした。実測入力で記録するか、写真を選び直してください。')
                else:
                    _log_estimate_outcome(request, '回帰による重量推定 完了' if succeeded else '回帰による重量推定を見送り')
        if not succeeded:recovery(product,rgb,synthetic=using_demo)
        return
    try:
        result=request.get('_detection_result')
        if result is None:
            with st.spinner('画像を解析しています…'):
                if mode=='広げた状態':result=detect_spread(rgb,**request['parameters'])
                elif mode=='束の断面':result=detect_bundle_ends(rgb,**request['parameters'])
                else:raise ValueError('この撮影状態は現在の測定に対応していません。')
                if not admin:
                    request['_geometry'] = mask_geometry(result.mask)
                    log_info(WEIGHT, '乾麺の画像解析 完了')
            # A request is replaced when its photo, placement or settings change.
            # Keep this private result in this session only, not a shared cache.
            request['_detection_result']=result
    except Exception:
        log_error(WEIGHT, '乾麺の本数検出 失敗')
        st.error('本数の検出を完了できませんでした。もう一度試すか、写真を選び直してください。')
        _unfinished_count_diagnostic(product, '本数の検出処理でエラーが発生しました。')
        if st.button('再試行する'):st.rerun()
        if not admin and 'saved_product' not in request:
            product = _optional_result_product(request)
        recovery(product,rgb,synthetic=using_demo)
        return
    if not admin:
        if '_geometry' not in request:  # A session may already hold the count result.
            request['_geometry'] = mask_geometry(result.mask)
        if request['_geometry']['mask_area_px']:
            st.caption('乾麺候補の領域と形状を解析しました。')
        if 'saved_product' not in request:
            product = _optional_result_product(request)
    try:
        decision, manual = _count_diagnostic(result, product, mode, using_demo)
    except Exception:
        log_error(WEIGHT, '検出本数による重量計算 失敗')
        st.error('写真の重量を計算できませんでした。撮り直すか、実測重量を入力してください。')
        recovery(product, rgb, synthetic=using_demo)
        return
    if st.session_state.get('_app_admin'):
        st.session_state['dev_last_detection'] = dict(
            overlay=result.overlay_rgb, mask=result.mask, count=result.count, parameters=request['parameters'],
            product_name=product['name'], calibration=decision.calibration_g_per_strand,
            reasons=decision.reasons, weight_g=decision.weight_g, manual=manual, using_demo=using_demo)
    if decision.weight_g is None:
        _log_estimate_outcome(request, '検出結果・校正条件を確認して重量推定を見送り')
        if admin:
            st.warning('重量を計算できません。' + ' '.join(decision.reasons))
        else:
            if area_reason:
                st.warning('面積から重量を推定できませんでした。' + area_reason)
            else:
                photo_reasons = [reason for code, reason in zip(decision.reason_codes, decision.reasons)
                                 if code != 'uncalibrated']
                st.warning('写真から重さを推定できませんでした。' +
                           (' '.join(photo_reasons) or 'この撮影方法では製品の1本重量が必要です。'))
        recovery(product, rgb, synthetic=using_demo)
        return

    if manual:
        _log_estimate_outcome(request, '手入力の本数による重量計算 完了')
        st.success('入力した本数から重量を計算しました。')
    elif using_demo:
        _log_estimate_outcome(request, 'CGサンプルの重量計算 完了')
        st.success('CGサンプルの重量計算ができました。')
    else:
        _log_estimate_outcome(request, '検出本数による重量計算 完了')
        st.success('写真の検出本数と製品の校正値から重量を推定しました。')
    estimator = 'manual_count' if manual else 'opencv'
    diagnostics = trace(rgb, estimator=estimator, product=product, mode=mode,
                        raw_sha256=request.get('raw_sha256'), parameters=dict(
                            automatic_count=decision.automatic_count, used_count=decision.used_count,
                            foreground_pixels=decision.foreground_pixels, image_pixels=decision.image_pixels,
                            detection=request['parameters'],
                            **({'projected_geometry': request['_geometry']} if not admin else {})))
    render_weight_result(weight=decision.weight_g, product=product, rgb=rgb, mode=mode,
                         using_demo=using_demo, detected_count=decision.used_count, estimator=estimator,
                         diagnostics=diagnostics,
                         explanation=f'計算：{decision.used_count}本 × '
                                     f'{decision.calibration_g_per_strand:.4f}g/本 = {decision.weight_g:.2f}g。')
