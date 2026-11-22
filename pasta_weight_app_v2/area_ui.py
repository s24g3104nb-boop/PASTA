"""Optional locally installed segmenter and calibrated A4 area measurement."""
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
from threading import Lock
from uuid import uuid4

import numpy as np
import sqlite3
import streamlit as st

from storage import save_measurement
from pasta_vision import image_to_rgb, nutrition_for_weight
from nutrition_state import has_nutrition, snapshot_for_nutrition
from operation_logs import WEIGHT, log_error, log_info
from capture_guide_ui import render_sheet_guide

from pasta_weight_app_v2.experiments.area_calibration import predict_area
from pasta_weight_app_v2.experiments.foundation_backends import MAX_SIDE, SEED, hsv_mask, start_segmenter
from pasta_weight_app_v2.experiments.foundation_masks import a4_projection
from pasta_weight_app_v2.experiments.foundation_resources import torch_runtime
from pasta_weight_app_v2.experiments.stability_pipeline import shape_features


@st.cache_resource
def _cached_segmenter(path, size, mtime, device):
    torch = torch_runtime(True, device)
    return start_segmenter('mobile_sam', Path(path), device, torch), Lock()


def _run_model(rgb, line, flat, layer):
    checkpoint = Path(os.environ['PASTA_MOBILESAM_CHECKPOINT']).resolve()
    info = checkpoint.stat()
    device = os.environ.get('PASTA_LOCAL_VISION_DEVICE', 'cpu')
    if device not in ('cpu', 'cuda'):
        raise ValueError('Device must be cpu or cuda')
    runner, lock = _cached_segmenter(str(checkpoint), info.st_size, info.st_mtime_ns, device)
    with lock:
        mask = runner(rgb)
    projection = a4_projection(rgb, mask, dict(printed_line_mm=line, flat_sheet_confirmed=flat) if line else None)
    projected_area = projection['features'][0] if projection['features'] and layer else None
    color_mask = hsv_mask(rgb, lab=True)
    union = np.count_nonzero(mask | color_mask)
    agreement = np.count_nonzero(mask & color_mask) / union if union else 0.

    overlay = rgb.copy()
    overlay[mask] = np.rint(rgb[mask]*.55 + np.array([0, 200, 200])*.45).astype(np.uint8)
    return dict(overlay=overlay, shape=shape_features(mask, SEED), projection=projection,
                backend=runner.metadata, area_mm2=projected_area, agreement=agreement, token=uuid4().hex, saved=None)


def render_area_candidate(preview, product, content):
    photo_id = sha256(content).hexdigest()
    with st.expander('A4シートを使って重さを推定（検証中）'):
        if not os.environ.get('PASTA_MOBILESAM_CHECKPOINT'):
            st.info('このPCでは画像認識モデルが未設定です。通常の推定か実測入力を使えます。')
            return
        st.caption('乾麺を重ねず、なるべく一層に広げます。紙全体と四隅のマーカーを真上から写してください。')
        render_sheet_guide()
        line = st.number_input('定規で測った印刷確認線（mm）', min_value=1., value=None, step=.1,
                               key='area_print_line')
        flat = st.checkbox('シートを平らな台に固定した', key='area_flat')
        layer = st.checkbox('乾麺をシートと同じ高さに一層で置いた', key='area_layer')
        signature = (photo_id, product['id'], line, flat, layer)
        if st.button('AIで乾麺の範囲を抽出する', key='area_run', width='stretch'):
            previous = st.session_state.get('a4_area_result')
            try:
                with st.spinner('このPCで乾麺の範囲を抽出しています…'):
                    result = _run_model(image_to_rgb(BytesIO(content), max_side=MAX_SIDE), line, flat, layer)
                if previous and previous['signature'] == signature:
                    result.update(token=previous['token'], saved=previous['saved'])
                result['signature'] = signature
                st.session_state['a4_area_result'] = result
                log_info(WEIGHT, '乾麺の面積抽出 完了')
            except (OSError, ValueError, ImportError, RuntimeError):
                log_error(WEIGHT, '乾麺の面積抽出 失敗')
                st.error('AIで抽出できませんでした。モデルの設定を確認するか、通常の解析・実測入力を使ってください。')
        result = st.session_state.get('a4_area_result')
        if not result or result['signature'] != signature:
            return
        if st.session_state.get('_app_admin'):
            st.image(result['overlay'], caption='青緑色がAIで抽出した可視領域', width='stretch')
            st.caption(f"画像内の占有率 {result['shape']['mask_fraction']:.1%}")
        area = result['area_mm2']
        if area is None:
            st.warning('実寸換算に必要な条件が揃っていません。印刷線、四隅のマーカー、全体の収まり、一層の配置を確認してください。')
            return
        st.metric('A4紙面上の投影面積', f'{area/100:.1f} cm²')
        if not st.session_state.get('_app_admin'):
            if result['agreement'] < .65:
                st.warning('背景や反射の影響で、麺の範囲を確認できませんでした。無地で光沢のない背景から撮り直してください。')
            else:
                return _render_estimate(result, preview, product, area)
            return
        _, length, width = result['projection']['features']
        st.caption(f'幅 {width:.0f} mm × 長さ {length:.0f} mm（紙面への投影）')
        if st.checkbox('抽出範囲に背景・反射の混入や麺の欠けがないことを確認した',
                       key='area_mask_review_'+result['token']):
            evidence = dict(original_sha256=photo_id, backend=result['backend'],
                            projection_status=result['projection']['status'], area_mm2=area,
                            flat_layer_and_mask_confirmed=True, product_name=product['name'])
            st.download_button('面積と検品記録を取得', json.dumps(evidence, ensure_ascii=False, indent=2),
                               file_name='pasta_area_review.json', mime='application/json',
                               key='area_review_download', on_click='ignore')
            return _render_estimate(result, preview, product, area)
        else:
            st.caption('写真と抽出範囲を見比べてから、実寸の検品記録を取得できます。')


def _render_estimate(result, preview, product, area):
    profile_path = os.environ.get('PASTA_AREA_CALIBRATION')
    weight = None
    if profile_path and not product['name'].startswith('デモ用'):
        try:
            profile = json.loads(Path(profile_path).read_text(encoding='utf-8-sig'))
            weight = predict_area(profile, product['name'], result['backend'], area)
        except (OSError, ValueError, KeyError, TypeError):
            log_error(WEIGHT, '面積換算条件の読み込み 失敗')
    if weight is None:
        st.info('この製品・範囲の重量換算は未設定または検証不足です。はかりで量った値は下の実測入力から記録できます。')
        return
    st.metric('乾麺の推定重量（面積換算・検証中）', f'{weight:.1f} g')
    actual = st.number_input('はかりで測った実測重量（任意・g）', min_value=.1, value=None, step=.1,
                             key='area_actual_'+result['token'], disabled=bool(result['saved']))
    nutrition = None
    if has_nutrition(product):
        nutrition = nutrition_for_weight(actual if actual is not None else weight, product['kcal_100'],
                                         product['protein_100'], product['fat_100'], product['carb_100'])
        st.caption(f"乾麺分 {nutrition['kcal']:.1f} kcal · {'実測重量' if actual is not None else '推定重量'}を使用")
    else:
        st.caption('栄養情報は未計算です。')
    if st.button('面積による測定を保存', key='area_save', disabled=bool(result['saved'])):
        try:
            result['saved'] = save_measurement(product_name=product['name'], mode='A4の一層配置',
                detected_count=0, estimated_g=weight, actual_g=actual, nutrition=nutrition, image_rgb=preview,
                source_kind='real', estimator='a4_area_calibrated', product_snapshot=snapshot_for_nutrition(product),
                submission_token=result['token'])
            st.rerun()
        except (OSError, sqlite3.Error, ValueError):
            st.error('測定を保存できませんでした。入力は残っています。保存先を確認してください。')
    if result['saved']:
        from diary_ui import start_diary
        st.success('面積による測定を保存しました。')
        st.button('この乾麺で料理を記録する', key='area_to_diary',
                  on_click=start_diary, kwargs={'measurement_id':result['saved']})
    return bool(result['saved'])
