"""Developers' A4 capture preview. Measurements stay separate from the diary."""
import json
from pathlib import Path
import sys

import streamlit as st

from capture_guide_ui import render_sheet_guide

# Streamlit can be started with the app folder as its only import root.
repository = str(Path(__file__).resolve().parent.parent)
if repository not in sys.path:
    sys.path.insert(0, repository)
from pasta_weight_app_v2.experiments.capture_preflight import pilot_template_csv, review_preflight

MARKER_MESSAGES = {
    'all_four_expected_markers_required': '四隅のマーカー0〜3をすべて入れてください。',
    'sheet_clipped': '用紙の外周まで余白を入れて撮影してください。',
    'marker_near_image_edge': 'マーカーの周りに余白を入れてください。',
    'markers_too_small': 'マーカーが小さく写っています。用紙全体が入る範囲で距離を縮めてください。',
    'sheet_too_small': 'シートが小さく写っています。近づいて撮影してください。',
}


def render_capture_preflight():
    st.subheader('A4基準の試し撮り')
    render_sheet_guide()
    st.markdown('まず、約55cmの真上から**シートだけ**を1枚撮ります。次に、'
                '**密集した200g前後の束**を置き、約35cmの真上から1枚撮ります。'
                '余裕があれば同じ束を約55cm・真上から15度傾けてもう1枚撮ります。')
    st.caption('距離はレンズから紙面中央までです。光沢トレーで反射が出る場合は、光源を拡散するか反射の少ない無地の台を使ってください。')
    st.download_button('6試料・20枚の撮影台帳をダウンロード', pilot_template_csv().encode('utf-8-sig'),
                       file_name='pasta_a4_pilot_20.csv', mime='text/csv', key='preflight_plan', on_click='ignore')
    photo = st.file_uploader('試し撮りの原本', type=['jpeg', 'jpg', 'png', 'webp'], key='preflight_photo')
    with st.form('capture_preflight'):
        role = st.selectbox('写真の種類', ['empty_sheet', 'dense'],
                            format_func={'empty_sheet': 'シートだけ', 'dense': '密集した乾麺'}.get)
        line = st.number_input('定規で測った印刷確認線（mm）', min_value=1., value=None, step=.1)
        flat = st.checkbox('紙を平らな台に固定した')
        distance = st.number_input('レンズから紙面中央までの実測距離（cm）', min_value=1., value=None, step=1.)
        tilt = st.number_input('真上からの角度（度）', min_value=0., max_value=90., value=None, step=1.)
        height = st.number_input('束の高さ（mm・乾麺を写した場合）', min_value=0., value=None, step=1.)
        submit = st.form_submit_button('マーカーと前景を確認', type='primary')
    if not submit:
        return
    if photo is None:
        st.warning('確認する写真を選んでください。')
        return
    try:
        report, overlay = review_preflight(photo.getvalue(), photo_role=role, printed_line_mm=line,
                    flat_sheet_confirmed=flat, camera_distance_cm=distance, tilt_deg=tilt, bundle_height_mm=height)
    except ValueError:
        st.warning('写真や計測値を読み込めませんでした。JPEG・PNG・WebPの原本を選び直し、計測値を確認してください。')
        return
    geometry = report['geometry']
    reference = geometry['reference']
    st.image(overlay, caption='マーカー、紙面の外周、麺の候補輪郭', width='stretch')
    st.write('検出したマーカーID', sorted(reference['detected_ids']))
    if geometry['reference_geometry_valid']:
        st.success('四隅を認識し、紙面の座標を計算できました。')
    else:
        for reason in reference['reasons']:
            st.warning(MARKER_MESSAGES.get(reason, '紙を平らに置き、反射を避けて真上から撮り直してください。'))
    if not report['within_print_line_tolerance'] or not flat:
        st.warning('印刷線を定規で測り、100mmからのずれが1mm以内か確認してください。紙を平らに置くと寸法比較ができます。')
    if role == 'dense':
        projection = geometry['sheet_plane_projection']
        if projection is None:
            st.warning('麺の候補を紙面に投影できませんでした。麺の両端を含め、輪郭が背景や画像端へはみ出していないか確認してください。')
        else:
            st.write(f'紙面への投影面積 {projection["nominal_area_mm2"]:.0f} mm²、'
                     f'投影長さ {projection["nominal_length_mm"]:.1f} mm、投影幅 {projection["nominal_width_mm"]:.1f} mm')
            if geometry['projection_scale_status'] == 'nominal_print_unverified':
                st.caption('仮寸法です。印刷寸法と紙面を確認してください。')
            st.caption('表示は候補輪郭の紙面投影です。原本と見比べて両端・反射を確認してください。')
    st.download_button('この写真の検品結果をダウンロード', json.dumps(report, indent=2, ensure_ascii=False),
                       file_name='a4_preflight_review.json', mime='application/json', key='preflight_review', on_click='ignore')
