"""Authenticated, local one-photo trial for the public-style capture flow."""

import hashlib

import streamlit as st

from .practical_capture import WidthReference, analyze_capture
from .strand_scale import AGREED


def render_practical_capture():
    if not st.session_state.get('_app_admin'):
        return
    st.subheader('概算重量の撮影試験')
    st.caption('管理者用の研究機能です。推定は検証中で、結果は食事記録に保存されません。')
    with st.expander('試験の参照係数', expanded='practical_width_profile' not in st.session_state):
        st.caption('「20g基準の面積換算」で作成した参照係数JSONを一度読み込みます。')
        reference = st.file_uploader('参照係数JSON', type=['json'], key='practical_width_file')
        if reference is not None:
            try:
                st.session_state['practical_width_profile'] = WidthReference.from_json(reference.getvalue())
            except ValueError as exc:
                st.session_state.pop('practical_width_profile', None)
                st.error(str(exc))
                return
        if 'practical_width_profile' in st.session_state:
            st.success('参照係数を読み込みました。')
    profile = st.session_state.get('practical_width_profile')
    if profile is None:
        st.info('参照係数を読み込むと、写真1枚で試せます。')
        return

    st.caption('乾麺を平らに広げ、真上から撮り、麺の両端を画面に入れてください。')
    source = st.radio('写真の取り込み方', ['写真を選ぶ', 'カメラで撮影する'], horizontal=True,
                      key='practical_photo_source')
    if source == 'カメラで撮影する':
        photo = st.camera_input('乾麺を撮影する', key='practical_camera')
    else:
        photo = st.file_uploader('乾麺の写真', type=['jpg', 'jpeg', 'png', 'webp'], key='practical_photo')
    if photo is None:
        return
    content = photo.getvalue()
    capture_key = (hashlib.sha256(content).hexdigest(), profile)
    cached = st.session_state.get('practical_last_capture')
    if not cached or cached[0] != capture_key:
        try:
            with st.spinner('麺の輪郭と幅を確認しています…'):
                result = analyze_capture(content, profile)
        except (OSError, ValueError) as exc:
            st.error(f'写真を解析できませんでした。{exc}')
            return
        st.session_state['practical_last_capture'] = (capture_key, result)
    else:
        result = cached[1]
    if result.estimated_g is None:
        st.warning(result.reason)
    else:
        st.metric('試験の概算重量', f'約{result.estimated_g:.0f}g')
        if result.method == AGREED:
            st.caption('色境界を複数の解像度で確認した研究用の推定です。')
        else:
            st.caption('現行の面積比方式による推定です。')

    with st.expander('判定の詳細'):
        if result.current_g is None:
            st.write(f'現行方式の見送り。{result.current_reason}')
        else:
            st.write(f'現行方式で表示した重量は約{result.current_g:.0f}gです。')
        if result.signals.wide_fraction is not None:
            st.write(f'広く連結した前景の割合 {result.signals.wide_fraction:.0%}')
        if result.signals.mixed_direction_fraction is not None:
            st.write(f'方向が混じる色境界の割合 {result.signals.mixed_direction_fraction:.0%}')
        st.caption('画像に見える特徴です。密集した麺の層数や隠れた重量は確認できません。')
