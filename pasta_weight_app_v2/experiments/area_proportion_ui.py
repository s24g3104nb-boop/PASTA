"""Local comparison of visible-area estimates for the authenticated admin app."""

import streamlit as st

from ..pasta_vision import image_to_rgb
from .area_proportion import DESCRIPTIONS, FEATURES, analyze, calibrate, overlay, predict, yellow_masks


def render_area_proportion():
    if not st.session_state.get('_app_admin'):
        return
    st.subheader('20g基準の面積換算')
    st.caption('同じ20gの乾麺を置き直した写真3枚を基準に、可視面積から概算します。'
               '実寸と隠れた麺の量は測れません。研究用の推定は食事記録に保存されません。')
    with st.form('area_proportion_research'):
        references = st.file_uploader('20gの基準写真（同じ試料の配置違い3枚）',
                                      type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
        target = st.file_uploader('推定する乾麺の写真', type=['jpg', 'jpeg', 'png'])
        submitted = st.form_submit_button('面積と重量を比較する')
    if not submitted:
        return
    if len(references) != 3 or target is None:
        st.info('基準写真を3枚と、推定する写真を1枚選んでください。')
        return
    try:
        with st.spinner('乾麺の領域と面積を調べています…'):
            model = calibrate([analyze(image_to_rgb(photo)) for photo in references])
            evidence = analyze(image_to_rgb(target))
    except (OSError, ValueError) as exc:
        st.error(f'写真を解析できませんでした。{exc}')
        return
    st.caption('120gなら基準面積の6倍という仮説です。100gを超える推定は外挿・未検証です。')
    if evidence.legacy_weight_g is None:
        st.write(f'現行方式は見送り。{evidence.legacy_reason}')
    else:
        st.write(f'現行方式の概算 {evidence.legacy_weight_g:.1f} g')
    for method in FEATURES:
        result = predict(model, evidence, method)
        st.markdown(f'**{DESCRIPTIONS[method]}**')
        if result['estimated_g'] is None:
            st.write(f"見送り。{result['reason']}")
        else:
            st.write(f"約 {result['estimated_g']:.1f} g。{result['status']}")
    st.write(f'抽出面積 {evidence.area_px:,} px²（長辺960px）')
    st.image(evidence.scene, caption='入力画像')
    for name, mask in yellow_masks(evidence.scene).items():
        mask_column, overlay_column = st.columns(2)
        mask_column.image(mask.astype('uint8') * 255, caption=f'{name}のマスク')
        overlay_column.image(overlay(evidence.scene, mask), caption=f'{name}の重ね合わせ')
