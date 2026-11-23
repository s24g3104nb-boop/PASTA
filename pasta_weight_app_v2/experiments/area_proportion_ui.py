"""Local comparison of visible-area estimates for the authenticated admin app."""

import streamlit as st

from ..pasta_vision import image_to_rgb
from .area_proportion import (DESCRIPTIONS, EVALUATED_MAX_G, FEATURES, PRIMARY, analyze, calibrate,
                              overlay, predict, yellow_masks)
from .strand_scale import (AGREED, COLOR_EDGE, MAX_SIDE, NATIVE_MASK, calibrate_widths,
                           canonical_width, choose_estimator, measure_for_area, visible_density,
                           width_outcomes)

_WIDTH_LABELS = {NATIVE_MASK: '高解像度のマスク境界', COLOR_EDGE: '高解像度の色境界',
                 AGREED: '別の解像度でも一致した色境界'}
_SELECTION_LABELS = {**_WIDTH_LABELS, PRIMARY: '長い線候補', 'legacy': '現行方式'}


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
        inspect_scale = st.checkbox('元画像と複数解像度で麺幅を比較する', value=True)
        submitted = st.form_submit_button('面積と重量を比較する')
    if not submitted:
        return
    if len(references) != 3 or target is None:
        st.info('基準写真を3枚と、推定する写真を1枚選んでください。')
        return
    try:
        with st.spinner('乾麺の領域と面積を調べています…'):
            reference_evidence = [analyze(image_to_rgb(photo)) for photo in references]
            model = calibrate(reference_evidence)
            evidence = analyze(image_to_rgb(target))
            if inspect_scale:
                pairs = [(ev, measure_for_area(image_to_rgb(photo, max_side=MAX_SIDE), ev))
                         for photo, ev in zip(references, reference_evidence, strict=True)]
                width_reference = calibrate_widths(pairs)
                scale = measure_for_area(image_to_rgb(target, max_side=MAX_SIDE), evidence)
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
    if evidence.strand_width.width_px is not None:
        st.write(f'分離した麺の代表幅 {evidence.strand_width.width_px:.2f} px')
    if inspect_scale:
        _show_scale(evidence, scale, width_reference, model)
    st.image(evidence.scene, caption='入力画像')
    for name, mask in yellow_masks(evidence.scene).items():
        mask_column, overlay_column = st.columns(2)
        mask_column.image(mask.astype('uint8') * 255, caption=f'{name}のマスク')
        overlay_column.image(overlay(evidence.scene, mask), caption=f'{name}の重ね合わせ')


def _show_scale(evidence, scale, reference, previous_model):
    st.subheader('元画像と複数解像度の麺幅')
    st.caption(f'解析時の長辺 {scale.long_side_px}px（上限{MAX_SIDE}px）、'
               f'照合時の長辺 {scale.check_long_side_px}px。幅は長辺960px相当で表示します。')
    outcomes = width_outcomes(evidence, scale, reference)
    for method, title in _WIDTH_LABELS.items():
        st.markdown(f'**{title}**')
        width = canonical_width(scale, method)
        if width is not None:
            st.write(f'代表幅 {width:.2f} px')
        result = outcomes[method]
        if result['estimated_g'] is None:
            st.write(f"見送り。{result['reason']}")
        else:
            st.write(f"約 {result['estimated_g']:.1f} g。{result['status']}")
    density = visible_density(evidence, scale)
    if density is not None:
        st.caption(f'幅広くつながる前景の割合 {density:.0%}。見えている領域から算出しています。')
    choice = choose_estimator(evidence, scale, outcomes, previous_model)
    if choice['estimated_g'] is not None:
        name = _SELECTION_LABELS[choice['method']]
        status = outcomes[choice['method']]['status'] if choice['method'] == AGREED else ''
        if choice['estimated_g'] > EVALUATED_MAX_G and not status:
            status = '外挿・未検証'
        st.write(f"切り替え方式　{name}で約 {choice['estimated_g']:.1f} g。{status or '探索的な概算・未検証'}")
    else:
        st.write(f"切り替え方式は見送り。{choice['reason']}")
