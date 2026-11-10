"""Shared instructions only; no change to calibration or inference."""
from pathlib import Path
import streamlit as st


def render_sheet_guide():
    st.write('A4補助シートは研究用の撮影基準です。重量推定の精度を保証するものではありません。')
    st.markdown('1. **A4・100%・実際のサイズ**で印刷します。「用紙に合わせる」は無効にします。\n2. 確認線を定規で測り、**100mm**あることを確認します。\n3. 平らな台に置き、四隅のマーカーを隠さず、麺の両端まで余白付きで写します。\n4. 暗い・ぼけている・端が切れている場合は撮り直してください。')
    st.caption('マーカーが見えても印刷倍率は自動保証できません。厚い束の下側は分かりません。実印刷・iPhone実機での寸法精度は未検証です。')
    pdf=Path(__file__).resolve().parents[1]/'docs/assets/capture_sheet/Pasta_Lens_A4_reference_v1.pdf'
    if pdf.is_file():
        st.download_button('A4撮影補助シートをダウンロード',pdf.read_bytes(),file_name=pdf.name,mime='application/pdf',key='sheet_guide_download')
