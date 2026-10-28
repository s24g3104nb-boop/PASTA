"""Pasta Lens: everyday shell, with research tools isolated in settings."""
from pathlib import Path
import streamlit as st
from storage import init_db
from generate_demo import generate_images
from ui_shell import shell, PAGES, persist_appearance
from home_ui import render_home
from capture_ui import render_capture
from history_ui import render_history
from cooking_ui import render_cooking_page
from settings_ui import render_products, render_guide
from training_ui import render_training_tab

st.set_page_config(page_title='Pasta Lens | ひと皿を、ちょうどよく。',page_icon='🍝',layout='centered')
init_db()
base=Path(__file__).resolve().parent
if not all((base/'sample_images'/name).is_file() for name in ('demo_spread_12.png','demo_bundle_16.png')):
    generate_images()
page=shell()
if page==PAGES[0]:
    if st.session_state.get('capture_source'):render_capture()
    else:render_home()
elif page==PAGES[1]:render_history()
elif page==PAGES[2]:
    st.title('料理と栄養')
    st.caption('乾麺の記録に、使った食材をプラス。')
    render_cooking_page()
else:
    st.title('設定')
    section=st.selectbox('設定メニュー',['表示・使い方','製品・校正','AI学習管理（管理者向け）'],key='settings_section')
    if section=='表示・使い方':
        with st.container(border=True,key="card_appearance"):
            st.subheader('あなたに合った表示')
            st.selectbox('カラーテーマ',['端末に合わせる','ライト','ダーク'],index=['端末に合わせる','ライト','ダーク'].index(st.session_state.get('theme_preference','端末に合わせる')),key='appearance',on_change=persist_appearance)
            st.caption('OSの「視差効果を減らす」設定に対応しています。')
        with st.expander('撮影・栄養計算のガイド'):render_guide()
        st.caption('写真と履歴はアプリを起動したPCに保存されます。管理画面の分離は認証機能ではありません。')
        with st.expander('スマートフォンの接続・カメラ'):
            st.write('カメラへのアクセスを許可してください。開けない場合はブラウザのサイト設定を確認するか、標準カメラで撮って「写真から選ぶ」を使ってください。ブラウザ内カメラにはHTTPSまたはlocalhostが必要です。')
    elif section=='製品・校正':render_products()
    else:
        st.caption('管理者向け · 学習データ、画像検品、評価とモデル管理')
        render_training_tab()
