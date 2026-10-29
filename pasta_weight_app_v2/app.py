"""Pasta Lens: everyday diary and measurement, with research tools in settings."""
from pathlib import Path
import logging
import sqlite3
import streamlit as st
from storage import init_db, DATA_DIR
from generate_demo import generate_images
from ui_shell import shell, PAGES, persist_appearance
from home_ui import render_home
from capture_ui import render_capture
from history_ui import render_history
from diary_ui import render_diary
from cooking_ui import render_cooking_page
from settings_ui import render_products, render_guide
from developer_ui import render_developer_tools

st.set_page_config(page_title='Pasta Lens | ひと皿を、思い出に。',page_icon='🍝',layout='centered')
try:
    init_db()
except (OSError, sqlite3.Error):
    logging.exception('Pasta Lens database initialization failed')
    st.error('データを開けませんでした。ほかのアプリで使用中の場合は閉じ、再試行してください。')
    if st.button('再試行する'):
        st.rerun()
    st.stop()
base=Path(__file__).resolve().parent
if not all((base/'sample_images'/name).is_file() for name in ('demo_spread_12.png','demo_bundle_16.png')):
    generate_images()
# Existing open browser sessions can move from the earlier four-page navigation.
if st.session_state.get('main_nav') == '履歴':
    st.session_state['main_nav']='記録'
    st.session_state['record_kind']='乾麺の測定履歴'
elif st.session_state.get('main_nav') == '料理・栄養':
    st.session_state['main_nav']='設定'
    st.session_state['settings_section']='料理の栄養計算'
page=shell()
if page==PAGES[0]:
    if st.session_state.get('capture_source'):
        render_capture()
    else:
        render_home()
elif page==PAGES[1]:
    kind=st.selectbox('記録の種類',['食事日記','乾麺の測定履歴'],key='record_kind')
    if kind=='食事日記':
        render_diary()
    else:
        render_history()
else:
    st.title('設定')
    section=st.selectbox('設定メニュー',['表示・使い方','製品・校正','料理の栄養計算','開発・検証ツール'],key='settings_section')
    if section=='表示・使い方':
        with st.container(border=True,key='card_appearance'):
            st.subheader('あなたに合った表示')
            st.selectbox('カラーテーマ',['端末に合わせる','ライト','ダーク'],index=['端末に合わせる','ライト','ダーク'].index(st.session_state.get('theme_preference','端末に合わせる')),key='appearance',on_change=persist_appearance)
        with st.expander('撮影・栄養計算のガイド'):
            render_guide()
        with st.expander('写真とデータの保存先'):
            st.write('写真と履歴はアプリを起動したPCに保存されます。食事日記の写真はmeal_photos、DB移行前の控えはDBと同じフォルダーの.backupファイルに保存します。')
            st.code(str(DATA_DIR),language=None)
            st.caption('接続セッションが切れると未保存の入力は失われます。保存済みの記録はPCに残ります。')
        st.caption('認証機能はありません。外部接続で実演するときは、個人データを含まない独立したデモ環境かアクセス制御を使用してください。')
        with st.expander('スマートフォンの接続・カメラ'):
            st.write('カメラへのアクセスを許可してください。開けない場合はブラウザのサイト設定を確認するか、標準カメラで撮って写真を選んでください。ブラウザ内カメラにはHTTPSまたはlocalhostが必要です。')
    elif section=='製品・校正':
        render_products()
    elif section=='料理の栄養計算':
        st.subheader('料理と栄養')
        st.caption('使った材料と分量から計算できます。計算しなくても食事は記録できます。')
        render_cooking_page()
    else:
        render_developer_tools()
