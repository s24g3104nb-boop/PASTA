"""Shared navigation; admin routes are opt-in and lazily loaded."""
from pathlib import Path
import sqlite3
import sys

REPOSITORY = str(Path(__file__).resolve().parent.parent)
if REPOSITORY not in sys.path:
    sys.path.insert(0, REPOSITORY)
import streamlit as st
from storage import init_db, DATA_DIR
from operation_logs import RECORDS, configure_operation_log, log_error
from ui_shell import COOKING_RECORD, PAGES, persist_appearance, shell
from home_ui import render_home
from capture_ui import render_capture
from history_ui import render_history
from diary_ui import render_diary, render_return_to_meal
from cooking_ui import render_cooking_page
from settings_ui import render_guide, render_products

def render_application(*, admin=False):
    st.session_state['_app_admin'] = admin
    if admin:
        def logout():
            st.session_state.clear()
        st.sidebar.button('管理者ログアウト', on_click=logout)
    else:
        if st.session_state.get('capture_source', '').startswith('サンプル'):
            st.session_state.pop('capture_source', None)
            st.session_state.pop('result_request', None)
        if st.session_state.get('result_request', {}).get('using_regression'):
            st.session_state.pop('result_request', None)
    configure_operation_log(DATA_DIR)
    try:
        init_db()
    except (OSError, sqlite3.Error):
        log_error(RECORDS, '記録データの初期化 失敗')
        st.error('データを開けませんでした。ほかのアプリで使用中の場合は閉じ、再試行してください。')
        if st.button('再試行する'):
            st.rerun()
        st.stop()
    base=Path(__file__).resolve().parent
    if admin and not all((base/'sample_images'/name).is_file() for name in ('demo_spread_12.png','demo_bundle_16.png')):
        from generate_demo import generate_images
        generate_images()
    _migrate_navigation()
    page = shell()
    if page == PAGES[0]:
        with st.container(key='view_home'):
            if st.session_state.get('capture_source'):
                render_capture()
            else:
                render_home()
    elif page == PAGES[1]:
        with st.container(key='view_records'):
            _render_records()
    else:
        with st.container(key='view_settings'):
            _render_settings(admin)


def _migrate_navigation():
    # Preserve the destination of sessions opened before the route changed.
    state = st.session_state
    page = state.get('main_nav')
    if page == '履歴':
        state.update(main_nav='記録', record_kind='乾麺の測定履歴')
    elif page == COOKING_RECORD or (page == '設定' and state.get('settings_section') == '料理の栄養計算'):
        state.update(main_nav='記録', record_kind=COOKING_RECORD)
        state.pop('settings_section', None)
    if state.get('diary_screen') == 'cooking':
        state.diary_screen = 'editor'
        state.cooking_return_draft_id = state.get('diary_active_draft')
        if state.get('main_nav') == '記録':
            state.record_kind = COOKING_RECORD
    state.pop('cooking_return_diary', None)


def _record_changed():
    st.session_state.pop('cooking_return_draft_id', None)


def _render_records():
    kind = st.selectbox('記録の種類', ['食事日記', '乾麺の測定履歴', COOKING_RECORD],
                        key='record_kind', on_change=_record_changed)
    if kind == COOKING_RECORD:
        with st.container(key='records_cooking'):
            render_return_to_meal()
            st.title('料理・栄養')
            render_cooking_page()
    elif kind == '乾麺の測定履歴':
        with st.container(key='records_history'):
            render_history()
    else:
        with st.container(key='records_diary'):
            render_diary()


def _render_preferences(admin):
    appearances = ['端末に合わせる', 'ライト', 'ダーク']
    theme = st.session_state.get('theme_preference', appearances[0])
    st.selectbox('カラーテーマ', appearances, index=appearances.index(theme),
                 key='appearance', on_change=persist_appearance)
    with st.expander('スマートフォンの接続・カメラ'):
        st.write('アプリ内のカメラにはHTTPSまたはlocalhostが必要です。'
                 '開けないときは、スマートフォンのカメラで撮り、写真を選んでください。')
    if admin:
        with st.expander('撮影・栄養計算のガイド'):
            render_guide()
        with st.expander('写真とデータの保存先'):
            st.code(str(DATA_DIR), language=None)


def _render_settings(admin):
    st.title('設定')
    if not admin:
        _render_preferences(admin=False)
        return
    sections = ['表示・使い方', '製品・校正', '開発・検証ツール']
    if st.session_state.get('settings_section') not in sections:
        st.session_state.pop('settings_section', None)
    section = st.selectbox('設定メニュー', sections, key='settings_section')
    if section == '製品・校正':
        with st.container(key='settings_products'):
            render_products()
    elif section == '開発・検証ツール':
        from developer_ui import render_developer_tools
        with st.container(key='settings_admin'):
            render_developer_tools()
    else:
        with st.container(key='settings_preferences'):
            _render_preferences(admin=True)
