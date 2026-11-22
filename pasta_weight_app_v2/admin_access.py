"""Host-local administration. The normal entry point cannot enable this mode."""
import hmac
import os
from pathlib import Path
from time import monotonic

import streamlit as st


# Captured before the isolated admin process changes storage environment.
PUBLIC_ROOTS = (
    os.environ.get('PASTA_DATA_DIR', str(Path(__file__).resolve().parent/'data')),
    os.environ.get('PASTA_RESEARCH_DATA_DIR'),
    str(Path(__file__).resolve().parent.parent/'user_data'),
)


def configure_admin():
    """Deny before any app/storage initialization, including on LAN binds."""
    if st.get_option('server.address') not in ('127.0.0.1', '::1'):
        st.error('管理用アプリはこのPCの127.0.0.1で起動してください。')
        st.stop()
    password = os.environ.get('PASTA_ADMIN_PASSWORD', '')
    admin_setting = os.environ.get('PASTA_ADMIN_DATA_DIR')
    if len(password) < 12 or not admin_setting:
        st.error('管理用パスワード（12文字以上）と専用保存先を起動環境に設定してください。')
        st.stop()

    try:
        public = [Path(value).resolve() for value in PUBLIC_ROOTS if value]
        root = Path(admin_setting).resolve()
        research = Path(os.environ.get('PASTA_ADMIN_RESEARCH_DIR', str(root/'research_collection'))).resolve()
    except (OSError, ValueError):
        st.error('管理用の保存先を確認してください。')
        st.stop()
    if any(a.is_relative_to(p) or p.is_relative_to(a) for a in (root, research) for p in public):
        st.error('管理用の保存先には通常の保存先と重ならない別フォルダーを指定してください。')
        st.stop()

    if st.session_state.get('_admin_authenticated') is not True:
        st.title('管理者ログイン')
        with st.form('admin_login', clear_on_submit=True):
            attempt = st.text_input('管理用パスワード', type='password', max_chars=256)
            submit = st.form_submit_button('ログイン')
        if submit:
            now = monotonic()
            if now < st.session_state.get('_admin_next_attempt', 0):
                st.error('少し待ってから再試行してください。')
            elif hmac.compare_digest(attempt.encode('utf-8'), password.encode('utf-8')):
                st.session_state['_admin_authenticated'] = True
                st.rerun()
            else:
                st.session_state['_admin_next_attempt'] = now + 2
                st.error('管理用パスワードを確認してください。')
        st.stop()
    # A separate externally launched process is required. Existing environment
    # and storage constants from the public server are never changed there.
    os.environ['PASTA_DATA_DIR'] = str(root)
    os.environ['PASTA_RESEARCH_DATA_DIR'] = str(research)
