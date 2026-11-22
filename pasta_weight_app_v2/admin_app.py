"""Start independently on loopback. Authenticate before importing storage/UI."""
import streamlit as st
from admin_access import configure_admin

st.set_page_config(page_title='Pasta Lens 管理', page_icon='🍝', layout='centered')
configure_admin()
from app_runtime import render_application

render_application(admin=True)
