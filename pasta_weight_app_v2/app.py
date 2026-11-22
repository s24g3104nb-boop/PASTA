"""Normal Pasta Lens entry point."""
import streamlit as st
from app_runtime import render_application

st.set_page_config(page_title='Pasta Lens | ひと皿を、思い出に。', page_icon='🍝', layout='centered')
render_application()
