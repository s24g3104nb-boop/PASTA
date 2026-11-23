"""Mobile-first shell, navigation and common presentation primitives."""
from html import escape
from pathlib import Path
import streamlit as st

from sheetless_area import ESTIMATOR as AREA_ESTIMATOR

PAGES = ['ホーム', '記録', '設定']
COOKING_RECORD = '料理・栄養'


def dry_weight_label(row):
    if row['actual_g'] is not None:
        return f"実測 {row['actual_g']:g} g"
    if row['estimated_g'] is None:
        return '重量未登録'
    if row.get('estimator') == AREA_ESTIMATOR:
        return f"推定 約{row['estimated_g']:.0f} g"
    return f"推定 {row['estimated_g']:g} g"


def shell():
    theme=st.session_state.get('theme_preference','端末に合わせる')
    st.html(Path(__file__).parent / 'ui' / 'design.css')
    mode={'ライト':'light','ダーク':'dark'}.get(theme,'auto')
    st.markdown('<div class="pl-brand pl-theme-'+mode+'"><span aria-hidden="true">≋</span>Pasta Lens<small>ひと皿を、ちょうどよく。</small></div>',unsafe_allow_html=True)
    return st.radio('メインナビゲーション',PAGES,format_func=lambda name: ':material/'+['home','calendar_month','settings'][PAGES.index(name)]+': '+name,horizontal=True,key='main_nav',label_visibility='collapsed',on_change=_nav_changed)


def navigate(page):
    st.session_state['main_nav']=PAGES[page]


def start_capture(source):
    st.session_state['main_nav']=PAGES[0]
    st.session_state['capture_source']=source
    st.session_state['capture_resume_source']=source
    st.session_state.pop('manual_only',None)
    st.session_state.pop('analysis_ready',None)
    st.session_state.pop('result_request',None)
    st.session_state.pop('editing_photo',None)
    st.session_state.pop('a4_area_result', None)
    st.session_state.pop('capture_photo',None)
    st.session_state.pop('measurement_submission_token',None)
    st.session_state['capture_generation']=st.session_state.get('capture_generation',0)+1


def resume_capture():
    """Returning from navigation must not discard an unsaved measurement."""
    st.session_state['main_nav']=PAGES[0]
    st.session_state['capture_source']=st.session_state.get('capture_resume_source','写真をアップロード')
    st.session_state.pop('manual_only',None)


def start_manual_weight():
    st.session_state['main_nav']=PAGES[0]
    st.session_state.pop('capture_source',None)
    st.session_state['manual_only']=True


def empty_state(title,body):
    st.markdown(f'<div class="pl-empty"><span aria-hidden="true">◌</span><strong>{escape(title)}</strong>{escape(body)}</div>',unsafe_allow_html=True)


def persist_appearance():
    st.session_state['theme_preference']=st.session_state['appearance']


def _nav_changed():
    st.session_state.pop('cooking_return_draft_id', None)
    st.session_state.pop('calibration_origin', None)
    if st.session_state.get('main_nav') == 'ホーム':
        st.session_state.pop('capture_source', None)
