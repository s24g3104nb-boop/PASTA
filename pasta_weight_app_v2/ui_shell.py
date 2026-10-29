"""Mobile-first shell, navigation and common presentation primitives."""
from html import escape
from pathlib import Path
import streamlit as st

PAGES = ['ホーム', '記録', '設定']
DARK = '''--pl-on-accent:#2b231e;--pl-bg:#181e1c;--pl-card:#232c28;--pl-text:#f2f3ee;--pl-muted:#bbc5bd;
--pl-line:#3b4841;--pl-orange:#f2a37e;--pl-orange-soft:#3c3029;--pl-green:#a8d3bf;--pl-sage:#2d4038;
--background-color:#181e1c;--secondary-background-color:#2d3831;--text-color:#f2f3ee;'''


def shell():
    theme=st.session_state.get('theme_preference','端末に合わせる')
    override = ':root {'+DARK+'}' if theme=='ダーク' else ('@media(prefers-color-scheme:dark){:root{'+DARK+'}}' if theme=='端末に合わせる' else '')
    css=(Path(__file__).parent/'ui'/'design.css').read_text()
    st.markdown('<style>'+css.replace('THEME_OVERRIDES',override)+'</style>',unsafe_allow_html=True)
    st.markdown('<div class="pl-brand"><span aria-hidden="true">≋</span>Pasta Lens<small>ひと皿を、ちょうどよく。</small></div>',unsafe_allow_html=True)
    return st.radio('メインナビゲーション',PAGES,format_func=lambda name: ':material/'+['home','calendar_month','settings'][PAGES.index(name)]+': '+name,horizontal=True,key='main_nav',label_visibility='collapsed',on_change=_nav_changed)


def navigate(page):
    st.session_state['main_nav']=PAGES[page]


def start_capture(source):
    st.session_state['main_nav']=PAGES[0]
    st.session_state['capture_source']=source
    st.session_state.pop('analysis_ready',None)
    st.session_state.pop('result_request',None)
    st.session_state.pop('editing_photo',None)
    st.session_state.pop('capture_photo',None)
    st.session_state.pop('measurement_submission_token',None)
    st.session_state['capture_generation']=st.session_state.get('capture_generation',0)+1


def empty_state(title,body):
    st.markdown(f'<div class="pl-empty"><span aria-hidden="true">◌</span><strong>{escape(title)}</strong>{escape(body)}</div>',unsafe_allow_html=True)


def hero():
    st.markdown('''<section class="pl-hero"><div class="pl-eyebrow">YOUR EVERYDAY PASTA</div>
<h1>今日のひと皿を、<br>思い出に。</h1><p>乾麺の重さをはかって、<br>食べたパスタを写真で記録。</p>
<svg viewBox="0 0 220 240" aria-hidden="true"><ellipse cx="135" cy="166" rx="86" ry="18" fill="#294f3d" opacity=".1"/>
<circle cx="130" cy="112" r="87" fill="#fffaf0"/><circle cx="130" cy="112" r="68" fill="none" stroke="#e5ddd0" stroke-width="2"/>
<g fill="none" stroke="#dca74f" stroke-width="5" stroke-linecap="round"><path d="M75 130Q72 72 128 65T174 105Q180 155 120 148T93 91Q130 62 157 99T111 133Q87 110 122 94T143 119"/>
<path d="M86 148Q55 108 92 80T169 80M101 155Q165 170 184 119M82 98Q106 60 145 78T158 140"/></g>
<path d="M151 70Q177 43 187 64Q177 86 151 70M74 145Q54 125 48 143Q55 160 74 145" fill="#5c8063"/>
<circle cx="102" cy="90" r="7" fill="#c76b46"/><circle cx="150" cy="139" r="6" fill="#c76b46"/></svg></section>''',unsafe_allow_html=True)


def persist_appearance():
    st.session_state['theme_preference']=st.session_state['appearance']


def _nav_changed():
    if st.session_state.get('main_nav') == 'ホーム':
        st.session_state.pop('capture_source', None)
