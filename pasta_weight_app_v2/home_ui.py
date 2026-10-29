"""Two everyday actions and recent meals; measurements stay a separate concept."""
import logging
import sqlite3
import streamlit as st
from meal_diary import entries, photo_path
from diary_ui import start_diary, open_diary_entry
from ui_shell import hero, empty_state, start_capture


def render_home():
    hero()
    st.button('乾麺の重さをはかる', icon=':material/photo_camera:', type='primary',
              width='stretch', on_click=start_capture, args=('スマホで撮影',), key='home_camera')
    st.button('食べたパスタを記録する', icon=':material/restaurant:', width='stretch',
              on_click=start_diary, key='home_meal')
    st.caption('写真なしでも記録できます。重さの推定精度は検証中です。')
    with st.expander('写真なしで操作を試す'):
        st.caption('CGの操作確認です。実物の推定精度を示しません。')
        st.button('サンプルで試す', key='home_demo', on_click=start_capture, args=('サンプル画像（今すぐ試す）',))
    st.subheader('最近食べたパスタ')
    try:
        rows = entries()
    except (OSError, sqlite3.Error):
        logging.exception('Unable to read recent meals')
        st.error('記録を読み込めませんでした。少し待ってから、記録画面で再試行してください。')
        return
    if not rows:
        empty_state('最初のひと皿を記録しよう', '料理名と食べた日だけでも残せます。')
    for row in rows[:3]:
        with st.container(border=True, key='card_recent_' + row['id']):
            path = photo_path(row['photo']) if row['photo'] else None
            if path and path.is_file():
                st.image(str(path), width='stretch')
            else:
                empty_state('🍝', '写真なしの記録')
            st.subheader(row['dish_name'])
            st.caption(row['eaten_date'] + ' ' + (row['eaten_time'] or '') + ' ' + row.get('meal_type', ''))
            if row.get('source_kind') == 'synthetic':
                st.caption('デモの記録')
            st.button('記録を見る', key='recent_' + row['id'], on_click=open_diary_entry, args=(row['id'],))
    if rows:
        st.button('すべての食事記録を見る', on_click=lambda: st.session_state.update(main_nav='記録', record_kind='食事日記', diary_screen='list'), key='home_history', width='stretch')
    st.markdown('<aside class="pl-tip"><b>きれいに撮るコツ</b><br>乾麺は無地の背景で、全体が写るように。広げて撮るときは、1本ずつ離しましょう。</aside>', unsafe_allow_html=True)
