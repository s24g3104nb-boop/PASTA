"""Two everyday actions and recent meals; measurements stay a separate concept."""
import sqlite3
import streamlit as st
from meal_diary import entries, photo_path
from diary_ui import open_diary_entry, open_diary_list, start_diary
from ui_shell import empty_state, resume_capture, start_capture, start_manual_weight
from operation_logs import RECORDS, log_error


def render_home():
    if st.session_state.get('manual_only'):
        from manual_weight_ui import render_without_photo
        render_without_photo()
        return
    st.title('今日のパスタ')
    with st.container(key='home_actions'):
        dry, dish = st.columns(2, gap='medium')
        with dry.container(border=True, key='home_dry_action'):
            st.button('乾麺を測定', icon=':material/photo_camera:', type='primary',
                      width='stretch', on_click=start_capture, args=('写真をアップロード',), key='home_camera')
            st.caption('ゆでる前の乾麺の重さの目安を確認。')
        with dish.container(border=True, key='home_dish_action'):
            st.button('完成料理を記録', icon=':material/restaurant:', width='stretch',
                      on_click=start_diary, key='home_meal')
            st.caption('料理の写真、材料、食べた分の栄養を記録。')
    with st.container(key='home_shortcuts'):
        diary, calendar = st.columns(2)
        diary.button('食事日記', icon=':material/menu_book:', key='home_history', width='stretch',
                     on_click=open_diary_list, args=('写真付き一覧',))
        calendar.button('カレンダー', icon=':material/calendar_month:', key='home_calendar', width='stretch',
                        on_click=open_diary_list, args=('カレンダー',))
    if st.session_state.get('capture_photo') is not None or st.session_state.get('result_request'):
        st.button('途中の測定を再開する', key='home_resume_capture', width='stretch', on_click=resume_capture)
    with st.expander('カメラを開く・はかりの数値を記録する'):
        st.button('ブラウザでカメラを開く', icon=':material/photo_camera:', width='stretch',
                  on_click=start_capture, args=('スマホで撮影',), key='home_camera_direct')
        st.button('写真なしで実測重量を記録する', key='home_manual_weight', width='stretch', on_click=start_manual_weight)
    if st.session_state.get('_app_admin'):
        with st.expander('サンプルで操作を確認する'):
            st.caption('CGの操作確認です。実物の推定精度を示しません。')
            st.button('サンプルで試す', key='home_demo', on_click=start_capture, args=('サンプル画像（今すぐ試す）',))
    st.subheader('最近の食事')
    try:
        rows = [row for row in entries() if st.session_state.get('_app_admin') or row.get('source_kind') != 'synthetic']
    except (OSError, sqlite3.Error, ValueError):
        st.error('記録を読み込めませんでした。少し待ってから、記録画面で再試行してください。')
        return
    if not rows:
        empty_state('最初のひと皿を記録しよう', '料理名と食べた日だけでも残せます。')
        return
    recent = rows[:3]
    for column, row in zip(st.columns(len(recent), gap='medium'), recent):
        with column.container(border=True, key='card_recent_' + row['id']):
            try:
                path = photo_path(row['photo']) if row['photo'] else None
            except (OSError,ValueError):
                log_error(RECORDS, '最近の食事写真の参照 失敗')
                path = None
            shown=False
            try:
                if path and path.is_file():
                    st.image(str(path), width='stretch')
                    shown=True
            except (OSError,ValueError):
                log_error(RECORDS, '最近の食事写真の表示 失敗')
            if not shown:
                empty_state('🍝', '写真なしの記録')
            st.subheader(row['dish_name'])
            st.caption(row['eaten_date'] + (' ' + row['eaten_time'][:5] if row['eaten_time'] else '')
                       + (' · ' + row['meal_type'] if row.get('meal_type') else ''))
            if row.get('source_kind') == 'synthetic':
                st.caption('デモの記録')
            st.button('記録を見る', key='recent_' + row['id'], on_click=open_diary_entry, args=(row['id'],),
                      width='stretch')
