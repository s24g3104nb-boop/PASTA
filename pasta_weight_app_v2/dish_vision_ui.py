"""Photo suggestions and selection for the three editable standard recipes."""

import cv2
import streamlit as st

from dish_classifier import STANDARD_DISH_NAMES
from local_dish_vision import suggest_dish_from_color
from operation_logs import RECORDS, log_error


def _remember_dish_choice(draft, key):
    # A photo change can clear this widget before Streamlit delivers its callback.
    if key in st.session_state:
        draft['manual_dish_choice'] = st.session_state[key]


def render_dish_choice(rgb, draft, *, prefer_photo=False):
    candidate_key = 'meal_dish_candidate_' + draft['id']
    if rgb is not None and not draft.get('photo_analyzed'):
        draft['photo_analyzed'] = True
        try:
            draft['photo_suggestion'] = suggest_dish_from_color(rgb)
        except (cv2.error, ValueError):
            log_error(RECORDS, '完成料理の色解析 失敗')
            draft['photo_analysis_failed'] = True

    suggested = getattr(draft.get('photo_suggestion'), 'name', None) if rgb is not None else None
    if suggested:
        st.info(f'写真からの料理候補：{suggested}')
    elif rgb is not None and draft.get('photo_analysis_failed'):
        st.caption('写真の色を解析できませんでした。料理を選んで続けられます。')

    names = [None, *STANDARD_DISH_NAMES]
    current_recipe = draft.get('finished_recipe', {}).get('dish')
    default = suggested if prefer_photo and suggested else current_recipe or suggested
    current = draft.get('manual_dish_choice', default)
    index = names.index(current) if current in names else 0
    return st.selectbox('標準レシピの料理', names, index=index, key=candidate_key,
                        on_change=_remember_dish_choice, args=(draft, candidate_key),
                        format_func=lambda name: name or '料理を選ぶ（任意）')
