"""Photo recognition suggestions for the three editable standard recipes."""
import os
from pathlib import Path

import streamlit as st

from dish_classifier import STANDARD_DISH_NAMES
from local_dish_vision import LocalDishClassifier
from operation_logs import RECORDS, log_error


@st.cache_resource
def _classifier(checkpoint, device):
    return LocalDishClassifier(checkpoint[0], device)


def _signature(config_path):
    path = Path(config_path).resolve(strict=True)
    stat = path.stat()
    return str(path), stat.st_size, stat.st_mtime_ns


def _remember_recognition(draft, toggle_key, candidate_key):
    enabled = st.session_state[toggle_key]
    draft['photo_recognition_enabled'] = enabled
    suggested = getattr(draft.get('photo_suggestion'), 'name', None)
    current = st.session_state.get(candidate_key)
    changed_candidate = (enabled and current is None) or (not enabled and suggested is not None and current == suggested)
    if not draft.get('finished_recipe') and changed_candidate:
        st.session_state.pop(candidate_key, None)


def render_dish_choice(rgb, draft):
    candidate_key = 'meal_dish_candidate_'+draft['id']
    model_path = os.environ.get('PASTA_OPENCLIP_CHECKPOINT')
    enabled = False
    if model_path:
        toggle_key = 'meal_photo_recognition_'+draft['id']
        enabled = st.toggle('写真から料理候補を表示', value=draft.get('photo_recognition_enabled', True),
                            key=toggle_key, on_change=_remember_recognition,
                            args=(draft, toggle_key, candidate_key))
        if rgb is not None and enabled and not draft.get('photo_analyzed'):
            draft['photo_analyzed'] = True
            try:
                with st.spinner('この端末で料理の候補を調べています…'):
                    model = _classifier(_signature(model_path), os.environ.get('PASTA_LOCAL_VISION_DEVICE', 'cpu'))
                    draft['photo_suggestion'] = model.predict(rgb)
            except (OSError, ValueError, ImportError, RuntimeError):
                log_error(RECORDS, '完成料理の画像認識 失敗')
                draft['photo_analysis_failed'] = True

    suggested = getattr(draft.get('photo_suggestion'), 'name', None) if enabled and rgb is not None else None
    if suggested not in STANDARD_DISH_NAMES:
        suggested = None
    if suggested:
        st.info(f'写真からの候補は「{suggested}」です。確認して料理を選んでください。')
        st.caption('写真は料理名の候補に使います。重量や実際に使った材料・分量は入力してください。')
    elif rgb is not None and (enabled or not model_path):
        reason = ''
        if not model_path:
            reason = '（この環境では写真認識を利用できません）'
        elif draft.get('photo_analysis_failed'):
            reason = '（写真認識を実行できませんでした）'
        st.info(f'判別できません{reason}。料理を選んで続けられます。')
    names = [None, *STANDARD_DISH_NAMES]
    current = draft.get('finished_recipe', {}).get('dish') or suggested
    return st.selectbox('標準レシピの料理', names, index=names.index(current), key=candidate_key,
                        format_func=lambda name: name or '料理を選ぶ（任意）')
