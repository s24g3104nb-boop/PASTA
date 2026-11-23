"""Photo recognition suggestions for the three editable standard recipes."""
import os
from pathlib import Path

import streamlit as st

from dish_classifier import STANDARD_DISH_NAMES
from local_dish_vision import LocalDishClassifier
from operation_logs import RECORDS, log_error


@st.cache_resource
def _classifier(checkpoint, head, device):
    return LocalDishClassifier(checkpoint[0], device, head_path=head[0] if head else None)


def _signature(config_path):
    path = Path(config_path).resolve(strict=True)
    stat = path.stat()
    return str(path), stat.st_size, stat.st_mtime_ns


def render_dish_choice(rgb, draft):
    candidate_key = 'meal_dish_candidate_'+draft['id']
    if rgb is not None and not draft.get('photo_analyzed'):
        draft['photo_analyzed'] = True
        model_path = os.environ.get('PASTA_OPENCLIP_CHECKPOINT')
        head_path = os.environ.get('PASTA_DISH_HEAD_PATH')
        if model_path:
            try:
                with st.spinner('この端末で料理の候補を調べています…'):
                    model = _classifier(_signature(model_path), _signature(head_path) if head_path else None,
                                        os.environ.get('PASTA_LOCAL_VISION_DEVICE', 'cpu'))
                    draft['photo_suggestion'] = model.predict(rgb)
            except (OSError, ValueError, ImportError, RuntimeError):
                log_error(RECORDS, '完成料理の画像認識 失敗')
                draft['photo_analysis_failed'] = True
        else:
            draft['photo_analysis_unavailable'] = True
    suggestion = draft.get('photo_suggestion')
    suggested = suggestion.name if suggestion and suggestion.name in STANDARD_DISH_NAMES else None
    if rgb is not None:
        if suggested:
            st.info(f'写真からの候補は「{suggested}」です。確認して料理を選んでください。')
            if suggestion.visible_options:
                st.caption('写っている具材の候補 · ' + '、'.join(suggestion.visible_options))
        else:
            reason = ''
            if draft.get('photo_analysis_unavailable'):
                reason = '（この環境では写真認識を利用できません）'
            if draft.get('photo_analysis_failed'):
                reason = '（写真認識を実行できませんでした）'
            st.info(f'判別できません{reason}。料理を選んで続けられます。')
    names = [None, *STANDARD_DISH_NAMES]
    current = draft.get('finished_recipe', {}).get('dish') or suggested
    return st.selectbox('標準レシピの料理', names, index=names.index(current), key=candidate_key,
                        format_func=lambda name: name or '料理を選ぶ（任意）')
