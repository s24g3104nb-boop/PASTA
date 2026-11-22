"""Optional local recognition in the meal draft."""
import os
from pathlib import Path

import streamlit as st
from local_dish_vision import DISH_PROMPTS, LocalDishClassifier, color_note

CHOICES = [name for name in DISH_PROMPTS if name != 'その他']


@st.cache_resource
def _classifier(path, size, mtime_ns, device):
    return LocalDishClassifier(path, device)


def _use_name(draft):
    selection = st.session_state['meal_dish_candidate_'+draft['id']]
    draft['dish_name'] = selection
    draft['photo_name_confirmed'] = selection == getattr(draft.get('photo_suggestion'), 'name', None)
    st.session_state['meal_dish_name_'+draft['id']] = selection


def render_dish_suggestion(rgb, draft):
    st.caption('料理写真から料理名と見えている具材の候補を探せます。材料の量は自分で確認してください。')
    model_path = os.environ.get('PASTA_OPENCLIP_CHECKPOINT')
    if model_path and st.button('写真から料理の候補を探す', key='dish_analyze_'+draft['id'], width='stretch'):
        try:
            path = Path(model_path).resolve()
            stat = path.stat()
            with st.spinner('このPCで料理の候補を調べています…'):
                model = _classifier(str(path), stat.st_size, stat.st_mtime_ns, os.environ.get('PASTA_LOCAL_VISION_DEVICE', 'cpu'))
                draft['photo_suggestion'] = model.predict(rgb)
                draft['photo_name_confirmed'] = False
        except (OSError, ValueError, ImportError, RuntimeError):
            draft.pop('photo_suggestion', None)
            st.warning('写真から候補を取得できませんでした。料理名を下から選べます。')
    suggestion = draft.get('photo_suggestion')
    if suggestion and suggestion.name:
        st.info(suggestion.name+'の可能性があります。写真と材料を確認して、料理名を選んでください。')
        if suggestion.visible_options:
            st.caption('見えている具材の候補 · '+'、'.join(suggestion.visible_options)+'。実際に使った材料だけを栄養計算に入れてください。')
    elif model_path and suggestion is not None:
        st.info('この写真では料理の種類を判断できませんでした。')
    elif not model_path:
        st.caption('このPCでは写真認識が未設定です。料理名を選んで続けられます。')
    color = color_note(rgb)
    if color:
        st.caption('写真中央の主な色 · '+color)
    names = [None, *CHOICES]
    suggested = suggestion.name if suggestion else None
    st.selectbox('料理名の候補', names, index=names.index(suggested), key='meal_dish_candidate_'+draft['id'],
                 format_func=lambda name: name or 'ここから選ぶ')
    if st.session_state['meal_dish_candidate_'+draft['id']]:
        st.button('選んだ料理名を入力する', key='dish_use_'+draft['id'], on_click=_use_name, args=(draft,))
