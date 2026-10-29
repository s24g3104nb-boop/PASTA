"""Research controls are opt-in; ordinary capture uses existing estimator defaults."""
import streamlit as st
from ui_shell import start_capture
from training_ui import render_training_tab

DEFAULT_SPREAD = dict(saturation_min=20, value_min=75, min_area=160, min_elongation=3.0)
DEFAULT_BUNDLE = dict(min_radius=8, max_radius=27, min_distance=32, hough_sensitivity=18)


def _remember(widget, durable):
    st.session_state[durable] = st.session_state[widget]


def render_developer_tools():
    st.caption('研究・検証用の設定です。画面を分けることはアクセス制御ではありません。')
    enabled = st.checkbox('測定で開発用設定を使う', value=st.session_state.get('dev_tools_enabled', False),
                          key='dev_tools_toggle', on_change=_remember,
                          args=('dev_tools_toggle', 'dev_tools_enabled'))
    st.session_state['dev_tools_enabled'] = enabled
    tools = st.selectbox('開発ツール', ['推定設定・検出結果', '学習データ・AIモデル管理'], key='developer_section')
    if tools == '学習データ・AIモデル管理':
        render_training_tab()
        return
    st.radio('推定方式', ['OpenCV本数検出', '学習モデル（実験用回帰）'],
             index=1 if st.session_state.get('research_method', '').startswith('学習') else 0,
             key='estimate_method', on_change=_remember, args=('estimate_method', 'research_method'))
    with st.expander('OpenCVの検出パラメータ'):
        spread = dict(st.session_state.get('research_spread', DEFAULT_SPREAD))
        bundle = dict(st.session_state.get('research_bundle', DEFAULT_BUNDLE))
        st.caption('広げた状態')
        for name, label, low, high, step in [('saturation_min','色の濃さ（最小彩度）',0,120,5),
                ('value_min','明るさ（最小輝度）',0,180,5),('min_area','領域の最小面積（画素）',30,2500,10),
                ('min_elongation','細長さの基準',1.5,8.0,.5)]:
            spread[name] = st.slider(label, low, high, spread[name], step, key='dev_' + name)
        st.caption('束の断面')
        for name,label,low,high in [('min_radius','円の最小半径（画素）',2,35),
                ('max_radius','円の最大半径（画素）',9,65),('min_distance','検出円の最小間隔（画素）',5,90),
                ('hough_sensitivity','円検出の厳しさ',5,45)]:
            bundle[name] = st.slider(label,low,high,bundle[name],key='dev_' + name)
        st.session_state['research_spread'] = spread
        st.session_state['research_bundle'] = bundle
    last = st.session_state.get('dev_last_detection')
    if last:
        with st.expander('最後の検出マスク・診断'):
            st.image(last['overlay'], caption='検出箇所', width='stretch')
            st.image(last['mask'], caption='検出対象', width='stretch')
            st.write(last['count'], '個')
            st.json(last['parameters'])
    st.caption('CGサンプルは操作確認用です。実写の推定精度を示すものではありません。')
    st.button('サンプルで試す', on_click=start_capture, args=('サンプル画像（今すぐ試す）',), key='home_demo')
