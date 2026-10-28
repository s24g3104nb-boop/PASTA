import streamlit as st
from storage import list_measurements
from cooking_storage import list_cooking_sessions
from ui_shell import hero, empty_state, start_capture, navigate
from history_ui import measurement_card


def render_home():
    hero()
    st.button('パスタを撮影する',icon=':material/photo_camera:',type='primary',width='stretch',on_click=start_capture,args=('スマホで撮影',),key='home_camera')
    st.button('写真から選ぶ',icon=':material/photo_library:',width='stretch',on_click=start_capture,args=('写真をアップロード',),key='home_upload')
    st.caption('推定精度は検証中です。実際の重量は、はかりで確認してください。')
    with st.expander('写真なしで操作を試す'):
        st.caption('CGのサンプルです。実物の測定精度を示すものではありません。')
        st.button('サンプルで試す',on_click=start_capture,args=('サンプル画像（今すぐ試す）',),key='home_demo')
    st.subheader('最近の測定')
    rows=list_measurements()
    if rows:
        for row in rows[:2]:measurement_card(row,compact=True)
        st.button('すべての履歴を見る',on_click=navigate,args=(1,),key='home_history')
    else:empty_state('まだ記録はありません','いつものパスタから、はじめてみませんか。')
    st.subheader('保存した料理')
    sessions=list_cooking_sessions()
    if sessions:
        for row in sessions[:2]:
            with st.container(border=True):
                st.write(row['selected_dish'])
                st.caption(row['created_at'])
                if row['dry_snapshot'].get('source_kind')!='real' or row['dry_snapshot']['product_name'].startswith('デモ用'):
                    st.caption('CG・デモ / 来歴未確認の計算')
                st.write(f"{row['calculation']['total']['kcal']:.0f} kcal · 料理全体")
        st.button('料理・栄養を見る',on_click=navigate,args=(2,),key='home_cooking')
    else:empty_state('ひと皿の栄養も、まとめて','乾麺の記録に食材を足して、料理全体の栄養を計算できます。')
    st.markdown('<aside class="pl-tip"><b>きれいに撮るコツ</b><br>無地の背景で、麺の全体を画面に。広げて撮るときは、1本ずつ離しましょう。</aside>',unsafe_allow_html=True)
