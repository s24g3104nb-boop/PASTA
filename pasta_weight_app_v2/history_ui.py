"""Photo cards, details and audited history actions."""
from pathlib import Path
import sqlite3
import pandas as pd
import streamlit as st
from storage import DEFAULT_DB, list_measurements
from history_actions import change_measurement
from ui_shell import empty_state


def measurement_card(row, db_path=DEFAULT_DB, *, compact=False):
    with st.container(border=True,key=f"card_history_{row['id']}"):
        left,right=st.columns([1,3])
        path=Path(db_path).parent/'photos'/Path(row.get('image_filename') or '').name
        with left:
            if path.is_file():st.image(str(path),width='stretch')
        with right:
            st.caption(row['created_at'])
            st.markdown(f"**{row['product_name']}**")
            st.markdown(f"### {row['estimated_g']:.1f} g　·　{row['kcal']:.0f} kcal")
            tags=[]
            if row['actual_g'] is not None:tags.append(f"実測値登録済み {row['actual_g']:.1f} g")
            if row['estimator']=='manual_count':tags.append('本数を手動修正')
            if row['source_kind']!='real':tags.append('CG・デモ / 来歴未確認')
            st.caption(' / '.join(tags) if tags else '推定値 · 実測未登録')
        if compact:return
        with st.expander('詳細・重量の修正'):
            if path.is_file():st.image(str(path),caption='保存した写真',width='stretch')
            st.write(f"撮影状態：{row['mode']} / 使用方式：{row['estimator']}")
            if not row['estimator'].startswith('ridge:'):
                st.write(f"計算に使用した本数：{row['detected_count']} 本")
            st.caption('元の推定重量・推定栄養値を保持し、はかりの実測値を追記します。料理計算では実測値を優先します。')
            if row['actual_g'] is not None:
                st.write(f"絶対誤差：{abs(row['estimated_g']-row['actual_g']):.1f} g")
            with st.form(f"edit_history_{row['id']}"):
                actual=st.number_input('はかりで測った重量（g）',min_value=.1,value=float(row['actual_g'] or max(.1,row['estimated_g'])),key=f"actual_{row['id']}")
                if st.form_submit_button('実測重量を保存'):
                    try:
                        change_measurement(row['id'],actual_g=actual,db_path=db_path)
                        st.session_state['history_feedback']='実測重量を保存しました。'
                        st.rerun()
                    except (ValueError,sqlite3.Error) as exc:st.error(str(exc))
            confirm=st.checkbox('この履歴を一覧から削除する',key=f"delete_confirm_{row['id']}")
            if st.button('削除を確定',disabled=not confirm,key=f"delete_{row['id']}"):
                try:
                    change_measurement(row['id'],archive=True,db_path=db_path)
                    st.session_state['history_feedback']='履歴を一覧から削除しました。元データは監査用に保持されています。'
                    st.rerun()
                except (ValueError,sqlite3.Error) as exc:st.error(str(exc))


def render_history(db_path=DEFAULT_DB):
    st.title('あなたの記録')
    st.caption('ひと皿ずつ、振り返ろう。')
    if st.session_state.get('history_feedback'):st.success(st.session_state.pop('history_feedback'))
    rows=list_measurements(db_path)
    if not rows:
        empty_state('最初の記録を残そう','ホームからパスタを撮影して、結果を保存してください。')
        return
    search=st.text_input('履歴を検索（製品・日時・方式）',key='history_search',placeholder='製品名や日付で探す')
    matches=[r for r in rows if search.strip().casefold() in ' '.join(str(r[k] or '') for k in ('product_name','created_at','estimator')).casefold()]
    if not matches:
        st.info('検索条件に一致する履歴はありません。')
        return
    last_page=max(1,(len(matches)+9)//10)
    if st.session_state.get('history_page',1)>last_page:
        st.session_state['history_page']=last_page
    page=st.number_input('ページ',min_value=1,max_value=last_page,value=1,key='history_page')
    for row in matches[(page-1)*10:page*10]:measurement_card(row,db_path)
    with st.expander('誤差の集計・一覧データ'):
        frame=pd.DataFrame(matches)
        shown=frame.copy()
        shown.loc[shown.estimator.str.startswith('ridge:'),'detected_count']=None
        st.dataframe(shown[['created_at','product_name','mode','detected_count','estimated_g','actual_g','kcal','estimator','source_kind']],hide_index=True,width='stretch')
        measured=frame[(frame.source_kind=='real') & (frame.actual_g>0)].copy()
        if len(measured):
            measured['absolute_error_g']=(measured.estimated_g-measured.actual_g).abs()
            measured['absolute_percentage_error']=measured.absolute_error_g/measured.actual_g*100
            st.metric('実写・実測値がある履歴のMAE',f'{measured.absolute_error_g.mean():.2f} g')
            st.dataframe(measured.groupby(['mode','estimator']).agg(件数=('id','count'),MAE_g=('absolute_error_g','mean'),最大誤差_g=('absolute_error_g','max'),MAPE_percent=('absolute_percentage_error','mean')).reset_index(),hide_index=True)
            st.caption('保存された推定成功分の参考値です。独立評価の精度ではありません。')
