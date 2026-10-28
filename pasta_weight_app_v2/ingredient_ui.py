"""One ingredient at a time, without a horizontally scrolling mobile table."""
import pandas as pd
import streamlit as st
from cooking import Ingredient

FIELDS=('amount_g','kcal_100','protein_100','fat_100','carb_100')
LABELS=('使った量（g）','100gあたり kcal','100gあたり たんぱく質（g）','100gあたり 脂質（g）','100gあたり 炭水化物（g）')


def render_ingredients(initial_rows, key):
    state_key=key+'_cards'
    if state_key not in st.session_state:st.session_state[state_key]=[dict(row) for row in initial_rows]
    rows=st.session_state[state_key]
    for i,row in enumerate(rows):
        with st.container(border=True,key=f"card_ingredient_{key}_{i}"):
            st.write(f"{row['name']} · {row['amount_g']:g} g")
            st.caption(f"{row['amount_g']*row['kcal_100']/100:.1f} kcal")
            c1,c2=st.columns(2)
            if c1.button('編集',key=f'{key}_edit_{i}'):
                st.session_state[key+'_editing']=i
                st.rerun()
            if c2.button('取り除く',key=f'{key}_remove_{i}'):
                rows.pop(i)
                st.session_state.pop(key+'_editing',None)
                st.rerun()
    editing=st.session_state.get(key+'_editing')
    current=rows[editing] if editing is not None and editing<len(rows) else {}
    revision=st.session_state.get(key+'_revision',0)
    with st.expander('食材を編集' if current else '食材を1つ追加',expanded=bool(current) or not rows):
        st.caption('商品ラベルの100gあたり栄養値を入力してください。不明な値は0で埋めないでください。')
        with st.form(f'{key}_ingredient_{editing}_{revision}',clear_on_submit=False):
            name=st.text_input('食材名',value=current.get('name',''),placeholder='例：トマトソース')
            values=[st.number_input(label,min_value=0.,value=float(current[field]) if field in current else None,step=.1) for field,label in zip(FIELDS,LABELS)]
            submitted=st.form_submit_button('この食材を保存',type='primary')
        if submitted:
            try:
                row=dict(name=name,**dict(zip(FIELDS,values)))
                Ingredient(**row).nutrition()
                if current:rows[editing]=row
                else:rows.append(row)
                st.session_state.pop(key+'_editing',None)
                st.session_state[key+'_revision']=revision+1
                st.rerun()
            except (ValueError,TypeError) as exc:st.error(f'食材を保存できませんでした: {exc}')
    return pd.DataFrame(rows,columns=['name',*FIELDS])
