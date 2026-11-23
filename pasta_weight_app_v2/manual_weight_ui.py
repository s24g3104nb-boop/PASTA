"""Measured-weight recovery, independent of image inference and calibration."""
import hashlib
import sqlite3
from uuid import uuid4
import streamlit as st
from storage import save_measurement,DEFAULT_DB,list_products
from pasta_vision import nutrition_for_weight
from nutrition_state import (STANDARD_DRY_PASTA, STANDARD_NUTRITION_NOTE,
                             has_nutrition, snapshot_for_nutrition, standard_nutrition_used)
from settings_ui import open_calibration, select_measurement_product, select_optional_product
from measurement_diagnostics import strand_calibration


def render_manual_weight(product, rgb=None, *, synthetic=False, db_path=DEFAULT_DB):
    signature=hashlib.sha256((str(product['id'])+str(synthetic)).encode()+(rgb.tobytes() if rgb is not None else b'no-photo')).hexdigest()
    drafts=st.session_state.setdefault('manual_weight_drafts',{})
    draft=drafts.setdefault(signature,dict(token=uuid4().hex,weight=0.,saved=None))
    if draft.get('saved_product'):
        product = draft['saved_product']
    key='manual_weight_'+draft['token']
    if key not in st.session_state:st.session_state[key]=draft['weight']
    st.subheader('実際の重さを入力する')
    st.caption('電子はかりの表示を入力してください。')
    value=st.number_input('はかりで測った実測重量（g）',min_value=0.,step=.1,key=key,disabled=bool(draft['saved']))
    draft['weight']=value
    known=has_nutrition(product)
    nutrition=nutrition_for_weight(value,product['kcal_100'],product['protein_100'],product['fat_100'],product['carb_100']) if known else None
    if nutrition:
        st.metric('実測重量から計算した乾麺分',f'{nutrition["kcal"]:.1f} kcal')
        st.write(f"たんぱく質 {nutrition['protein_g']:.1f} g / 脂質 {nutrition['fat_g']:.1f} g / 炭水化物 {nutrition['carb_g']:.1f} g")
        if standard_nutrition_used(product):
            st.caption(STANDARD_NUTRITION_NOTE)
    else:st.info('重量は保存できます。この乾麺の栄養は未計算です。')
    if synthetic:st.warning('CG写真を使用した操作確認です。実写の記録ではありません。')
    if st.button('実測重量を履歴に保存',type='primary',width='stretch',key='manual_weight_save',disabled=bool(draft['saved'])):
        try:
            if value<=0:raise ValueError('はかりで測った0gより大きい重量を入力してください。')
            snapshot=snapshot_for_nutrition(product)
            draft['saved']=save_measurement(product_name=product['name'],mode='実測重量の入力',detected_count=0,
                estimated_g=None,actual_g=value,nutrition=nutrition,image_rgb=rgb,source_kind='synthetic' if synthetic else 'real',
                estimator='manual_weight',product_snapshot=snapshot,submission_token=draft['token'],db_path=db_path)
            draft['saved_product'] = dict(product)
            request = st.session_state.get('result_request')
            if request and rgb is request['rgb']:
                request['product'] = product
                request['saved_product'] = dict(product)
            st.rerun()
        except ValueError as exc:st.error(str(exc))
        except (OSError,sqlite3.Error):st.error('保存できませんでした。写真と入力値は残っています。空き容量を確認して、もう一度保存してください。')
    if draft['saved']:
        from diary_ui import start_diary
        st.success('実測重量を保存しました。推定重量は記録していません。')
        st.button('この乾麺で料理を記録する',key='manual_weight_to_diary',type='primary',width='stretch',on_click=start_diary,kwargs={'measurement_id':draft['saved']})
        if st.button('別の実測重量を記録する',key='manual_weight_new'):
            drafts.pop(signature);st.rerun()


def recovery(product,rgb,*,synthetic=False):
    if st.session_state.get('_app_admin') and (
            product['name'].startswith('デモ用') or strand_calibration(product) is None):
        st.button('校正設定へ',key='recovery_calibration',width='stretch',
                  on_click=open_calibration, args=(product,))
    from ui_shell import start_capture
    st.button('撮り直す・写真を選び直す',key='recovery_retake',on_click=start_capture,args=('写真をアップロード',),width='stretch')
    render_manual_weight(product,rgb,synthetic=synthetic)


def render_without_photo():
    st.button('ホームに戻る',on_click=lambda:st.session_state.pop('manual_only',None))
    admin = st.session_state.get('_app_admin', False)
    try:
        products = {p['id']: p for p in list_products() if admin or not p['name'].startswith('デモ用')}
    except (OSError, sqlite3.Error):
        st.warning('登録済みの製品を読み込めませんでした。一般的な乾麺として記録できます。')
        product = STANDARD_DRY_PASTA
    else:
        if admin:
            product = select_measurement_product(products, '使用した製品', 'manual_only_product')
            if product is None:
                return
        else:
            product = select_optional_product(products, '使った乾麺の製品（任意）', 'manual_only_product',
                                              st.session_state.get('last_product_id'))
    render_manual_weight(product)
