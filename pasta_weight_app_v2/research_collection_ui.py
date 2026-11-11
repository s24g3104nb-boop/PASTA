"""Developer-only collection of measured photographs; never trains a model."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from io import BytesIO
from hashlib import sha256
import logging
from pathlib import Path
import sqlite3
import uuid

from PIL import Image, ImageOps, UnidentifiedImageError
import streamlit as st

from research_collection import CollectionStore, CollectionValidationError

LOGGER = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))
ANGLES = {'unknown': '未確認', 'overhead': '真上', 'oblique': '斜め上', 'side': '側面・端面'}
REFERENCES = {'none': '基準なし', 'a4_markers': 'A4の印刷マーカー',
              'known_board': '寸法を測った撮影台・背景板', 'noodle_length': '実測した麺の長さ',
              'fixed_distance': '撮影距離をそろえる', 'other': 'その他'}
RELATIONS = {'unknown': '以前の麺との関係は未確認',
             'new_disjoint': '未使用の別の麺を取り分けて秤量した',
             'reuse': '登録済みの麺を再使用・取り分けた',
             'added': '登録済みの麺に追加・混合した'}


def _new_draft(previous=None, *, same_sample=False):
    now = datetime.now(JST)
    previous = previous or {}
    draft = dict(token=uuid.uuid4().hex, photo_bytes=None, photo_name=None, saved=None,
                 measured_weight_g=0.0, product_name=previous.get('product_name', ''),
                 sample_id='sample-' + uuid.uuid4().hex[:12],
                 capture_session_id=previous.get('capture_session_id') or
                 'session-' + now.strftime('%Y%m%d') + '-' + uuid.uuid4().hex[:6],
                 captured_date=now.date(), captured_time=now.time().replace(tzinfo=None, microsecond=0),
                 angle='unknown', scale_reference='none', scale_verified=False,
                 coplanar=False, relation='unknown', parent_sample_ids=[], lineage_note='',
                 lineage_confirmed_by='', sample_choice='new', input_source='写真を選択',
                 upload_generation=0)
    if same_sample:
        for field in ('measured_weight_g', 'product_name', 'sample_id', 'relation',
                      'parent_sample_ids', 'lineage_note', 'lineage_confirmed_by'):
            draft[field] = previous[field]
        draft['sample_choice'] = previous['sample_id']
    return draft


def _draft():
    if 'research_collection_draft' not in st.session_state:
        st.session_state.research_collection_draft = _new_draft()
    return st.session_state.research_collection_draft


def _remember(field, key):
    draft = _draft()
    draft[field] = st.session_state[key]
    if field == 'relation' and draft[field] not in {'reuse','added'}:
        draft['parent_sample_ids'] = []
    if field == 'scale_reference':
        draft['scale_verified'] = False
        draft['coplanar'] = False
        for name in ('scale_verified','coplanar'):
            st.session_state.pop('collection_' + name + '_' + draft['token'], None)


def _field(widget, label, draft, field, **kwargs):
    key = 'collection_' + field + '_' + draft['token']
    if key not in st.session_state:
        st.session_state[key] = draft[field]
    return widget(label, key=key, on_change=_remember, args=(field, key), **kwargs)


def _remember_photo(key):
    uploaded = st.session_state.get(key)
    if uploaded is not None:
        draft = _draft()
        draft['photo_bytes'] = uploaded.getvalue()
        draft['photo_name'] = Path(uploaded.name).name


def _clear_photo():
    draft = _draft()
    draft.pop('quality_cache', None)
    draft['photo_bytes'] = None
    draft['photo_name'] = None
    draft['upload_generation'] += 1


def _start_next(same_sample=False):
    st.session_state.research_collection_draft = _new_draft(_draft(), same_sample=same_sample)


def _preview(photo_bytes):
    """Bound decoded images before passing them to Streamlit's media renderer."""
    if len(photo_bytes) > 20 * 1024 * 1024:
        raise ValueError('写真は20MB以下のJPEGまたはPNGにしてください。')
    with Image.open(BytesIO(photo_bytes)) as image:
        if (image.format not in {'JPEG', 'PNG'} or image.width * image.height > 40_000_000
                or min(image.size) < 16 or getattr(image, 'n_frames', 1) != 1):
            raise ValueError('写真は16px以上・4,000万画素以下の静止画JPEGまたはPNGにしてください。')
        image.load()
        preview = ImageOps.exif_transpose(image).convert('RGB')
        preview.thumbnail((1000, 1000))
        return preview


def _photo_diagnostic(draft):
    """Keep only the latest result inside this session, never across users."""
    from collection_quality import inspect_photo
    data = draft['photo_bytes']
    options = (draft['scale_reference'] == 'a4_markers',
               bool(draft['scale_verified']), bool(draft['coplanar']))
    signature = (sha256(data).hexdigest(), options)
    cache = draft.get('quality_cache')
    if cache and cache['signature'] == signature:
        return cache['result']
    # No history of full-size arrays: the overlay is bounded to 1600px.
    draft.pop('quality_cache', None)
    result = inspect_photo(data, reference=options[0], printed=options[1], coplanar=options[2])
    draft['quality_cache'] = {'signature': signature, 'result': result}
    return result


def _metadata(draft, samples):
    if draft['sample_choice'] != 'new':
        sample = samples[draft['sample_choice']]
        physical = {key: sample[key] for key in ('sample_id', 'measured_weight_g', 'product_name',
                    'relation', 'parent_sample_ids', 'lineage_note', 'lineage_confirmed_by')}
    else:
        physical = {key: draft[key] for key in ('sample_id', 'measured_weight_g', 'product_name',
                    'relation', 'parent_sample_ids', 'lineage_note', 'lineage_confirmed_by')}
    return dict(physical, captured_at=datetime.combine(draft['captured_date'], draft['captured_time'],
                tzinfo=JST).isoformat(), capture_session_id=draft['capture_session_id'],
                angle=draft['angle'], scale_reference=draft['scale_reference'],
                scale_verified=(draft['scale_verified'] and draft['scale_reference'] != 'none'),
                reference_coplanar=draft['coplanar'], original_filename=draft['photo_name'])


def _render_saved(draft, store):
    row = draft['saved']
    st.success('登録済みの写真です。重複登録は行いませんでした。' if row.get('status') == 'duplicate'
               else '写真と実測ラベルを保存しました。')
    st.info('学習・評価への利用は未承認です。独立性とマスクを別途確認してから利用します。')
    st.caption('登録ID：' + str(row['id']))
    quality = row.get('quality')
    if quality:
        with st.expander('撮影品質の確認結果'):
            st.json(quality)
    with st.expander('原本・プレビューの照合情報'):
        st.write('原本 SHA-256：' + row['original_sha256'])
        st.write('プレビュー SHA-256：' + row['preview_sha256'])
        st.caption('プレビューは向き・サイズ・位置情報などの付加情報を整理した別ファイルです。')
    st.button('同じ試料の別の写真を登録', key='collection_next_photo', on_click=_start_next,
              args=(True,), type='primary')
    st.button('別の秤量試料を登録', key='collection_next_sample', on_click=_start_next)


def render_research_collection():
    st.subheader('独立試料・撮影データ登録')
    st.write('写真とはかりで測った重量を保存します。画像からの重量推定や自動学習は行いません。')
    st.caption('同じ麺の別写真・取り分け・積み増しは関連付けて管理します。'
               '登録だけで独立した評価データにはなりません。')
    try:
        store = CollectionStore()
        listed_samples = store.list_samples()
        samples = {row['sample_id']: row for row in listed_samples}
    except (OSError, sqlite3.Error, ValueError):
        LOGGER.exception('Cannot open the separate research collection')
        st.error('撮影データの保存先を開けませんでした。保存先の空き容量とアクセス権を確認し、再度開いてください。')
        return
    from capture_guide_ui import render_sheet_guide
    with st.expander('撮影の準備・A4シートの使い方'):
        render_sheet_guide()
    st.caption('保存先（このPC内）：' + str(store.registration_root))
    draft = _draft()
    if draft['saved']:
        _render_saved(draft, store)
    else:
        choices = ['new'] + list(samples)
        if draft['sample_choice'] not in choices:
            draft['sample_choice'] = 'new'
        _field(st.selectbox, '撮影する試料', draft, 'sample_choice', options=choices,
               format_func=lambda value: '新しい秤量試料' if value == 'new' else
               f"{value} · {samples[value]['measured_weight_g']:g} g · {samples[value]['product_name']}")
        if draft['sample_choice'] == 'new':
            _field(st.number_input, 'はかりに表示された実測重量（g）', draft, 'measured_weight_g',
                   min_value=0.0, max_value=2000.0, step=0.1, format='%.2f')
            _field(st.text_input, 'パスタの製品名', draft, 'product_name', max_chars=200)
            with st.expander('試料の識別・以前の麺との関係', expanded=True):
                _field(st.text_input, '物理試料ID', draft, 'sample_id', max_chars=100, help='同じ麺の別写真では同じID。麺を追加・取り分けたら新しいIDを付け、元のIDに関連付けます。')
                _field(st.selectbox, '以前に撮った麺との関係', draft, 'relation',
                       options=list(RELATIONS), format_func=RELATIONS.get)
                if draft['relation'] in {'reuse', 'added'}:
                    _field(st.multiselect, '再使用・追加元の試料ID', draft, 'parent_sample_ids', options=list(samples))
                    st.caption('追加元が未登録の場合、先にその試料を登録するか「関係は未確認」で保存してください。')
                if draft['relation'] != 'unknown':
                    _field(st.text_input, '関係を確認した人（記録用）', draft, 'lineage_confirmed_by', max_chars=100)
                    _field(st.text_area, '取り分け・再使用の確認根拠', draft, 'lineage_note', max_chars=2000,
                           placeholder='例：未使用の麺を別容器に取り分け、他の試料には戻していない。')
                else:
                    st.caption('未確認の試料は学習用・評価用に自動で分割しません。')
        else:
            existing = samples[draft['sample_choice']]
            st.info(f"同じ試料：実測 {existing['measured_weight_g']:g} g / {existing['product_name']}")
            st.caption('麺を足した・減らした場合は、新しい秤量試料として元の試料に関連付けてください。')
        _field(st.radio, '写真の入力', draft, 'input_source', options=['写真を選択', 'カメラで撮影'], horizontal=True)
        upload_key = 'collection_upload_' + draft['token'] + '_' + str(draft['upload_generation'])
        if draft['input_source'] == '写真を選択':
            st.file_uploader('乾麺の写真（JPEG・PNG）', type=['jpg', 'jpeg', 'png'], key=upload_key,
                             on_change=_remember_photo, args=(upload_key,))
        else:
            upload_key += '_camera'
            st.camera_input('乾麺を撮影', key=upload_key, on_change=_remember_photo, args=(upload_key,))
        photo_valid = False
        if draft['photo_bytes']:
            try:
                st.image(_preview(draft['photo_bytes']), caption='保存予定の写真', width='stretch')
                photo_valid = True
            except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError):
                st.error('写真を開けません。20MB・4,000万画素以下のJPEGまたはPNGを選び直してください。')
            st.button('写真を選び直す', key='collection_clear_photo', on_click=_clear_photo)
        with st.expander('撮影日時・条件を確認', expanded=True):
            _field(st.date_input, '撮影日（日本時間）', draft, 'captured_date')
            _field(st.time_input, '撮影時刻（日本時間）', draft, 'captured_time')
            _field(st.text_input, '撮影セッションID', draft, 'capture_session_id', max_chars=100)
            st.caption('同じ撮影会ではセッションIDをそろえてください。日時の初期値は現在日時です。')
            _field(st.selectbox, '撮影角度', draft, 'angle', options=list(ANGLES), format_func=ANGLES.get)
            _field(st.selectbox, '写真内のスケール基準・撮影条件', draft, 'scale_reference',
                   options=list(REFERENCES), format_func=REFERENCES.get)
            if draft['scale_reference'] != 'none':
                _field(st.checkbox, '基準の実寸・条件を実際に確認した', draft, 'scale_verified')
                if draft['scale_reference'] == 'a4_markers':
                    st.caption('A4を100%で印刷し、確認線が100mmあることを定規で測ってください。'
                               'チェックだけでマーカー検出や寸法精度が保証されるわけではありません。')
                    _field(st.checkbox, '紙は平らで、乾麺と基準を同じ撮影面に置いた', draft, 'coplanar')
        if photo_valid:
            try:
                quality, overlay = _photo_diagnostic(draft)
            except Exception:
                LOGGER.exception('Research photo diagnostic failed')
                st.error('写真の状態を確認できませんでした。入力内容は残っています。'
                         '別の写真を選ぶか、この画面を開き直してください。')
                photo_valid = False
            else:
                st.subheader('写真の確認結果')
                st.warning(quality['status'])
                messages={'foreground_touches_frame':'麺らしい領域が画像の端に接しています。両端が入るように撮り直してください。',
                    'dark_image':'写真が暗すぎます。明るい場所で撮影してください。',
                    'blur_review':'ピントが甘い可能性があります。麺にピントを合わせてください。',
                    'reference_unavailable':'寸法基準を利用できません。4個のマーカーと印刷寸法を確認してください。',
                    'insufficient_foreground':'麺の領域を十分に確認できません。'}
                for flag in quality['flags']:st.caption(messages.get(flag,'画質を目視で確認してください：'+flag))
                st.caption('パスタ全体が写っているかは自動で保証できません。隠れた麺・切れた端を原本で確認してください。')
                if draft['scale_reference']=='a4_markers':
                    st.image(overlay,caption='マーカーの検出結果',width='stretch')
                    st.write('検出数：',quality['reference']['detected_marker_count'])
                    st.write('寸法基準：' + ('研究用の紙面基準として利用可能（申告条件付き）' if quality['reference']['usable_for_metric_scale'] else '利用不可・確認が必要'))
                st.caption('確認が必要な写真も未承認資料として保存できます。学習には自動使用しません。')
        if photo_valid:
            st.caption('保存前の確認：実測重量と製品名は、はかりの表示・パッケージと照合してください。')
        if st.button('写真と実測ラベルを保存', key='collection_save', type='primary', disabled=not photo_valid):
            try:
                metadata = _metadata(draft, samples)
                with st.spinner('写真と実測ラベルを保存しています…'):
                    row = store.register(draft['photo_bytes'], metadata, submission_token=draft['token'])
                draft.update({key: metadata[key] for key in ('sample_id', 'product_name', 'measured_weight_g',
                    'relation', 'parent_sample_ids', 'lineage_note', 'lineage_confirmed_by')})
                draft['saved'] = row
                st.rerun()
            except CollectionValidationError as exc:
                st.error(str(exc))
            except (OSError, sqlite3.Error):
                LOGGER.exception('Research collection save failed')
                st.error('保存できませんでした。入力は保持しています。保存先の空き容量を確認して、もう一度保存してください。')
    try:
        recent = store.list_records(limit=10)
        with st.expander('最近登録した写真'):
            st.caption(f'直近 {len(recent)} 件・すべて学習利用未承認')
            for row in recent:
                meta = row['metadata']
                st.write(f"{meta['measured_weight_g']:g} g · {meta['product_name']} · {meta['sample_id']}")
                st.caption(meta['captured_at'])
                st.image(store.read_preview(row['id']), width=180)
    except (OSError, sqlite3.Error, ValueError):
        LOGGER.exception('Cannot read recent research records')
        st.warning('最近の登録一覧を開けませんでした。保存済みデータは削除していません。')
