"""Photo-first meal diary. Drafts live independently of Streamlit widget lifetime."""
from __future__ import annotations

import calendar
from datetime import date, datetime, time, timedelta, timezone
from io import BytesIO
import logging
import sqlite3
import uuid

import streamlit as st

from cooking_storage import list_cooking_sessions
from meal_diary import delete, entries, get, photo_path, save
from pasta_vision import image_to_rgb
from storage import DEFAULT_DB, list_measurements

LOGGER = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))
MEAL_TYPES = ['', '朝食', '昼食', '夕食', '間食', 'その他']
NUTRITION_LABELS = {
    'unknown': '未計算',
    'dry_only': '乾麺分のみ',
    'confirmed_amounts': '材料・使用量に基づく計算値',
    'template_estimate': '参考レシピによる概算',
    'validated_image_ai': '検証済み写真AIによる推定',
}


def _drafts():
    if 'diary_drafts' not in st.session_state:
        st.session_state.diary_drafts = {}
    return st.session_state.diary_drafts


def _new_draft(*, day=None, measurement_id=None, cooking_session_id=None):
    now = datetime.now(JST)
    ident = uuid.uuid4().hex
    return {
        'id': ident, 'edit': False, 'dish_name': '', 'eaten_date': day or now.date(),
        'eaten_time': now.time().replace(second=0, microsecond=0, tzinfo=None),
        'time_known': True, 'meal_type': '', 'memo': '', 'photo_bytes': None,
        'photo_name': None, 'photo': None, 'remove_photo': False,
        'measurement_id': measurement_id, 'cooking_session_id': cooking_session_id,
        'expected_updated_at': None, 'saved': False,
    }


def _activate(draft):
    _drafts()[draft['id']] = draft
    st.session_state.diary_active_draft = draft['id']
    st.session_state.diary_screen = 'editor'


def _route_diary():
    st.session_state.main_nav = '記録'
    st.session_state.record_kind = '食事日記'


def start_diary(measurement_id=None, cooking_session_id=None, day=None):
    """Navigation callback: resume an unsaved draft, or start a contextual meal."""
    drafts = _drafts()
    active = drafts.get(st.session_state.get('diary_active_draft'))
    returning = st.session_state.pop('cooking_return_diary', False)
    if returning and active and not active['saved']:
        active['measurement_id'] = measurement_id
        active['cooking_session_id'] = cooking_session_id
        _activate(active)
    elif measurement_id is not None or cooking_session_id is not None or day is not None:
        matching = next((d for d in drafts.values() if not d['saved'] and not d['edit']
                         and d['measurement_id'] == measurement_id
                         and d['cooking_session_id'] == cooking_session_id
                         and (day is None or d['eaten_date'] == day)), None)
        _activate(matching or _new_draft(day=day, measurement_id=measurement_id,
                                       cooking_session_id=cooking_session_id))
    elif active and not active['saved']:
        _activate(active)
    else:
        _activate(_new_draft())
    _route_diary()


def _edit_entry(row):
    drafts = _drafts()
    current = drafts.get(row['id'])
    if current and current['edit'] and not current['saved']:
        _activate(current)
        return
    draft = _new_draft(day=date.fromisoformat(row['eaten_date']))
    draft.update({
        'id': row['id'], 'edit': True, 'dish_name': row['dish_name'],
        'eaten_time': time.fromisoformat(row['eaten_time']) if row['eaten_time'] else draft['eaten_time'],
        'time_known': bool(row['eaten_time']), 'meal_type': row.get('meal_type') or '',
        'memo': row['memo'], 'photo': row['photo'],
        'measurement_id': row.get('measurement_id'),
        'cooking_session_id': row.get('cooking_session_id'),
        'expected_updated_at': row['updated_at'],
    })
    _activate(draft)


def open_diary_entry(ident):
    st.session_state.diary_detail_id = ident
    st.session_state.diary_screen = 'detail'
    _route_diary()


def _show_list():
    st.session_state.diary_screen = 'list'


def _remember_field(draft_id, field, widget_key):
    _drafts()[draft_id][field] = st.session_state[widget_key]


def _field(widget, label, draft, field, **kwargs):
    key = 'meal_' + field + '_' + draft['id']
    if key not in st.session_state:
        st.session_state[key] = draft[field]
    return widget(label, key=key, on_change=_remember_field,
                  args=(draft['id'], field, key), **kwargs)


def _upload_changed(draft_id, key):
    uploaded = st.session_state.get(key)
    if uploaded is not None:
        draft = _drafts()[draft_id]
        draft['photo_bytes'] = uploaded.getvalue()
        draft['photo_name'] = uploaded.name
        draft['remove_photo'] = False


def _clear_photo(draft_id):
    draft = _drafts()[draft_id]
    draft['photo_bytes'] = None
    draft['photo_name'] = None
    draft['remove_photo'] = True
    draft['upload_generation'] = draft.get('upload_generation', 0) + 1


def _photo(row, db_path, *, preview=False):
    if row.get('photo'):
        try:
            path = photo_path(row['photo'], db_path)
            if path.is_file():
                st.image(str(path), width='stretch')
                return
        except (ValueError, OSError):
            LOGGER.warning('Diary photo could not be opened', exc_info=True)
        st.caption('写真を開けませんでした。記録の内容は確認できます。')
    if not preview:
        st.markdown('<div class="pl-meal-placeholder" role="img" aria-label="料理写真なし">'
                    '<span aria-hidden="true">🍝</span><small>写真なし</small></div>', unsafe_allow_html=True)


def _show_measurement(snapshot):
    if not snapshot:
        return
    st.caption('使用した乾麺 · ' + str(snapshot.get('product_name') or '製品名なし'))
    estimated = snapshot.get('estimated_g')
    actual = snapshot.get('actual_g')
    if estimated is not None:
        st.write(f'推定重量 {estimated:g} g')
    if actual is not None:
        st.write(f'実測重量 {actual:g} g')
    else:
        st.caption('実測重量は未登録です。')
    if snapshot.get('source_kind') == 'synthetic' or 'デモ' in str(snapshot.get('product_name', '')):
        st.warning('この乾麺はデモデータです。実際の食事の測定値ではありません。')


def _nutrition(row, *, detailed=False):
    status = row.get('nutrition_status', 'unknown')
    snapshot = row.get('nutrition_snapshot')
    label = NUTRITION_LABELS.get(status, '未計算')
    if not snapshot or status == 'unknown':
        st.caption('栄養：未計算')
        return
    kcal = snapshot.get('kcal')
    if kcal is None:
        st.caption('栄養：未計算')
        return
    st.write(f'{kcal:g} kcal · {label}')
    if detailed:
        values = [(name, snapshot.get(key)) for name, key in
                  [('たんぱく質', 'protein_g'), ('脂質', 'fat_g'), ('炭水化物', 'carb_g')]]
        st.caption(' / '.join(f'{name} {value:g} g' for name, value in values if value is not None))
        source = row.get('nutrition_source')
        if source:
            st.caption(source)
        if status == 'dry_only':
            st.info('乾麺分だけの栄養です。ソースや具材は含まれません。')
        if status == 'template_estimate':
            st.info('参考レシピの分量を前提とした概算です。写真から計算した値ではありません。')


def _go_cooking(draft):
    _activate(draft)
    st.session_state.cooking_return_diary = True
    st.session_state.cooking_source_measurement_id = draft.get('measurement_id')
    st.session_state.main_nav = '設定'
    st.session_state.settings_section = '料理の栄養計算'


def _link_options(draft, db_path):
    """Reuse saved source records, leaving all calculation and source rows unchanged."""
    measurements = list_measurements(db_path)
    sessions = list_cooking_sessions(db_path)
    measurement_map = {r['id']: r for r in measurements}
    session_map = {r['id']: r for r in sessions}
    with st.expander('乾麺・栄養の記録を関連付ける（任意）', expanded=bool(draft.get('measurement_id') or draft.get('cooking_session_id'))):
        options = [None] + list(measurement_map)
        if draft['measurement_id'] not in options:
            options.append(draft['measurement_id'])
        _field(st.selectbox, '乾麺の記録', draft, 'measurement_id', options=options,
               format_func=lambda ident: '関連付けない' if ident is None else
               (f"{measurement_map[ident]['product_name']} · 推定 {measurement_map[ident]['estimated_g']:g} g · #{ident}"
                if ident in measurement_map else f'保存済みの乾麺 #{ident}'))
        chosen = measurement_map.get(draft['measurement_id'])
        if chosen:
            _show_measurement(chosen)
        compatible = {ident: row for ident, row in session_map.items()
                      if draft['measurement_id'] is None or row['measurement_id'] == draft['measurement_id']}
        options = [None] + list(compatible)
        if draft['cooking_session_id'] not in options:
            # Keep the user's choice visible until they explicitly resolve a mismatched link.
            options.append(draft['cooking_session_id'])
        _field(st.selectbox, '保存した料理の栄養計算', draft, 'cooking_session_id', options=options,
               format_func=lambda ident: 'まだ計算しない' if ident is None else
               (f"{session_map[ident]['selected_dish']} · 料理 #{ident}" if ident in session_map
                else f'保存済みの料理 #{ident}'))
        cooking = session_map.get(draft['cooking_session_id'])
        if cooking:
            if draft['measurement_id'] is not None and cooking['measurement_id'] != draft['measurement_id']:
                st.warning('乾麺の記録と料理の計算が一致していません。どちらかを選び直してください。')
            calculation = cooking['calculation']
            _nutrition({'nutrition_status': calculation.get('nutrition_basis', 'confirmed_amounts'),
                        'nutrition_snapshot': calculation['portion'],
                        'nutrition_source': '保存済み料理の、食べた分として計算した値'}, detailed=True)
        elif chosen:
            st.caption('保存すると乾麺分のみの栄養が引き継がれます。')
        else:
            st.caption('栄養が未計算のままでも保存できます。')
        st.button('材料を入力して栄養を計算する', key='meal_open_cooking',
                  on_click=_go_cooking, args=(draft,), width='stretch')


def _draft_selector():
    drafts = [d for d in _drafts().values() if not d['saved']]
    if len(drafts) < 2:
        return
    with st.expander('入力途中の記録'):
        for draft in drafts:
            label = draft['dish_name'].strip() or '料理名未入力'
            st.button(f"{draft['eaten_date'].isoformat()} · {label}",
                      key='resume_' + draft['id'], on_click=_activate, args=(draft,), width='stretch')


def _render_editor(db_path):
    draft = _drafts().get(st.session_state.get('diary_active_draft'))
    if not draft:
        _activate(_new_draft())
        draft = _drafts()[st.session_state.diary_active_draft]
    st.button('記録一覧へ戻る', key='meal_editor_back', on_click=_show_list)
    st.subheader('食事記録を編集' if draft['edit'] else '食べたパスタを記録する')
    st.caption('料理名と日付だけでも保存できます。入力途中の内容は、この接続中は保持されます。')
    _draft_selector()
    if draft['saved']:
        st.success('食事記録を保存しました。')
        st.button('保存した記録を見る', key='meal_view_saved', type='primary',
                  on_click=open_diary_entry, args=(draft['id'],), width='stretch')
        st.button('別の食事を記録する', key='meal_start_another', on_click=start_diary, width='stretch')
        return
    upload_key = f"meal_photo_{draft['id']}_{draft.get('upload_generation', 0)}"
    st.file_uploader('完成料理の写真（任意）', type=['jpg', 'jpeg', 'png'], key=upload_key,
                     on_change=_upload_changed, args=(draft['id'], upload_key),
                     help='スマートフォンのカメラで撮影した写真も選べます。')
    valid_photo = True
    if draft['photo_bytes'] is not None:
        try:
            rgb = image_to_rgb(BytesIO(draft['photo_bytes']))
            st.image(rgb, width='stretch')
            st.caption('選択中：' + (draft['photo_name'] or '料理写真'))
        except (ValueError, OSError):
            valid_photo = False
            st.error('写真を読み込めませんでした。JPEG または PNG の写真を選び直してください。')
    elif draft['photo'] and not draft['remove_photo']:
        _photo(draft, db_path, preview=True)
    if draft['photo_bytes'] is not None or (draft['photo'] and not draft['remove_photo']):
        st.button('写真を外す', key='meal_remove_photo', on_click=_clear_photo, args=(draft['id'],))
    _field(st.text_input, '料理名', draft, 'dish_name', placeholder='例：トマトのパスタ', max_chars=120)
    _field(st.date_input, '食べた日', draft, 'eaten_date', min_value=date(1900, 1, 1),
           max_value=date(2100, 12, 31), format='YYYY/MM/DD')
    _field(st.checkbox, '食べた時刻も記録する', draft, 'time_known')
    if draft['time_known']:
        _field(st.time_input, '食べた時刻（日本時間）', draft, 'eaten_time')
    _field(st.selectbox, '食事の区分（任意）', draft, 'meal_type', options=MEAL_TYPES,
           format_func=lambda value: value or '指定しない')
    _field(st.text_area, 'ひとこと（任意）', draft, 'memo', max_chars=2000)
    try:
        _link_options(draft, db_path)
    except (sqlite3.Error, OSError, ValueError):
        LOGGER.warning('Meal source records unavailable', exc_info=True)
        st.warning('関連する記録を読み込めません。接続先の保存領域を確認して、もう一度お試しください。')
    st.caption('写真AIによるカロリー推定は準備中です。写真だけでは材料や分量を判断できません。')
    if st.button('食事記録を保存', key='meal_save', type='primary', width='stretch', disabled=not valid_photo):
        try:
            with st.spinner('食事記録を保存しています…'):
                saved = save(draft['id'], draft['eaten_date'], draft['eaten_time'] if draft['time_known'] else None,
                             draft['dish_name'], draft['memo'], draft['photo_bytes'], db_path,
                             meal_type=draft['meal_type'], measurement_id=draft['measurement_id'],
                             cooking_session_id=draft['cooking_session_id'], edit=draft['edit'],
                             expected_updated_at=draft['expected_updated_at'], remove_photo=draft['remove_photo'])
            draft['saved'] = True
            draft['expected_updated_at'] = saved['updated_at']
            st.session_state.diary_notice = '食事記録を保存しました。'
            st.session_state.diary_detail_id = saved['id']
            st.session_state.diary_screen = 'detail'
            # Keep the exact submission id as a receipt; a new id requires an explicit new action.
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
        except (sqlite3.Error, OSError):
            LOGGER.warning('Meal save failed; draft retained', exc_info=True)
            st.error('保存できませんでした。入力内容は残っています。保存先の空き容量や接続を確認し、もう一度保存してください。')


def _render_detail(rows, db_path):
    row = next((r for r in rows if r['id'] == st.session_state.get('diary_detail_id')), None)
    st.button('記録一覧へ戻る', key='meal_detail_back', on_click=_show_list)
    if row is None:
        st.info('この食事記録は見つからないか、削除されています。')
        return
    notice = st.session_state.pop('diary_notice', None)
    if notice:
        st.success(notice)
    with st.container(border=True, key='diary_detail'):
        _photo(row, db_path)
        st.subheader(row['dish_name'])
        st.write(row['eaten_date'] + (' ' + row['eaten_time'][:5] if row['eaten_time'] else '') + '（日本時間）')
        if row.get('meal_type'):
            st.caption(row['meal_type'])
        if row['memo']:
            st.write(row['memo'])
        _show_measurement(row.get('measurement_snapshot'))
        _nutrition(row, detailed=True)
        if row.get('source_kind') == 'synthetic' and not row.get('measurement_snapshot'):
            st.warning('デモ用の記録です。')
    st.button('記録を編集する', key='meal_edit', on_click=_edit_entry, args=(row,), width='stretch')
    st.button('栄養計算を追加・変更する', key='meal_detail_cooking', width='stretch',
              on_click=_detail_to_cooking, args=(row,))
    st.button('別の食事を記録する', key='meal_detail_new', on_click=start_diary, width='stretch')
    with st.expander('この記録を削除する'):
        confirmed = st.checkbox('この食事記録を一覧から削除する', key='meal_delete_confirm_' + row['id'])
        st.caption('関連する乾麺の測定履歴や栄養計算は削除しません。')
        if st.button('削除を確定', key='meal_delete', disabled=not confirmed):
            try:
                delete(row['id'], db_path, expected_updated_at=row['updated_at'])
                _drafts().pop(row['id'], None)
                st.session_state.diary_notice = '食事記録を削除しました。'
                _show_list()
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
            except (sqlite3.Error, OSError):
                LOGGER.warning('Meal delete failed', exc_info=True)
                st.error('削除できませんでした。記録は残っています。もう一度お試しください。')


def _select_day(day):
    st.session_state.diary_day = day


def _shift_month(delta):
    month = st.session_state.get('diary_month_value', datetime.now(JST).date()).replace(day=1)
    absolute = month.year * 12 + month.month - 1 + delta
    st.session_state.diary_month_value = date(absolute // 12, absolute % 12 + 1, 1)
    st.session_state.diary_day = st.session_state.diary_month_value
    st.session_state.pop('diary_month', None)


def _month_changed():
    selected = st.session_state.diary_month
    st.session_state.diary_month_value = selected.replace(day=1)
    day = st.session_state.get('diary_day')
    if day is None or (day.year, day.month) != (selected.year, selected.month):
        st.session_state.diary_day = selected


def _calendar(rows):
    now = datetime.now(JST).date()
    month = st.session_state.setdefault('diary_month_value', now.replace(day=1))
    st.session_state.setdefault('diary_day', now)
    with st.container(key='diary_calendar'):
        left, right = st.columns(2)
        left.button('前の月', key='diary_previous_month', on_click=_shift_month, args=(-1,), width='stretch')
        right.button('次の月', key='diary_next_month', on_click=_shift_month, args=(1,), width='stretch')
        st.date_input('表示する月', value=month, key='diary_month', format='YYYY/MM/DD',
                      on_change=_month_changed, min_value=date(1900, 1, 1), max_value=date(2100, 12, 31))
        st.caption(f'{month.year}年{month.month}月 · 印がある日は食事の記録があります。')
        for column, weekday in zip(st.columns(7), ['月', '火', '水', '木', '金', '土', '日']):
            column.markdown(f'**{weekday}**')
        counts = {}
        for row in rows:
            counts[row['eaten_date']] = counts.get(row['eaten_date'], 0) + 1
        for week in calendar.Calendar().monthdatescalendar(month.year, month.month):
            for column, day in zip(st.columns(7), week):
                count = counts.get(day.isoformat(), 0)
                column.button(str(day.day) + (' •' if count else ''),
                              key='diary_day_' + day.isoformat(), help=f'{day.isoformat()} · {count}件',
                              on_click=_select_day, args=(day,), width='stretch',
                              type='primary' if day == st.session_state.diary_day else 'secondary',
                              disabled=day.month != month.month)
    selected = st.session_state.diary_day
    st.subheader(f'{selected.year}年{selected.month}月{selected.day}日')
    st.button('この日に食事を記録する', key='diary_add_selected_day', on_click=start_diary,
              kwargs={'day': selected}, width='stretch')
    return selected


def _filter_changed(key):
    st.session_state.setdefault('diary_filters', {})[key] = st.session_state['diary_filter_' + key]


def _render_list(rows, db_path):
    notice = st.session_state.pop('diary_notice', None)
    if notice:
        st.success(notice)
    st.button('食べたパスタを記録する', key='diary_add', type='primary', width='stretch', on_click=start_diary)
    _draft_selector()
    view = st.radio('記録の表示', ['写真付き一覧', 'カレンダー'], key='diary_view', horizontal=True)
    selected = _calendar(rows) if view == 'カレンダー' else None
    now = datetime.now(JST).date()
    oldest = min((date.fromisoformat(r['eaten_date']) for r in rows), default=now)
    newest = max((date.fromisoformat(r['eaten_date']) for r in rows), default=now)
    filters = st.session_state.setdefault('diary_filters', {'query': '', 'start': min(oldest, now), 'end': max(newest, now)})
    st.text_input('料理名で検索', value=filters['query'], key='diary_filter_query',
                  on_change=_filter_changed, args=('query',))
    with st.expander('期間で絞り込む', expanded=bool(filters['start'] > oldest or filters['end'] < newest)):
        columns = st.columns(2)
        columns[0].date_input('期間の開始', value=filters['start'], key='diary_filter_start',
                              on_change=_filter_changed, args=('start',), min_value=date(1900, 1, 1), max_value=date(2100, 12, 31))
        columns[1].date_input('期間の終了', value=filters['end'], key='diary_filter_end',
                              on_change=_filter_changed, args=('end',), min_value=date(1900, 1, 1), max_value=date(2100, 12, 31))
        if st.button('検索条件をリセット', key='diary_reset_filters'):
            st.session_state.diary_filters = {'query': '', 'start': min(oldest, now), 'end': max(newest, now)}
            for key in ('query', 'start', 'end'):
                st.session_state.pop('diary_filter_' + key, None)
            st.rerun()
    if filters['start'] > filters['end']:
        st.warning('期間の終了を、開始と同じ日か後の日にしてください。')
        return
    shown = [r for r in rows if filters['start'].isoformat() <= r['eaten_date'] <= filters['end'].isoformat()
             and filters['query'].strip().casefold() in r['dish_name'].casefold()
             and (selected is None or r['eaten_date'] == selected.isoformat())]
    st.caption(f'{len(shown)}件の食事記録')
    if not shown:
        st.info('この条件の食事記録はありません。別の日付・料理名で探すか、新しく記録できます。')
    # Page size bounds image loading on long diaries while preserving chronological order.
    pages = max(1, (len(shown) + 11) // 12)
    page = st.selectbox('表示ページ', range(1, pages + 1), key='diary_list_page') if pages > 1 else 1
    with st.container(key='diary_list'):
        for row in shown[(page - 1) * 12:page * 12]:
            with st.container(border=True):
                _photo(row, db_path)
                st.subheader(row['dish_name'])
                st.caption(row['eaten_date'] + (' ' + row['eaten_time'][:5] if row['eaten_time'] else '')
                           + (' · ' + row['meal_type'] if row.get('meal_type') else ''))
                _nutrition(row)
                st.button('この食事を見る', key='meal_detail_' + row['id'],
                          on_click=open_diary_entry, args=(row['id'],), width='stretch')


def render_diary(db_path=DEFAULT_DB):
    st.title('パスタ日記')
    try:
        rows = entries(db_path)
    except (sqlite3.Error, OSError, ValueError):
        LOGGER.warning('Meal diary unavailable', exc_info=True)
        st.error('記録を読み込めませんでした。保存先を確認して、もう一度お試しください。入力途中の内容は保持しています。')
        if st.button('もう一度読み込む', key='diary_retry'):
            st.rerun()
        return
    screen = st.session_state.get('diary_screen', 'list')
    if screen == 'editor':
        _render_editor(db_path)
    elif screen == 'detail':
        _render_detail(rows, db_path)
    else:
        _render_list(rows, db_path)


def _detail_to_cooking(row):
    _edit_entry(row)
    _go_cooking(_drafts()[st.session_state.diary_active_draft])
