"""学習データ管理と非同期CPU学習の画面。測定画面から分離する。"""
from __future__ import annotations

import sqlite3

import pandas as pd
import streamlit as st

from storage import DEFAULT_DB, init_db, list_products
from training_data import (delete_sample, init_training_db, list_samples,
                           register_sample, update_sample)
from training_manager import (activate_model, get_active_model, init_training_manager,
                              list_jobs, list_models, start_training)


def _fields(prefix, products, current=None):
    current = current or {}
    product_names = {str(p['id']): p['name'] for p in products}
    if current.get('product_id') and current['product_id'] not in product_names:
        product_names[current['product_id']] = current['product_id']
    options = list(product_names)
    selected_product = current.get('product_id')
    product_id = st.selectbox(
        '製品・銘柄', options, index=options.index(selected_product) if selected_product in options else 0,
        format_func=lambda key: product_names[key], key=f'{prefix}_product')
    if product_names[product_id].startswith('デモ用'):
        st.warning('この製品は仮のデモ値です。実写の登録前に「パスタデータ」で銘柄・栄養・校正値を登録してください。')
    actual = st.number_input('電子はかりで測った実測重量（g）', min_value=0.001,
                             value=float(current['actual_weight_g']) if current else None,
                             step=0.1, key=f'{prefix}_weight',
                             help='未入力の写真は登録・学習できません。推定値を正解値にしないでください。')
    mode = st.selectbox('撮影状態', ['spread', 'bundle_end'],
                        index=1 if current.get('mode') == 'bundle_end' else 0,
                        format_func=lambda value: '広げた状態' if value == 'spread' else '束の状態',
                        key=f'{prefix}_mode')
    pasta_type = st.text_input('パスタの種類', value=current.get('pasta_type', ''),
                               placeholder='スパゲッティ', key=f'{prefix}_type')
    thickness = st.number_input('麺の太さ（mm、不明なら空欄）', min_value=0.001,
                                value=current.get('thickness_mm'), step=0.1,
                                key=f'{prefix}_thickness')
    captured = st.text_input('撮影日時（ISO形式、不明なら空欄）',
                             value=current.get('captured_at') or '',
                             placeholder='2026-10-28T12:30:00+09:00', key=f'{prefix}_captured')
    batch = st.text_input('秤量バッチID', value=current.get('batch_id', ''),
                          help='同じ乾麺を繰り返し撮る場合は同じID。重量が同じでも別に量った試料は別IDです。',
                          key=f'{prefix}_batch')
    session = st.text_input('撮影セッションID', value=current.get('session_id', ''),
                            help='同じ撮影セッションの写真は学習用と評価用に分けません。',
                            key=f'{prefix}_session')
    source = st.selectbox('画像の由来', ['real', 'synthetic'],
                          index=1 if current.get('source') == 'synthetic' else 0,
                          format_func=lambda value: '実写' if value == 'real' else 'CG・合成画像（テスト専用）',
                          key=f'{prefix}_source')
    lighting = st.text_input('照明条件', value=current.get('lighting', ''), key=f'{prefix}_lighting')
    background = st.text_input('背景', value=current.get('background', ''), key=f'{prefix}_background')
    notes = st.text_area('撮影条件・メモ', value=current.get('notes', ''), key=f'{prefix}_notes')
    parent_id = st.number_input('データ拡張元の登録ID（元画像なら空欄）', min_value=1,
                                value=current.get('parent_sample_id'), step=1,
                                key=f'{prefix}_parent',
                                help='加工・拡張画像は元画像と同じ分割に固定します。')
    return dict(actual_weight_g=actual, mode=mode, product_id=product_id, pasta_type=pasta_type,
                thickness_mm=thickness, captured_at=captured, batch_id=batch, session_id=session,
                source=source, lighting=lighting, background=background, notes=notes,
                parent_sample_id=parent_id)


def render_training_tab(db_path=DEFAULT_DB):
    init_db(db_path)
    init_training_db(db_path)
    init_training_manager(db_path)
    st.subheader('AI学習管理')
    st.caption('ローカル管理者向けです。写真と実測重量を登録し、CPU回帰モデルを別プロセスで学習します。')
    st.warning('実写真による精度は未検証です。5重量の写真だけで35g・75g等への精度を保証できません。'
               '公開認証は未実装のため、この管理画面を無認証で外部公開しないでください。')
    products = list_products(db_path)
    with st.expander('写真と実測重量を登録', expanded=True):
        with st.form('training_register'):
            image = st.file_uploader('乾麺の写真（JPEG / PNG）', type=['jpg', 'jpeg', 'png'], key='training_upload')
            metadata = _fields('training_new', products)
            submitted = st.form_submit_button('学習データを登録')
        if submitted:
            try:
                if image is None:
                    raise ValueError('写真をアップロードしてください。')
                ident = register_sample(image, metadata, db_path=db_path)
                st.success(f'学習データを登録しました（ID {ident}）。')
            except (ValueError, TypeError, OSError, sqlite3.Error) as exc:
                st.error(f'登録できませんでした: {exc}')
    samples = list_samples(db_path)
    real = [row for row in samples if row['source'] == 'real']
    st.write(f'登録中: {len(samples)}枚 ／ 実写: {len(real)}枚 ／ 合成: {len(samples) - len(real)}枚')
    st.caption(f"実写の秤量バッチ数: {len({r['batch_id'] for r in real})} ／ "
               f"撮影セッション数: {len({r['session_id'] for r in real})}。実際の分割では同一・近似画像も検査します。")
    _render_samples(samples, products, db_path)
    st.divider()
    st.subheader('学習と実行状況')
    st.caption('画面の学習は実写のみを使用します。合成画像は精度評価に混ぜません。'
               '既存の分割と実験を保持し、画面にはval評価を表示します。最終testはCLIで別途実施します。')
    if len(real) < 5:
        st.info('学習・val・testの独立グループが不足する可能性があります。'
                '各モード・製品の訓練に独立3グループ以上・2重量以上と、別の評価写真が必要です。')
    alpha = st.number_input('回帰の正則化係数 alpha', min_value=0.001, value=1.0,
                            step=0.1, key='training_alpha', help='調整はvalだけで行い、testを見て選ばないでください。')
    if st.button('CPU学習を開始', key='training_start', disabled=not real):
        try:
            ident = start_training(db_path=db_path, alpha=float(alpha))
            st.success(f'学習を受け付けました（ジョブID {ident}）。進行状況を更新して確認してください。')
        except (ValueError, TypeError, OSError, sqlite3.Error, RuntimeError) as exc:
            st.error(f'学習を開始できませんでした: {exc}')
    st.button('進行状況・モデル一覧を更新', key='training_refresh')
    jobs = list_jobs(db_path)
    if not jobs:
        st.info('まだ学習ジョブはありません。')
    else:
        st.dataframe(pd.DataFrame([{k: row.get(k) for k in
                     ('id', 'status', 'progress', 'sample_count', 'created_at', 'finished_at', 'message')}
                     for row in jobs]), hide_index=True, width='stretch')
        for job in jobs[:3]:
            st.write(f"ジョブ {job['id']}: {job['status']} — {job.get('message') or ''}")
            st.progress(max(0, min(100, int(job.get('progress') or 0))))
            if job.get('error'):
                st.error(f"ジョブ {job['id']}: {job['error']}")
    _render_models(db_path)


def _render_samples(samples, products, db_path):
    st.subheader('登録データの一覧・編集・削除')
    if not samples:
        st.info('学習データはまだありません。写真と実測重量を登録してください。')
        return
    shown = pd.DataFrame([{k: row.get(k) for k in
             ('id', 'actual_weight_g', 'mode', 'product_id', 'pasta_type', 'thickness_mm',
              'source', 'captured_at', 'registered_at', 'session_id', 'batch_id', 'version')}
             for row in samples])
    st.dataframe(shown, hide_index=True, width='stretch')
    mapping = {row['id']: row for row in samples}
    chosen = st.selectbox('編集・確認する学習データID', [None] + list(mapping),
                          format_func=lambda value: '選択してください' if value is None else str(value),
                          key='training_selected_sample')
    if chosen is None:
        return
    sample = mapping[chosen]
    try:
        st.image(sample['image_path'], caption=f"登録ID {chosen} / {sample['source']}", width='stretch')
    except (OSError, ValueError) as exc:
        st.error(f'登録写真を表示できませんでした: {exc}')
    with st.expander('このデータのラベルを編集', expanded=True):
        with st.form(f'training_edit_{chosen}_{sample["version"]}'):
            changes = _fields(f'training_edit_{chosen}_{sample["version"]}', products, sample)
            update = st.form_submit_button('ラベル変更を保存')
        if update:
            try:
                update_sample(chosen, changes, db_path=db_path, expected_version=sample['version'])
                st.success('ラベルを更新しました。既存モデルの実験スナップショットは変更しません。')
            except (ValueError, TypeError, OSError, sqlite3.Error) as exc:
                st.error(f'更新できませんでした: {exc}')
    confirm = st.checkbox('この画像を今後の学習対象から削除する', key=f'training_delete_confirm_{chosen}')
    if st.button('選択した学習データを削除', key=f'training_delete_{chosen}', disabled=not confirm):
        try:
            delete_sample(chosen, db_path=db_path, expected_version=sample['version'])
            st.success('学習対象から削除しました。既存実験の再現に必要な画像は保持します。')
        except (ValueError, TypeError, OSError, sqlite3.Error) as exc:
            st.error(f'削除できませんでした: {exc}')


def _render_models(db_path):
    st.subheader('モデル管理・評価結果')
    models = list_models(db_path)
    try:
        active = get_active_model(db_path)
    except (OSError, ValueError, sqlite3.Error) as exc:
        st.error(f'選択中のモデルを読み込めません。別のモデルを選択してください: {exc}')
        active = None
    if active:
        st.caption(f"測定に使用するモデル: ID {active['id']}。モデルを有効にしても推定精度を保証しません。")
    else:
        st.info('有効な学習モデルはありません。OpenCVによる既存の測定は引き続き利用できます。')
    if not models:
        st.caption('モデルは学習成功時に保存され、次のアプリ起動時も読み込めます。')
        return
    mapping = {model['id']: model for model in models}
    st.dataframe(pd.DataFrame([{k: row.get(k) for k in
                 ('id', 'job_id', 'created_at', 'source', 'active')} for row in models]),
                 hide_index=True, width='stretch')
    chosen = st.selectbox('確認・切り替えるモデルID', [None] + list(mapping),
                          format_func=lambda value: '選択してください' if value is None else str(value),
                          key='training_selected_model')
    if chosen is None:
        return
    model = mapping[chosen]
    if st.button('このモデルを測定画面で使用', key='training_activate', disabled=model.get('source') != 'real'):
        try:
            activate_model(chosen, db_path=db_path)
            st.success(f'モデルID {chosen}を選択しました。測定画面で学習モデル方式を選べます。')
        except (ValueError, TypeError, OSError, sqlite3.Error) as exc:
            st.error(f'モデルを読み込めませんでした: {exc}')
    if model.get('source') != 'real':
        st.warning('このモデルは合成画像テスト用です。実写真の測定用には切り替えられません。')
    if model.get('report'):
        _render_report(model['report'])


def _render_report(report):
    st.caption(report.get('notice', 'val評価。testは別途実施してください。'))
    st.write('データ分割:', report.get('split_counts', {}))
    rows = []
    details = []
    for method, label in (('regression', '画像回帰'), ('opencv', 'OpenCV本数検出')):
        result = report.get(method) or {}
        overall = result.get('overall') or {}
        rows.append({'方式': label, **overall})
        for item in result.get('rows', []):
            pred = item.get('predicted_g')
            details.append({'方式': label, '撮影状態': item.get('mode'),
                            '実測g': item.get('actual_weight_g'), '推定g': pred,
                            '絶対誤差g': abs(pred - item['actual_weight_g']) if pred is not None else None,
                            '成功': item.get('success'), '理由': item.get('reason')})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
    st.caption('MAE/RMSE/MAPEは成功画像について計算します。失敗数・成功率も併せて確認してください。'
               'この画面の値はval評価であり、未使用実写testでの性能確認に代わるものではありません。')
    if details:
        frame = pd.DataFrame(details)
        st.dataframe(frame, hide_index=True, width='stretch')
        chart = frame[frame['成功'] == True].dropna(subset=['実測g', '推定g'])
        if len(chart):
            st.scatter_chart(chart, x='実測g', y='推定g', color='方式', width='stretch')
        else:
            st.info('推定成功がないため、比較グラフは表示しません。失敗理由と撮影条件を確認してください。')
    with st.expander('撮影状態・重量別の誤差'):
        for method, label in (('regression', '画像回帰'), ('opencv', 'OpenCV本数検出')):
            by = (report.get(method) or {}).get('by') or {}
            for field, title in (('mode', '撮影状態'), ('actual_weight_g', '実測重量')):
                entries = by.get(field) or {}
                if entries:
                    st.write(f'{label}: {title}別')
                    st.dataframe(pd.DataFrame([{title: key, **value} for key, value in entries.items()]),
                                 hide_index=True, width='stretch')
