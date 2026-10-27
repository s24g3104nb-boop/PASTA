"""私有写真パッケージの受領、重量確認、マスク検品を学習登録から分離する。"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    from .storage import DEFAULT_DB, init_db, list_products
    from .photo_intake import (add_original, confirm_group, list_intake, load_manifest,
                               load_original, load_preview, merge_zip, package_inventory, promote_photo,
                               reassign_photo, save_review, set_photo_mode)
    from .photo_qc import (apply_mask_edits, composite_rgb, make_overlay, propose_mask,
                           validate_uploaded_mask)
    from .preprocessing import propose_conservative_mask, audit_mask
except ImportError:
    from storage import DEFAULT_DB, init_db, list_products
    from photo_intake import (add_original, confirm_group, list_intake, load_manifest,
                              load_original, load_preview, merge_zip, package_inventory, promote_photo,
                              reassign_photo, save_review, set_photo_mode)
    from photo_qc import (apply_mask_edits, composite_rgb, make_overlay, propose_mask,
                          validate_uploaded_mask)
    from preprocessing import propose_conservative_mask, audit_mask


ERRORS = (OSError, ValueError, TypeError, sqlite3.Error)


def _refresh(message):
    st.session_state['intake_flash'] = message
    st.rerun()


def _verified(value):
    return value is True or value == 1 or isinstance(value, str) and value.upper() in ('YES', 'TRUE')


def _json(text, label):
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        raise ValueError(f'{label}はJSON形式で入力してください。') from None
    return value


def render_photo_intake(db_path=DEFAULT_DB):
    init_db(db_path)
    root = Path(db_path).resolve().parent / 'Pasta_Lens_42_Photos'
    st.subheader('写真パッケージ：受領・確認待ち・検品')
    if st.session_state.get('intake_flash'):
        st.success(st.session_state.pop('intake_flash'))
    st.caption('写真・マスク・重量確認表はローカルの同じPasta_Lens_42_Photosフォルダへ統合します。'
               'この段階では学習データへ登録せず、重量や麺の形を推測で確定しません。')
    _render_uploads(root, db_path)
    manifest = root / 'dataset' / 'metadata' / 'labels_NEED_CONFIRMATION.csv'
    if not manifest.is_file():
        st.info('重量確認用マニフェストが未受信です。00番ZIPまたはメタデータを配置してから検査してください。')
        return
    try:
        load_manifest(root, db_path=db_path)
        inventory = package_inventory(root)
        rows = [row for row in list_intake(db_path) if Path(row['package_root']).resolve() == root.resolve()]
    except ERRORS as exc:
        st.error(f'受領データを検査できません: {exc}')
        return
    st.button('取込状況を再検査', key='intake_refresh')
    _show_inventory(inventory, rows)
    if not rows:
        st.info('確認待ちの写真台帳はありません。')
        return
    st.write(f'台帳: {len(rows)}枚 ／ 重量確認済み: '
             f"{sum(_verified(row.get('weight_verified')) for row in rows)}枚 ／ "
             f"学習登録済み: {sum(row.get('training_sample_id') is not None for row in rows)}枚")
    verified = [r for r in rows if _verified(r.get('weight_verified')) and _verified(r.get('group_verified'))]
    st.caption(f"確認済み計量バッチ: {len({r['batch_id'] for r in verified})} ／ "
               f"確認済み撮影セッション: {len({r['session_id'] for r in verified})}。枚数と独立した試料数は異なります。")
    table = [{key: row.get(key) for key in
              ('filename', 'capture_group_id', 'weight_g', 'weight_verified', 'group_verified',
               'batch_id', 'session_id', 'mode', 'qc_status', 'leakage_checked', 'training_sample_id')}
             for row in rows]
    st.dataframe(pd.DataFrame(table), hide_index=True, width='stretch')
    _render_group_confirmation(rows, root, db_path)
    mapping = {row['id']: row for row in rows}
    selected = st.selectbox('検品・分類する写真', [None] + list(mapping),
                            format_func=lambda value: '写真を選択してください' if value is None
                            else mapping[value]['filename'], key='intake_selected_photo')
    if selected is not None:
        _render_photo(mapping[selected], db_path)


def _render_uploads(root, db_path):
    with st.expander('分割ZIP・元JPEGを受け取る'):
        st.caption('00〜06は独立ZIPです。同名原本や異なる内容を無断で上書きしません。'
                   '写真を公開GitHubへ送信する操作はありません。添付ツールは実行しません。')
        with st.form('intake_upload_form'):
            archives = st.file_uploader('分割ZIP（00〜06）', type=['zip'], accept_multiple_files=True,
                                         key='intake_zip_upload')
            originals = st.file_uploader('元JPEG（マニフェストのSHA-256と一致するファイル）',
                                          type=['jpg', 'jpeg'], accept_multiple_files=True,
                                          key='intake_original_upload')
            submitted = st.form_submit_button('受信ファイルを統合')
        if submitted:
            if not archives and not originals:
                st.error('ZIPまたは元JPEGを選択してください。')
            for archive in archives or []:
                try:
                    added = merge_zip(archive, root)
                    st.success(f'{archive.name}: {len(added)}ファイルを統合しました。')
                except ERRORS as exc:
                    st.error(f'{archive.name}: 統合できません: {exc}')
            for original in originals or []:
                try:
                    filename = add_original(original, root)
                    st.success(f'{original.name}: 原本 {filename} と照合しました。')
                except ERRORS as exc:
                    st.error(f'{original.name}: 原本として登録できません: {exc}')


def _size(value):
    return value if isinstance(value, int) else len(value or [])


def _show_inventory(inventory, rows):
    st.write(f"プレビュー受領: {_size(inventory.get('preview_received'))}／{_size(inventory.get('expected'))}枚")
    if inventory.get('preview_group_counts'):
        st.write('プレビュー群別受領枚数', inventory['preview_group_counts'])
    st.write(f"原本受領: {_size(inventory.get('received'))}／{_size(inventory.get('expected'))}枚")
    labels = {'missing': '不足', 'damaged': '破損', 'hash_mismatch': 'SHA-256不一致',
              'duplicate_bytes': '同一バイトの重複', 'unexpected': '台帳にないファイル'}
    for key, label in labels.items():
        items = inventory.get(key) or []
        if items:
            st.warning(f'{label}: {_size(items)}件')
            st.write(items)
    received = set(inventory.get('received') or [])
    groups = sorted({row['capture_group_id'] for row in rows})
    if groups:
        st.write('撮影群別の台帳件数・原本受領数')
        st.dataframe(pd.DataFrame([
            {'撮影群': group, '台帳件数': sum(row['capture_group_id'] == group for row in rows),
             '原本受領数': sum(row['capture_group_id'] == group and row['filename'] in received for row in rows)}
            for group in groups]), hide_index=True, width='stretch')
    st.caption('元JPEGのファイル名・SHA-256・デコード検査による状況です。'
               'チャットの縮小表示や検品一覧を原本受領と数えません。')


def _render_group_confirmation(rows, root, db_path):
    with st.expander('撮影グループと実測重量を確認する'):
        groups = sorted({row['capture_group_id'] for row in rows})
        group_id = st.selectbox('確認する撮影グループ', groups, key='intake_group')
        group_rows = [row for row in rows if row['capture_group_id'] == group_id]
        st.write(f'対象: {len(group_rows)}枚', [row['filename'] for row in group_rows])
        st.warning('グループ名に20g等が含まれても確定値ではありません。'
                   '異なる計量が混在する場合は下の写真別設定でグループを分けてから確認してください。')
        product_names = {str(product['id']): product['name'] for product in list_products(db_path)
                         if not product['name'].startswith('デモ用')}
        with st.form(f'intake_confirm_group_{group_id}'):
            weight = st.number_input('このグループの実測重量（g、確認した値のみ）', min_value=0.001,
                                      value=None, step=.1, key=f'intake_group_weight_{group_id}')
            batch_id = st.text_input('実際の秤量バッチID', key=f'intake_group_batch_{group_id}')
            session_id = st.text_input('実際の撮影セッションID', key=f'intake_group_session_{group_id}',
                                       help='別グループでも同じ撮影セッションなら同じIDにしてください。')
            product_id = st.selectbox('このグループの実製品', [None] + list(product_names),
                                       format_func=lambda key: '実製品を選択してください' if key is None
                                       else product_names[key], key=f'intake_group_product_{group_id}')
            reviewer = st.text_input('重量を確認した人・確認記録', key=f'intake_group_reviewer_{group_id}')
            same = st.checkbox('群内の全写真が同じ計量の乾麺で、実測重量と撮影セッションの対応を確認しました',
                                key=f'intake_group_same_{group_id}')
            submit = st.form_submit_button('このグループの重量と対応を確定')
        if not product_names:
            st.info('先に「パスタデータ」で実製品を登録してください。仮のデモ製品では確定できません。')
        if submit:
            try:
                if product_id is None:
                    raise ValueError('実製品を選択してください。')
                confirm_group(root, group_id, weight, batch_id=batch_id, session_id=session_id,
                              product_id=product_id, confirmed_same_batch=same,
                              confirmed_by=reviewer, db_path=db_path)
                _refresh('グループの実測重量と計量の対応を保存しました。学習はまだ開始していません。')
            except ERRORS as exc:
                st.error(f'重量を確定できません: {exc}')


def _render_photo(row, db_path):
    ident = row['id']
    st.markdown(f"**{row['filename']}** ／ {row['capture_group_id']}")
    with st.expander('この写真の撮影グループ・状態を設定'):
        with st.form(f'intake_photo_group_form_{ident}'):
            new_group = st.text_input('撮影グループID', value=row['capture_group_id'], key=f'intake_reassign_{ident}')
            change_group = st.form_submit_button('写真のグループを変更')
        st.caption('計量が混在している場合に分割します。変更後は重量・グループの確認をやり直してください。')
        if change_group:
            try:
                reassign_photo(ident, new_group, db_path=db_path)
                _refresh('撮影グループを変更しました。重量の確認をやり直してください。')
            except ERRORS as exc:
                st.error(f'グループを変更できません: {exc}')
        modes = {None: '未確認', 'spread': '広げた状態', 'bundle_end': '束の端面', 'bundle_side': '束の側面'}
        with st.form(f'intake_mode_form_{ident}'):
            mode = st.selectbox('この写真の撮影状態', list(modes),
                                 index=list(modes).index(row.get('mode')) if row.get('mode') in modes else 0,
                                 format_func=lambda value: modes[value], key=f'intake_mode_{ident}')
            set_mode = st.form_submit_button('撮影状態を保存')
        if set_mode:
            try:
                if mode is None:
                    raise ValueError('写真を確認して撮影状態を選択してください。')
                set_photo_mode(ident, mode, db_path=db_path)
                _refresh('写真の撮影状態を保存しました。')
            except ERRORS as exc:
                st.error(f'状態を保存できません: {exc}')
    preview_only = False
    try:
        rgb = load_original(row)
    except ERRORS as exc:
        try:
            rgb = load_preview(row)
            preview_only = True
            st.warning('受領済みプレビューを表示しています。原本SHA-256とは別の画像です。検品する場合はプレビュー版として明示確認し、原本とは区別して保存します。')
        except ERRORS:
            st.info(f'この写真の検証済み原本をまだ読み込めません。元JPEGが必要です: {exc}')
            return
    st.caption(f'EXIF正規化後の画素寸法: 幅{rgb.shape[1]} × 高さ{rgb.shape[0]}。'
               'ROI・マスクはこの画素座標で指定し、画像サイズは変更しません。')
    digest = hashlib.sha256(rgb.tobytes()).hexdigest()
    candidate_key = f'intake_candidate_{ident}_{digest}'
    prepared = None
    prepared_file = Path(row['package_root'])/'dataset/preview_qc_v2/manifest.json'
    if prepared_file.exists():
        prepared = json.loads(prepared_file.read_text()).get(row['filename'])
        if prepared and prepared.get('source_rgb_sha256') == digest and prepared.get('status') == 'UNREVIEWED':
            try:
                path = Path(row['package_root'])/prepared['paths']['masks']
                if not path.resolve().is_relative_to(Path(row['package_root']).resolve()):
                    raise ValueError('検品候補のパスが不正です。')
                if hashlib.sha256(path.read_bytes()).hexdigest() != prepared['mask_sha256']:
                    raise ValueError('生成後の候補マスクが変更されています。')
                if candidate_key not in st.session_state:
                    st.session_state[candidate_key] = dict(mask=validate_uploaded_mask(path,rgb.shape[:2]),
                        excluded_rects=[],warnings=prepared['warnings'],
                        processing_context=dict(transform=prepared['transform'],reference=prepared['reference'],method='prepared-v2'))
                st.caption(f"ROI変換: {prepared['transform']['reference_size']} → {prepared['transform']['target_size']} / {prepared['reference']['basis']}")
                if not prepared['reference']['confirmed']:
                    st.warning('ROIの座標基準は未確認です。変換候補の位置と麺の先端を目視確認してください。')
            except ERRORS as exc:
                st.error(f'保存した候補を読み込めません: {exc}')
        else:
            prepared = None
    _render_candidate_editor(row, rgb, candidate_key,prepared=prepared)
    candidate = st.session_state.get(candidate_key)
    if candidate is None and row.get('mask_path'):
        try:
            review = json.loads(row.get('review_json') or '{}')
            candidate = dict(mask=validate_uploaded_mask(row['mask_path'], rgb.shape[:2]),
                             excluded_rects=review.get('excluded_rects', []),
                             warnings=['保存済みマスクです。変更する場合は再検品してください。'])
        except ERRORS as exc:
            st.error(f'保存マスクを読み込めません: {exc}')
    mask = candidate['mask'] if candidate is not None else None
    first, second, third = st.columns(3)
    first.image(rgb, caption='プレビュー・検品用' if preview_only else '元画像・EXIF正規化', width='stretch')
    if mask is not None:
        second.image(composite_rgb(rgb, mask), caption='加工後・原画素を保持', width='stretch')
        third.image(mask, caption='抽出マスク・白が保持対象', width='stretch')
        with st.expander('背景処理候補・注意事項'):
            st.image(make_overlay(rgb,mask), caption='境界確認用オーバーレイ',width='stretch')
            audit = audit_mask(rgb,mask,forbidden_rects=candidate.get('excluded_rects',[]))
            st.write('除外領域の残存画素:',audit['forbidden_foreground_pixels'])
            if not audit['forbidden_regions_supplied']:
                st.caption('除外矩形は未指定です。秤数字・小物・文字の漏洩がないかを別途確認してください。')
            st.caption(f'保持画素: {np.count_nonzero(mask)} / {mask.size}。画素数は重量・本数の真値ではありません。')
            for warning in candidate.get('warnings', []):
                st.warning(warning)
        _render_review(row, candidate, db_path,preview_only=preview_only)
    else:
        second.info('まだ候補マスクがありません。')
        third.info('仮マスクを生成するか、修正済み二値PNGを読み込んでください。')
    candidate_approved = False
    if mask is not None:
        encoded = io.BytesIO()
        Image.fromarray(mask).save(encoded, 'PNG')
        candidate_approved = hashlib.sha256(encoded.getvalue()).hexdigest() == row.get('mask_sha256')
        if not candidate_approved:
            st.info('表示中の候補はまだ承認されていません。変更後のマスクを検品して保存してください。')
    review_variant = json.loads(row.get('review_json') or '{}').get('source_variant','original')
    ready = (review_variant == ('preview' if preview_only else 'original') and candidate_approved and _verified(row.get('weight_verified')) and _verified(row.get('group_verified'))
             and row.get('qc_status') == 'APPROVED' and _verified(row.get('leakage_checked'))
             and bool(row.get('mode')) and bool(row.get('product_id')))
    if row.get('training_sample_id') is not None:
        st.success(f"学習候補へ登録済み: ID {row['training_sample_id']}")
    else:
        st.caption('重量・計量対応・実製品・撮影状態・マスク・秤数字の除外を確認した写真だけ登録できます。')
        if st.button('確認済み写真を学習候補へ登録', key=f'intake_promote_{ident}', disabled=not ready):
            try:
                sample_id = promote_photo(ident, db_path=db_path)
                _refresh(f'学習候補へ登録しました（ID {sample_id}）。学習開始は別の操作です。')
            except ERRORS as exc:
                st.error(f'学習候補へ登録できません: {exc}')


def _render_candidate_editor(row, rgb, candidate_key,prepared=None):
    ident = row['id']
    full_roi = json.dumps(prepared['transform']['roi_xyxy'] if prepared else [0, 0, rgb.shape[1], rgb.shape[0]])
    review = json.loads(row.get('review_json') or '{}')
    with st.expander('仮マスクを作成・修正する', expanded=True):
        st.caption('色抽出は仮の候補です。自動の膨張・穴埋めはせず、手動ポリゴンで追加・消去します。'
                   '秤表示などの除外矩形は最後に適用します。矩形xyxyは右端・下端を含まない座標です。')
        with st.form(f'intake_candidate_form_{ident}'):
            method = st.selectbox('候補作成方式',['従来のHSV候補','支持領域で背景を抑える候補'],key=f'intake_method_{ident}')
            support = st.text_area('支持ポリゴン（任意・背景を抑える方式用、[[x,y],...]）',value='',key=f'intake_support_{ident}')
            roi = st.text_input('候補抽出ROI [x1,y1,x2,y2]', value=full_roi, key=f'intake_roi_{ident}')
            exclusions = st.text_area('秤表示などの除外矩形 [[x1,y1,x2,y2],...]',
                                      value=json.dumps(review.get('excluded_rects', [])),
                                      key=f'intake_exclusions_{ident}')
            additions = st.text_area('マスクに追加するポリゴン [[[x,y],...],...]', value='[]',
                                     key=f'intake_additions_{ident}')
            erasures = st.text_area('マスクから消すポリゴン [[[x,y],...],...]', value='[]',
                                    key=f'intake_erasures_{ident}')
            hue_min = st.number_input('最小色相', min_value=0, max_value=179, value=13, step=1,
                                      key=f'intake_hmin_{ident}')
            hue_max = st.number_input('最大色相', min_value=0, max_value=179, value=36, step=1,
                                      key=f'intake_hmax_{ident}')
            saturation = st.number_input('最小彩度', min_value=0, max_value=255, value=60, step=1,
                                         key=f'intake_sat_{ident}')
            brightness = st.number_input('最小明度', min_value=0, max_value=255, value=55, step=1,
                                         key=f'intake_val_{ident}')
            uploaded = st.file_uploader('手修正済み二値PNG（任意・元画像と同寸法）', type=['png'],
                                         key=f'intake_mask_upload_{ident}')
            generate = st.form_submit_button('仮マスクを生成・修正内容を反映')
        if generate:
            try:
                rectangles = _json(exclusions, '除外矩形')
                if uploaded is not None:
                    mask = validate_uploaded_mask(uploaded, rgb.shape[:2])
                    notes = ['アップロードしたマスクです。ROIで切り抜かず、手動修正と除外矩形のみ適用します。']
                elif method.startswith('支持'):
                    proposed = propose_conservative_mask(rgb,_json(roi,'ROI'),excluded_rects=rectangles,
                        support_polygon=_json(support,'支持ポリゴン') if support.strip() else None,
                        min_saturation=int(saturation),min_value=int(brightness))
                    mask,notes=proposed['mask'],proposed['warnings']
                else:
                    proposed = propose_mask(rgb, roi=_json(roi, 'ROI'), excluded_rects=rectangles,
                                            hue_min=int(hue_min), hue_max=int(hue_max),
                                            saturation_min=int(saturation), value_min=int(brightness))
                    mask, notes = proposed['mask'], proposed['warnings']
                mask = apply_mask_edits(mask, _json(additions, '追加ポリゴン'),
                                       _json(erasures, '消去ポリゴン'), excluded_rects=rectangles)
                st.session_state[candidate_key] = dict(mask=mask, excluded_rects=rectangles, warnings=notes,
                    processing_context=dict(roi_xyxy=_json(roi,'ROI'),method='uploaded-mask' if uploaded else method))
                st.session_state.pop(f'intake_shape_confirmed_{ident}', None)
                st.session_state.pop(f'intake_leakage_confirmed_{ident}', None)
                st.success('未検品の候補を作成しました。下の元画像・マスク・オーバーレイで確認してください。')
            except ERRORS as exc:
                st.error(f'候補を作成できません: {exc}')


def _render_review(row, candidate, db_path,preview_only=False):
    ident = row['id']
    with st.form(f'intake_review_form_{ident}'):
        variant_confirmed = not preview_only
        if preview_only:
            variant_confirmed = st.checkbox('原本ではなく、受領ハッシュで識別したプレビュー版の検品として保存します',key=f'intake_preview_confirm_{ident}')
        reviewer = st.text_input('検品した人・確認記録', key=f'intake_review_by_{ident}')
        notes = st.text_area('欠落・混入の確認と修正内容', key=f'intake_review_notes_{ident}')
        shape = st.checkbox('加工前と比較し、麺の先端・形状・本数の見え方を損なわず、背景混入もないと確認しました',
                             key=f'intake_shape_confirmed_{ident}')
        leakage = st.checkbox('秤の数字など重量が分かる領域がマスク内に残っていないことを確認しました',
                               key=f'intake_leakage_confirmed_{ident}')
        approve = st.form_submit_button('プレビュー版マスクを検品済みとして保存' if preview_only else 'このマスクを検品済みとして保存')
    if approve:
        try:
            if not variant_confirmed:
                raise ValueError('プレビュー版としての検品であることを確認してください。')
            save_review(ident, candidate['mask'], excluded_rects=candidate.get('excluded_rects', []),
                        reviewer=reviewer, shape_confirmed=shape, leakage_checked=leakage,
                        notes=notes, db_path=db_path,source_variant='preview' if preview_only else 'original',
                        processing_context=candidate.get('processing_context'))
            _refresh('検品内容を保存しました。重量等の確認後、別ボタンで学習候補へ登録できます。')
        except ERRORS as exc:
            st.error(f'検品済みとして保存できません: {exc}')
