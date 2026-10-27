"""原本や未確認重量を学習へ流さず、受領と確認待ちを表示する。"""
import csv
import hashlib
import io
from pathlib import Path
import sqlite3

import numpy as np
from PIL import Image

from streamlit.testing.v1 import AppTest


def renderer(db):
    app_dir = str(Path(__file__).resolve().parents[1])
    return f'''
import sys
sys.path.insert(0, {app_dir!r})
from photo_intake_ui import render_photo_intake
render_photo_intake({str(db)!r})
'''


def create_pending_manifest(root, first_image=None):
    metadata = root / 'dataset' / 'metadata'
    metadata.mkdir(parents=True)
    rows = []
    for group, first, last in [('A_20g_candidate', 7231, 7240), ('B_weight_unknown', 7241, 7253),
                               ('C_weight_unknown', 7261, 7270), ('D_weight_unknown', 7281, 7289)]:
        for index in range(first, last + 1):
            filename = f'IMG_{index}.jpeg'
            rows.append(dict(filename=filename, weight_g='', weight_verified='NO',
                             session_id=group, capture_group_id=group,
                             original_sha256=hashlib.sha256(filename.encode()).hexdigest(),
                             mask_quality='UNREVIEWED', train_ready='NO', label_leakage_check='REQUIRED'))
    path = metadata / 'labels_NEED_CONFIRMATION.csv'
    if first_image is not None:
        rows[0]['original_sha256'] = hashlib.sha256(first_image).hexdigest()
        originals = root / 'dataset' / 'originals'
        originals.mkdir(parents=True)
        (originals / rows[0]['filename']).write_bytes(first_image)
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (metadata / 'originals_SHA256.txt').write_text(
        ''.join(row['original_sha256'] + '  ' + row['filename'] + '\n' for row in rows))
    return path


def test_no_manifest_is_reported_and_no_training_buttons(tmp_path):
    at = AppTest.from_string(renderer(tmp_path / 'missing.sqlite3')).run(timeout=15)
    assert not at.exception
    assert any('マニフェストが未受信' in item.value for item in at.info)
    assert not any('学習候補へ登録' in item.label for item in at.button)
    next(button for button in at.button if button.label == '受信ファイルを統合').click().run(timeout=15)
    assert not at.exception
    assert any('選択してください' in item.value for item in at.error)


def test_preview_can_be_inspected_but_not_approved_or_trained(tmp_path):
    import io
    import zipfile
    from PIL import Image
    from photo_intake import merge_zip
    db = tmp_path / 'preview.sqlite3'
    root = tmp_path / 'Pasta_Lens_42_Photos'
    create_pending_manifest(root)
    image = io.BytesIO()
    Image.new('RGB', (80,80), (190,150,50)).save(image, 'JPEG')
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('A_CANDIDATE/IMG_7231.jpeg', image.getvalue())
    merge_zip(archive.getvalue(), root)
    at = AppTest.from_string(renderer(db)).run(timeout=15)
    at.selectbox(key='intake_selected_photo').set_value(1).run(timeout=15)
    assert not at.exception
    assert any('受領済みプレビュー' in row.value for row in at.warning)
    next(b for b in at.button if b.label == '仮マスクを生成・修正内容を反映').click().run(timeout=15)
    assert not at.exception
    assert not any(b.label == 'このマスクを検品済みとして保存' for b in at.button)
    assert at.button(key='intake_promote_1').disabled


def test_preview_review_requires_explicit_variant_confirmation_and_no_training(tmp_path):
    import zipfile
    from photo_intake import merge_zip,list_intake
    db=tmp_path/'preview_review.sqlite3';root=tmp_path/'Pasta_Lens_42_Photos'
    create_pending_manifest(root)
    image=io.BytesIO();Image.new('RGB',(80,80),(190,150,50)).save(image,'JPEG')
    archive=io.BytesIO()
    with zipfile.ZipFile(archive,'w') as z: z.writestr('A_CANDIDATE/IMG_7231.jpeg',image.getvalue())
    merge_zip(archive.getvalue(),root)
    at=AppTest.from_string(renderer(db)).run(timeout=15)
    at.selectbox(key='intake_selected_photo').set_value(1).run(timeout=15)
    next(b for b in at.button if b.label=='仮マスクを生成・修正内容を反映').click().run(timeout=15)
    at.text_input(key='intake_review_by_1').set_value('fixture reviewer')
    at.checkbox(key='intake_shape_confirmed_1').check()
    at.checkbox(key='intake_leakage_confirmed_1').check()
    next(b for b in at.button if b.label=='プレビュー版マスクを検品済みとして保存').click().run(timeout=15)
    assert any('プレビュー版として' in e.value for e in at.error)
    assert list_intake(db)[0]['qc_status']=='UNREVIEWED'
    at.checkbox(key='intake_preview_confirm_1').check()
    next(b for b in at.button if b.label=='プレビュー版マスクを検品済みとして保存').click().run(timeout=15)
    assert not at.exception
    assert list_intake(db)[0]['qc_status']=='APPROVED'
    assert at.button(key='intake_promote_1').disabled
    assert not list_intake(db)[0]['weight_verified']


def test_42_metadata_only_rows_stay_pending_with_blank_weight(tmp_path):
    db = tmp_path / 'pending.sqlite3'
    create_pending_manifest(tmp_path / 'Pasta_Lens_42_Photos')
    at = AppTest.from_string(renderer(db)).run(timeout=15)
    assert not at.exception
    assert any('不足: 42件' in item.value for item in at.warning)
    assert any('原本受領: 0／42枚' in item.value for item in at.markdown)
    assert at.number_input(key='intake_group_weight_A_20g_candidate').value is None
    from photo_intake import list_intake
    rows = list_intake(db)
    assert len(rows) == 42
    assert all(not row['weight_verified'] and row['weight_g'] is None for row in rows)
    first = rows[0]['id']
    at.selectbox(key='intake_selected_photo').set_value(first).run(timeout=15)
    assert not at.exception
    assert any('検証済み原本をまだ読み込めません' in item.value for item in at.info)
    assert not any('確認済み写真を学習候補へ登録' == item.label for item in at.button)
    with sqlite3.connect(db) as con:
        exists = con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='training_samples'").fetchone()
        assert not exists or con.execute('SELECT COUNT(*) FROM training_samples').fetchone()[0] == 0


def test_review_then_explicit_weight_confirmation_then_promotion(tmp_path):
    # 合成画素で承認経路だけを試す。実物マスク精度・実測重量の試験ではない。
    rgb = np.zeros((40, 60, 3), dtype=np.uint8)
    rgb[5:35, 22:25] = (230, 190, 65)
    image = io.BytesIO()
    Image.fromarray(rgb).save(image, 'JPEG', quality=95)
    db = tmp_path / 'review.sqlite3'
    create_pending_manifest(tmp_path / 'Pasta_Lens_42_Photos', first_image=image.getvalue())
    from storage import init_db, save_product
    init_db(db)
    save_product('テスト用実製品1.6mm', .9, 350, 12, 2, 70, db_path=db)
    at = AppTest.from_string(renderer(db)).run(timeout=15)
    at.selectbox(key='intake_selected_photo').set_value(1).run(timeout=15)
    assert not at.exception
    at.selectbox(key='intake_mode_1').set_value('spread')
    next(item for item in at.button if item.label == '撮影状態を保存').click().run(timeout=15)
    next(item for item in at.button if item.label == '仮マスクを生成・修正内容を反映').click().run(timeout=15)
    assert not at.exception
    assert at.button(key='intake_promote_1').disabled
    next(item for item in at.button if item.label == 'このマスクを検品済みとして保存').click().run(timeout=15)
    assert any('確認' in item.value for item in at.error)
    at.text_input(key='intake_review_by_1').set_value('fixture reviewer')
    at.checkbox(key='intake_shape_confirmed_1').check()
    at.checkbox(key='intake_leakage_confirmed_1').check()
    next(item for item in at.button if item.label == 'このマスクを検品済みとして保存').click().run(timeout=15)
    assert not at.exception
    assert at.button(key='intake_promote_1').disabled
    group = 'A_20g_candidate'
    at.number_input(key=f'intake_group_weight_{group}').set_value(20)
    at.text_input(key=f'intake_group_batch_{group}').set_value('fixture-batch')
    at.text_input(key=f'intake_group_session_{group}').set_value('fixture-session')
    at.text_input(key=f'intake_group_reviewer_{group}').set_value('fixture label confirmation')
    at.selectbox(key=f'intake_group_product_{group}').set_value('2')
    at.checkbox(key=f'intake_group_same_{group}').check()
    next(item for item in at.button if item.label == 'このグループの重量と対応を確定').click().run(timeout=15)
    assert not at.exception
    assert not at.button(key='intake_promote_1').disabled
    at.button(key='intake_promote_1').click().run(timeout=15)
    assert not at.exception
    with sqlite3.connect(db) as con:
        result = con.execute('SELECT actual_weight_g,batch_id,session_id,mode FROM training_samples').fetchall()
    assert result == [(20, 'fixture-batch', 'fixture-session', 'spread')]
