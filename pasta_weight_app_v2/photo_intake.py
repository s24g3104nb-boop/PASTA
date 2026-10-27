"""重量未確認の写真を学習台帳と分けて保管し、明示確認後だけ登録する。"""
from __future__ import annotations

from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sqlite3
import stat
import re
import zipfile

import numpy as np
from PIL import Image

try:
    from .storage import DEFAULT_DB, list_products
    from .training_data import _image_rgb, _positive, register_sample
    from .photo_qc import apply_mask_edits, composite_rgb
except ImportError:
    from storage import DEFAULT_DB, list_products
    from training_data import _image_rgb, _positive, register_sample
    from photo_qc import apply_mask_edits, composite_rgb


def _now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _connection(db_path):
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=15)
    con.row_factory = sqlite3.Row
    try:
        with con:
            yield con
    finally:
        con.close()


def _init(db_path):
    with _connection(db_path) as con:
        con.executescript('''
            CREATE TABLE IF NOT EXISTS photo_intake (
                id INTEGER PRIMARY KEY, package_root TEXT NOT NULL,
                filename TEXT NOT NULL, expected_sha256 TEXT NOT NULL,
                capture_group_id TEXT NOT NULL, metadata_json TEXT NOT NULL,
                weight_g REAL, weight_verified INTEGER NOT NULL DEFAULT 0,
                group_verified INTEGER NOT NULL DEFAULT 0,
                batch_id TEXT, session_id TEXT, product_id TEXT, mode TEXT,
                confirmed_by TEXT, qc_status TEXT NOT NULL DEFAULT 'UNREVIEWED',
                leakage_checked INTEGER NOT NULL DEFAULT 0,
                mask_path TEXT, mask_sha256 TEXT, processed_path TEXT,
                reviewed_original_sha256 TEXT, review_json TEXT,
                training_sample_id INTEGER, created_at TEXT NOT NULL,
                UNIQUE(package_root,filename)
            );
            CREATE TABLE IF NOT EXISTS photo_intake_audit (
                id INTEGER PRIMARY KEY, photo_id INTEGER NOT NULL,
                action TEXT NOT NULL, at TEXT NOT NULL, details_json TEXT NOT NULL
            );
        ''')


def _audit(con, photo_id, action, details):
    con.execute('INSERT INTO photo_intake_audit(photo_id,action,at,details_json) VALUES(?,?,?,?)',
                (photo_id, action, _now(), json.dumps(details, ensure_ascii=False, allow_nan=False)))


def _safe_name(name):
    if not isinstance(name, str) or not name or Path(name).name != name or '\\' in name or ':' in name:
        raise ValueError('原写真名はディレクトリを含まない名前が必要です。')
    if name in ('.', '..'):
        raise ValueError('原写真名が不正です。')
    return name


def _manifest(root, expected_count=None):
    root = Path(root).resolve()
    path = root / 'dataset/metadata/labels_NEED_CONFIRMATION.csv'
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'filename', 'original_sha256', 'capture_group_id', 'session_id'}
        if (not required <= set(reader.fieldnames or ()) or
                len(reader.fieldnames) != len(set(reader.fieldnames))):
            raise ValueError('重量確認CSVの必須列が不足しています。')
        rows = list(reader)
    if not rows or (expected_count is not None and len(rows) != expected_count):
        raise ValueError(f'期待する写真件数と一致しません: {len(rows)}件。')
    names, digests = set(), set()
    for row in rows:
        if None in row or any(row.get(k) is None for k in required):
            raise ValueError('CSVの列数が一致しません。')
        name = _safe_name(row['filename'])
        digest = row['original_sha256']
        if name in names or digest in digests:
            raise ValueError('CSVにファイル名または原写真SHA-256の重複があります。')
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('原写真SHA-256が不正です。')
        if not row['capture_group_id'].strip() or not row['session_id'].strip():
            raise ValueError('仮の撮影グループとセッションIDが必要です。')
        names.add(name)
        digests.add(digest)
    sha_path = path.parent / 'originals_SHA256.txt'
    if sha_path.exists():
        supplied = {}
        for line in sha_path.read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():
                continue
            digest, name = line.split(None, 1)
            name = name.strip().lstrip('*')
            if name in supplied:
                raise ValueError('ハッシュ一覧に重複名があります。')
            supplied[_safe_name(name)] = digest
        if supplied != {r['filename']: r['original_sha256'] for r in rows}:
            raise ValueError('CSVと原写真ハッシュ一覧が一致しません。')
    return rows


def package_inventory(root):
    root = Path(root).resolve()
    rows = _manifest(root)
    directory = root / 'dataset/originals'
    received, missing, damaged, mismatched, duplicates = [], [], [], [], []
    observed = {}
    for row in rows:
        name = row['filename']
        path = directory / name
        if not path.is_file():
            missing.append(name)
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('原写真が保管領域外を参照しています。')
        received.append(name)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row['original_sha256']:
            mismatched.append(name)
        if digest in observed:
            duplicates.append([observed[digest], name])
        observed[digest] = name
        try:
            _image_rgb(path)
        except (ValueError, OSError) as exc:
            damaged.append(dict(filename=name, error=str(exc)))
    names = {row['filename'] for row in rows}
    unexpected = [p.name for p in directory.iterdir() if p.is_file() and p.name not in names] if directory.exists() else []
    groups = {}
    for row in rows:
        groups[row['capture_group_id']] = groups.get(row['capture_group_id'], 0) + 1
    previews, preview_missing, preview_damaged, preview_groups = [], [], [], {}
    for row in rows:
        try:
            load_preview(dict(row, package_root=str(root)))
            previews.append(row['filename'])
            group = row['capture_group_id']
            preview_groups[group] = preview_groups.get(group, 0) + 1
        except (OSError, ValueError, KeyError):
            if (root / 'dataset/previews' / row['filename']).exists():
                preview_damaged.append(row['filename'])
            else:
                preview_missing.append(row['filename'])
    return dict(expected=len(rows), received=received, missing=missing, damaged=damaged,
                hash_mismatch=mismatched, duplicate_bytes=duplicates, unexpected=unexpected, groups=groups,
                preview_received=previews, preview_missing=preview_missing,
                preview_damaged=preview_damaged, preview_group_counts=preview_groups)


def load_manifest(root, db_path=DEFAULT_DB, expected_count=42):
    root = Path(root).resolve()
    rows = _manifest(root, expected_count)
    _init(db_path)
    with _connection(db_path) as con:
        con.execute('BEGIN IMMEDIATE')
        for row in rows:
            previous = con.execute('SELECT * FROM photo_intake WHERE package_root=? AND filename=?',
                                   (str(root), row['filename'])).fetchone()
            if previous is not None:
                if previous['expected_sha256'] != row['original_sha256']:
                    raise ValueError('既存台帳の原写真ハッシュを変更しません。')
                continue
            # CSVのYESや候補重量は明示確認の代わりにしない。
            cursor = con.execute('''INSERT INTO photo_intake
                (package_root,filename,expected_sha256,capture_group_id,metadata_json,created_at)
                VALUES(?,?,?,?,?,?)''',
                (str(root), row['filename'], row['original_sha256'], row['capture_group_id'],
                 json.dumps(row, ensure_ascii=False), _now()))
            _audit(con, cursor.lastrowid, 'received_metadata_pending', row)
    return package_inventory(root)


def list_intake(db_path=DEFAULT_DB):
    _init(db_path)
    with _connection(db_path) as con:
        return [dict(row) for row in con.execute('SELECT * FROM photo_intake ORDER BY id')]


def _photo(photo_id, db_path):
    _init(db_path)
    with _connection(db_path) as con:
        row = con.execute('SELECT * FROM photo_intake WHERE id=?', (photo_id,)).fetchone()
    if row is None:
        raise ValueError('受領写真IDが存在しません。')
    return dict(row)


def _original_path(row):
    root = Path(row['package_root']).resolve()
    path = root / 'dataset/originals' / _safe_name(row['filename'])
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError('原写真の保存先が不正です。')
    return path


def load_original(row):
    path = _original_path(row)
    if not path.is_file():
        raise ValueError('元JPEGファイルをまだ受信していません。表示用一覧画像からは復元しません。')
    if hashlib.sha256(path.read_bytes()).hexdigest() != row['expected_sha256']:
        raise ValueError('元JPEGのSHA-256が受領マニフェストと一致しません。')
    return _image_rgb(path)


def _bytes(upload, limit):
    if isinstance(upload, (bytes, bytearray)):
        data = bytes(upload)
    else:
        upload.seek(0)
        data = upload.read(limit + 1)
    if not data or len(data) > limit:
        raise ValueError('ファイルが空、または上限サイズを超えています。')
    return data


def add_original(upload, root):
    data = _bytes(upload, 30_000_000)
    digest = hashlib.sha256(data).hexdigest()
    matches = [r for r in _manifest(root) if r['original_sha256'] == digest]
    if len(matches) != 1:
        raise ValueError('このファイルは原写真SHA-256一覧と一致しません。再圧縮せずJPEG原本を指定してください。')
    _image_rgb(data)
    path = Path(root).resolve() / 'dataset/originals' / matches[0]['filename']
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == data:
            return path.name
        raise ValueError('同名の原写真を上書きしません。')
    with path.open('xb') as stream:
        stream.write(data)
    return path.name


def merge_zip(upload, root):
    root = Path(root).resolve()
    data = _bytes(upload, 64_000_000)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        infos = archive.infolist()
        if len(infos) > 5000 or sum(i.file_size for i in infos) > 512_000_000:
            raise ValueError('ZIP展開後の件数・サイズが上限を超えています。')
        files = [i for i in infos if not i.is_dir()]
        metadata_names = {'00_README_FIRST.md', '01_photo_manifest_NEEDS_CONFIRMATION.csv',
                          '02_confirm_groups_FILL_IN.csv', '03_CODEX_MESSAGE_COPY.md'}
        if files and all(i.filename in metadata_names for i in files):
            added = []
            for info in files:
                target = root / 'received_preview_metadata' / info.filename
                content = archive.read(info)
                if target.exists() and target.read_bytes() != content:
                    raise ValueError('受領済みメタデータを上書きしません。')
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)
                    added.append(str(target.relative_to(root)))
            return added
        if files and all(re.fullmatch(r'[ABCD]_[A-Za-z0-9_]+/[^/]+\.jpe?g', i.filename) for i in files):
            return _merge_previews(archive, files, root)
        planned, seen = [], set()
        for info in infos:
            parts = PurePosixPath(info.filename).parts
            if (not parts or parts[0] != 'Pasta_Lens_42_Photos' or '..' in parts
                    or '\\' in info.filename or ':' in info.filename
                    or stat.S_ISLNK(info.external_attr >> 16)):
                raise ValueError('ZIPに想定外の保存先またはリンクが含まれています。')
            if info.is_dir():
                continue
            relative = Path(*parts[1:])
            destination = root / relative
            if not relative.parts or relative in seen or not destination.resolve().is_relative_to(root):
                raise ValueError('ZIPに重複名または保管領域外のパスがあります。')
            seen.add(relative)
            content = archive.read(info)
            if destination.exists() and destination.read_bytes() != content:
                raise ValueError('受領済みファイルと内容が異なります。上書きはしません。')
            planned.append((destination, content))
        added = []
        try:
            for destination, content in planned:
                if destination.exists():
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open('xb') as stream:
                    stream.write(content)
                added.append(destination)
        except Exception:
            for path in added:
                path.unlink(missing_ok=True)
            raise
    return [str(path.relative_to(root)) for path in added]


def _merge_previews(archive, files, root):
    expected = {r['filename']: r for r in _manifest(root)}
    directory = root / 'dataset/previews'
    receipt_path = directory / 'receipts.json'
    receipts = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    planned, names = [], set()
    for info in files:
        name = _safe_name(PurePosixPath(info.filename).name)
        if name not in expected or name in names or stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError('プレビューに予定外・重複名・リンクがあります。')
        if info.filename[0] != expected[name]['capture_group_id'][0]:
            raise ValueError('プレビューの撮影群がマニフェストと一致しません。')
        names.add(name)
        content = archive.read(info)
        rgb = _image_rgb(content)
        digest = hashlib.sha256(content).hexdigest()
        target = directory / name
        if target.exists() and target.read_bytes() != content:
            raise ValueError('受領済みプレビューを上書きしません。')
        receipts[name] = dict(sha256=digest, shape=list(rgb.shape[:2]), variant='preview',
                              original_sha256_match=digest == expected[name]['original_sha256'])
        planned.append((target, content))
    directory.mkdir(parents=True, exist_ok=True)
    added = []
    for target, content in planned:
        if not target.exists():
            with target.open('xb') as stream:
                stream.write(content)
            added.append(str(target.relative_to(root)))
    temporary = directory / 'receipts.json.tmp'
    temporary.write_text(json.dumps(receipts, ensure_ascii=False, indent=2))
    temporary.replace(receipt_path)
    return added


def load_preview(row):
    root = Path(row['package_root']).resolve()
    directory = root / 'dataset/previews'
    name = _safe_name(row['filename'])
    receipts = json.loads((directory / 'receipts.json').read_text())
    path = directory / name
    if (path.is_symlink() or not path.resolve().is_relative_to(root) or
            hashlib.sha256(path.read_bytes()).hexdigest() != receipts.get(name, {}).get('sha256')):
        raise ValueError('プレビューの受領ハッシュが一致しません。')
    return _image_rgb(path)


def confirm_group(root, group_id, weight_g, *, batch_id, session_id, product_id,
                  confirmed_same_batch, confirmed_by, db_path=DEFAULT_DB):
    weight = _positive(weight_g, '確認した実測重量')
    if confirmed_same_batch is not True:
        raise ValueError('同じ実測の乾麺であることを確認してください。混在する場合は先に群を分けてください。')
    if not all(isinstance(v, str) and v.strip() for v in (batch_id, session_id, product_id, confirmed_by)):
        raise ValueError('計量バッチ・撮影セッション・製品・確認者が必要です。')
    product = next((p for p in list_products(db_path) if str(p['id']) == product_id), None)
    if product is None or product['name'].startswith('デモ用'):
        raise ValueError('実際の製品をパスタデータへ登録して選択してください。')
    _init(db_path)
    with _connection(db_path) as con:
        con.execute('BEGIN IMMEDIATE')
        rows = con.execute('SELECT * FROM photo_intake WHERE package_root=? AND capture_group_id=?',
                           (str(Path(root).resolve()), group_id)).fetchall()
        if not rows:
            raise ValueError('対象の撮影グループがありません。')
        if any(row['training_sample_id'] is not None for row in rows):
            raise ValueError('学習登録済みの群は受領台帳から変更できません。学習台帳で変更履歴を管理してください。')
        for row in rows:
            con.execute('''UPDATE photo_intake SET weight_g=?,weight_verified=1,group_verified=1,
                batch_id=?,session_id=?,product_id=?,confirmed_by=? WHERE id=?''',
                (weight, batch_id.strip(), session_id.strip(), product_id, confirmed_by.strip(), row['id']))
            _audit(con, row['id'], 'group_weight_confirmed', dict(weight_g=weight, batch_id=batch_id,
                                                               session_id=session_id, confirmed_by=confirmed_by))


def reassign_photo(photo_id, new_group, db_path=DEFAULT_DB):
    if not isinstance(new_group, str) or not new_group.strip():
        raise ValueError('新しい撮影グループIDが必要です。')
    row = _photo(photo_id, db_path)
    if row['training_sample_id'] is not None:
        raise ValueError('学習登録済みの写真です。')
    with _connection(db_path) as con:
        con.execute('''UPDATE photo_intake SET capture_group_id=?,weight_g=NULL,weight_verified=0,
            group_verified=0,batch_id=NULL,session_id=NULL,product_id=NULL,confirmed_by=NULL WHERE id=?''',
            (new_group.strip(), photo_id))
        _audit(con, photo_id, 'group_reassigned_pending', dict(group=new_group))


def set_photo_mode(photo_id, mode, db_path=DEFAULT_DB):
    if mode not in ('spread', 'bundle_end', 'bundle_side'):
        raise ValueError('写真ごとに撮影状態を明示してください。')
    row = _photo(photo_id, db_path)
    if row['training_sample_id'] is not None:
        raise ValueError('学習登録済みの写真です。')
    with _connection(db_path) as con:
        con.execute('UPDATE photo_intake SET mode=? WHERE id=?', (mode, photo_id))
        _audit(con, photo_id, 'capture_mode_confirmed', dict(mode=mode))


def save_review(photo_id, mask, *, excluded_rects=(), reviewer, shape_confirmed,
                leakage_checked, notes='', db_path=DEFAULT_DB):
    if (shape_confirmed is not True or leakage_checked is not True or
            not isinstance(reviewer, str) or not reviewer.strip()):
        raise ValueError('麺の形状・漏洩領域の確認と検品者の記録が必要です。')
    row = _photo(photo_id, db_path)
    if row['training_sample_id'] is not None:
        raise ValueError('学習登録済みの検品結果は上書きしません。')
    rgb = load_original(row)
    clean = apply_mask_edits(mask, excluded_rects=excluded_rects)
    if clean.shape != rgb.shape[:2] or not np.array_equal(clean, mask) or not clean.any():
        raise ValueError('マスク寸法・二値性・秤除外領域・前景を確認してください。')
    png = io.BytesIO()
    Image.fromarray(clean).save(png, 'PNG')
    digest = hashlib.sha256(png.getvalue()).hexdigest()
    directory = Path(row['package_root']) / 'dataset/preprocessing_reviewed'
    directory.mkdir(parents=True, exist_ok=True)
    mask_path = directory / f'{photo_id}_{digest}_mask.png'
    processed_path = directory / f'{photo_id}_{digest}_foreground.png'
    if mask_path.exists() and mask_path.read_bytes() != png.getvalue():
        raise ValueError('保存済み検品マスクが変更されています。上書きしません。')
    if not mask_path.exists():
        mask_path.write_bytes(png.getvalue())
    if not processed_path.exists():
        Image.fromarray(composite_rgb(rgb, clean)).save(processed_path, 'PNG')
    details = dict(reviewer=str(reviewer).strip(), notes=notes, excluded_rects=excluded_rects,
                   shape_confirmed=True, leakage_checked=True, original_sha256=row['expected_sha256'])
    with _connection(db_path) as con:
        con.execute('''UPDATE photo_intake SET qc_status='APPROVED',leakage_checked=1,
            mask_path=?,mask_sha256=?,processed_path=?,reviewed_original_sha256=?,review_json=? WHERE id=?''',
            (str(mask_path), digest, str(processed_path), row['expected_sha256'],
             json.dumps(details, ensure_ascii=False), photo_id))
        _audit(con, photo_id, 'mask_and_leakage_reviewed', details)


def promote_photo(photo_id, db_path=DEFAULT_DB):
    row = _photo(photo_id, db_path)
    if row['training_sample_id'] is not None:
        return row['training_sample_id']
    if not row['weight_verified'] or not row['group_verified']:
        raise ValueError('重量と計量・撮影グループが未確認です。教師あり学習へ登録しません。')
    weight = _positive(row['weight_g'], '実測重量')
    if row['mode'] not in ('spread', 'bundle_end', 'bundle_side'):
        raise ValueError('撮影状態を写真ごとに確認してください。')
    if row['qc_status'] != 'APPROVED' or not row['leakage_checked']:
        raise ValueError('マスク検品・秤表示などの漏洩確認が未完了です。')
    rgb = load_original(row)
    mask_path = Path(row['mask_path'])
    expected_root = Path(row['package_root']).resolve() / 'dataset/preprocessing_reviewed'
    if not mask_path.resolve().is_relative_to(expected_root) or mask_path.is_symlink():
        raise ValueError('検品マスクの保存先が不正です。')
    content = mask_path.read_bytes()
    if (hashlib.sha256(content).hexdigest() != row['mask_sha256']
            or row['reviewed_original_sha256'] != row['expected_sha256']):
        raise ValueError('検品後の原写真またはマスクが変更されています。')
    try:
        from .photo_qc import validate_uploaded_mask
    except ImportError:
        from photo_qc import validate_uploaded_mask
    mask = validate_uploaded_mask(io.BytesIO(content), rgb.shape[:2])
    processed = composite_rgb(rgb, mask)
    output = io.BytesIO()
    Image.fromarray(processed).save(output, 'PNG')
    metadata = json.loads(row['metadata_json'])
    ident = register_sample(output.getvalue(), dict(actual_weight_g=weight, mode=row['mode'],
        source='real', product_id=row['product_id'], batch_id=row['batch_id'], session_id=row['session_id'],
        notes=f"受領台帳ID {photo_id} / 原写真SHA256 {row['expected_sha256']} / 検品マスク {row['mask_sha256']}",
        thickness_mm=metadata.get('diameter_mm') or None), db_path=db_path)
    with _connection(db_path) as con:
        con.execute('UPDATE photo_intake SET training_sample_id=? WHERE id=?', (ident, photo_id))
        _audit(con, photo_id, 'registered_for_training', dict(training_sample_id=ident))
    return ident
