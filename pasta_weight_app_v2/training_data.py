"""学習写真の登録台帳。実験スナップショットと元画像は変更しない。"""
from __future__ import annotations

from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import shutil
import sqlite3
import warnings

import numpy as np
from PIL import Image, ImageOps

try:
    from .storage import DEFAULT_DB
    from .pasta_ml.dataset import MAX_BYTES, MAX_PIXELS, hashes, normalized_rgb
except ImportError:
    from storage import DEFAULT_DB
    from pasta_ml.dataset import MAX_BYTES, MAX_PIXELS, hashes, normalized_rgb


TEXT_FIELDS = ('product_id', 'batch_id', 'session_id', 'pasta_type', 'captured_at',
               'lighting', 'background', 'occlusion_level', 'device_label', 'notes')
META_FIELDS = ('actual_weight_g', 'mode', 'source', 'thickness_mm',
               'individual_strand_count', 'parent_sample_id', *TEXT_FIELDS)
REQUIRED_FIELDS = ('actual_weight_g', 'mode', 'source', 'product_id', 'batch_id', 'session_id')


@contextmanager
def _connection(db_path):
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), timeout=15)
    con.row_factory = sqlite3.Row
    con.execute('PRAGMA foreign_keys=ON')
    try:
        with con:
            yield con
    finally:
        con.close()


def _now():
    return datetime.now(timezone.utc).isoformat()


def init_training_db(db_path=DEFAULT_DB):
    with _connection(db_path) as con:
        con.executescript('''
            CREATE TABLE IF NOT EXISTS training_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pixel_hash TEXT UNIQUE NOT NULL,
                perceptual_hash TEXT NOT NULL,
                image_filename TEXT NOT NULL,
                actual_weight_g REAL NOT NULL CHECK(actual_weight_g > 0),
                mode TEXT NOT NULL CHECK(mode IN ('spread', 'bundle_end')),
                source TEXT NOT NULL CHECK(source IN ('real', 'synthetic')),
                product_id TEXT NOT NULL,
                batch_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                pasta_type TEXT NOT NULL DEFAULT '',
                thickness_mm REAL,
                captured_at TEXT NOT NULL DEFAULT '',
                lighting TEXT NOT NULL DEFAULT '',
                background TEXT NOT NULL DEFAULT '',
                occlusion_level TEXT NOT NULL DEFAULT '',
                device_label TEXT NOT NULL DEFAULT '',
                notes TEXT NOT NULL DEFAULT '',
                individual_strand_count INTEGER,
                parent_sample_id INTEGER REFERENCES training_samples(id),
                version INTEGER NOT NULL DEFAULT 1,
                registered_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                deleted_at TEXT
            );
            CREATE TABLE IF NOT EXISTS training_sample_revisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_id INTEGER NOT NULL REFERENCES training_samples(id),
                version INTEGER NOT NULL,
                changed_at TEXT NOT NULL,
                action TEXT NOT NULL,
                snapshot_json TEXT NOT NULL,
                UNIQUE(sample_id, version)
            );
        ''')


def _positive(value, label):
    if isinstance(value, bool):
        raise ValueError(f'{label}は有限の正数にしてください。')
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'{label}は有限の正数にしてください。') from None
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f'{label}は有限の正数にしてください。')
    return number


def _optional_id(value, label):
    if value is None or value == '':
        return None
    number = _positive(value, label)
    if not number.is_integer():
        raise ValueError(f'{label}は正の整数にしてください。')
    return int(number)


def _validate_metadata(metadata):
    if not isinstance(metadata, dict):
        raise ValueError('学習ラベルは辞書形式で指定してください。')
    unknown = set(metadata) - set(META_FIELDS)
    if unknown:
        raise ValueError('未対応の登録項目: ' + ', '.join(sorted(unknown)))
    if any(key not in metadata for key in REQUIRED_FIELDS):
        raise ValueError('画像の実測重量・状態・製品・秤量・撮影セッション・由来が必要です。')
    data = {key: metadata.get(key, '') for key in TEXT_FIELDS}
    for key, value in data.items():
        if not isinstance(value, str):
            raise ValueError(f'{key}は文字列にしてください。')
        data[key] = value.strip()
        if len(data[key]) > 2000:
            raise ValueError(f'{key}が長すぎます（2000文字以内）。')
    if any(not data[key] for key in ('product_id', 'batch_id', 'session_id')):
        raise ValueError('製品・秤量バッチ・撮影セッションIDは必須です。')
    if data['captured_at']:
        try:
            datetime.fromisoformat(data['captured_at'].replace('Z', '+00:00'))
        except ValueError:
            raise ValueError('撮影日時はISO形式（例2026-10-28T10:30:00）にしてください。') from None
    mode = metadata['mode']
    if mode == 'bundle':
        mode = 'bundle_end'
    if mode not in ('spread', 'bundle_end'):
        raise ValueError('撮影状態はspreadまたはbundle_endにしてください。')
    source = metadata['source']
    if source not in ('real', 'synthetic'):
        raise ValueError('写真の由来はrealまたはsyntheticにしてください。')
    thickness = metadata.get('thickness_mm')
    data.update(actual_weight_g=_positive(metadata['actual_weight_g'], '実測重量'),
                mode=mode, source=source,
                thickness_mm=None if thickness in (None, '') else _positive(thickness, '太さ(mm)'),
                individual_strand_count=_optional_id(metadata.get('individual_strand_count'), '本数'),
                parent_sample_id=_optional_id(metadata.get('parent_sample_id'), '元画像ID'))
    return data


def _image_rgb(image):
    if isinstance(image, (str, Path)):
        path = Path(image)
        if path.stat().st_size > MAX_BYTES:
            raise ValueError('画像は30MB以下にしてください。')
        raw = path.read_bytes()
    elif isinstance(image, (bytes, bytearray, memoryview)):
        raw = bytes(image)
    elif hasattr(image, 'read'):
        previous = image.tell() if hasattr(image, 'tell') else None
        if hasattr(image, 'seek'):
            image.seek(0)
        try:
            raw = image.read(MAX_BYTES + 1)
        finally:
            if previous is not None and hasattr(image, 'seek'):
                image.seek(previous)
    else:
        raise ValueError('画像ファイルまたは画像バイト列が必要です。')
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError('画像は空でない30MB以下のファイルにしてください。')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as im:
                if im.format not in ('JPEG', 'PNG'):
                    raise ValueError('JPEG/PNG画像を登録してください。')
                if im.width * im.height > MAX_PIXELS or min(im.size) < 16:
                    raise ValueError('画像は各辺16画素以上・3000万画素以下にしてください。')
                im.load()
                return normalized_rgb(im)
    except (OSError, Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise ValueError('破損または大きすぎる画像は登録できません。') from exc


def _photo_dir(db_path):
    return Path(db_path).resolve().parent / 'training' / 'images'


def _row_to_dict(row, db_path):
    result = dict(row)
    directory = _photo_dir(db_path).resolve()
    filename = result['image_filename']
    path = (directory / filename).resolve()
    if Path(filename).name != filename or not path.is_relative_to(directory):
        raise ValueError('保存された学習画像パスが不正です。')
    result['image_path'] = str(path)
    return result


def _journal(con, sample_id, action):
    row = dict(con.execute('SELECT * FROM training_samples WHERE id=?', (sample_id,)).fetchone())
    con.execute('''INSERT INTO training_sample_revisions
        (sample_id, version, changed_at, action, snapshot_json) VALUES (?, ?, ?, ?, ?)''',
        (sample_id, row['version'], row['updated_at'], action,
         json.dumps(row, ensure_ascii=False, allow_nan=False)))


def _check_parent(con, data, sample_id=None):
    parent_id = data['parent_sample_id']
    if parent_id is None:
        return
    if parent_id == sample_id:
        raise ValueError('自分自身を元画像には指定できません。')
    parent = con.execute('SELECT * FROM training_samples WHERE id=?', (parent_id,)).fetchone()
    if parent is None or parent['deleted_at']:
        raise ValueError('有効な元画像を指定してください。')
    for key in ('batch_id', 'session_id', 'product_id', 'mode', 'source', 'actual_weight_g'):
        if data[key] != parent[key]:
            raise ValueError('拡張画像は元画像と同じ重量・製品・状態・由来・秤量・撮影セッションにしてください。')
    # 元画像変更によって循環関係を作らない。
    seen = {sample_id}
    while parent is not None:
        if parent['id'] in seen:
            raise ValueError('元画像の参照が循環しています。')
        seen.add(parent['id'])
        parent = con.execute('SELECT * FROM training_samples WHERE id=?',
                             (parent['parent_sample_id'],)).fetchone()


def register_sample(image, metadata, *, db_path=DEFAULT_DB):
    data = _validate_metadata(metadata)
    rgb = _image_rgb(image)
    digest, dhash = hashes(rgb)
    init_training_db(db_path)
    directory = _photo_dir(db_path)
    directory.mkdir(parents=True, exist_ok=True)
    filename = digest + '.png'
    destination = directory / filename
    created_photo = False
    try:
        with _connection(db_path) as con:
            con.execute('BEGIN IMMEDIATE')
            if con.execute('SELECT id FROM training_samples WHERE pixel_hash=?', (digest,)).fetchone():
                raise ValueError('同じ画像が既に登録されています（削除済みを含む）。')
            _check_parent(con, data)
            # 排他的な生成。登録済みの画像ファイルを上書きしない。
            with destination.open('xb') as photo:
                created_photo = True
                Image.fromarray(rgb).save(photo, format='PNG')
            timestamp = _now()
            columns = ['pixel_hash', 'perceptual_hash', 'image_filename', *data, 'registered_at', 'updated_at']
            values = [digest, str(dhash), filename, *data.values(), timestamp, timestamp]
            query = f"INSERT INTO training_samples ({','.join(columns)}) VALUES ({','.join('?' for _ in values)})"
            cursor = con.execute(query, values)
            ident = cursor.lastrowid
            _journal(con, ident, 'register')
            return ident
    except Exception:
        if created_photo:
            destination.unlink(missing_ok=True)
        raise


def list_samples(db_path=DEFAULT_DB, *, include_deleted=False):
    init_training_db(db_path)
    with _connection(db_path) as con:
        query = 'SELECT * FROM training_samples'
        if not include_deleted:
            query += ' WHERE deleted_at IS NULL'
        return [_row_to_dict(row, db_path) for row in con.execute(query + ' ORDER BY id')]


def update_sample(sample_id, changes, *, db_path=DEFAULT_DB, expected_version=None):
    if isinstance(sample_id, bool) or not isinstance(sample_id, int) or sample_id < 1:
        raise ValueError('有効な画像IDが必要です。')
    if not isinstance(changes, dict) or not changes:
        raise ValueError('変更する項目を指定してください。')
    init_training_db(db_path)
    with _connection(db_path) as con:
        con.execute('BEGIN IMMEDIATE')
        row = con.execute('SELECT * FROM training_samples WHERE id=? AND deleted_at IS NULL',
                          (sample_id,)).fetchone()
        if row is None:
            raise ValueError('有効な登録画像がありません。')
        if expected_version is not None and row['version'] != expected_version:
            raise ValueError('他の操作で更新されています。一覧を更新してください。')
        if set(changes) - set(META_FIELDS):
            raise ValueError('変更できない項目が含まれています。')
        merged = {key: row[key] for key in META_FIELDS}
        merged.update(changes)
        data = _validate_metadata(merged)
        _check_parent(con, data, sample_id)
        children = con.execute('SELECT id FROM training_samples WHERE parent_sample_id=? AND deleted_at IS NULL',
                               (sample_id,)).fetchall()
        if children and any(row[key] != data[key] for key in
                            ('batch_id', 'session_id', 'product_id', 'mode', 'source', 'actual_weight_g')):
            raise ValueError('拡張画像があるため、元画像のグループ・ラベル変更はできません。先に拡張画像を編集・削除してください。')
        con.execute(f"UPDATE training_samples SET {','.join(key+'=?' for key in data)},"
                    'version=version+1,updated_at=? WHERE id=?', (*data.values(), _now(), sample_id))
        _journal(con, sample_id, 'update')
    return next(item for item in list_samples(db_path) if item['id'] == sample_id)


def delete_sample(sample_id, *, db_path=DEFAULT_DB, expected_version=None):
    if isinstance(sample_id, bool) or not isinstance(sample_id, int) or sample_id < 1:
        raise ValueError('有効な画像IDが必要です。')
    init_training_db(db_path)
    with _connection(db_path) as con:
        con.execute('BEGIN IMMEDIATE')
        row = con.execute('SELECT * FROM training_samples WHERE id=? AND deleted_at IS NULL',
                          (sample_id,)).fetchone()
        if row is None:
            raise ValueError('有効な登録画像がありません。')
        if expected_version is not None and row['version'] != expected_version:
            raise ValueError('他の操作で更新されています。一覧を更新してください。')
        if con.execute('SELECT id FROM training_samples WHERE parent_sample_id=? AND deleted_at IS NULL',
                       (sample_id,)).fetchone():
            raise ValueError('元画像より先に拡張画像を削除してください。')
        timestamp = _now()
        con.execute('UPDATE training_samples SET deleted_at=?,updated_at=?,version=version+1 WHERE id=?',
                    (timestamp, timestamp, sample_id))
        _journal(con, sample_id, 'delete')


def export_dataset(destination, *, db_path=DEFAULT_DB, sample_ids=None):
    """現在の有効ラベルと写真を新規ディレクトリへ固定保存する。"""
    samples = list_samples(db_path)
    if sample_ids is not None:
        wanted = set(sample_ids)
        samples = [sample for sample in samples if sample['id'] in wanted]
        if {sample['id'] for sample in samples} != wanted:
            raise ValueError('選択した学習画像が存在しないか削除されています。')
    if not samples:
        raise ValueError('実測ラベル付きの学習画像が登録されていません。')
    destination = Path(destination)
    if destination.exists():
        raise ValueError('スナップショット先は新しいディレクトリを指定してください。')
    destination.mkdir(parents=True)
    try:
        images = destination / 'images'
        images.mkdir()
        columns = ['image_path', 'training_sample_id', 'sample_version', *META_FIELDS, 'registered_at']
        with (destination / 'dataset.csv').open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for sample in samples:
                source = Path(sample['image_path'])
                # 登録後の改変や破損を検知し、誤った写真で実験を開始しない。
                rgb = _image_rgb(source)
                if hashes(rgb)[0] != sample['pixel_hash']:
                    raise ValueError('登録画像が変更されています。新しい写真として登録し直してください。')
                filename = sample['pixel_hash'] + '.png'
                Image.fromarray(rgb).save(images / filename, format='PNG')
                writer.writerow(dict(image_path='images/' + filename,
                                     training_sample_id=sample['id'], sample_version=sample['version'],
                                     **{key: sample[key] for key in META_FIELDS},
                                     registered_at=sample['registered_at']))
        return destination / 'dataset.csv'
    except Exception:
        shutil.rmtree(destination)
        raise
