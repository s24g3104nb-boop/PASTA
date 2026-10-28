"""SQLiteによる食品データと測定履歴の保存。"""
from __future__ import annotations

import sqlite3
import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from PIL import Image
import numpy as np
try:
    from .pasta_vision import finite_nonnegative
except ImportError:
    from pasta_vision import finite_nonnegative

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get('PASTA_DATA_DIR', str(BASE_DIR / 'data'))).resolve()
DEFAULT_DB = DATA_DIR / 'measurements.sqlite3'


def _connect(db_path=DEFAULT_DB):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    return connection


def init_db(db_path=DEFAULT_DB):
    with _connect(db_path) as con:
        con.executescript('''
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                g_per_strand REAL NOT NULL DEFAULT 0,
                kcal_100 REAL NOT NULL DEFAULT 0,
                protein_100 REAL NOT NULL DEFAULT 0,
                fat_100 REAL NOT NULL DEFAULT 0,
                carb_100 REAL NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                product_name TEXT NOT NULL,
                mode TEXT NOT NULL,
                detected_count INTEGER NOT NULL,
                estimated_g REAL NOT NULL,
                actual_g REAL,
                kcal REAL NOT NULL,
                protein_g REAL NOT NULL,
                fat_g REAL NOT NULL,
                carb_g REAL NOT NULL,
                image_filename TEXT
            );
        ''')
        existing = {r['name'] for r in con.execute('PRAGMA table_info(measurements)')}
        additions = {
            'source_kind': "TEXT NOT NULL DEFAULT 'unknown'",
            'estimator': "TEXT NOT NULL DEFAULT 'opencv'",
            **{key: 'REAL' for key in ('g_per_strand', 'kcal_100', 'protein_100', 'fat_100', 'carb_100')},
        }
        for name, declaration in additions.items():
            if name not in existing:
                con.execute(f'ALTER TABLE measurements ADD COLUMN {name} {declaration}')
        con.execute('''
            INSERT OR IGNORE INTO products
                (name, g_per_strand, kcal_100, protein_100, fat_100, carb_100)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', ('デモ用（仮の栄養・重量データ）', 1.0, 350.0, 12.0, 2.0, 70.0))


def list_products(db_path=DEFAULT_DB):
    with _connect(db_path) as con:
        return [dict(r) for r in con.execute('SELECT * FROM products ORDER BY id')]


def save_product(name, g_per_strand, kcal_100, protein_100, fat_100, carb_100,
                 *, product_id=None, db_path=DEFAULT_DB):
    if not name.strip():
        raise ValueError('商品名を入力してください。')
    for v in (g_per_strand, kcal_100, protein_100, fat_100, carb_100):
        finite_nonnegative(v, '重量・栄養情報')
    with _connect(db_path) as con:
        if product_id is None:
            con.execute('''INSERT INTO products
                (name, g_per_strand, kcal_100, protein_100, fat_100, carb_100)
                VALUES (?, ?, ?, ?, ?, ?)''',
                (name.strip(), g_per_strand, kcal_100, protein_100, fat_100, carb_100))
        else:
            con.execute('''UPDATE products SET name=?, g_per_strand=?, kcal_100=?,
                protein_100=?, fat_100=?, carb_100=? WHERE id=?''',
                (name.strip(), g_per_strand, kcal_100, protein_100, fat_100, carb_100,
                 int(product_id)))


def save_measurement(*, product_name, mode, detected_count, estimated_g, actual_g,
                     nutrition, image_rgb, db_path=DEFAULT_DB, source_kind='unknown',
                     estimator='opencv', product_snapshot=None, diagnostics=None):
    # 履歴はローカル保存。Web公開環境での複数人利用には認証が必要。
    if source_kind not in {'unknown', 'real', 'synthetic'}:
        raise ValueError('画像の由来が不正です。')
    if not product_name.strip() or not mode.strip():
        raise ValueError('製品名と撮影状態が必要です。')
    if finite_nonnegative(detected_count, '本数') != int(detected_count):
        raise ValueError('本数は整数にしてください。')
    finite_nonnegative(estimated_g, '推定重量')
    if actual_g is not None:
        if finite_nonnegative(actual_g, '実測重量') <= 0:
            raise ValueError('実測重量は正数にしてください。')
    for k in ('kcal', 'protein_g', 'fat_g', 'carb_g'):
        finite_nonnegative(nutrition[k], k)
    snapshot_keys = ('g_per_strand', 'kcal_100', 'protein_100', 'fat_100', 'carb_100')
    snapshot = [finite_nonnegative(product_snapshot[k], k) if product_snapshot else None
                for k in snapshot_keys]
    rgb = np.asarray(image_rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size:
        raise ValueError('保存画像はRGB uint8形式にしてください。')
    db_path = Path(db_path)
    photo_dir = db_path.parent / 'photos'
    photo_dir.mkdir(parents=True, exist_ok=True)
    filename = f'{uuid4().hex}.jpg'
    exact_filename=filename.removesuffix('.jpg')+'.png'
    if diagnostics is not None:
        import json
        diagnostics=dict(diagnostics,inference_image_filename=exact_filename)
        serialized=json.dumps(diagnostics,ensure_ascii=False,allow_nan=False)
    try:
        if diagnostics is not None:Image.fromarray(rgb).save(photo_dir/exact_filename,'PNG')
        Image.fromarray(rgb).save(photo_dir / filename, 'JPEG', quality=82)
        with _connect(db_path) as con:
            cursor = con.execute('''
                INSERT INTO measurements (created_at, product_name, mode, detected_count,
                estimated_g, actual_g, kcal, protein_g, fat_g, carb_g, image_filename,
                source_kind, estimator, g_per_strand, kcal_100, protein_100, fat_100, carb_100)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), product_name, mode,
                  int(detected_count), float(estimated_g),
                  (float(actual_g) if actual_g is not None else None),
                  nutrition['kcal'], nutrition['protein_g'], nutrition['fat_g'],
                  nutrition['carb_g'], filename, source_kind, estimator, *snapshot))
            ident=cursor.lastrowid
            if diagnostics is not None:
                con.execute('CREATE TABLE IF NOT EXISTS measurement_diagnostics (measurement_id INTEGER PRIMARY KEY REFERENCES measurements(id),details_json TEXT NOT NULL)')
                con.execute('INSERT INTO measurement_diagnostics VALUES (?,?)',(ident,serialized))
            return ident
    except Exception:
        (photo_dir / filename).unlink(missing_ok=True)
        if diagnostics is not None:(photo_dir/exact_filename).unlink(missing_ok=True)
        raise


def list_measurements(db_path=DEFAULT_DB):
    with _connect(db_path) as con:
        return [dict(row) for row in con.execute(
            'SELECT * FROM measurements ORDER BY id DESC')]
