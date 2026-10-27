"""SQLiteによる食品データと測定履歴の保存。"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from PIL import Image
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
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
    if any(v < 0 for v in (g_per_strand, kcal_100, protein_100, fat_100, carb_100)):
        raise ValueError('重量・栄養情報に負数は使用できません。')
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
                     nutrition, image_rgb, db_path=DEFAULT_DB):
    # 履歴はローカル保存。Web公開環境での複数人利用には認証が必要。
    db_path = Path(db_path)
    photo_dir = db_path.parent / 'photos'
    photo_dir.mkdir(parents=True, exist_ok=True)
    filename = f'{uuid4().hex}.jpg'
    Image.fromarray(np.asarray(image_rgb, dtype=np.uint8)).save(photo_dir / filename,
                                                               'JPEG', quality=82)
    try:
        with _connect(db_path) as con:
            con.execute('''
                INSERT INTO measurements (created_at, product_name, mode, detected_count,
                estimated_g, actual_g, kcal, protein_g, fat_g, carb_g, image_filename)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), product_name, mode,
                  int(detected_count), float(estimated_g),
                  (float(actual_g) if actual_g is not None else None),
                  nutrition['kcal'], nutrition['protein_g'], nutrition['fat_g'],
                  nutrition['carb_g'], filename))
    except Exception:
        (photo_dir / filename).unlink(missing_ok=True)
        raise


def list_measurements(db_path=DEFAULT_DB):
    with _connect(db_path) as con:
        return [dict(row) for row in con.execute(
            'SELECT * FROM measurements ORDER BY id DESC')]
