"""既存の計測を変更せず、料理の入力値・計算条件を別テーブルに保存する。"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

from cooking import Ingredient, calculate_meal, dry_snapshot
from dish_classifier import DISH_NAMES
from storage import DEFAULT_DB


@contextmanager
def _connection(db_path):
    con = sqlite3.connect(str(db_path), timeout=10)
    con.row_factory = sqlite3.Row
    con.execute('PRAGMA foreign_keys = ON')
    try:
        with con:
            yield con
    finally:
        con.close()


def init_cooking_db(db_path=DEFAULT_DB):
    """追加のみの移行。既存products/measurementsの構造・行は変更しない。"""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with _connection(db_path) as con:
        con.executescript('''
            CREATE TABLE IF NOT EXISTS cooking_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                measurement_id INTEGER NOT NULL REFERENCES measurements(id),
                created_at TEXT NOT NULL,
                selected_dish TEXT NOT NULL,
                same_batch_confirmed INTEGER NOT NULL CHECK(same_batch_confirmed = 1),
                classifier_status TEXT NOT NULL,
                dry_snapshot_json TEXT NOT NULL,
                calculation_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS cooking_ingredients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL REFERENCES cooking_sessions(id),
                name TEXT NOT NULL,
                amount_g REAL NOT NULL,
                kcal_100 REAL NOT NULL,
                protein_100 REAL NOT NULL,
                fat_100 REAL NOT NULL,
                carb_100 REAL NOT NULL
            );
        ''')


def save_cooking_session(*, measurement_id: int, selected_dish: str,
                        confirmed_same_batch: bool, ingredients: list[Ingredient],
                        servings: int | None = None, consumed_fraction: float | None = None,
                        db_path=DEFAULT_DB) -> int:
    if confirmed_same_batch is not True:
        raise ValueError('同一調理バッチの乾麺であることを確認してください。')
    if isinstance(measurement_id, bool) or not isinstance(measurement_id, int):
        raise ValueError('計測履歴IDを指定してください。')
    if selected_dish not in DISH_NAMES:
        raise ValueError('料理名を一覧から選択してください。')
    init_cooking_db(db_path)
    with _connection(db_path) as con:
        row = con.execute('SELECT * FROM measurements WHERE id=?', (measurement_id,)).fetchone()
        if row is None:
            raise ValueError('指定した乾麺の計測履歴がありません。')
        snapshot = dry_snapshot(dict(row))
        calculation = calculate_meal(snapshot, ingredients, servings=servings,
                                     consumed_fraction=consumed_fraction)
        cursor = con.execute('''INSERT INTO cooking_sessions
            (measurement_id, created_at, selected_dish, same_batch_confirmed,
             classifier_status, dry_snapshot_json, calculation_json)
            VALUES (?, ?, ?, 1, ?, ?, ?)''',
            (measurement_id, datetime.now(timezone.utc).isoformat(), selected_dish,
             'untrained/manual', json.dumps(snapshot, ensure_ascii=False, allow_nan=False),
             json.dumps(calculation, ensure_ascii=False, allow_nan=False)))
        session_id = cursor.lastrowid
        con.executemany('''INSERT INTO cooking_ingredients
            (session_id, name, amount_g, kcal_100, protein_100, fat_100, carb_100)
            VALUES (?, ?, ?, ?, ?, ?, ?)''',
            [(session_id, item['name'], item['amount_g'], item['kcal_100'],
              item['protein_100'], item['fat_100'], item['carb_100'])
             for item in calculation['ingredients']])
        return session_id


def list_cooking_sessions(db_path=DEFAULT_DB) -> list[dict]:
    init_cooking_db(db_path)
    with _connection(db_path) as con:
        rows = con.execute('SELECT * FROM cooking_sessions ORDER BY id DESC').fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item['dry_snapshot'] = json.loads(item.pop('dry_snapshot_json'))
            item['calculation'] = json.loads(item.pop('calculation_json'))
            result.append(item)
        return result
