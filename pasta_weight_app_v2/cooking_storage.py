"""既存の計測を変更せず、料理の入力値・計算条件を別テーブルに保存する。"""
from __future__ import annotations

from contextlib import contextmanager
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

from cooking import Ingredient, calculate_meal, dry_snapshot, nonnegative
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
            CREATE TABLE IF NOT EXISTS recipe_templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                selected_dish TEXT NOT NULL,
                reference_dry_g REAL NOT NULL,
                default_servings INTEGER NOT NULL,
                source TEXT NOT NULL,
                ingredients_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
        ''')


def save_cooking_session(*, measurement_id: int, selected_dish: str,
                        confirmed_same_batch: bool, ingredients: list[Ingredient],
                        servings: int | None = None, consumed_fraction: float | None = None,
                        recipe_template_id: int | None = None,
                        db_path=DEFAULT_DB) -> int:
    if confirmed_same_batch is not True:
        raise ValueError('同一調理バッチの乾麺であることを確認してください。')
    if isinstance(measurement_id, bool) or not isinstance(measurement_id, int):
        raise ValueError('計測履歴IDを指定してください。')
    if selected_dish not in DISH_NAMES:
        raise ValueError('料理名を一覧から選択してください。')
    init_cooking_db(db_path)
    with _connection(db_path) as con:
        # 同時送信も直列化し、旧DBに重複がある場合もデータは削除しない。
        con.execute('BEGIN IMMEDIATE')
        existing = con.execute('SELECT id FROM cooking_sessions WHERE measurement_id=?',
                               (measurement_id,)).fetchone()
        if existing is not None:
            raise ValueError(f'この乾麺履歴は料理ID {existing["id"]} に使用済みです。二重計上を防ぐため新規保存できません。')
        row = con.execute('SELECT * FROM measurements WHERE id=?', (measurement_id,)).fetchone()
        if row is None:
            raise ValueError('指定した乾麺の計測履歴がありません。')
        snapshot = dry_snapshot(dict(row))
        template = None
        if recipe_template_id is not None:
            if isinstance(recipe_template_id, bool) or not isinstance(recipe_template_id, int):
                raise ValueError('レシピテンプレートIDが不正です。')
            found = con.execute('SELECT * FROM recipe_templates WHERE id=?',
                                (recipe_template_id,)).fetchone()
            if found is None:
                raise ValueError('選択したレシピテンプレートは存在しません。')
            template = _decode_template(found)
            if template['selected_dish'] != selected_dish:
                raise ValueError('テンプレートと料理名が一致しません。料理名に合うテンプレートを選択してください。')
        calculation = calculate_meal(snapshot, ingredients, servings=servings,
                                     consumed_fraction=consumed_fraction,
                                     nutrition_basis='template_estimate' if template else 'confirmed_amounts')
        if template:
            calculation['recipe_template'] = template
            calculation['assumptions'] = ('利用者登録レシピの材料量を参考にした概算です。'
                                           '今回の使用量を確認・修正した値ですが、写真から量を実測したものではありません。')
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
        counts = Counter(row['measurement_id'] for row in rows)
        result = []
        for row in rows:
            item = dict(row)
            item['dry_snapshot'] = json.loads(item.pop('dry_snapshot_json'))
            item['calculation'] = json.loads(item.pop('calculation_json'))
            item['duplicate_measurement_count'] = counts[item['measurement_id']]
            result.append(item)
        return result


def _decode_template(row) -> dict:
    item = dict(row)
    item['ingredients'] = json.loads(item.pop('ingredients_json'))
    return item


def save_recipe_template(*, name: str, selected_dish: str, ingredients: list[Ingredient],
                         reference_dry_g: float, default_servings: int, source: str,
                         db_path=DEFAULT_DB) -> int:
    if not isinstance(name, str) or not name.strip():
        raise ValueError('テンプレート名を入力してください。')
    if not isinstance(source, str) or not source.strip():
        raise ValueError('材料・栄養値の出典と分量の前提を入力してください。')
    if selected_dish not in DISH_NAMES:
        raise ValueError('料理名を一覧から選択してください。')
    weight = nonnegative(reference_dry_g, '基準乾麺g')
    if weight <= 0:
        raise ValueError('基準乾麺gは0より大きい必要があります。')
    if isinstance(default_servings, bool) or not isinstance(default_servings, int) or default_servings < 1:
        raise ValueError('基準人数は1以上の整数で指定してください。')
    if not ingredients:
        raise ValueError('テンプレートには材料を1件以上登録してください。')
    payload = [asdict(item.validated()) for item in ingredients]
    init_cooking_db(db_path)
    with _connection(db_path) as con:
        cursor = con.execute('''INSERT INTO recipe_templates
            (name, selected_dish, reference_dry_g, default_servings, source, ingredients_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)''',
            (name.strip(), selected_dish, weight, default_servings, source.strip(),
             json.dumps(payload, ensure_ascii=False, allow_nan=False),
             datetime.now(timezone.utc).isoformat()))
        return cursor.lastrowid


def list_recipe_templates(db_path=DEFAULT_DB) -> list[dict]:
    init_cooking_db(db_path)
    with _connection(db_path) as con:
        return [_decode_template(row) for row in con.execute('SELECT * FROM recipe_templates ORDER BY id')]


def delete_recipe_template(template_id: int, db_path=DEFAULT_DB):
    if isinstance(template_id, bool) or not isinstance(template_id, int):
        raise ValueError('テンプレートIDが不正です。')
    init_cooking_db(db_path)
    with _connection(db_path) as con:
        con.execute('DELETE FROM recipe_templates WHERE id=?', (template_id,))
