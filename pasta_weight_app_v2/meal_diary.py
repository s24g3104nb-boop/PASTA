"""Independent, idempotent local meals with immutable calculation snapshots."""
from datetime import date as Date, time as Time, datetime, timezone, timedelta
from pathlib import Path
import json
import sqlite3
from storage import DEFAULT_DB
from data_backup import backup_database
from meal_photos import prepare_photo, persist_photo, photo_path as _photo_path

ADDITIONS = {'meal_type': "TEXT NOT NULL DEFAULT ''", 'measurement_id': 'INTEGER',
 'cooking_session_id':'INTEGER', 'measurement_snapshot_json':'TEXT',
 'nutrition_status':"TEXT NOT NULL DEFAULT 'unknown'", 'nutrition_source':'TEXT',
 'nutrition_snapshot_json':'TEXT', 'eaten_at':'TEXT', 'source_kind':"TEXT NOT NULL DEFAULT 'unknown'"}


def initialize(db=DEFAULT_DB):
    db=Path(db); db.parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(db) as c:
        c.execute('BEGIN IMMEDIATE')
        columns={r[1] for r in c.execute('PRAGMA table_info(meal_entries)')}
        if not columns or set(ADDITIONS)-columns:
            backup_database(db)
            c.execute('''CREATE TABLE IF NOT EXISTS meal_entries(id TEXT PRIMARY KEY, eaten_date TEXT NOT NULL,
              eaten_time TEXT, dish_name TEXT NOT NULL, memo TEXT NOT NULL, photo TEXT,
              created_at TEXT NOT NULL, updated_at TEXT NOT NULL, deleted INTEGER NOT NULL DEFAULT 0)''')
            columns={r[1] for r in c.execute('PRAGMA table_info(meal_entries)')}
            for name,definition in ADDITIONS.items():
                if name not in columns:c.execute(f'ALTER TABLE meal_entries ADD COLUMN {name} {definition}')
            c.execute('CREATE INDEX IF NOT EXISTS meal_dates ON meal_entries(deleted,eaten_date)')


def _decode(row):
    if row is None:return None
    result=dict(row)
    for name in ('measurement_snapshot','nutrition_snapshot'):
        result[name]=json.loads(result.pop(name+'_json') or 'null')
    return result


def photo_path(key,db=DEFAULT_DB):return _photo_path(key,db)


def get(ident,db=DEFAULT_DB):
    initialize(db)
    with sqlite3.connect(db) as c:
        c.row_factory=sqlite3.Row
        return _decode(c.execute('SELECT * FROM meal_entries WHERE id=? AND deleted=0',(ident,)).fetchone())


def entries(db=DEFAULT_DB):
    initialize(db)
    with sqlite3.connect(db) as c:
        c.row_factory=sqlite3.Row
        return [_decode(r) for r in c.execute('SELECT * FROM meal_entries WHERE deleted=0 ORDER BY eaten_date DESC,eaten_time DESC,created_at DESC')]


def _sources(c,measurement_id,cooking_session_id):
    snapshot=None;nutrition=None;status='unknown';source=None
    for ident in (measurement_id,cooking_session_id):
        if ident is not None and (type(ident) is not int or ident<1):raise ValueError('関連する記録を選び直してください。')
    if cooking_session_id is not None:
        tables={r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        row=c.execute('SELECT * FROM cooking_sessions WHERE id=?',(cooking_session_id,)).fetchone() if 'cooking_sessions' in tables else None
        if row is None:raise ValueError('関連する料理の計算が見つかりません。')
        if measurement_id is not None and measurement_id!=row['measurement_id']:raise ValueError('乾麺の記録と料理の計算が一致していません。')
        measurement_id=row['measurement_id']
        calculation=json.loads(row['calculation_json'])
        status=calculation.get('nutrition_basis','confirmed_amounts')
        nutrition=dict(calculation['portion'],calculation=calculation)
        source='保存済み料理 #'+str(cooking_session_id)+' の取り分け量に基づく計算'
    if measurement_id is not None:
        row=c.execute('SELECT * FROM measurements WHERE id=?',(measurement_id,)).fetchone()
        if row is None:raise ValueError('関連する乾麺の記録が見つかりません。')
        snapshot=dict(row)
        if nutrition is None and all(snapshot.get(k) is not None for k in ('kcal_100','protein_100','fat_100','carb_100')):
            weight=snapshot['actual_g'] if snapshot['actual_g'] is not None else snapshot['estimated_g']
            nutrition={out:weight*snapshot[key]/100 for out,key in [('kcal','kcal_100'),('protein_g','protein_100'),('fat_g','fat_100'),('carb_g','carb_100')]}
            status='dry_only';source='乾麺の実測重量' if snapshot['actual_g'] is not None else '乾麺の推定重量'
    return measurement_id,snapshot,status,source,nutrition


def save(ident,date,time,name,memo='',photo=None,db=DEFAULT_DB,*,meal_type='',measurement_id=None,
         cooking_session_id=None,edit=False,expected_updated_at=None,remove_photo=False):
    if not isinstance(ident,str) or not ident or len(ident)>128:raise ValueError('記録の識別情報を確認してください。')
    if type(date) is not Date or not Date(1900,1,1)<=date<=Date(2100,12,31):raise ValueError('食べた日を確認してください。')
    if time is not None and not isinstance(time,Time):raise ValueError('食べた時刻を確認してください。')
    if not isinstance(name,str) or not name.strip() or len(name)>120:raise ValueError('料理名を120文字以内で入力してください。')
    if not isinstance(memo,str) or len(memo)>2000:raise ValueError('メモは2000文字以内にしてください。')
    if meal_type not in ('','朝食','昼食','夕食','間食','その他'):raise ValueError('食事の区分を選び直してください。')
    initialize(db);filename=None
    try:
        with sqlite3.connect(db) as c:
            c.row_factory=sqlite3.Row;c.execute('BEGIN IMMEDIATE')
            old=c.execute('SELECT * FROM meal_entries WHERE id=?',(ident,)).fetchone()
            if old and not edit:
                if old['deleted']:raise ValueError('削除済みの記録です。別の食事として記録してください。')
                return _decode(old)
            if edit and (old is None or old['deleted']):raise ValueError('編集する記録が見つかりません。')
            if edit and expected_updated_at and expected_updated_at!=old['updated_at']:raise ValueError('別の画面で更新されています。一覧から記録を開き直してください。')
            mid,snapshot,status,source,nutrition=_sources(c,measurement_id,cooking_session_id)
            if old and old['measurement_id']==mid and old['cooking_session_id']==cooking_session_id:
                snapshot=json.loads(old['measurement_snapshot_json'] or 'null');status=old['nutrition_status']
                source=old['nutrition_source'];nutrition=json.loads(old['nutrition_snapshot_json'] or 'null')
            if photo is not None:filename=persist_photo(prepare_photo(photo),db)
            stamp=datetime.now(timezone.utc).isoformat()
            payload=dict(id=ident,eaten_date=date.isoformat(),eaten_time=time.replace(tzinfo=None).isoformat() if time else None,
              dish_name=name.strip(),memo=memo,photo=filename or (old['photo'] if old and not remove_photo else None),
              created_at=old['created_at'] if old else stamp,updated_at=stamp,deleted=0,
              meal_type=meal_type,measurement_id=mid,cooking_session_id=cooking_session_id,
              measurement_snapshot_json=json.dumps(snapshot,ensure_ascii=False,allow_nan=False),nutrition_status=status,
              nutrition_source=source,nutrition_snapshot_json=json.dumps(nutrition,ensure_ascii=False,allow_nan=False),
              eaten_at=datetime.combine(date,time.replace(tzinfo=None),timezone(timedelta(hours=9))).isoformat() if time else None,
              source_kind=snapshot.get('source_kind','unknown') if snapshot else 'unknown')
            columns=','.join(payload);marks=','.join('?' for _ in payload)
            c.execute(f'INSERT INTO meal_entries ({columns}) VALUES ({marks}) ON CONFLICT(id) DO UPDATE SET '+','.join(k+'=excluded.'+k for k in payload if k!='id'),tuple(payload.values()))
        return _decode(payload)
    except Exception:
        if filename:photo_path(filename,db).unlink(missing_ok=True)
        raise


def delete(ident,db=DEFAULT_DB,expected_updated_at=None):
    initialize(db)
    with sqlite3.connect(db) as c:
        c.execute('BEGIN IMMEDIATE')
        row=c.execute('SELECT updated_at FROM meal_entries WHERE id=? AND deleted=0',(ident,)).fetchone()
        if not row:return
        if expected_updated_at and row[0]!=expected_updated_at:raise ValueError('記録が更新されています。開き直してください。')
        c.execute('UPDATE meal_entries SET deleted=1,updated_at=? WHERE id=?',(datetime.now(timezone.utc).isoformat(),ident))
