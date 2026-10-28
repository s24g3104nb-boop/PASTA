"""Audited edits and reversible archival of local measurements."""
import json
import sqlite3
from datetime import datetime, timezone
from storage import DEFAULT_DB


def change_measurement(ident, *, actual_g=None, archive=False, db_path=DEFAULT_DB):
    if not archive:
        import math
        if actual_g is None or not math.isfinite(actual_g) or actual_g<=0:
            raise ValueError('実測重量は0より大きい有限数で入力してください。')
    with sqlite3.connect(db_path) as con:
        con.row_factory=sqlite3.Row
        row=con.execute('SELECT * FROM measurements WHERE id=?',(ident,)).fetchone()
        if row is None:raise ValueError('履歴が見つかりません。')
        tables={r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'cooking_sessions' in tables and con.execute('SELECT 1 FROM cooking_sessions WHERE measurement_id=?',(ident,)).fetchone():
            raise ValueError('保存した料理で使用中の履歴です。料理との対応を保つため変更・削除できません。')
        con.execute('CREATE TABLE IF NOT EXISTS measurement_edits (id INTEGER PRIMARY KEY, measurement_id INTEGER NOT NULL, created_at TEXT NOT NULL, action TEXT NOT NULL, previous_json TEXT NOT NULL, actual_g REAL)')
        con.execute('INSERT INTO measurement_edits(measurement_id,created_at,action,previous_json,actual_g) VALUES(?,?,?,?,?)',
                    (ident,datetime.now(timezone.utc).isoformat(),'archive' if archive else 'actual_weight',json.dumps(dict(row),ensure_ascii=False),actual_g))
        if archive:
            con.execute('UPDATE measurements SET archived=1 WHERE id=?',(ident,))
        else:
            con.execute('UPDATE measurements SET actual_g=? WHERE id=?',(actual_g,ident))
