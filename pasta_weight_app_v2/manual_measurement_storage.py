"""Backed-up, atomic migration: absent predictions/nutrition are SQL NULL."""
import re
import sqlite3
from data_backup import backup_database


def allow_unestimated_measurements(db_path):
    con=sqlite3.connect(db_path,timeout=10)
    try:
        columns={r[1]:r for r in con.execute('PRAGMA table_info(measurements)')}
        fields=('estimated_g','kcal','protein_g','fat_g','carb_g')
        if not any(columns[k][3] for k in fields):return
        backup_database(db_path,label='manual-weight')
        con.execute('BEGIN IMMEDIATE')
        sql=con.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='measurements'").fetchone()[0]
        for field in fields:
            sql=re.sub(r'\b'+field+r'\s+REAL\s+NOT\s+NULL',field+' REAL',sql,flags=re.I)
        sql=re.sub(r'CREATE TABLE\s+["\[]?measurements["\]]?', 'CREATE TABLE measurements_manual_new',sql,count=1,flags=re.I)
        objects=[r[0] for r in con.execute("SELECT sql FROM sqlite_master WHERE tbl_name='measurements' AND type IN ('index','trigger') AND sql IS NOT NULL")]
        seq=con.execute("SELECT seq FROM sqlite_sequence WHERE name='measurements'").fetchone()
        con.execute(sql)
        names=','.join('"'+k+'"' for k in columns)
        con.execute(f'INSERT INTO measurements_manual_new ({names}) SELECT {names} FROM measurements')
        con.execute('DROP TABLE measurements')
        con.execute('ALTER TABLE measurements_manual_new RENAME TO measurements')
        for statement in objects:con.execute(statement)
        if seq:con.execute("UPDATE sqlite_sequence SET seq=max(seq,?) WHERE name='measurements'",seq)
        if con.execute('PRAGMA foreign_key_check').fetchall():raise sqlite3.IntegrityError('Foreign key integrity check failed')
        con.commit()
    except Exception:
        con.rollback();raise
    finally:con.close()
