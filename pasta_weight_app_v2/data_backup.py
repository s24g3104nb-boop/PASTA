"""Verified local SQLite snapshots for additive migrations."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
import sqlite3
from uuid import uuid4


def backup_database(db_path, label: str = 'diary-v2') -> Path:
    """Back up committed SQLite/WAL data without overwriting any recovery copy.

    A separate reader permits callers to hold BEGIN IMMEDIATE before migration.
    """
    db = Path(db_path)
    if not db.is_file():
        raise FileNotFoundError('バックアップするデータベースがありません。')
    if not re.fullmatch(r'[a-z0-9-]{1,40}', label):
        raise ValueError('バックアップ名が不正です。')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    target = db.with_name(f'{db.name}.before-{label}-{stamp}-{uuid4().hex[:8]}.backup')
    try:
        with target.open('xb'):
            pass
        with sqlite3.connect(db, timeout=10) as source:
            with sqlite3.connect(target) as destination:
                source.backup(destination)
                if destination.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                    raise sqlite3.DatabaseError('バックアップの整合性を確認できませんでした。')
        return target
    except Exception:
        target.unlink(missing_ok=True)
        raise
