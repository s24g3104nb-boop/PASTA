"""画面から独立したCPU学習ジョブ、固定分割、モデル選択を管理する。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import subprocess
import sys
from uuid import uuid4

APP_DIR = Path(__file__).resolve().parent
REPO_DIR = APP_DIR.parent
for directory in (REPO_DIR, APP_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from pasta_weight_app_v2.pasta_ml.dataset import (
    Sample, dataset_hash, grouping, load_dataset, validate_splits, write_csv,
)
from pasta_weight_app_v2.pasta_ml.evaluate import evaluate,compare_reports
from pasta_weight_app_v2.pasta_ml.model import load_model, train
from pasta_weight_app_v2.storage import DEFAULT_DB
from pasta_weight_app_v2.training_data import export_dataset, list_samples


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


def _now():
    return datetime.now(timezone.utc).isoformat()


def init_training_manager(db_path=DEFAULT_DB):
    with _connection(db_path) as con:
        con.executescript('''
            CREATE TABLE IF NOT EXISTS training_jobs (
                id TEXT PRIMARY KEY, status TEXT NOT NULL, progress INTEGER NOT NULL,
                message TEXT NOT NULL, created_at TEXT NOT NULL, finished_at TEXT,
                worker_pid INTEGER, alpha REAL NOT NULL, allow_synthetic INTEGER NOT NULL,
                sample_count INTEGER NOT NULL DEFAULT 0, model_id TEXT, report_json TEXT,
                error TEXT
            );
            CREATE TABLE IF NOT EXISTS training_models (
                id TEXT PRIMARY KEY, job_id TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL,
                source TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 0,
                model_path TEXT NOT NULL, artifact_sha256 TEXT NOT NULL,
                dataset_sha256 TEXT NOT NULL, report_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS training_split_assignments (
                fingerprint TEXT PRIMARY KEY, pixel_hash TEXT NOT NULL, batch_id TEXT NOT NULL,
                session_id TEXT NOT NULL, perceptual_hash TEXT NOT NULL, split TEXT NOT NULL,
                physical_sample_id TEXT NOT NULL DEFAULT '',
                assigned_at TEXT NOT NULL
            );
        ''')
        columns = {r['name'] for r in con.execute('PRAGMA table_info(training_split_assignments)')}
        if 'physical_sample_id' not in columns:
            backups=Path(db_path).resolve().parent/'backups'
            backups.mkdir(parents=True,exist_ok=True)
            with sqlite3.connect(backups/f'{Path(db_path).stem}.before-split-physical-{uuid4().hex}.sqlite3') as backup:
                con.backup(backup)
            con.execute("ALTER TABLE training_split_assignments ADD COLUMN physical_sample_id TEXT NOT NULL DEFAULT ''")


def _job_dir(job_id, db_path):
    if not isinstance(job_id, str) or len(job_id) != 32 or any(c not in '0123456789abcdef' for c in job_id):
        raise ValueError('学習ジョブIDが不正です。')
    return Path(db_path).resolve().parent / 'training' / 'jobs' / job_id


def _process_matches(pid, job_id):
    if not pid:
        return False
    if os.name == 'posix' and Path('/proc').is_dir():
        try:
            return job_id.encode() in Path(f'/proc/{int(pid)}/cmdline').read_bytes()
        except OSError:
            return False
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def _recover_interrupted(con):
    for row in con.execute("SELECT * FROM training_jobs WHERE status IN ('queued','running')").fetchall():
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(row['created_at'])).total_seconds()
        if (row['worker_pid'] and not _process_matches(row['worker_pid'], row['id'])) or (not row['worker_pid'] and age > 30):
            con.execute("UPDATE training_jobs SET status='failed', message=?, error=?, finished_at=? WHERE id=?",
                        ('学習プロセスが停止しました。新しいジョブで再実行してください。',
                         'worker_interrupted', _now(), row['id']))


def start_training(*, db_path=DEFAULT_DB, alpha=1.0, allow_synthetic=False):
    """スキャンや学習は子プロセスで行う。画面への復帰時に待たない。"""
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('正則化係数は有限の正数にしてください。')
    init_training_manager(db_path)
    job_id = uuid4().hex
    with _connection(db_path) as con:
        con.execute('BEGIN IMMEDIATE')
        _recover_interrupted(con)
        if con.execute("SELECT 1 FROM training_jobs WHERE status IN ('queued','running')").fetchone():
            raise ValueError('学習ジョブが実行中です。完了後に次の学習を開始してください。')
        con.execute('''INSERT INTO training_jobs
            (id,status,progress,message,created_at,alpha,allow_synthetic)
            VALUES (?,'queued',0,?,?,?,?)''',
            (job_id, '学習を準備しています。', _now(), alpha, int(allow_synthetic)))
    directory = _job_dir(job_id, db_path)
    try:
        directory.mkdir(parents=True)
        with (directory / 'worker.log').open('ab') as log:
            process = subprocess.Popen(
                [sys.executable, '-m', 'pasta_weight_app_v2.training_manager',
                 '--worker', job_id, '--db', str(Path(db_path).resolve())],
                cwd=REPO_DIR, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                start_new_session=(os.name != 'nt'))
        with _connection(db_path) as con:
            con.execute('UPDATE training_jobs SET worker_pid=? WHERE id=?', (process.pid, job_id))
    except Exception as exc:
        _update_job(job_id, db_path, status='failed', message='学習プロセスを開始できませんでした。',
                    error=str(exc), finished_at=_now())
        raise
    return job_id


def _update_job(job_id, db_path, **fields):
    allowed = {'status', 'progress', 'message', 'worker_pid', 'sample_count', 'model_id',
               'report_json', 'error', 'finished_at'}
    if not fields or not set(fields) <= allowed:
        raise ValueError('ジョブ更新項目が不正です。')
    with _connection(db_path) as con:
        con.execute('UPDATE training_jobs SET ' + ','.join(f'{key}=?' for key in fields) + ' WHERE id=?',
                    (*fields.values(), job_id))


def list_jobs(db_path=DEFAULT_DB):
    init_training_manager(db_path)
    with _connection(db_path) as con:
        _recover_interrupted(con)
        rows = con.execute('SELECT * FROM training_jobs ORDER BY created_at DESC').fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item['report'] = json.loads(item.pop('report_json')) if item['report_json'] else None
        result.append(item)
    return result


def _fingerprint(sample):
    values = [sample.pixel_hash, sample.values['batch_id'], sample.values['session_id'],
              sample.values.get('physical_sample_id',''), str(sample.perceptual_hash)]
    return hashlib.sha256(json.dumps(values).encode()).hexdigest()


def assign_persistent_splits(samples, db_path):
    """過去のval/testをtrainへ移さず、追加画像にも関連グループの分割を継承する。"""
    with _connection(db_path) as con:
        con.execute('BEGIN IMMEDIATE')
        history = con.execute('SELECT * FROM training_split_assignments').fetchall()
        combined = list(samples) + [Sample(
            dict(batch_id=r['batch_id'], session_id=r['session_id'], physical_sample_id=r['physical_sample_id']), Path('.'),
            r['pixel_hash'], int(r['perceptual_hash'])) for r in history]
        current_groups, unknown = [], []
        counts = dict(train=0, val=0, test=0)
        for group in grouping(combined):
            current = [i for i in group if i < len(samples)]
            if not current:
                continue
            known = {history[i - len(samples)]['split'] for i in group if i >= len(samples)}
            if len(known) > 1:
                raise ValueError('追加画像が既存のtrain/val/testを結び付けています。分割リークを防ぐため停止しました。')
            current_groups.append(current)
            if known:
                split = next(iter(known))
                for i in current:
                    samples[i].values['split'] = split
                counts[split] += 1
            else:
                unknown.append(current)
        if len(current_groups) < 3:
            raise ValueError('独立した秤量・撮影・画像グループが3つ以上必要です。')
        random.Random(42).shuffle(unknown)
        target = dict(val=max(1, round(len(current_groups) * .15)),
                      test=max(1, round(len(current_groups) * .15)))
        target['train'] = len(current_groups) - target['val'] - target['test']
        # First fill missing partitions, then their desired group counts.
        for split in ('train', 'val', 'test'):
            if not counts[split]:
                if not unknown:
                    raise ValueError('以前の分割を保つと評価用データが不足します。独立した写真を追加してください。')
                group = unknown.pop()
                for i in group:
                    samples[i].values['split'] = split
                counts[split] += 1
        for group in unknown:
            split = max(('train', 'val', 'test'), key=lambda key: target[key] - counts[key])
            for i in group:
                samples[i].values['split'] = split
            counts[split] += 1
        validate_splits(samples)
        for sample in samples:
            con.execute('''INSERT OR IGNORE INTO training_split_assignments
                (fingerprint,pixel_hash,batch_id,session_id,perceptual_hash,split,assigned_at,physical_sample_id)
                VALUES (?,?,?,?,?,?,?,?)''',
                (_fingerprint(sample), sample.pixel_hash, sample.values['batch_id'],
                 sample.values['session_id'], str(sample.perceptual_hash), sample.values['split'], _now(),
                 sample.values.get('physical_sample_id','')))
    return samples


def _calibrations(db_path):
    with _connection(db_path) as con:
        if not con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='products'").fetchone():
            return {}
        rows = con.execute('SELECT id,g_per_strand,name FROM products').fetchall()
    return {str(row['id']): row['g_per_strand'] for row in rows
            if math.isfinite(row['g_per_strand']) and row['g_per_strand'] > 0
            and not row['name'].startswith('デモ用')}


def run_job(job_id, db_path=DEFAULT_DB):
    """ワーカープロセス専用。失敗も状態として保存し、モデルを有効化しない。"""
    init_training_manager(db_path)
    with _connection(db_path) as con:
        job = con.execute('SELECT * FROM training_jobs WHERE id=?', (job_id,)).fetchone()
    if job is None or job['status'] != 'queued':
        raise ValueError('待機中のジョブがありません。')
    directory = _job_dir(job_id, db_path)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        _update_job(job_id, db_path, status='running', progress=5, worker_pid=os.getpid(),
                    message='登録画像を確認し、変更されない実験スナップショットを作成しています。')
        registered = list_samples(db_path)
        selected = registered if job['allow_synthetic'] else [r for r in registered if r['source'] == 'real']
        if not selected:
            raise ValueError('実測ラベル付きの実写学習画像がありません。写真を登録してください。')
        _update_job(job_id, db_path, sample_count=len(selected))
        csv_path = export_dataset(directory / 'dataset', db_path=db_path,
                                  sample_ids=[r['id'] for r in selected])
        samples = load_dataset(csv_path)
        _update_job(job_id, db_path, progress=25, message='秤量・撮影セッション・近似画像をまとめ、固定分割を検査しています。')
        assign_persistent_splits(samples, db_path)
        split_csv = directory / 'dataset' / 'split.csv'
        write_csv(samples, split_csv)
        # Reload the persisted split exactly as future CLI evaluations will.
        samples = load_dataset(split_csv)
        _update_job(job_id, db_path, progress=45, message='CPUでモード・製品別の連続重量回帰を学習しています。')
        model_path = directory / 'model.json'
        artifact = train(samples, model_path, alpha=job['alpha'], allow_synthetic=bool(job['allow_synthetic']))
        _update_job(job_id, db_path, progress=75, message='同じval写真で回帰とOpenCVを比較しています。testは開封しません。')
        calibrations = _calibrations(db_path)
        report = {
            'regression': artifact['validation_report'],
            'opencv': evaluate(samples, 'val', calibrations=calibrations,
                               allow_synthetic=bool(job['allow_synthetic'])),
            'split_counts': {split: sum(s.values['split'] == split for s in samples)
                             for split in ('train', 'val', 'test')},
            'sample_count': len(samples), 'excluded_synthetic_count': len(registered) - len(selected),
            'dataset_sha256': dataset_hash(samples), 'calibrations': calibrations,
            'notice': 'valのみの開発用比較です。testは未開封。実写での汎化性能は別の最終評価が必要です。',
        }
        report['comparison']=compare_reports(report['regression'],report['opencv'])
        report['split_group_counts']={split:len(grouping([s for s in samples if s.values['split']==split])) for split in ('train','val','test')}
        report['reproducibility']=artifact['reproducibility']
        serialized = json.dumps(report, ensure_ascii=False, allow_nan=False)
        (directory / 'report.json').write_text(serialized, encoding='utf-8')
        model_id = uuid4().hex
        with _connection(db_path) as con:
            con.execute('''INSERT INTO training_models
                (id,job_id,created_at,source,model_path,artifact_sha256,dataset_sha256,report_json)
                VALUES (?,?,?,?,?,?,?,?)''',
                (model_id, job_id, _now(), artifact['source'], str(model_path.resolve()),
                 hashlib.sha256(model_path.read_bytes()).hexdigest(), artifact['dataset_sha256'], serialized))
            con.execute('''UPDATE training_jobs SET status='succeeded',progress=100,message=?,
                model_id=?,report_json=?,finished_at=? WHERE id=?''',
                ('学習とval比較が完了しました。利用前にモデルと誤差を確認してください。',
                 model_id, serialized, _now(), job_id))
        return model_id
    except Exception as exc:
        _update_job(job_id, db_path, status='failed', message='学習を完了できませんでした。',
                    error=str(exc), finished_at=_now())
        return None


def list_models(db_path=DEFAULT_DB):
    init_training_manager(db_path)
    with _connection(db_path) as con:
        rows = con.execute('SELECT * FROM training_models ORDER BY created_at DESC').fetchall()
        revoked={}
        if con.execute("SELECT 1 FROM sqlite_master WHERE name='model_revocations'").fetchone():
            revoked={r['model_id']:r['reason'] for r in con.execute('SELECT * FROM model_revocations')}
    result = []
    for row in rows:
        item = dict(row)
        item['active'] = bool(item['active'])
        item['revoked_reason']=revoked.get(item['id'])
        item['report'] = json.loads(item.pop('report_json'))
        item['test_opened'] = Path(item['model_path'] + '.test-used.json').exists()
        if item['test_opened']:
            item['report']['notice'] = ('このモデルの最終testはCLIで開封済みです。以下は保存済みval比較です。'
                                        '再調整後の最終評価には新しい未使用実写testが必要です。')
        result.append(item)
    return result


def _validated_artifact(metadata, db_path):
    with _connection(db_path) as con:
        if con.execute("SELECT 1 FROM sqlite_master WHERE name='model_revocations'").fetchone():
            if con.execute('SELECT 1 FROM model_revocations WHERE model_id=?',(metadata['id'],)).fetchone():
                raise ValueError('ラベル訂正で利用を撤回したモデルです。新しい台帳で再学習してください。')
    path = Path(metadata['model_path']).resolve()
    expected_root = Path(db_path).resolve().parent / 'training' / 'jobs'
    if not path.is_relative_to(expected_root) or path.name != 'model.json':
        raise ValueError('管理モデルの保存パスが不正です。')
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['artifact_sha256']:
        raise ValueError('モデルファイルが保存時から変更されています。')
    artifact = load_model(path)
    if artifact['source'] != metadata['source'] or artifact['dataset_sha256'] != metadata['dataset_sha256']:
        raise ValueError('モデルの来歴が管理情報と一致しません。')
    return artifact


def activate_model(model_id, db_path=DEFAULT_DB):
    init_training_manager(db_path)
    with _connection(db_path) as con:
        row = con.execute('SELECT * FROM training_models WHERE id=?', (model_id,)).fetchone()
        if row is None:
            raise ValueError('モデルが存在しません。')
        metadata = dict(row)
        artifact = _validated_artifact(metadata, db_path)
        if artifact['source'] != 'real':
            raise ValueError('合成画像専用モデルは実写測定用に選択できません。')
        report = json.loads(metadata['report_json'])
        if report['regression']['overall']['succeeded'] == 0:
            raise ValueError('val画像で推定成功がありません。独立した検証写真と撮影条件を見直してください。')
        con.execute('UPDATE training_models SET active=0')
        con.execute('UPDATE training_models SET active=1 WHERE id=?', (model_id,))


def get_active_model(db_path=DEFAULT_DB):
    models = [model for model in list_models(db_path) if model['active']]
    if not models:
        return None
    model = models[0]
    model['artifact'] = _validated_artifact(model, db_path)
    return model


def deactivate_model(db_path=DEFAULT_DB):
    init_training_manager(db_path)
    with _connection(db_path) as con:
        con.execute('UPDATE training_models SET active=0')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', required=True)
    parser.add_argument('--db', required=True)
    args = parser.parse_args()
    run_job(args.worker, args.db)


if __name__ == '__main__':
    main()
