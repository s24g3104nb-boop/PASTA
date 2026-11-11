"""Local, unapproved research intake; never opens the application's database.

Original bytes are immutable. User-entered weights are declarations, not model
predictions. Registration never establishes physical independence or approves
training, even when the operator declares a new separately weighed sample.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path
import sqlite3
import uuid

from PIL import Image, ImageOps, ImageStat, UnidentifiedImageError

SCHEMA_VERSION = 1
MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 40_000_000
RELATIONS = frozenset({'unknown', 'new_disjoint', 'reuse', 'added'})
ANGLES = frozenset({'overhead', 'oblique', 'side', 'unknown'})
REFERENCES = frozenset({'none', 'a4_markers', 'known_board', 'noodle_length', 'fixed_distance', 'other'})
SAMPLE_FIELDS = ('sample_id', 'measured_weight_g', 'product_name', 'relation',
                 'parent_sample_ids', 'lineage_note', 'lineage_confirmed_by')


class CollectionValidationError(ValueError):
    """An intake must be corrected before it can be registered."""


class DuplicateConflictError(CollectionValidationError):
    """An existing immutable record conflicts with a new submission."""


class ExistingDatasetDuplicateError(CollectionValidationError):
    def __init__(self, matches):
        self.matches = matches
        super().__init__('この原本は既存データ内にあります。統合やラベル変更は行わず登録を中止しました。')


def _now():
    return datetime.now(timezone.utc).isoformat()


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def _text(value, name, *, optional=False, maximum=200):
    if not isinstance(value, str) or '\x00' in value or len(value) > maximum:
        raise CollectionValidationError(f'{name}を正しく入力してください。')
    value = value.strip()
    if not value and not optional:
        raise CollectionValidationError(f'{name}を入力してください。')
    return value


def validate_metadata(source):
    if not isinstance(source, dict):
        raise CollectionValidationError('撮影情報を入力してください。')
    weight = source.get('measured_weight_g')
    if isinstance(weight, bool) or not isinstance(weight, (float, int)) or not math.isfinite(weight) or weight <= 0:
        raise CollectionValidationError('実測重量は0より大きい有限の数値で入力してください。')
    row = {'measured_weight_g': float(weight)}
    for field, label in [('product_name', '製品名'), ('sample_id', '物理試料ID'),
                         ('capture_session_id', '撮影セッションID')]:
        row[field] = _text(source.get(field), label)
    for field, allowed, label in [('angle', ANGLES, '撮影角度'),
                                  ('scale_reference', REFERENCES, 'スケール基準'),
                                  ('relation', RELATIONS, '試料の関係')]:
        if source.get(field) not in allowed:
            raise CollectionValidationError(f'{label}を選択してください。')
        row[field] = source[field]
    stamp = source.get('captured_at')
    try:
        parsed = datetime.fromisoformat(stamp)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError()
    except (ValueError, TypeError):
        raise CollectionValidationError('撮影日時はタイムゾーンを含む日時で入力してください。') from None
    row['captured_at'] = parsed.isoformat()
    if not isinstance(source.get('scale_verified'), bool):
        raise CollectionValidationError('基準の確認状態を指定してください。')
    if source['scale_verified'] and source['scale_reference'] == 'none':
        raise CollectionValidationError('基準物なしでは寸法確認済みにできません。')
    row['scale_verified'] = source['scale_verified']
    parents = source.get('parent_sample_ids')
    if not isinstance(parents, list):
        raise CollectionValidationError('再利用・追加元の試料IDを指定してください。')
    row['parent_sample_ids'] = sorted(_text(value, '元の試料ID') for value in parents)
    if len(set(row['parent_sample_ids'])) != len(parents) or row['sample_id'] in parents:
        raise CollectionValidationError('同じ試料自身や重複した試料を親に指定できません。')
    row['lineage_note'] = _text(source.get('lineage_note', ''), '試料の根拠', optional=True, maximum=2000)
    row['lineage_confirmed_by'] = _text(source.get('lineage_confirmed_by', ''), '確認者', optional=True)
    if row['relation'] in {'reuse', 'added'} and not parents:
        raise CollectionValidationError('再利用・追加元の登録済み試料を選択してください。')
    if row['relation'] == 'new_disjoint':
        if parents or not row['lineage_note'] or not row['lineage_confirmed_by']:
            raise CollectionValidationError('独立試料の申告には確認者と、未使用の麺を別々に量った根拠が必要です。')
    if row['relation'] == 'unknown' and parents:
        raise CollectionValidationError('親試料が分かる場合は再利用または追加を選択してください。')
    filename = _text(source.get('original_filename', ''), 'ファイル名', optional=True, maximum=500)
    # Filename is display metadata only; it is never used to select a path.
    row['original_filename'] = filename.replace('\\', '/').split('/')[-1]
    row['reference_coplanar'] = source.get('reference_coplanar') is True
    row['label_source'] = 'user_entered_measured_weight'
    row['lineage_verified'] = False
    row['training_approved'] = False
    row['scale_verification_source'] = 'operator_declaration_only'
    return row


def prepare_image(image_bytes):
    if not isinstance(image_bytes, bytes) or not image_bytes or len(image_bytes) > MAX_BYTES:
        raise CollectionValidationError('写真は20MB以内のJPEG・PNG・WebPを選択してください。')
    try:
        with Image.open(BytesIO(image_bytes)) as source:
            if source.format not in {'JPEG', 'PNG', 'WEBP'} or getattr(source, 'n_frames', 1) != 1:
                raise CollectionValidationError('静止画のJPEG・PNG・WebPを選択してください。')
            if source.width * source.height > MAX_PIXELS or min(source.size) < 16:
                raise CollectionValidationError('写真の画素数が対応範囲外です（40メガ画素以内）。')
            fmt = source.format
            source.load()
            oriented = ImageOps.exif_transpose(source).convert('RGB')
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise CollectionValidationError('写真を読み込めませんでした。別のJPEG・PNG画像で試してください。') from exc
    clean = Image.new('RGB', oriented.size)
    clean.paste(oriented)
    width, height = clean.size
    clean.thumbnail((768, 768), Image.Resampling.LANCZOS)
    stream = BytesIO()
    clean.save(stream, format='PNG')
    preview = stream.getvalue()
    stat = ImageStat.Stat(clean.convert('L'))
    reasons = []
    if min(width, height) < 256:
        reasons.append('low_resolution')
    if stat.mean[0] < 35:
        reasons.append('dark_image')
    if stat.stddev[0] < 10:
        reasons.append('low_contrast')
    quality = dict(status='review_required', flags=reasons, width=width, height=height,
                   brightness_mean=stat.mean[0], brightness_std=stat.stddev[0],
                   pasta_mask_reviewed=False, physical_scale_validated=False,
                   limitation='Basic image checks only; overlap, scale, blur and label correctness need review.')
    return dict(original_sha256=sha256(image_bytes).hexdigest(),
                original_extension={'JPEG': '.jpg', 'PNG': '.png', 'WEBP': '.webp'}[fmt],
                preview_bytes=preview, preview_sha256=sha256(preview).hexdigest(), quality=quality)


def default_root():
    data = Path(os.environ.get('PASTA_DATA_DIR', str(Path(__file__).resolve().parent / 'data')))
    return Path(os.environ.get('PASTA_RESEARCH_DATA_DIR', str(data / 'research_collection'))).resolve()


def default_duplicate_roots():
    data = Path(os.environ.get('PASTA_DATA_DIR', str(Path(__file__).resolve().parent / 'data')))
    return [data / 'Pasta_Lens_Private_Dataset' / 'photos',
            data / 'intake_series_nov05' / 'originals',
            data / 'calibration_50_34_nov08' / 'originals',
            data / 'direct_weight_nov08' / 'originals']


class CollectionStore:
    def __init__(self, root=None, duplicate_roots=None):
        self.root = Path(root).resolve() if root is not None else default_root()
        self.registration_root = self.root
        self.db_path = self.root / 'collection.sqlite3'
        self.duplicate_roots = list(default_duplicate_roots() if duplicate_roots is None else map(Path, duplicate_roots))
        self.root.mkdir(parents=True, exist_ok=True)
        for folder in ('originals', 'previews', 'backups'):
            (self.root / folder).mkdir(exist_ok=True)
        self._initialize()

    @contextmanager
    def _connection(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA foreign_keys=ON')
        try:
            yield conn
        finally:
            conn.close()

    def backup(self):
        destination = self.root / 'backups' / ('collection_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.sqlite3')
        with self._connection() as source, sqlite3.connect(destination) as output:
            source.backup(output)
        return destination

    def _initialize(self):
        with self._connection() as conn:
            version = conn.execute('PRAGMA user_version').fetchone()[0]
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        if version == SCHEMA_VERSION:
            return
        if tables or version:
            self.backup()
            raise CollectionValidationError('研究用DBの形式が対応していません。バックアップを作成しました。既存DBは変更していません。')
        # Even an existing empty database is backed up before adding the schema.
        self.backup()
        with self._connection() as conn:
            conn.executescript('''
                BEGIN IMMEDIATE;
                CREATE TABLE IF NOT EXISTS samples(sample_id TEXT PRIMARY KEY, declaration TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS entries(
                    id TEXT PRIMARY KEY, original_sha256 TEXT NOT NULL UNIQUE,
                    preview_sha256 TEXT NOT NULL, original_key TEXT NOT NULL,
                    preview_key TEXT NOT NULL, metadata TEXT NOT NULL,
                    quality TEXT NOT NULL, sample_id TEXT NOT NULL REFERENCES samples(sample_id),
                    created_at TEXT NOT NULL, submission_token TEXT NOT NULL UNIQUE,
                    training_approved INTEGER NOT NULL DEFAULT 0 CHECK(training_approved=0));
                CREATE TABLE IF NOT EXISTS audit_events(
                    id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, kind TEXT NOT NULL, details TEXT NOT NULL);
                PRAGMA user_version=1;
                COMMIT;
            ''')

    def check_existing_duplicates(self, image_bytes):
        digest = sha256(image_bytes).hexdigest()
        matches = []
        for root in self.duplicate_roots:
            if not root.is_dir():
                continue
            for path in sorted(root.rglob('*')):
                if path.is_symlink() or not path.is_file() or path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.webp'}:
                    continue
                if path.stat().st_size == len(image_bytes) and sha256(path.read_bytes()).hexdigest() == digest:
                    matches.append(str(path))
        return matches

    @staticmethod
    def _record(row, status='stored'):
        return dict(id=row['id'], status=status, original_sha256=row['original_sha256'],
                    preview_sha256=row['preview_sha256'], metadata=json.loads(row['metadata']),
                    quality=json.loads(row['quality']), created_at=row['created_at'],
                    training_approved=False, lineage_status='unreviewed_declaration',
                    conservative_split_group='unresolved_physical_relationships',
                    original_key=row['original_key'], preview_key=row['preview_key'])

    def _safe_file(self, key):
        path = (self.root / key).resolve()
        if self.root not in path.parents:
            raise CollectionValidationError('保存先を確認できませんでした。')
        return path

    def _write_new(self, key, data, created):
        path = self._safe_file(key)
        try:
            with path.open('xb') as stream:
                created.append(path)
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError:
            if path.read_bytes() != data:
                raise CollectionValidationError('保存済み画像のハッシュ整合性を確認できませんでした。') from None

    def _commit(self, conn):
        conn.commit()

    def register(self, image_bytes, metadata, *, submission_token):
        metadata = validate_metadata(metadata)
        token = _text(submission_token, '送信ID', maximum=150)
        image = prepare_image(image_bytes)
        from collection_quality import inspect_photo
        diagnostic, _ = inspect_photo(image_bytes, reference=metadata['scale_reference']=='a4_markers', printed=metadata['scale_verified'], coplanar=metadata['reference_coplanar'])
        image['quality']['capture_diagnostic'] = diagnostic
        matches = self.check_existing_duplicates(image_bytes)
        if matches:
            raise ExistingDatasetDuplicateError(matches)
        canonical = _json(metadata)
        created = []
        with self._connection() as conn:
            try:
                conn.execute('BEGIN IMMEDIATE')
                by_token = conn.execute('SELECT * FROM entries WHERE submission_token=?', (token,)).fetchone()
                by_hash = conn.execute('SELECT * FROM entries WHERE original_sha256=?', (image['original_sha256'],)).fetchone()
                previous = by_token or by_hash
                if previous:
                    if previous['original_sha256'] == image['original_sha256'] and previous['metadata'] == canonical:
                        conn.rollback()
                        return self._record(previous, 'duplicate')
                    conn.execute('INSERT INTO audit_events(created_at,kind,details) VALUES(?,?,?)',
                                 (_now(), 'conflicting_submission', _json(dict(existing_id=previous['id'],
                                   original_sha256=image['original_sha256'], previous_metadata=json.loads(previous['metadata']),
                                   attempted_metadata=metadata))))
                    conn.commit()
                    raise DuplicateConflictError('登録済み写真または送信IDと内容が異なります。元データは変更せず、差分を監査履歴に保存しました。')
                declaration = {key: metadata[key] for key in SAMPLE_FIELDS}
                previous_sample = conn.execute('SELECT declaration FROM samples WHERE sample_id=?', (metadata['sample_id'],)).fetchone()
                if previous_sample and previous_sample['declaration'] != _json(declaration):
                    raise CollectionValidationError('同じ物理試料IDの重量・製品・系譜が異なります。量を追加した場合は新しい試料IDと追加元を指定してください。')
                for parent in metadata['parent_sample_ids']:
                    if not conn.execute('SELECT 1 FROM samples WHERE sample_id=?', (parent,)).fetchone():
                        raise CollectionValidationError('追加・再利用元の試料IDが未登録です。先に元の試料を登録してください。')
                if not previous_sample:
                    conn.execute('INSERT INTO samples VALUES(?,?)', (metadata['sample_id'], _json(declaration)))
                identifier = uuid.uuid4().hex
                original_key = 'originals/' + image['original_sha256'] + image['original_extension']
                preview_key = 'previews/' + image['preview_sha256'] + '.png'
                self._write_new(original_key, image_bytes, created)
                self._write_new(preview_key, image['preview_bytes'], created)
                conn.execute('''INSERT INTO entries(id,original_sha256,preview_sha256,original_key,preview_key,
                             metadata,quality,sample_id,created_at,submission_token) VALUES(?,?,?,?,?,?,?,?,?,?)''',
                             (identifier, image['original_sha256'], image['preview_sha256'], original_key,
                              preview_key, canonical, _json(image['quality']), metadata['sample_id'], _now(), token))
                conn.execute('INSERT INTO audit_events(created_at,kind,details) VALUES(?,?,?)',
                             (_now(), 'registered_unapproved', _json(dict(id=identifier, original_sha256=image['original_sha256']))))
                receipt = self._record(conn.execute('SELECT * FROM entries WHERE id=?', (identifier,)).fetchone(), 'registered')
                self._commit(conn)
                return receipt
            except Exception:
                conn.rollback()
                # A commit may succeed even if its acknowledgement fails.
                # Keep files whenever durable state is uncertain.
                try:
                    with self._connection() as check:
                        # A different original can share this clean preview.
                        # Hold the writer lock while checking each file so a
                        # concurrent registration cannot adopt it mid-cleanup.
                        check.execute('BEGIN IMMEDIATE')
                        for path in created:
                            key = path.relative_to(self.root).as_posix()
                            referenced = check.execute(
                                'SELECT 1 FROM entries WHERE original_key=? OR preview_key=? LIMIT 1',
                                (key, key)).fetchone()
                            if referenced is None:
                                path.unlink(missing_ok=True)
                        check.rollback()
                except sqlite3.Error:
                    pass
                raise

    def list_records(self, limit=30):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1 or limit > 10000:
            raise CollectionValidationError('表示件数は1〜10000件で指定してください。')
        with self._connection() as conn:
            return [self._record(row) for row in conn.execute('SELECT * FROM entries ORDER BY created_at DESC,id DESC LIMIT ?', (limit,))]

    def list_samples(self):
        with self._connection() as conn:
            return [json.loads(row['declaration']) for row in conn.execute('SELECT declaration FROM samples ORDER BY sample_id')]

    def read_preview(self, record_id):
        with self._connection() as conn:
            row = conn.execute('SELECT preview_key,preview_sha256 FROM entries WHERE id=?', (record_id,)).fetchone()
        if not row:
            raise CollectionValidationError('写真記録が見つかりません。')
        data = self._safe_file(row['preview_key']).read_bytes()
        if sha256(data).hexdigest() != row['preview_sha256']:
            raise CollectionValidationError('プレビューのハッシュが一致しません。原本からの確認が必要です。')
        return data

    def export_manifest(self):
        """Private inventory only. Deliberately not a ready-to-train manifest."""
        records = self.list_records(10000)
        return dict(schema_version=SCHEMA_VERSION, created_at=_now(), purpose='private_review_inventory',
                    training_approved=False, independent_evaluation_ready=False,
                    records=records, samples=self.list_samples(),
                    instruction='Review labels and complete physical lineage before using the independent split validator.')
