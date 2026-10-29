"""Separate cooked-photo decoding and private storage; no image nutrition inference."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
from pathlib import Path
import re
from uuid import uuid4
from PIL import Image, UnidentifiedImageError
from pasta_vision import image_to_rgb

MAX_UPLOAD_BYTES = 25 * 1024 * 1024


@dataclass(frozen=True)
class PreparedMealPhoto:
    content: bytes
    sha256: str
    width: int
    height: int


def prepare_photo(upload) -> PreparedMealPhoto:
    """Validate bytes and produce bounded RGB PNG without camera EXIF or GPS."""
    if isinstance(upload, (bytes, bytearray)):
        payload = bytes(upload)
    elif hasattr(upload, 'getvalue'):
        payload = upload.getvalue()
    elif isinstance(upload, Image.Image):
        buffer = BytesIO()
        upload.save(buffer, 'PNG')
        payload = buffer.getvalue()
    else:
        raise ValueError('JPEG・PNG・WebPの写真を選び直してください。')
    if not payload or len(payload) > MAX_UPLOAD_BYTES:
        raise ValueError('写真は25MiB以下の画像を選んでください。')
    try:
        with Image.open(BytesIO(payload)) as probe:
            if probe.format not in ('JPEG', 'PNG', 'WEBP'):
                raise ValueError('JPEG・PNG・WebPの写真を選び直してください。')
            if probe.width * probe.height > 25_000_000:
                raise ValueError('写真は2500万画素以下にしてください。')
            probe.verify()
        rgb = image_to_rgb(BytesIO(payload), max_side=1600)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError('写真を読み込めません。別のJPEG・PNG・WebP画像を選んでください。') from exc
    buffer = BytesIO()
    Image.fromarray(rgb).save(buffer, 'PNG')
    content = buffer.getvalue()
    return PreparedMealPhoto(content, hashlib.sha256(content).hexdigest(), rgb.shape[1], rgb.shape[0])


def photo_path(key: str, db_path) -> Path:
    if not isinstance(key, str) or not re.fullmatch(r'[0-9a-f]{32}\.png', key):
        raise ValueError('写真の保存情報を確認できません。')
    root = Path(db_path).parent / 'meal_photos'
    target = root / key
    if target.resolve().parent != root.resolve() or target.is_symlink():
        raise ValueError('写真の保存情報を確認できません。')
    return target


def persist_photo(prepared: PreparedMealPhoto, db_path) -> str:
    """Exclusive create with partial-write cleanup; caller rolls back on DB failure."""
    key = uuid4().hex + '.png'
    target = photo_path(key, db_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open('xb') as stream:
            stream.write(prepared.content)
            stream.flush()
        return key
    except Exception:
        target.unlink(missing_ok=True)
        raise
