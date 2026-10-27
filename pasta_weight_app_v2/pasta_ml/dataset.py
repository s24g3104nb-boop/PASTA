from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

REQUIRED = ('image_path', 'batch_id', 'session_id', 'product_id', 'mode',
            'actual_weight_g', 'source')
MAX_PIXELS = 30_000_000
MAX_BYTES = 30_000_000


def normalized_rgb(image):
    image = ImageOps.exif_transpose(image)
    if 'A' in image.getbands() or 'transparency' in image.info:
        rgba = image.convert('RGBA')
        image = Image.alpha_composite(Image.new('RGBA', rgba.size, (255, 255, 255, 255)), rgba)
    return np.asarray(image.convert('RGB')).copy()


@dataclass
class Sample:
    values: dict
    path: Path
    pixel_hash: str
    perceptual_hash: int

    @property
    def weight(self):
        return float(self.values['actual_weight_g'])

    @property
    def pair(self):
        return self.values['mode'] + ':' + self.values['product_id']


def image_rgb(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise ValueError(f'画像が存在しないか30MBを超えています: {path.name}')
    with warnings.catch_warnings():
        warnings.simplefilter('error', Image.DecompressionBombWarning)
        with Image.open(path) as im:
            if im.format not in ('JPEG', 'PNG'):
                raise ValueError('JPEG/PNGの画像が必要です')
            if im.width * im.height > MAX_PIXELS or min(im.size) < 16:
                raise ValueError('画像寸法は各辺16px以上・合計3000万画素以下が必要です')
            im.load()
            return normalized_rgb(im)


def hashes(rgb):
    digest = hashlib.sha256(str(rgb.shape).encode() + rgb.tobytes()).hexdigest()
    small = np.asarray(Image.fromarray(rgb).convert('L').resize((9, 8)))
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    dhash = sum(int(b) << i for i, b in enumerate(bits))
    return digest, dhash


def load_dataset(csv_path, image_root=None):
    csv_path = Path(csv_path)
    root = Path(image_root or csv_path.parent).resolve()
    samples = []
    with csv_path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError('CSVに一意の列名が必要です')
        missing = set(REQUIRED) - set(reader.fieldnames)
        if missing:
            raise ValueError('CSV必須列が不足: ' + ', '.join(sorted(missing)))
        for line, values in enumerate(reader, 2):
            if None in values or any(v is None for v in values.values()):
                raise ValueError(f'CSV {line}行目: 列数が一致しません')
            values = {k: v.strip() for k, v in values.items()}
            try:
                if any(not values[k] for k in REQUIRED):
                    raise ValueError('必須値が空です')
                if values['mode'] == 'bundle':
                    values['mode'] = 'bundle_end'
                if values['mode'] not in ('spread', 'bundle_end'):
                    raise ValueError('modeはspread/bundle_endです（bundleも可）')
                if values['source'] not in ('real', 'synthetic'):
                    raise ValueError('sourceはreal/syntheticです')
                weight = float(values['actual_weight_g'])
                if not np.isfinite(weight) or weight <= 0:
                    raise ValueError('実測重量は有限の正数が必要です')
                if values.get('individual_strand_count'):
                    count = float(values['individual_strand_count'])
                    if not np.isfinite(count) or count <= 0 or not count.is_integer():
                        raise ValueError('本数は正の整数です')
                relative = Path(values['image_path'])
                if relative.is_absolute() or '..' in relative.parts or '\\' in values['image_path'] or ':' in values['image_path']:
                    raise ValueError('画像パスは画像ルート内の相対パスで指定してください')
                path = (root / relative).resolve()
                if not path.is_relative_to(root):
                    raise ValueError('画像パスがルート外を参照しています')
                rgb = image_rgb(path)
                digest, dhash = hashes(rgb)
                samples.append(Sample(values, path, digest, dhash))
            except Exception as exc:
                raise ValueError(f'CSV {line}行目: {exc}') from exc
    if not samples:
        raise ValueError('CSVにデータがありません。画像・実測ラベルを用意してください')
    seen = {}
    for s in samples:
        label = (s.weight, s.pair, s.values['source'])
        if s.pixel_hash in seen and seen[s.pixel_hash] != label:
            raise ValueError('同一画像に矛盾する重量・製品・モード・sourceがあります')
        seen[s.pixel_hash] = label
    return samples


def grouping(samples):
    """秤量・撮影セッション・同一/近似画像を推移的に同一グループにする。"""
    parent = list(range(len(samples)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(samples):
        for j in range(i):
            b = samples[j]
            same = any(a.values[k] == b.values[k] for k in ('batch_id', 'session_id'))
            near = (a.perceptual_hash ^ b.perceptual_hash).bit_count() <= 4
            if same or a.pixel_hash == b.pixel_hash or near:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(len(samples)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def validate_splits(samples):
    for s in samples:
        if s.values.get('split') not in ('train', 'val', 'test'):
            raise ValueError('全行にsplit=train/val/testが必要です。先にsplitを実行してください')
    if {s.values['split'] for s in samples} != {'train', 'val', 'test'}:
        raise ValueError('train/val/testはそれぞれ空でない必要があります')
    for group in grouping(samples):
        if len({samples[i].values['split'] for i in group}) != 1:
            raise ValueError('分割リーク: 同一秤量・撮影セッション・同一/近似画像が分割を跨いでいます')


def write_csv(samples, output, root=None):
    output = Path(output)
    if output.exists():
        raise ValueError('出力先が既に存在します。新しいパスを指定してください')
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in samples:
        row = dict(s.values)
        # Split files may be placed elsewhere; paths remain relative to output CSV.
        try:
            row['image_path'] = s.path.relative_to(Path(root or output.parent).resolve()).as_posix()
        except ValueError as exc:
            raise ValueError('出力CSVは画像ルートに置くか --image-root を指定してください') from exc
        rows.append(row)
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with output.open('x', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def split_dataset(samples, seed=42):
    if any(s.values.get('split') for s in samples):
        raise ValueError('既存の分割を作り直しません。元の未分割CSVを使用してください')
    groups = grouping(samples)
    if len(groups) < 3:
        raise ValueError('独立した秤量/撮影/画像グループが3つ以上必要です。連写や近似画像は同じグループです')
    rng = random.Random(seed)
    rng.shuffle(groups)
    holdout = max(1, round(len(groups) * .15))
    train_n = len(groups) - 2 * holdout
    for n, group in enumerate(groups):
        split = 'train' if n < train_n else ('val' if n < train_n + holdout else 'test')
        for i in group:
            samples[i].values['split'] = split
    validate_splits(samples)
    return samples


def dataset_hash(samples):
    content = [(s.pixel_hash, s.values) for s in samples]
    return hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def import_dataset(samples, destination):
    destination = Path(destination)
    if destination.exists():
        raise ValueError('取込先は存在しない新しいディレクトリを指定してください')
    destination.mkdir(parents=True)
    try:
        images = destination / 'images'
        images.mkdir()
        imported = []
        for s in samples:
            rgb = image_rgb(s.path)
            path = images / (s.pixel_hash + '.png')
            if not path.exists():
                # Re-encoding avoids copying location/device EXIF into managed images.
                Image.fromarray(rgb).save(path)
            values = dict(s.values, image_path='images/' + path.name)
            imported.append(Sample(values, path.resolve(), s.pixel_hash, s.perceptual_hash))
        write_csv(imported, destination / 'dataset.csv')
    except Exception:
        shutil.rmtree(destination)
        raise
    return destination / 'dataset.csv'
