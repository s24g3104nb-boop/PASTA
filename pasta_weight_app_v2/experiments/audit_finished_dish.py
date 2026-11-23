"""Check the private finished-dish receipt and prepare a reviewed training manifest."""
from argparse import ArgumentParser
from collections import Counter
import csv
from hashlib import file_digest
import json
from pathlib import Path
import re

from PIL import Image

from .train_finished_dish import SPLITS

CLASSES = {
    'carbonara': (0, 'カルボナーラ'),
    'genovese': (1, 'ジェノベーゼ'),
    'tomato_pasta': (2, 'トマトパスタ'),
}


def _read(path, fields):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not set(fields).issubset(reader.fieldnames or ()):
            raise ValueError(f'{path.name}: 必須列がありません。')
        return list(reader)


def _indexed(rows, key):
    index = {row[key]: row for row in rows}
    if len(index) != len(rows) or '' in index or None in index:
        raise ValueError(f'{key}: 空欄か重複があります。')
    return index


def audit_receipt(root, review=None):
    root = Path(root).resolve(strict=True)
    rows = _read(root/'labels.csv', ('image_path', 'class_label', 'class_id', 'original_sha256',
                                    'training_image_sha256', 'width_px', 'height_px', 'converted_from_heic'))
    photos = _indexed(rows, 'image_path')
    reference = _indexed(_read(root/'split_manifest.csv', ('path', 'sha256', 'bytes')), 'path')
    visual = _indexed(_read(Path(review), ('file', 'group', 'split', 'group_status')), 'file') if review else {}
    if review and set(photos) != set(visual):
        raise ValueError('目視レビューのファイルとラベル台帳が一致しません。')
    known_sha = set()
    groups = {}
    counts = Counter()
    converted = 0
    manifest = []
    for file, row in photos.items():
        member = Path(file)
        target = (root/member).resolve()
        info = CLASSES.get(row['class_label'])
        if not (info and row['class_id'] == str(info[0]) and member.parts[:2] == ('images', row['class_label'])
                and len(member.parts) == 3 and target.is_relative_to(root/'images')):
            raise ValueError(f'{file}: 写真のパスかクラスIDが不一致です。')
        declared = reference.get(file)
        if declared is None or target.stat().st_size != int(declared['bytes']):
            raise ValueError(f'{file}: 分割台帳のファイルサイズと一致しません。')
        with target.open('rb') as stream:
            digest = file_digest(stream, 'sha256').hexdigest()
        if digest != row['training_image_sha256'] or digest != declared['sha256'] or digest in known_sha:
            raise ValueError(f'{file}: SHA-256 不一致か完全重複です。')
        known_sha.add(digest)
        original = row['original_sha256']
        conversion = row['converted_from_heic']
        if conversion not in ('0', '1') or not re.fullmatch(r'[0-9a-f]{64}', original):
            raise ValueError(f'{file}: 元画像のハッシュか変換フラグが不正です。')
        if conversion == '0' and original != digest:
            raise ValueError(f'{file}: 未変換の画像が元画像の SHA-256 と一致しません。')
        with Image.open(target) as image:
            if image.size != (int(row['width_px']), int(row['height_px'])):
                raise ValueError(f'{file}: 縦横のサイズが一致しません。')
            image.load()
        counts[info[1]] += 1
        converted += conversion == '1'
        if review:
            group, split, status = (visual[file][field].strip() for field in ('group', 'split', 'group_status'))
            if not group or split not in SPLITS or status not in ('confirmed', 'inferred'):
                raise ValueError(f'{file}: 目視レビューの群、分割か確度が不正です。')
            if groups.setdefault(group, split) != split:
                raise ValueError(f'{file}: 同じ目視レビューの群が複数の分割に含まれています。')
            manifest.append(dict(file=file, label=info[1], sha256=digest, session_id=group, split=split,
                                 group_status=status))
    if set(counts) != {name for _, name in CLASSES.values()}:
        raise ValueError('ラベル台帳に3クラスすべての写真が必要です。')
    report = dict(photos=len(photos), by_class=dict(counts), verified_training_hashes=len(known_sha),
                  verified_unconverted_original_hashes=len(photos)-converted,
                  conversion_originals_not_checked=converted)
    if review:
        report['group_status'] = ('inferred' if any(row['group_status'].strip() == 'inferred'
                                                  for row in visual.values()) else 'confirmed')
        report['groups_by_split'] = dict(Counter(groups.values()))
    return report, manifest


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--review', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        report, manifest = audit_receipt(args.dataset, args.review)
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output/'receipt.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        if manifest:
            with (args.output/'split.csv').open('x', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
                writer.writeheader()
                writer.writerows(manifest)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
