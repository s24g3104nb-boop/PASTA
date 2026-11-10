"""Offline receipt for one nominal-weight photo series; never approves learning data.

All generated files contain private metadata and belong under the ignored data
folder. Original archive members are preserved byte-for-byte. No app DB is opened.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
from io import BytesIO, StringIO
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import zipfile

from PIL import Image, ImageOps

WEIGHTS = {20, 40, 60, 80, 100}
MODES = {'spread_overlap', 'bundle_ends_low_angle', 'handheld_bundle_ends'}
MAX_MEMBER = 25 * 1024 * 1024
MAX_TOTAL = 128 * 1024 * 1024


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(name):
    if not name or '\\' in name or '\x00' in name or ':' in name:
        raise ValueError('Unsafe archive member path')
    parts = name.rstrip('/').split('/')
    if any(p in ('', '.', '..') for p in parts) or PurePosixPath(name).is_absolute():
        raise ValueError('Unsafe archive member path')
    return name.rstrip('/')


def _archive(zip_path):
    payloads = {}
    names = set()
    total = 0
    with zipfile.ZipFile(zip_path) as archive:
        if len(archive.infolist()) > 100:
            raise ValueError('Too many archive members')
        for info in archive.infolist():
            name = _safe_name(info.filename)
            if name.casefold() in names:
                raise ValueError('Duplicate archive member path')
            names.add(name.casefold())
            file_type = stat.S_IFMT(info.external_attr >> 16)
            if file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise ValueError('Archive links and special files are not accepted')
            if info.flag_bits & 1:
                raise ValueError('Encrypted archives are not accepted')
            total += info.file_size
            if info.file_size > MAX_MEMBER or total > MAX_TOTAL:
                raise ValueError('Archive exceeds size limit')
            if info.is_dir():
                continue
            suffix = PurePosixPath(name).suffix.lower()
            if suffix not in {'.jpeg', '.jpg', '.csv', '.txt', '.md'}:
                raise ValueError('Unexpected archive member type')
            payload = archive.read(info)  # verifies member CRC
            if len(payload) != info.file_size:
                raise ValueError('Archive size mismatch')
            payloads[name] = payload
    return payloads


def _validate(payloads):
    if 'labels.csv' not in payloads or 'SHA256SUMS.txt' not in payloads:
        raise ValueError('labels.csv and SHA256SUMS.txt are required')
    if not any('readme' in PurePosixPath(n).name.lower() for n in payloads):
        raise ValueError('README is required')
    rows = list(csv.DictReader(StringIO(payloads['labels.csv'].decode('utf-8-sig'))))
    required = {'image_path', 'original_filename', 'nominal_weight_g', 'measured_exact_g',
                'capture_mode', 'physical_family_group_id', 'image_width_height', 'sha256'}
    if len(rows) != 15 or any(not required.issubset(row) for row in rows):
        raise ValueError('Expected 15 complete label rows')
    sums = {}
    for line in payloads['SHA256SUMS.txt'].decode('utf-8-sig').splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r'([0-9a-fA-F]{64})\s+\*?(.+)', line)
        if not match or match[2] in sums:
            raise ValueError('Invalid or duplicate checksum entry')
        sums[_safe_name(match[2])] = match[1].lower()
    image_members = {n for n in payloads if PurePosixPath(n).suffix.lower() in {'.jpg', '.jpeg'}}
    paths = [r['image_path'] for r in rows]
    if len(set(paths)) != 15 or set(paths) != image_members or set(sums) != image_members:
        raise ValueError('Image, label and checksum inventories differ')
    families = {r['physical_family_group_id'] for r in rows}
    if len(families) != 1 or not next(iter(families)):
        raise ValueError('All photos must remain in one physical-family group')
    combinations = Counter()
    images = []
    for row in rows:
        name = _safe_name(row['image_path'])
        if PurePosixPath(name).name != row['original_filename']:
            raise ValueError('Filename and label disagree')
        try:
            weight = int(row['nominal_weight_g'])
        except (TypeError, ValueError):
            raise ValueError('Invalid nominal weight') from None
        if weight not in WEIGHTS or row['capture_mode'] not in MODES:
            raise ValueError('Unexpected weight or mode')
        combinations[(weight, row['capture_mode'])] += 1
        digest = _sha(payloads[name])
        if digest != row['sha256'].lower() or digest != sums[name]:
            raise ValueError('SHA-256 mismatch')
        with Image.open(BytesIO(payloads[name])) as probe:
            if probe.format != 'JPEG' or probe.width * probe.height > 25_000_000:
                raise ValueError('Expected bounded JPEG image')
            stored_size = list(probe.size)
            orientation = probe.getexif().get(274)
            probe.verify()
        with Image.open(BytesIO(payloads[name])) as decoded:
            decoded.load()
            oriented_size = list(ImageOps.exif_transpose(decoded).size)
        if row['image_width_height'] != f'{stored_size[0]}x{stored_size[1]}':
            raise ValueError('Label pixel dimensions mismatch')
        images.append(dict(
            image_path=name, original_path='originals/' + name,
            received_sha256=digest, supplied_sha256=row['sha256'].lower(),
            checksum_file_sha256=sums[name], stored_dimensions=stored_size,
            oriented_dimensions=oriented_size, exif_orientation=orientation,
            size_bytes=len(payloads[name]), nominal_weight_g=weight,
            measured_exact_g=None if not row['measured_exact_g'] else row['measured_exact_g'],
            label_status='user_confirmed_nominal_not_exact', capture_mode=row['capture_mode'],
            mode_status='provisional_requires_visual_review',
            physical_family_group_id=row['physical_family_group_id'],
            qc_status='PENDING_REVIEW', foreground_mask_status='NOT_VERIFIED',
            train_split='UNASSIGNED', training_eligible=False, supplied_metadata=row,
        ))
    if combinations != Counter({(weight, mode): 1 for weight in WEIGHTS for mode in MODES}):
        raise ValueError('Expected one image per weight and mode')
    return images


def _snapshot(root):
    files, symlinks = [], []
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            symlinks.append(dict(path=relative, target=str(path.readlink())))
        elif path.is_file():
            files.append(dict(path=relative, sha256=_file_sha(path), size_bytes=path.stat().st_size))
    return dict(root=str(root), files=files, symlinks=symlinks)


def receive(zip_path, existing_root, output):
    """Validate, hash existing data, then create a new private receipt directory.

    Caller must keep the original app stopped while using filesystem fingerprints
    to prove protection. This does not register or authorize supervised samples.
    """
    zip_path, existing_root, output = map(lambda p: Path(p).resolve(), (zip_path, existing_root, output))
    if output.exists():
        raise FileExistsError('Receipt output already exists; refusing to overwrite')
    if not existing_root.is_dir():
        raise ValueError('Existing data root must exist')
    payloads = _archive(zip_path)
    images = _validate(payloads)
    before = _snapshot(existing_root)
    by_hash = defaultdict(list)
    for item in before['files']:
        by_hash[item['sha256']].append(item['path'])
    received = defaultdict(list)
    for image in images:
        received[image['received_sha256']].append(image['image_path'])
    canonical = 'Pasta_Lens_Private_Dataset/photos/'
    for image in images:
        matches = by_hash.get(image['received_sha256'], [])
        image['existing_duplicate_matches'] = matches
        image['canonical_existing_duplicate_matches'] = [p for p in matches if p.startswith(canonical)]
        image['within_receipt_duplicate_matches'] = [p for p in received[image['received_sha256']] if p != image['image_path']]
    receipt = dict(
        schema_version=1, purpose='receipt_and_diagnosis_only',
        received_at=datetime.now(timezone.utc).isoformat(), archive_sha256=_file_sha(zip_path),
        archive_size_bytes=zip_path.stat().st_size,
        metadata_sha256={n: _sha(b) for n, b in payloads.items() if n not in {i['image_path'] for i in images}},
        image_count=len(images), unique_received_sha256_count=len(received),
        existing_file_count=len(before['files']),
        canonical_existing_image_count=sum(p['path'].startswith(canonical) for p in before['files']),
        exact_duplicate_images=sum(bool(i['existing_duplicate_matches']) for i in images),
        canonical_exact_duplicate_images=sum(bool(i['canonical_existing_duplicate_matches']) for i in images),
        physical_family_group_count=1, physical_family_group_id=images[0]['physical_family_group_id'],
        supervised_training_images=0, independent_evaluation_images=0,
        training_eligible=False, train_split='UNASSIGNED', images=images,
        notes=['Nominal user labels are not exact decimal scale readings.',
               'Checksum agreement proves consistency with supplied files, not camera provenance.',
               'Exact-byte duplicate check does not detect recompressed or cropped near-duplicates.',
               'QC, masks, calibration and supervised registration remain unapproved.'],
    )
    output.mkdir(parents=True, exist_ok=False)
    try:
        for name, payload in payloads.items():
            target = output / 'originals' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(payload)
        with (output / 'protection_before.json').open('x', encoding='utf-8') as stream:
            json.dump(before, stream, ensure_ascii=False, indent=2)
        with (output / 'receipt.json').open('x', encoding='utf-8') as stream:
            json.dump(receipt, stream, ensure_ascii=False, indent=2)
    except Exception:
        # Only this newly created output is ours to remove. Existing data untouched.
        shutil.rmtree(output)
        raise
    return receipt


def verify_protection(output):
    """Compare all preexisting regular files/symlinks; allow new receipt artifacts."""
    before = json.loads((Path(output) / 'protection_before.json').read_text(encoding='utf-8'))
    root = Path(before['root'])
    changed, missing = [], []
    for item in before['files']:
        path = root / item['path']
        if not path.is_file() or path.is_symlink():
            missing.append(item['path'])
        elif _file_sha(path) != item['sha256']:
            changed.append(item['path'])
    for item in before['symlinks']:
        path = root / item['path']
        if not path.is_symlink() or str(path.readlink()) != item['target']:
            changed.append(item['path'])
    return dict(existing_regular_files=len(before['files']), unchanged=not changed and not missing,
                changed=changed, missing=missing)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', required=True)
    parser.add_argument('--existing', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = receive(args.zip, args.existing, args.output)
    print(json.dumps({k: v for k, v in result.items() if k not in {'images', 'metadata_sha256'}}, ensure_ascii=False, indent=2))
