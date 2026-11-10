"""Artificial archives only; no private images, CSVs or DBs are loaded."""
import csv
from io import BytesIO, StringIO
import hashlib
import json
from pathlib import Path
import stat
import zipfile

from PIL import Image
import pytest
from pasta_weight_app_v2.experiments.intake_receipt import receive, verify_protection


def archive(tmp_path, *, mutate=None):
    payloads, rows = {}, []
    for wi, weight in enumerate((20, 40, 60, 80, 100)):
        for mi, mode in enumerate(('spread_overlap', 'bundle_ends_low_angle', 'handheld_bundle_ends')):
            name = f'images/{weight}/sample_{mi}.jpeg'
            buffer = BytesIO()
            Image.new('RGB', (32, 24), (wi * 40, mi * 60, 120)).save(buffer, 'JPEG')
            payloads[name] = buffer.getvalue()
            rows.append(dict(image_path=name, original_filename=Path(name).name,
                nominal_weight_g=str(weight), measured_exact_g='', capture_mode=mode,
                physical_family_group_id='TEST_SINGLE_FAMILY', image_width_height='32x24',
                sha256=hashlib.sha256(payloads[name]).hexdigest(), qc_status='PENDING_REVIEW'))
    if mutate:
        mutate(payloads, rows)
    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    payloads['labels.csv'] = buf.getvalue().encode()
    payloads['SHA256SUMS.txt'] = ''.join(r['sha256'] + '  ' + r['image_path'] + '\n' for r in rows).encode()
    payloads['README.md'] = b'Synthetic fixture, no real measurements.'
    path = tmp_path / 'input.zip'
    with zipfile.ZipFile(path, 'w') as z:
        for name, data in payloads.items():z.writestr(name, data)
    return path, payloads


def test_receipt_preserves_bytes_labels_and_single_group(tmp_path):
    z, data = archive(tmp_path)
    old = tmp_path / 'existing';old.mkdir()
    (old / 'database.sqlite3').write_bytes(b'Existing fixture bytes')
    receipt_path = old / 'new_receipt'
    result = receive(z, old, receipt_path)
    assert result['image_count'] == 15 and result['unique_received_sha256_count'] == 15
    assert result['training_eligible'] is False and result['independent_evaluation_images'] == 0
    assert result['physical_family_group_count'] == 1 and result['train_split'] == 'UNASSIGNED'
    for row in result['images']:
        assert (receipt_path / row['original_path']).read_bytes() == data[row['image_path']]
        assert row['measured_exact_g'] is None and row['qc_status'] == 'PENDING_REVIEW'
    assert verify_protection(receipt_path)['unchanged']
    (old / 'database.sqlite3').write_bytes(b'Changed')
    assert verify_protection(receipt_path)['changed'] == ['database.sqlite3']
    with pytest.raises(FileExistsError):receive(z, old, receipt_path)


def test_duplicate_search_includes_existing_canonical_and_internal(tmp_path):
    def repeat(payloads, rows):
        payloads[rows[1]['image_path']] = payloads[rows[0]['image_path']]
        rows[1]['sha256'] = rows[0]['sha256']
    z, data = archive(tmp_path, mutate=repeat)
    old = tmp_path / 'old';canonical = old / 'Pasta_Lens_Private_Dataset/photos'
    canonical.mkdir(parents=True)
    (canonical / 'older.jpeg').write_bytes(data['images/20/sample_0.jpeg'])
    result = receive(z, old, tmp_path / 'received')
    assert result['exact_duplicate_images'] == 2 and result['canonical_exact_duplicate_images'] == 2
    assert result['unique_received_sha256_count'] == 14
    assert result['images'][0]['within_receipt_duplicate_matches'] == ['images/20/sample_1.jpeg']


@pytest.mark.parametrize('change', ['hash', 'dimensions', 'family', 'missing'])
def test_invalid_metadata_writes_nothing(tmp_path, change):
    def mutate(payloads, rows):
        if change == 'hash':rows[0]['sha256'] = '0' * 64
        if change == 'dimensions':rows[0]['image_width_height'] = '12x24'
        if change == 'family':rows[0]['physical_family_group_id'] = 'INDEPENDENT_TEST_LIE'
        if change == 'missing':del payloads[rows[0]['image_path']]
    z, _ = archive(tmp_path, mutate=mutate)
    old = tmp_path / 'old';old.mkdir()
    output = tmp_path / 'received'
    with pytest.raises(ValueError):receive(z, old, output)
    assert not output.exists()


@pytest.mark.parametrize('name', ['../outside.jpeg', '/outside.jpeg', 'folder\\outside.jpeg'])
def test_archive_traversal_rejected(tmp_path, name):
    z, _ = archive(tmp_path)
    with zipfile.ZipFile(z, 'a') as archive_file:archive_file.writestr(name, b'bad')
    old = tmp_path / 'old';old.mkdir()
    with pytest.raises(ValueError):receive(z, old, tmp_path / 'received')
    assert not (tmp_path / 'received').exists()


def test_archive_symlink_rejected(tmp_path):
    z, _ = archive(tmp_path)
    link = zipfile.ZipInfo('link.jpeg');link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(z, 'a') as archive_file:archive_file.writestr(link, '../private')
    old = tmp_path / 'old';old.mkdir()
    with pytest.raises(ValueError):receive(z, old, tmp_path / 'received')
