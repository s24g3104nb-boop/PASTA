from pathlib import Path
import csv
from hashlib import sha256

from PIL import Image
import pytest

from pasta_weight_app_v2.experiments.audit_finished_dish import CLASSES, audit_receipt
from pasta_weight_app_v2.experiments.fine_tune_finished_dish import _weights
from pasta_weight_app_v2.experiments.train_finished_dish import Example


def write_rows(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fixture_receipt(tmp_path):
    labels, split, review = [], [], []
    for label, (idx, _) in CLASSES.items():
        for n in range(2):
            file = f'images/{label}/{n}.png'
            photo = tmp_path / file
            photo.parent.mkdir(exist_ok=True, parents=True)
            Image.new('RGB', (20, 15), (idx, n, 100)).save(photo)
            digest = sha256(photo.read_bytes()).hexdigest()
            flag = label == 'carbonara' and n == 0
            labels.append(dict(image_path=file, class_label=label, class_id=idx, width_px=20, height_px=15,
                               converted_from_heic=int(flag), original_sha256='1'*64 if flag else digest,
                               training_image_sha256=digest))
            split.append(dict(path=file, bytes=photo.stat().st_size, sha256=digest))
            review.append(dict(file=file, group=label, split='train', group_status='inferred'))
    write_rows(tmp_path/'labels.csv', labels)
    write_rows(tmp_path/'split_manifest.csv', split)
    write_rows(tmp_path/'review.csv', review)
    return review


def test_receipt_verifies_raw_and_training_bytes(tmp_path):
    review = fixture_receipt(tmp_path)
    report, manifest = audit_receipt(tmp_path, tmp_path/'review.csv')
    assert report['verified_training_hashes'] == 6
    assert report['verified_unconverted_original_hashes'] == 5
    assert report['conversion_originals_not_checked'] == 1
    assert report['group_status'] == 'inferred'
    assert len(manifest) == 6 and manifest[0]['label'] == 'カルボナーラ'
    assert all(row['group_status'] == 'inferred' for row in manifest)
    path = tmp_path/review[0]['file']
    contents = path.read_bytes()
    path.write_bytes(contents[:-1]+bytes([contents[-1] ^ 1]))
    with pytest.raises(ValueError, match='SHA-256'):
        audit_receipt(tmp_path)


def test_visual_group_cannot_be_split(tmp_path):
    review = fixture_receipt(tmp_path)
    review[1]['split'] = 'test'
    write_rows(tmp_path/'review.csv', review)
    with pytest.raises(ValueError, match='複数の分割'):
        audit_receipt(tmp_path, tmp_path/'review.csv')


def test_fine_tuning_weights_bound_repeated_images_per_class():
    items = [Example(Path('same'), 'カルボナーラ', 'repeated', 'train')] * 9
    items += [Example(Path('other'), 'カルボナーラ', 'second', 'train'),
              Example(Path('tomato'), 'トマトパスタ', 'single', 'train')]
    weights = _weights(items)
    assert sum(weights[:9]) == pytest.approx(weights[9])
    assert sum(weights[:10]) == pytest.approx(weights[-1])
    assert sum(weights) == pytest.approx(len(items))
