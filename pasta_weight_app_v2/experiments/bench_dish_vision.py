"""Measure the deployed, frozen OpenCLIP model. Photo data stays on the machine."""

from argparse import ArgumentParser
from collections import Counter
import csv
from hashlib import file_digest
import json
from pathlib import Path
import platform
from statistics import median
from time import perf_counter

import cv2
import numpy as np

from ..local_dish_vision import CHECKPOINT_SHA256, LocalDishClassifier, MODEL
from ..pasta_vision import image_to_rgb
from .audit_finished_dish import CLASSES, audit_receipt
from .foundation_resources import device_sync, memory_report

ABSTAIN = '判別不能'
CONDITIONS = {
    'original': 'Original RGB as accepted by image_to_rgb',
    'brightness_60': 'RGB multiplied by 0.60',
    'brightness_130': 'RGB multiplied by 1.30, clipping at 255',
    'warm_cast': 'RGB gains 1.12, 1.00, 0.86',
    'cool_cast': 'RGB gains 0.86, 1.00, 1.12',
    'roll_15': '15 degrees in-plane with reflecting border',
    'perspective': 'Top edge narrowed by 12% for planar projection',
    'long_edge_224': 'Downsample to at most 224 px on the long edge',
    'long_edge_112': 'Downsample to at most 112 px on the long edge',
    'light_frame': '12% neutral light border on each side',
    'dark_frame': '12% neutral dark border on each side',
}


def transform(rgb, condition):
    if condition == 'original':
        return rgb
    gains = {
        'brightness_60': .60, 'brightness_130': 1.30,
        'warm_cast': (1.12, 1, .86), 'cool_cast': (.86, 1, 1.12),
    }
    if condition in gains:
        return np.clip(np.rint(rgb * np.float32(gains[condition])), 0, 255).astype(np.uint8)
    h, w = rgb.shape[:2]
    if condition.startswith('long_edge_'):
        ratio = min(1., int(condition.rsplit('_', 1)[1]) / max(h, w))
        return cv2.resize(rgb, (max(1, round(w * ratio)), max(1, round(h * ratio))),
                          interpolation=cv2.INTER_AREA)
    if condition == 'roll_15':
        matrix = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), 15, 1)
        return cv2.warpAffine(rgb, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    if condition == 'perspective':
        x, y = w - 1, h - 1
        corners = np.float32([(0, 0), (x, 0), (x, y), (0, y)])
        projected = np.float32([(.06 * x, 0), (.94 * x, 0), (x, y), (0, y)])
        matrix = cv2.getPerspectiveTransform(corners, projected)
        return cv2.warpPerspective(rgb, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    if condition in {'light_frame', 'dark_frame'}:
        dy, dx = round(.12 * h), round(.12 * w)
        level = 235 if condition == 'light_frame' else 32
        return cv2.copyMakeBorder(rgb, dy, dy, dx, dx, cv2.BORDER_CONSTANT, value=(level,) * 3)
    raise ValueError(f'Unknown transform {condition}')


def summarize(rows):
    total = len(rows)
    abstained = sum(row['prediction'] == ABSTAIN for row in rows)
    return {
        'photos': total,
        'correct': sum(row['truth'] == row['prediction'] for row in rows),
        'abstained': abstained,
        'false_reject_rate': abstained / total if total else None,
        'wrong_candidate': sum(row['prediction'] not in (row['truth'], ABSTAIN) for row in rows),
        'changed_from_original': sum(row['prediction'] != row['original'] for row in rows),
        'new_abstentions': sum(row['original'] != ABSTAIN and row['prediction'] == ABSTAIN for row in rows),
        'recovered_from_abstention': sum(row['original'] == ABSTAIN and row['truth'] == row['prediction']
                                         for row in rows),
    }


def _checked_controls(manifest, root, supported_hashes):
    with manifest.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'file', 'kind', 'sha256'}.issubset(reader.fieldnames or ()):
            raise ValueError('Controls need file, kind and sha256 columns')
        seen = set(supported_hashes)
        for row in reader:
            relative = Path(row['file'] or '.')
            kind = (row['kind'] or '').strip()
            path = (root / relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(root) or not path.is_file() or not kind:
                raise ValueError('Invalid control path or kind')
            with path.open('rb') as photo:
                digest = file_digest(photo, 'sha256').hexdigest()
            if digest != row['sha256'] or digest in seen:
                raise ValueError('Control hash mismatch or duplicate')
            seen.add(digest)
            yield relative.as_posix(), kind, path


def summarize_controls(counts):
    total = counts.total()
    abstained = counts[ABSTAIN]
    false_accepts = total - abstained
    return dict(photos=total, abstained=abstained, false_accepts=false_accepts,
                false_accept_rate=false_accepts / total if total else None,
                rejection_rate=abstained / total if total else None, candidates=dict(counts))


def evaluate_dataset(model, dataset, controls=None, control_root=None, *, original_only=False):
    receipt, _ = audit_receipt(dataset)
    conditions = {'original': CONDITIONS['original']} if original_only else CONDITIONS
    rows = {name: [] for name in conditions}
    supported_hashes = set()
    with (dataset / 'labels.csv').open(encoding='utf-8-sig', newline='') as stream:
        for item in csv.DictReader(stream):
            supported_hashes.add(item['training_image_sha256'])
            rgb = image_to_rgb(dataset / item['image_path'])
            truth = CLASSES[item['class_label']][1]
            original = model.predict(rgb).name or ABSTAIN
            for condition in conditions:
                prediction = (original if condition == 'original' else
                              (model.predict(transform(rgb, condition)).name or ABSTAIN))
                rows[condition].append(dict(file=item['image_path'], truth=truth,
                                            prediction=prediction, original=original))
    summaries = {}
    for condition, photos in rows.items():
        summaries[condition] = summarize(photos)
        summaries[condition]['by_class'] = {label: summarize([row for row in photos if row['truth'] == label])
                                             for _, label in CLASSES.values()}
    frame_disagreements = None if original_only else sum(
        a['prediction'] != b['prediction'] for a, b in zip(rows['light_frame'], rows['dark_frame'], strict=True))
    controls_by_kind = {}
    controls_rows = []
    if controls:
        for file, kind, path in _checked_controls(controls, control_root.resolve(strict=True), supported_hashes):
            prediction = model.predict(image_to_rgb(path)).name or ABSTAIN
            controls_rows.append(dict(file=file, kind=kind, prediction=prediction))
            counts = controls_by_kind.setdefault(kind, Counter())
            counts[prediction] += 1
    control_metrics = {kind: summarize_controls(counts) for kind, counts in controls_by_kind.items()}
    return dict(receipt=receipt, transforms=conditions, by_condition=summaries,
                frame_disagreements=frame_disagreements, photo_results=rows,
                controls=control_metrics, control_photo_results=controls_rows,
                all_controls=summarize_controls(Counter(row['prediction'] for row in controls_rows)))


def run(checkpoint, device='cpu', *, dataset=None, photo=None, controls=None, control_root=None, original_only=False):
    started = perf_counter()
    model = LocalDishClassifier(checkpoint, device)
    device_sync(model.torch, device)
    result = dict(model=MODEL, checkpoint_sha256=CHECKPOINT_SHA256, platform=platform.platform(),
                  torch_version=model.torch.__version__, cold_load_s=perf_counter()-started)
    if photo:
        started = perf_counter()
        rgb = image_to_rgb(photo)
        result['decode_s'] = perf_counter()-started
        prediction = None
        times = []
        for _ in range(6):
            started = perf_counter()
            prediction = model.predict(rgb).name or ABSTAIN
            device_sync(model.torch, device)
            times.append(perf_counter()-started)
        result.update(prediction=prediction, first_inference_s=times[0], warm_inference_median_s=median(times[1:]),
                      warm_inference_samples_s=times[1:])
    if dataset:
        result.update(evaluate_dataset(model, dataset, controls, control_root, original_only=original_only))
    result.update(memory_report(model.torch, device))
    return result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--dataset', type=Path)
    inputs.add_argument('--photo', type=Path)
    parser.add_argument('--controls', type=Path)
    parser.add_argument('--control-root', type=Path)
    parser.add_argument('--original-only', action='store_true', help='Compare rates without image transformations')
    args = parser.parse_args()
    if (args.controls is None) != (args.control_root is None) or (args.controls and not args.dataset):
        parser.error('--controls and --control-root require --dataset')
    if args.original_only and not args.dataset:
        parser.error('--original-only requires --dataset')
    if args.output.exists():
        parser.error('Choose a new output file')
    try:
        result = run(args.checkpoint, args.device, dataset=args.dataset, photo=args.photo,
                     controls=args.controls, control_root=args.control_root, original_only=args.original_only)
        with args.output.open('x', encoding='utf-8') as out:
            json.dump(result, out, ensure_ascii=False, indent=2)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
