"""Offline supervised head training and session-separated evaluation."""
import argparse
import csv
from dataclasses import dataclass
from hashlib import file_digest
import json
from pathlib import Path
import re

import numpy as np

from ..dish_classifier import EMBEDDING_SIZE, LinearDishHead, STANDARD_DISH_NAMES
from ..local_dish_vision import CHECKPOINT_SHA256, LocalDishClassifier
from ..pasta_vision import image_to_rgb

SPLITS = ('train', 'validation', 'test')
OUT_OF_SCOPE = '対象外'
ABSTAIN = '判別不能'
FIELDS = {'file', 'label', 'session_id', 'sha256', 'split'}


@dataclass(frozen=True)
class Example:
    path: Path
    label: str
    session_id: str
    split: str


def read_splits(manifest, photos):
    """Verify original bytes and require explicit cooking-session splits."""
    root = Path(photos).resolve(strict=True)
    sets = {name: [] for name in SPLITS}
    assigned = {}
    hashes = set()
    with Path(manifest).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not FIELDS.issubset(reader.fieldnames or ()):
            raise ValueError('CSV に file,label,session_id,sha256,split の列が必要です。')
        for line, row in enumerate(reader, start=2):
            file, label, session, split, digest = (str(row.get(field) or '').strip() for field in
                                                   ('file', 'label', 'session_id', 'split', 'sha256'))
            if not session or split not in sets or label not in (*STANDARD_DISH_NAMES, OUT_OF_SCOPE):
                raise ValueError(f'台帳 {line} 行目の料理・調理回・分割を確認してください。')
            if split == 'train' and label == OUT_OF_SCOPE:
                raise ValueError(f'台帳 {line} 行目: 対象外の写真は検証または評価に使用してください。')
            if assigned.setdefault(session, split) != split:
                raise ValueError(f'台帳 {line} 行目: 同一の調理回が複数の分割に含まれています。')
            relative = Path(file)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError(f'台帳 {line} 行目: 画像は写真フォルダからの相対パスで指定してください。')
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f'台帳 {line} 行目の画像を開けません。')
            if not re.fullmatch(r'[0-9a-f]{64}', digest) or digest in hashes:
                raise ValueError(f'台帳 {line} 行目: SHA-256 が不正か、同一画像が重複しています。')
            with path.open('rb') as image:
                if file_digest(image, 'sha256').hexdigest() != digest:
                    raise ValueError(f'台帳 {line} 行目の画像の SHA-256 が一致しません。')
            hashes.add(digest)
            sets[split].append(Example(path, label, session, split))
    for split, examples in sets.items():
        if not set(STANDARD_DISH_NAMES).issubset({item.label for item in examples}):
            raise ValueError(f'{split} に3種類すべての写真が必要です。確認できた別々の調理回を割り当ててください。')
    return sets


def fit_head(embeddings, labels):
    """Optimize only the three-class head, weighting each cooking group upstream."""
    import torch

    features = np.asarray(embeddings, dtype=np.float32)
    if features.shape != (len(labels), EMBEDDING_SIZE) or not np.isfinite(features).all():
        raise ValueError('Invalid image embeddings')
    targets = torch.tensor([STANDARD_DISH_NAMES.index(label) for label in labels])
    torch.manual_seed(42)
    torch.set_num_threads(1)
    head = torch.nn.Linear(EMBEDDING_SIZE, len(STANDARD_DISH_NAMES))
    optimizer = torch.optim.AdamW(head.parameters(), lr=.02, weight_decay=.01)
    inputs = torch.from_numpy(features)
    # Fixed before validation/test; testing does not select hyperparameters.
    for _ in range(100):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(head(inputs), targets)
        loss.backward()
        optimizer.step()
    return LinearDishHead(head.weight.detach().numpy(), head.bias.detach().numpy())


def evaluate(examples, predictions):
    if len(examples) != len(predictions) or not examples:
        raise ValueError('Each photo must have a prediction, including abstentions')
    truth_order = [*STANDARD_DISH_NAMES, OUT_OF_SCOPE]
    prediction_order = [*STANDARD_DISH_NAMES, ABSTAIN]
    matrix = {actual: dict.fromkeys(prediction_order, 0) for actual in truth_order}
    groups = {}
    for item, predicted in zip(examples, predictions):
        prediction = predicted or ABSTAIN
        matrix[item.label][prediction] += 1
        correct = predicted == (None if item.label == OUT_OF_SCOPE else item.label)
        groups.setdefault(item.session_id, []).append(correct)

    def ratio(n, d):
        return n / d if d else None

    per_class = {}
    for label in STANDARD_DISH_NAMES:
        correct = matrix[label][label]
        support = sum(matrix[label].values())
        proposed = sum(row[label] for row in matrix.values())
        per_class[label] = dict(support=support, correct=correct, class_accuracy=ratio(correct, support),
                                precision=ratio(correct, proposed), recall=ratio(correct, support),
                                f1=ratio(2 * correct, support + proposed))
    total_correct = sum(sum(values) for values in groups.values())
    out_of_scope = sum(matrix[OUT_OF_SCOPE].values())
    return dict(photos=len(examples), sessions=len(groups), overall_accuracy=total_correct/len(examples),
                mean_session_accuracy=float(np.mean([np.mean(values) for values in groups.values()])),
                coverage=sum(name is not None for name in predictions)/len(examples),
                out_of_scope_rejection=ratio(matrix[OUT_OF_SCOPE][ABSTAIN], out_of_scope),
                per_class=per_class, truth_order=truth_order, prediction_order=prediction_order,
                confusion_matrix=[[matrix[row][col] for col in prediction_order] for row in truth_order])


def run(manifest, photos, checkpoint, output, device='cpu'):
    sets = read_splits(manifest, photos)
    destination = Path(output)
    if destination.exists():
        raise FileExistsError('出力先は新しいフォルダを指定してください。')
    model = LocalDishClassifier(checkpoint, device)
    evidence = {}
    for split, examples in sets.items():
        evidence[split] = [model.image_evidence(image_to_rgb(item.path)) for item in examples]
    # Average repeated views within each cooking session and class before fitting.
    # This prevents a long series of near-identical shots dominating the conversion.
    views = {}
    for item, (embedding, _) in zip(sets['train'], evidence['train']):
        views.setdefault((item.session_id, item.label), []).append(embedding)
    head = fit_head([np.mean(items, axis=0) for items in views.values()], [key[1] for key in views])
    reports = {}
    for split, examples in sets.items():
        features = evidence[split]
        reports[split] = dict(
            zero_shot=evaluate(examples, [suggestion.name for _, suggestion in features]),
            linear_head=evaluate(examples, [head.select(embedding, suggestion.name) for embedding, suggestion in features]))
    destination.mkdir(parents=True, exist_ok=False)
    for name, payload in [('head.json', head.artifact(CHECKPOINT_SHA256)), ('metrics.json', reports)]:
        (destination / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return reports


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'photos', 'checkpoint', 'output'):
        cli.add_argument('--'+name, required=True, type=Path)
    cli.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = cli.parse_args()
    try:
        run(args.manifest, args.photos, args.checkpoint, args.output, args.device)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        cli.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
