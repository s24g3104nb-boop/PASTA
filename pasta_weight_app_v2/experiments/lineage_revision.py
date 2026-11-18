"""Record a user-confirmed cumulative series beside an immutable photo batch."""
from __future__ import annotations

import argparse
from datetime import date
from hashlib import sha256
import json
import math
from pathlib import Path

from tools.diagnose_photo_manifest import manifest_records, photo_spec

VERSION = 'cumulative-material-revision-v1'


def build_cumulative_revision(batch, *, stage_weights_g, series_id, confirmed_on, evidence):
    """Bind a confirmation to the submitted manifest, labels, and original bytes.

    The weights describe the order of material additions. They preserve the
    supplied labels and do not attest to an exact scale reading or relationships
    outside this batch. The resulting records are not assigned to a split.
    """
    if (not isinstance(series_id, str) or not series_id.strip() or not isinstance(evidence, str)
            or not evidence.strip()):
        raise ValueError('Series ID and user confirmation are required')
    confirmed_on = date.fromisoformat(confirmed_on).isoformat()
    weights = tuple(stage_weights_g)
    if (len(weights) < 2 or any(type(w) not in (int, float) or not math.isfinite(w) or w <= 0 for w in weights)
            or any(b <= a for a, b in zip(weights, weights[1:]))):
        raise ValueError('Supply two or more increasing user-confirmed stage labels')
    root = Path(batch).resolve()
    manifest = (root / 'manifest.json').read_bytes()
    groups = {weight: [] for weight in weights}
    seen = set()
    for item in manifest_records(json.loads(manifest.decode('utf-8-sig'))):
        relative, weight, count = photo_spec(item)
        path = root.joinpath(*relative.parts)
        if count is not None or weight not in groups:
            raise ValueError('The supplied labels do not match the confirmed cumulative stages')
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError('Original must be inside the submitted batch')
        digest = sha256(path.read_bytes()).hexdigest()
        if digest != item['sha256'] or digest in seen:
            raise ValueError('Original SHA-256 mismatch or duplicate')
        seen.add(digest)
        groups[weight].append(dict(filename=item['filename'], original_sha256=digest,
                                   submitted_label=item.get('label', dict(weight_g=weight))))
    if not all(groups.values()):
        raise ValueError('Every stage must have a submitted original')
    stages, material_parents = [], {}
    previous = None
    for weight in weights:
        material = f'{series_id}/{weight:g}g'
        material_parents[material] = [previous] if previous else []
        stages.append(dict(nominal_weight_g=weight, material_id=material, previous_material_id=previous,
                           relation='added' if previous else 'start_of_confirmed_series', photos=groups[weight]))
        previous = material
    return dict(version=VERSION, source_manifest_sha256=sha256(manifest).hexdigest(),
                prior_status='sharing_unresolved', new_status='cumulative_sharing_confirmed_by_submitter',
                confirmed_on=confirmed_on, evidence=evidence, physical_family_id=series_id,
                relations_to_other_batches='unknown', stages=stages, material_parents_within_batch=material_parents,
                independent_evaluation_eligible=False, assigned_split=None)


def verify_revision(revision_path, batch):
    revision = json.loads(Path(revision_path).read_text(encoding='utf-8'))
    rebuilt = build_cumulative_revision(batch,
        stage_weights_g=[stage['nominal_weight_g'] for stage in revision['stages']],
        series_id=revision['physical_family_id'], confirmed_on=revision['confirmed_on'], evidence=revision['evidence'])
    if revision != rebuilt:
        raise ValueError('Revision does not match the immutable batch and cumulative contract')
    return revision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--weights', type=float, nargs='+', required=True)
    parser.add_argument('--series-id', required=True)
    parser.add_argument('--confirmed-on', required=True)
    parser.add_argument('--evidence', required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.batch.resolve()):
        raise ValueError('Save the revision outside the immutable originals folder')
    result = build_cumulative_revision(args.batch, stage_weights_g=args.weights, series_id=args.series_id,
                                      confirmed_on=args.confirmed_on, evidence=args.evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(sum(len(stage['photos']) for stage in result['stages']), 'originals in one cumulative series')


if __name__ == '__main__':
    main()
