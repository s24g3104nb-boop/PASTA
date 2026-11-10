"""Validate a future, explicitly reviewed split without assigning current images.

This checks a declared lineage graph. It cannot establish whether the physical
strands were actually disjoint or whether the submitted evidence is truthful.
No files, labels, approval flags, models or database records are modified.
"""
from collections.abc import Mapping, Sequence
import hashlib
import json
import re

PROTOCOL_VERSION = 'physical-lineage-split-v1'
SPLITS = frozenset({'train', 'validation', 'test'})
_UNKNOWN = frozenset({'', 'none', 'null', 'n/a', 'na', 'unassigned', '未確認', '不明', '確認待ち'})
_IDENTIFIERS = ('image_id', 'original_id', 'sample_id', 'capture_group_id', 'physical_family_id')


def _identifier(value, field):
    if not isinstance(value, str) or value != value.strip() or value.lower() in _UNKNOWN:
        raise ValueError(f'{field}: explicit nonempty value required')
    if value.lower().startswith(('unknown', 'unconfirmed', 'pending')):
        raise ValueError(f'{field}: unresolved placeholder is not evidence of independence')
    return value


def _ids(value, field, *, allow_empty=False):
    if not isinstance(value, (list, tuple)) or (not value and not allow_empty):
        raise ValueError(f'{field}: explicit ID list required')
    result = [_identifier(item, field) for item in value]
    if len(set(result)) != len(result):
        raise ValueError(f'{field}: duplicate IDs')
    return result


def _material_ancestors(material_parents):
    if not isinstance(material_parents, Mapping) or not material_parents:
        raise ValueError('Complete explicit material_parents graph required')
    graph = {_identifier(key, 'material ID'): _ids(value, 'material parents', allow_empty=True)
             for key, value in material_parents.items()}
    missing = {parent for parents in graph.values() for parent in parents} - graph.keys()
    if missing:
        raise ValueError('Unspecified material ancestor: ' + ', '.join(sorted(missing)))
    state, ancestors = {}, {}

    def visit(node):
        if state.get(node) == 'visiting':
            raise ValueError('Material ancestry contains a cycle')
        if node in ancestors:
            return ancestors[node]
        state[node] = 'visiting'
        result = {node}
        for parent in graph[node]:
            result.update(visit(parent))
        state[node] = 'complete'
        ancestors[node] = result
        return result

    for node in graph:
        visit(node)
    return graph, ancestors


def validate_split(records, assignments, material_parents=None):
    """Reject unknown lineage and any declared sharing across the three splits.

    ``records`` is a nonempty sequence of JSON-compatible mappings. Required:
    image_id, original_id, original_sha256, sample_id, capture_group_id,
    physical_family_id, parent_material_ids (nonempty list), lineage_verified
    (the bool True), lineage_evidence, and lineage_confirmed_by.

    ``assignments`` must map every image_id, and no extra IDs, to train,
    validation or test. All three partitions must be present. An augmented
    image has its own image_id but keeps the source original_id/original hash,
    sample, capture and material references.

    ``material_parents`` must explicitly define every material node and all its
    ancestors; an empty list declares a reviewed root with no shared ancestor.
    Missing nodes are never inferred to be independent roots. Product/package
    labels alone do not establish which actual strands were shared.
    """
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)) or not records:
        raise ValueError('Nonempty record sequence required')
    if not isinstance(assignments, Mapping):
        raise ValueError('Explicit assignment mapping required')
    graph, ancestors = _material_ancestors(material_parents)
    checked, by_id, original_hashes = [], {}, {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError('Each record must be a mapping')
        row = {key: _identifier(record.get(key), key) for key in _IDENTIFIERS}
        if row['image_id'] in by_id:
            raise ValueError('Duplicate image_id: ' + row['image_id'])
        if record.get('lineage_verified') is not True:
            raise ValueError('lineage_verified must be explicitly True after human confirmation')
        _identifier(record.get('lineage_evidence'), 'lineage_evidence')
        _identifier(record.get('lineage_confirmed_by'), 'lineage_confirmed_by')
        digest = record.get('original_sha256')
        if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', digest):
            raise ValueError('original_sha256 must identify the original bytes')
        row['original_sha256'] = digest.lower()
        previous = original_hashes.setdefault(row['original_id'], row['original_sha256'])
        if previous != row['original_sha256']:
            raise ValueError('One original_id refers to different original hashes')
        parents = _ids(record.get('parent_material_ids'), 'parent_material_ids')
        missing = set(parents) - ancestors.keys()
        if missing:
            raise ValueError('Record has unspecified material nodes: ' + ', '.join(sorted(missing)))
        row['material_ancestors'] = set().union(*(ancestors[parent] for parent in parents))
        by_id[row['image_id']] = row
        checked.append(row)
    if set(assignments) != set(by_id):
        raise ValueError('Assignments must match image IDs exactly; missing/extra assignments are forbidden')
    if any(not isinstance(value, str) or value not in SPLITS for value in assignments.values()):
        raise ValueError('Allowed splits are train, validation and test')
    if set(assignments.values()) != SPLITS:
        raise ValueError('All train/validation/test partitions must be present')

    # Connected components are only declared groups, never measured evidence.
    parent = {row['image_id']: row['image_id'] for row in checked}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    seen = {}
    group_fields = ('original_id', 'original_sha256', 'sample_id', 'capture_group_id', 'physical_family_id')
    for row in checked:
        image_id = row['image_id']
        tokens = [(key, row[key]) for key in group_fields]
        tokens.extend(('material_ancestor', key) for key in row['material_ancestors'])
        for token in tokens:
            if token in seen:
                other = seen[token]
                if assignments[other] != assignments[image_id]:
                    raise ValueError(f'Shared {token[0]} crosses splits: {other}, {image_id}')
                parent[find(image_id)] = find(other)
            else:
                seen[token] = image_id
    groups = {}
    for image_id in by_id:
        groups.setdefault(find(image_id), []).append(image_id)
    # Include complete records, so changing labels or evidence also changes the
    # manifest fingerprint. It is not a signature or a human approval itself.
    try:
        canonical = json.dumps(dict(protocol=PROTOCOL_VERSION,
                                    records=sorted((dict(record) for record in records), key=lambda row: row['image_id']),
                                    assignments=dict(assignments), material_parents=graph),
                               ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError('Records and metadata must be finite JSON-compatible values') from exc
    return dict(protocol_version=PROTOCOL_VERSION, status='consistent_declared_split',
                image_records=len(checked), distinct_originals=len({row['original_sha256'] for row in checked}),
                declared_disjoint_groups=len(groups),
                groups_by_split={split: sum(assignments[items[0]] == split for items in groups.values()) for split in sorted(SPLITS)},
                split_manifest_sha256=hashlib.sha256(canonical.encode('utf-8')).hexdigest(),
                independence_verified_by_code=False,
                limitation='Checks declared sharing and evidence fields; does not verify the physical history or approve labels.')
