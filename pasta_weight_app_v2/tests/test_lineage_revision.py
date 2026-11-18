import copy
from hashlib import sha256
import json

import pytest

from pasta_weight_app_v2.experiments.lineage_protocol import validate_split
from pasta_weight_app_v2.experiments.lineage_revision import build_cumulative_revision, verify_revision


def test_confirmed_additions_are_bound_to_originals_and_cannot_cross_partitions(tmp_path):
    raw = tmp_path / 'originals'
    raw.mkdir()
    recorded = []
    for i, weight in enumerate((120, 120, 140, 140, 160, 160)):
        name = f'fixture{i}.jpg'
        content = f'unique provenance fixture {i}'.encode()
        (raw/name).write_bytes(content)
        recorded.append(dict(filename=name, file=name, image_sha256=sha256(content).hexdigest(),
                             weight_g_user_reported=weight))
    initial = json.dumps(dict(images_count=len(recorded), images=recorded)).encode()
    (raw/'manifest.json').write_bytes(initial)
    revision = build_cumulative_revision(raw, stage_weights_g=(120, 140, 160), series_id='fixture-series',
                                         confirmed_on='2026-11-18', evidence='Synthetic lineage test')
    assert (raw/'manifest.json').read_bytes() == initial
    assert revision['source_manifest_sha256'] == sha256(initial).hexdigest()
    assert revision['relations_to_other_batches'] == 'unknown'
    assert not revision['independent_evaluation_eligible'] and revision['assigned_split'] is None
    assert [len(stage['photos']) for stage in revision['stages']] == [2, 2, 2]
    correction = tmp_path/'correction.json'
    correction.write_text(json.dumps(revision), encoding='utf-8')
    assert verify_revision(correction, raw) == revision

    rows, splits = [], {}
    for index, (stage, split) in enumerate(zip(revision['stages'], ['train', 'validation', 'test'])):
        ident = f'image{index}'
        # Even a mistaken per-stage family ID cannot get around a shared ancestor.
        rows.append(dict(image_id=ident, original_id=ident, original_sha256=stage['photos'][0]['original_sha256'],
                         sample_id=ident, capture_group_id=ident, physical_family_id=ident,
                         parent_material_ids=[stage['material_id']], lineage_verified=True,
                         lineage_evidence='test-only declaration', lineage_confirmed_by='test fixture'))
        splits[ident] = split
    with pytest.raises(ValueError, match='Shared material_ancestor crosses splits'):
        validate_split(rows, splits, revision['material_parents_within_batch'])
    tampered = copy.deepcopy(revision)
    tampered['assigned_split'] = 'test'
    correction.write_text(json.dumps(tampered), encoding='utf-8')
    with pytest.raises(ValueError, match='Revision does not match'):
        verify_revision(correction, raw)
