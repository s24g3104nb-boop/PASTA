"""Synthetic metadata only: no private photo, DB or automatic split assignment."""
import copy
import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pasta_weight_app_v2.experiments.lineage_protocol import validate_split


def fixture():
    rows = [dict(image_id=f'image{i}', original_id=f'original{i}',
                 original_sha256=hashlib.sha256(f'original{i}'.encode()).hexdigest(),
                 sample_id=f'sample{i}', capture_group_id=f'capture{i}', physical_family_id=f'family{i}',
                 parent_material_ids=[f'portion{i}'], lineage_verified=True,
                 lineage_evidence='synthetic-confirmation-record', lineage_confirmed_by='synthetic-reviewer',
                 measured_weight_g=20*(i+1)) for i in range(3)]
    assignments = dict(zip((row['image_id'] for row in rows), ('train', 'validation', 'test')))
    graph = {f'portion{i}': [] for i in range(3)}
    return rows, assignments, graph


def test_valid_explicit_split_is_repeatable_and_does_not_approve_or_mutate():
    rows, assignments, graph = fixture()
    before = copy.deepcopy((rows, assignments, graph))
    result = validate_split(rows, assignments, graph)
    assert result == validate_split(list(reversed(rows)), assignments, graph)
    assert (rows, assignments, graph) == before
    assert result['groups_by_split'] == {'test': 1, 'train': 1, 'validation': 1}
    assert not result['independence_verified_by_code']
    assert result['distinct_originals'] == 3
    rows[0]['measured_weight_g'] = 21
    assert result['split_manifest_sha256'] != validate_split(rows, assignments, graph)['split_manifest_sha256']


@pytest.mark.parametrize('field,value', [
    ('lineage_verified', False), ('lineage_verified', 'True'),
    ('lineage_evidence', ''), ('lineage_evidence', 'pending_review'),
    ('lineage_confirmed_by', None), ('physical_family_id', 'UNKNOWN_RELATIONSHIP_ALL_WEIGHTS'),
    ('sample_id', ' sample0'), ('parent_material_ids', []),
])
def test_unknown_lineage_or_missing_confirmation_is_rejected(field, value):
    rows, assignments, graph = fixture(); rows[0][field] = value
    with pytest.raises(ValueError): validate_split(rows, assignments, graph)


@pytest.mark.parametrize('field', ['original_id', 'original_sha256', 'sample_id', 'capture_group_id', 'physical_family_id'])
def test_shared_declared_group_cannot_cross_splits(field):
    rows, assignments, graph = fixture(); rows[1][field] = rows[0][field]
    if field == 'original_id': rows[1]['original_sha256'] = rows[0]['original_sha256']
    with pytest.raises(ValueError, match='crosses splits'): validate_split(rows, assignments, graph)


def test_derived_images_remain_with_original_even_under_different_file_id():
    rows, assignments, graph = fixture()
    derived = dict(rows[0], image_id='image0-resized')
    rows.append(derived); assignments[derived['image_id']] = 'train'
    result = validate_split(rows, assignments, graph)
    assert result['image_records'] == 4 and result['distinct_originals'] == 3
    assignments[derived['image_id']] = 'test'
    with pytest.raises(ValueError, match='crosses splits'): validate_split(rows, assignments, graph)


def test_transitive_material_sharing_overrides_different_family_names():
    rows, assignments, graph = fixture()
    graph.update(portion0=['reallocation'], reallocation=['shared_pool'], portion2=['shared_pool'], shared_pool=[])
    with pytest.raises(ValueError, match='material_ancestor'): validate_split(rows, assignments, graph)


def test_combined_material_pools_link_multiple_parent_paths():
    rows, assignments, graph = fixture()
    rows[1]['parent_material_ids'] = ['portion1', 'portion0']
    with pytest.raises(ValueError, match='material_ancestor'): validate_split(rows, assignments, graph)


@pytest.mark.parametrize('kind', ['absent', 'missing_ancestor', 'cycle', 'unknown_root', 'duplicate_parent', 'implicit_root'])
def test_material_graph_never_invents_missing_or_independent_roots(kind):
    rows, assignments, graph = fixture()
    if kind == 'absent': graph = None
    elif kind == 'missing_ancestor': graph['portion0'] = ['omitted']
    elif kind == 'cycle': graph.update(portion0=['parent'], parent=['portion0'])
    elif kind == 'unknown_root': graph['portion0'] = ['unknown_parent']; graph['unknown_parent'] = []
    elif kind == 'duplicate_parent': graph['portion0'] = ['portion1', 'portion1']
    elif kind == 'implicit_root': del graph['portion0']
    with pytest.raises(ValueError): validate_split(rows, assignments, graph)


@pytest.mark.parametrize('kind', ['missing', 'extra', 'invalid', 'only_two', 'duplicate_image', 'conflicting_original'])
def test_malformed_records_and_assignments_are_rejected(kind):
    rows, assignments, graph = fixture()
    if kind == 'missing': del assignments['image0']
    elif kind == 'extra': assignments['not-an-image'] = 'train'
    elif kind == 'invalid': assignments['image0'] = 'TRAIN'
    elif kind == 'only_two': assignments['image2'] = 'train'
    elif kind == 'duplicate_image': rows.append(dict(rows[0]))
    elif kind == 'conflicting_original': rows[1]['original_id'] = rows[0]['original_id']
    with pytest.raises(ValueError): validate_split(rows, assignments, graph)


def test_nonfinite_metadata_cannot_enter_frozen_manifest():
    rows, assignments, graph = fixture(); rows[0]['measured_weight_g'] = float('nan')
    with pytest.raises(ValueError, match='JSON-compatible'): validate_split(rows, assignments, graph)
