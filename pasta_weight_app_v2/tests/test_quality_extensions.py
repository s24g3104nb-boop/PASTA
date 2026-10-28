"""人工画像のみでQC編集、撮影グループ分割、MPO読込と評価の保護を検証する。"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
import random

import numpy as np
from PIL import Image
import pytest

from pasta_weight_app_v2.qc_editor import brush_edit, put_candidate, history_step
from pasta_weight_app_v2.preprocessing import propose_lab_mask, quality_flags
from pasta_weight_app_v2.training_data import _image_rgb
from pasta_weight_app_v2.pasta_ml.dataset import Sample, image_rgb, grouping, validate_splits
from pasta_weight_app_v2.pasta_ml.evaluate import compare_reports
from pasta_weight_app_v2.pasta_ml.leakage import background_diagnostic
from pasta_weight_app_v2.training_manager import init_training_manager, assign_persistent_splits
from pasta_weight_app_v2 import learning_readiness


def test_brush_line_preserves_input_and_cannot_restore_forbidden_region():
    mask = np.zeros((80, 100), dtype=np.uint8)
    before = mask.copy()
    edited = brush_edit(mask, [[10, 40], [90, 40]], 3,
                        excluded_rects=[[40, 20, 60, 60]])
    assert np.array_equal(mask, before)
    assert edited[40, 10] == edited[40, 90] == 255
    assert np.all(edited[20:60, 40:60] == 0)
    assert np.all(edited[:30] == 0)
    erased = brush_edit(edited, [[20, 40], [30, 40]], 4, erase=True)
    assert erased[40, 20] == erased[40, 30] == 0
    assert edited[40, 20] == 255


@pytest.mark.parametrize('points,radius,erase', [
    ([], 2, False), ([[100, 20]], 2, False), ([[20, -1]], 2, False),
    ([[True, 20]], 2, False), ([[20.0, 20]], 2, False),
    ([[20, 20]], 0, False), ([[20, 20]], 101, False),
    ([[20, 20]], 2, 'erase'),
])
def test_brush_invalid_coordinates_and_actions_are_refused(points, radius, erase):
    with pytest.raises(ValueError):
        brush_edit(np.zeros((80, 100), np.uint8), points, radius, erase=erase)


def test_undo_redo_roundtrip_and_new_change_invalidates_redo():
    state = {}
    a = dict(mask=np.zeros((20, 20), np.uint8), warnings=['first'])
    b = dict(mask=brush_edit(a['mask'], [[10, 10]], 2), warnings=['second'])
    put_candidate(state, 'mask', a)
    put_candidate(state, 'mask', b)
    assert history_step(state, 'mask', 'undo')
    assert state['mask'] is a
    assert history_step(state, 'mask', 'redo')
    assert state['mask'] is b
    assert history_step(state, 'mask', 'undo')
    put_candidate(state, 'mask', dict(mask=a['mask'].copy(), warnings=['third']))
    assert not history_step(state, 'mask', 'redo')
    with pytest.raises(ValueError):
        history_step(state, 'mask', 'delete')


def test_undo_history_is_bounded():
    state = {}
    for i in range(12):
        put_candidate(state, 'x', dict(version=i))
    assert len(state['x_undo']) == 6
    for expected in range(10, 4, -1):
        assert history_step(state, 'x', 'undo')
        assert state['x']['version'] == expected
    assert not history_step(state, 'x', 'undo')


def test_lab_mask_does_not_change_thin_strand_geometry_or_source_pixels():
    rgb = np.zeros((100, 140, 3), np.uint8)
    rgb[10:90, 50:52] = (220, 180, 60)
    rgb[10:90, 57:59] = (220, 180, 60)
    before = rgb.copy()
    result = propose_lab_mask(rgb, [0, 0, 140, 100], excluded_rects=[[45, 80, 65, 100]])
    mask = result['mask']
    assert result['status'] == 'UNREVIEWED'
    assert np.array_equal(rgb, before)
    assert np.all(mask[10:80, 50:52] == 255)
    assert np.all(mask[10:80, 57:59] == 255)
    assert np.all(mask[:, 52:57] == 0)
    assert np.all(mask[80:100, 45:65] == 0)


def test_quality_flags_empty_clipped_and_bright_are_observations_only():
    rgb = np.zeros((60, 80, 3), np.uint8)
    mask = np.zeros((60, 80), np.uint8)
    empty = quality_flags(rgb, mask)
    assert 'empty_foreground' in empty['flags']
    assert empty['foreground_laplacian_variance'] is None
    assert empty['status'] == 'UNREVIEWED'
    rgb[:20, :20] = 255
    mask[:20, :20] = 255
    result = quality_flags(rgb, mask)
    assert 'subject_may_be_cut_off' in result['flags']
    assert 'possible_glare' in result['flags']
    assert result['touches_image_boundary']
    assert result['status'] == 'UNREVIEWED'
    with pytest.raises(ValueError, match='寸法'):
        quality_flags(rgb, np.zeros((10, 10), np.uint8))


@pytest.mark.parametrize('orientation,expected_shape', [(1, (90, 80, 3)), (6, (80, 90, 3))])
def test_mpo_primary_frame_exif_normalization_and_source_hash(tmp_path, orientation, expected_shape):
    exif = Image.Exif()
    exif[274] = orientation
    stream = io.BytesIO()
    Image.new('RGB', (80, 90), (240, 210, 20)).save(
        stream, format='MPO', save_all=True,
        append_images=[Image.new('RGB', (80, 90), (200, 10, 10))], exif=exif)
    raw = stream.getvalue()
    with Image.open(io.BytesIO(raw)) as picture:
        assert picture.format == 'MPO' and picture.n_frames == 2
    path = tmp_path / 'phone.jpeg'
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    memory = _image_rgb(raw)
    disk = image_rgb(path)
    assert memory.shape == disk.shape == expected_shape
    assert np.array_equal(memory, disk)
    assert float(memory[:, :, 1].mean()) > 190  # 主画像の黄色。赤い第2フレームは読まない。
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def sample(i, *, physical='', session=None, batch=None):
    return Sample(dict(batch_id=batch or f'b{i}', session_id=session or f's{i}',
                       physical_sample_id=physical, actual_weight_g='40', mode='spread',
                       product_id='fixture', source='synthetic'), Path(f'fixture{i}.png'),
                  f'{i:064x}', random.Random(i + 91827).getrandbits(64))


def test_physical_sample_id_unifies_separate_shoot_and_weighing_transitively():
    samples = [sample(0, physical='reused'), sample(1, physical='reused'),
               sample(2, session='s1'), sample(3)]
    assert grouping(samples) == [[0, 1, 2], [3]]
    empty = [sample(4), sample(5)]
    assert grouping(empty) == [[0], [1]]  # 空の任意フィールドは共有試料にしない。
    for i, s in enumerate(samples):
        s.values['split'] = ['train', 'test', 'test', 'val'][i]
    with pytest.raises(ValueError, match='リーク'):
        validate_splits(samples)


def test_physical_sample_id_persistent_history_inherits_holdout_and_blocks_bridge(tmp_path):
    db = tmp_path / 'training.sqlite3'
    init_training_manager(db)
    first = [sample(i, physical=f'p{i}') for i in range(9)]
    assign_persistent_splits(first, db)
    held = next(s for s in first if s.values['split'] == 'test')
    extra = sample(40, physical=held.values['physical_sample_id'])
    second = first + [extra]
    assign_persistent_splits(second, db)
    assert extra.values['split'] == 'test'
    # 新規写真の同じ撮影sessionがtrain履歴と同じ試料のtest履歴を結ぶ場合は停止。
    train = next(s for s in first if s.values['split'] == 'train')
    bridge = sample(41, physical=held.values['physical_sample_id'], session=train.values['session_id'])
    with pytest.raises(ValueError, match='結び付け'):
        assign_persistent_splits(first + [bridge], db)


def report(rows):
    return dict(rows=rows, dataset_sha256='fixture', independent_group_count=2)


def prediction_row(name, actual, predicted, success=True):
    return dict(image_path=name, actual_weight_g=actual, predicted_g=predicted, success=success)


def test_fair_comparison_reports_common_success_and_refuses_different_images():
    regression = report([prediction_row('a', 20, 22), prediction_row('b', 40, 39),
                         prediction_row('c', 60, None, False)])
    opencv = report([prediction_row('a', 20, 25), prediction_row('b', 40, None, False),
                    prediction_row('c', 60, 50)])
    compared = compare_reports(regression, opencv)
    assert compared['evaluation_images'] == 3
    assert compared['common_success_images'] == 1
    assert compared['regression_common']['mae_g'] == 2
    assert compared['opencv_common']['mae_g'] == 5
    assert compared['independent_group_count'] == 2
    with pytest.raises(ValueError, match='一致'):
        compare_reports(regression, report(opencv['rows'][:-1]))


def leakage_case():
    rgb = np.full((30, 40, 3), 200, np.uint8)
    rgb[10:20, 10:30] = (240, 210, 20)
    mask = np.zeros((30, 40), np.uint8)
    mask[10:20, 10:30] = 255
    return dict(rgb=rgb, mask=mask, actual_weight_g=40, label_confirmed=True,
                qc_status='APPROVED', group_id='evaluation', image_id='fixture')


@pytest.mark.parametrize('changes', [dict(qc_status='UNREVIEWED'), dict(label_confirmed=False),
                                    dict(group_id='training'), dict(actual_weight_g=float('nan'))])
def test_background_diagnostic_refuses_unapproved_tentative_or_training_overlap(changes):
    case = leakage_case()
    case.update(changes)
    with pytest.raises(ValueError):
        background_diagnostic([case], lambda rgb: dict(success=True, predicted_g=40),
                              training_group_ids={'training'})


def test_background_diagnostic_retains_failed_predictions_and_marks_shortcut_suspicion():
    case = leakage_case()
    before = case['rgb'].copy()
    # 常に40gと返す無意味な予測器も背景対照で疑いを示す。実精度の主張ではない。
    result = background_diagnostic([case], lambda rgb: dict(success=True, predicted_g=40),
                                   training_group_ids={'training'})
    assert result['background_shortcut_suspected']
    assert result['mean_background_change_g'] == 0
    assert result['independent_group_count'] == 1
    assert np.array_equal(case['rgb'], before)
    failed = background_diagnostic([case], lambda rgb: dict(success=False, predicted_g=None),
                                   training_group_ids={'training'})
    assert failed['scores']['background_only']['failed'] == 1
    assert failed['scores']['background_only']['mae_g'] is None
    assert failed['mean_background_change_g'] is None


def intake_row(i, *, confirmed=True, approved=False, physical=None):
    return dict(weight_verified=int(confirmed), weight_g=20.0 if confirmed else None,
                qc_status='APPROVED' if approved else 'UNREVIEWED', leakage_checked=int(approved),
                group_verified=int(physical is not None), batch_id=f'b{i}', session_id=f's{i}',
                product_id='fixture', mode='spread', metadata_json=__import__('json').dumps(
                    dict(latest_label=dict(physical_sample_id=physical))))


def test_readiness_distinguishes_confirmed_labels_qc_and_independent_samples(monkeypatch):
    rows = [intake_row(i) for i in range(33)] + [intake_row(i, confirmed=False) for i in range(25)]
    monkeypatch.setattr(learning_readiness, 'list_intake', lambda db: rows)
    result = learning_readiness.readiness('fixture')
    assert result['total'] == 58 and result['confirmed_labels'] == 33
    assert result['tentative_or_unconfirmed'] == 25
    assert result['eligible_images'] == result['verified_independent_groups'] == 0
    assert not result['training_permitted']
    # 写真が多くても全て同じ麺なら独立1群として数える。
    shared = [intake_row(i, approved=True, physical='same_material') for i in range(10)]
    monkeypatch.setattr(learning_readiness, 'list_intake', lambda db: shared)
    result = learning_readiness.readiness('fixture')
    assert result['eligible_images'] == 10
    assert result['verified_independent_groups'] == 1
    assert not result['training_permitted']
