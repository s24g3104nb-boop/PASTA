import io
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from photo_qc import (apply_mask_edits, composite_rgb, make_overlay, propose_mask,
                      validate_uploaded_mask)


def rgb_fixture():
    rgb = np.zeros((20, 30, 3), dtype=np.uint8)
    rgb[3:17, 5] = (230, 190, 65)
    rgb[3:17, 7] = (230, 190, 65)
    rgb[7:9, 22:24] = (230, 190, 65)  # 背景小物の模擬色候補。
    return rgb


def png(mask, **kwargs):
    stream = io.BytesIO()
    Image.fromarray(mask).save(stream, 'PNG', **kwargs)
    return stream.getvalue()


def test_candidate_has_no_closing_and_is_unreviewed():
    rgb = rgb_fixture()
    before = rgb.copy()
    result = propose_mask(rgb)
    assert result['status'] == 'UNREVIEWED'
    assert np.array_equal(rgb, before)
    assert np.all(result['mask'][3:17, 6] == 0)
    assert result['stats']['candidate_components'] == 3
    assert result['stats']['foreground_pixels'] == 32
    assert any('本数ではありません' in text for text in result['warnings'])


def test_roi_rejection_and_edge_warning_are_visible():
    result = propose_mask(rgb_fixture(), roi=[5, 3, 15, 17])
    assert result['stats']['discarded_by_roi_pixels'] == 4
    assert result['stats']['touches_roi_boundary']
    assert result['mask'].shape == (20, 30)
    assert any('ROI外' in text for text in result['warnings'])
    assert any('先端' in text for text in result['warnings'])


def test_empty_and_broad_masks_are_not_approved():
    empty = propose_mask(np.zeros((10, 10, 3), dtype=np.uint8))
    assert empty['stats']['foreground_pixels'] == 0
    assert any('抽出画素がありません' in text for text in empty['warnings'])
    broad = propose_mask(np.full((10, 10, 3), (230, 190, 65), dtype=np.uint8))
    assert broad['stats']['foreground_fraction'] == 1
    assert any('65%' in text for text in broad['warnings'])
    assert broad['status'] == 'UNREVIEWED'


@pytest.mark.parametrize('roi', [[-1, 0, 10, 10], [0, 0, 31, 10], [5, 3, 5, 10],
                               [0, 0, 10.0, 10], [0, False, 10, 10], [0, 0, 10]])
def test_invalid_roi_refuses_silent_clipping(roi):
    with pytest.raises(ValueError):
        propose_mask(rgb_fixture(), roi=roi)


@pytest.mark.parametrize('kwargs', [dict(hue_min=-1), dict(hue_min=40, hue_max=30),
                                  dict(value_min=float('nan')), dict(saturation_min=True)])
def test_invalid_color_thresholds(kwargs):
    with pytest.raises(ValueError):
        propose_mask(rgb_fixture(), **kwargs)


def test_exclusions_apply_after_manual_add_and_erase():
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[0, 0] = 255
    before = mask.copy()
    result = apply_mask_edits(mask, add_polygons=[[[2, 2], [9, 2], [9, 9], [2, 9]]],
                             erase_polygons=[[[3, 3], [5, 3], [5, 5], [3, 5]]],
                             excluded_rects=[[7, 0, 12, 12]])
    assert np.array_equal(mask, before)
    assert result[0, 0] == 255 and result[11, 0] == 0
    assert result[2, 2] == 255 and result[4, 4] == 0
    assert np.all(result[:, 7:] == 0)
    assert set(np.unique(result)) == {0, 255}
    proposed = propose_mask(rgb_fixture(), excluded_rects=[[22, 7, 24, 9]])
    assert proposed['stats']['excluded_candidate_pixels'] == 4
    assert np.all(proposed['mask'][7:9, 22:24] == 0)


def test_overlapping_additions_are_union_not_even_odd_holes():
    mask = np.zeros((15, 15), dtype=np.uint8)
    first = [[2, 2], [8, 2], [8, 8], [2, 8]]
    second = [[5, 5], [11, 5], [11, 11], [5, 11]]
    result = apply_mask_edits(mask, add_polygons=[first, second])
    assert np.all(result[5:9, 5:9] == 255)


@pytest.mark.parametrize('polygon', [[[0, 0], [2, 0]], [[-1, 0], [2, 0], [2, 2]],
                                   [[0.5, 0], [2, 0], [2, 2]], [[0, 0], [1, 1], [2, 2]]])
def test_invalid_polygons_are_rejected(polygon):
    with pytest.raises(ValueError):
        apply_mask_edits(np.zeros((10, 10), dtype=np.uint8), add_polygons=[polygon])


def test_composite_retains_exact_original_pixels_and_png_roundtrip():
    rgb = np.random.default_rng(2).integers(0, 256, (20, 30, 3), dtype=np.uint8)
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[2:14, 9:13] = 255
    result = composite_rgb(rgb, mask)
    assert np.array_equal(result[mask == 255], rgb[mask == 255])
    assert np.all(result[mask == 0] == 24)
    assert np.array_equal(np.asarray(Image.open(io.BytesIO(png(result)))), result)
    overlay = make_overlay(rgb, mask)
    assert np.array_equal(overlay[mask == 0], rgb[mask == 0])
    assert not np.array_equal(overlay[mask == 255], result[mask == 255])


@pytest.mark.parametrize('mask', [np.zeros((19, 30), dtype=np.uint8),
                               np.ones((20, 30), dtype=np.uint8),
                               np.zeros((20, 30, 3), dtype=np.uint8)])
def test_mask_geometry_and_binary_values_are_required(mask):
    with pytest.raises(ValueError):
        composite_rgb(rgb_fixture(), mask)


def test_uploaded_mask_requires_png_geometry_and_binary_pixels(tmp_path):
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[2:7, 9:15] = 255
    assert np.array_equal(validate_uploaded_mask(png(mask), mask.shape), mask)
    stream = io.BytesIO(png(mask))
    stream.seek(3)
    assert np.array_equal(validate_uploaded_mask(stream, mask.shape), mask)
    assert stream.tell() == 3
    path = tmp_path / 'mask.png'
    path.write_bytes(png(mask))
    assert np.array_equal(validate_uploaded_mask(path, mask.shape), mask)
    with pytest.raises(ValueError, match='寸法'):
        validate_uploaded_mask(png(mask), (30, 20))
    with pytest.raises(ValueError, match='0/255'):
        validate_uploaded_mask(png(np.full((20, 30), 127, dtype=np.uint8)), mask.shape)
    jpeg = io.BytesIO()
    Image.fromarray(mask).save(jpeg, 'JPEG')
    with pytest.raises(ValueError, match='PNG'):
        validate_uploaded_mask(jpeg.getvalue(), mask.shape)
    with pytest.raises(ValueError, match='破損'):
        validate_uploaded_mask(b'not an image', mask.shape)
    with pytest.raises(ValueError, match='1チャンネル'):
        validate_uploaded_mask(png(np.zeros((20, 30, 3), dtype=np.uint8)), mask.shape)


def test_uploaded_rotated_mask_is_not_auto_reoriented():
    mask = np.zeros((20, 30), dtype=np.uint8)
    exif = Image.Exif()
    exif[274] = 6
    with pytest.raises(ValueError, match='回転EXIF'):
        validate_uploaded_mask(png(mask, exif=exif), mask.shape)
