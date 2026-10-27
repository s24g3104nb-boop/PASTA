"""人工的な線・画像で座標と保存不変性を確認する。実写真の精度ではない。"""
import io
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from preprocessing import transform_roi,propose_conservative_mask,audit_mask,background_controls
from photo_qc import composite_rgb


@pytest.mark.parametrize('orientation,target,expected',[
    (1,(50,100),[5,10,30,90]),(2,(50,100),[20,10,45,90]),
    (3,(50,100),[20,10,45,90]),(4,(50,100),[5,10,30,90]),
    (5,(100,50),[10,5,90,30]),(6,(100,50),[10,5,90,30]),
    (7,(100,50),[10,20,90,45]),(8,(100,50),[10,20,90,45])])
def test_exif_all_orientations_map_corners(orientation,target,expected):
    r=transform_roi([10,20,60,180],(100,200),target,orientation=orientation,roi_space='raw')
    assert r['roi_xyxy']==expected


def test_oriented_coordinates_are_not_rotated_twice():
    r=transform_roi([10,20,60,180],(100,200),(50,100),orientation=6,roi_space='oriented')
    assert r['roi_xyxy']==[5,10,30,90]
    assert transform_roi([406,7,1936,3263],(2448,3264),(1152,1536))['roi_xyxy']==[191,3,912,1536]


@pytest.mark.parametrize('kwargs',[
    dict(roi=[0,0,101,200],reference_size=(100,200),target_size=(50,100)),
    dict(roi=[0,0,100,200],reference_size=(100,200),target_size=(100,50)),
    dict(roi=[0,0,100,200],reference_size=None,target_size=(50,100)),
    dict(roi=[0,0,100,200],reference_size=(100,200),target_size=(50,100),orientation=9)])
def test_ambiguous_and_invalid_coordinates_refuse_silent_clipping(kwargs):
    with pytest.raises(ValueError): transform_roi(**kwargs)


def strands():
    rgb=np.zeros((160,180,3),np.uint8)
    rgb[20:140,50:52]=(220,180,60)
    rgb[20:140,56:58]=(220,180,60)
    rgb[3:5,145:147]=(220,180,60)
    return rgb


def test_separate_thin_strands_and_gap_preserved_while_distant_object_removed():
    rgb=strands(); before=rgb.copy()
    result=propose_conservative_mask(rgb,[0,0,180,160],support_margin=3)
    mask=result['mask']
    assert np.all(mask[20:140,50:52]==255)
    assert np.all(mask[20:140,56:58]==255)
    assert np.all(mask[20:140,52:56]==0)
    assert np.all(mask[3:5,145:147]==0)
    assert np.array_equal(rgb,before)
    foreground=composite_rgb(rgb,mask)
    assert np.array_equal(foreground[mask>0],rgb[mask>0])
    assert result['status']=='UNREVIEWED'


def test_forbidden_scale_region_remains_removed_even_with_manual_support():
    rgb=strands()
    result=propose_conservative_mask(rgb,[0,0,180,160],excluded_rects=[[45,100,65,150]],
                                    support_polygon=[[0,0],[179,0],[179,159],[0,159]])
    assert not result['mask'][100:150,45:65].any()
    audit=audit_mask(rgb,result['mask'],forbidden_rects=[[45,100,65,150]])
    assert audit['forbidden_regions_clear'] and audit['forbidden_regions_supplied']
    assert audit['foreground_rgb_unchanged'] and not audit['subject_count_verified']


def test_audit_detects_digit_region_and_lost_reference_pixels():
    rgb=strands(); mask=np.zeros(rgb.shape[:2],np.uint8);mask[10:12,10:12]=255
    reference=mask.copy();reference[50:52,50:52]=255
    audit=audit_mask(rgb,mask,forbidden_rects=[[0,0,20,20]],reference_mask=reference)
    assert not audit['forbidden_regions_clear'] and audit['forbidden_foreground_pixels']==4
    assert audit['lost_reference_pixels']==4 and audit['status']=='UNREVIEWED'


def test_background_controls_do_not_modify_input_or_foreground():
    rgb=strands();before=rgb.copy()
    mask=np.zeros(rgb.shape[:2],np.uint8);mask[20:140,50:58]=255
    controls=background_controls(rgb,mask)
    assert np.array_equal(rgb,before)
    for key in ('foreground_dark','foreground_gray'):
        assert np.array_equal(controls[key][mask>0],rgb[mask>0])
    assert np.array_equal(controls['background_only'][mask==0],rgb[mask==0])
    assert np.all(controls['background_only'][mask>0]==24)


def test_empty_input_candidate_is_never_approved():
    result=propose_conservative_mask(np.zeros((100,100,3),np.uint8),[0,0,100,100])
    assert not result['mask'].any() and result['status']=='UNREVIEWED'
