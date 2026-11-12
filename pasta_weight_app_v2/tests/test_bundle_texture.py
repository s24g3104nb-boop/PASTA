"""Synthetic imaging contracts, not real-image counting or weight accuracy."""
import sys
from pathlib import Path
import numpy as np
import pytest
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.bundle_texture import extract_bundle_texture,TEXTURE_KEYS


def fixture(period=8, count=20):
    image=np.full((280,620,3),20,np.uint8)
    mask=np.zeros(image.shape[:2],np.uint8)
    h=period*count
    row=(155+25*np.cos(2*np.pi*np.arange(h)/period)).astype(np.uint8)
    image[50:50+h,70:550,0]=row[:,None]+40
    image[50:50+h,70:550,1]=row[:,None]+20
    image[50:50+h,70:550,2]=30
    mask[50:50+h,70:550]=255
    return image,mask


def test_touching_foreground_retains_periodic_ridges_without_claiming_count_or_weight():
    rgb,mask=fixture()
    assert cv2.connectedComponents(mask)[0]-1==1
    result=extract_bundle_texture(rgb,mask)
    assert result['texture_available']==1
    assert result['pitch_px']==pytest.approx(8,rel=.08)
    assert result['texture_width_over_pitch']==pytest.approx(20,rel=.12)
    assert result['texture_directional_coherence']>.98
    assert not any(k in result for k in ('predicted_g','count','weight_g'))
    assert np.isfinite([result[k] for k in TEXTURE_KEYS]).all()


def test_resize_rotation_and_background_do_not_create_model_scale_features():
    rgb,mask=fixture();ref=extract_bundle_texture(rgb,mask)
    half=extract_bundle_texture(cv2.resize(rgb,None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA),cv2.resize(mask,None,fx=.5,fy=.5,interpolation=cv2.INTER_NEAREST))
    matrix=cv2.getRotationMatrix2D((310,140),18,1)
    rotated=extract_bundle_texture(cv2.warpAffine(rgb,matrix,(620,280)),cv2.warpAffine(mask,matrix,(620,280),flags=cv2.INTER_NEAREST))
    for result in (half,rotated):
        assert result['texture_available']==1
        assert result['texture_width_over_pitch']==pytest.approx(ref['texture_width_over_pitch'],rel=.15)
    altered=rgb.copy();altered[mask==0]=(255,230,200)
    other=extract_bundle_texture(altered,mask)
    np.testing.assert_allclose([ref[k] for k in TEXTURE_KEYS],[other[k] for k in TEXTURE_KEYS])


def test_uniform_block_and_missing_foreground_do_not_invent_pitch():
    rgb,mask=fixture();rgb[mask>0]=(220,180,30)
    for m in (mask,np.zeros_like(mask)):
        result=extract_bundle_texture(rgb,m)
        assert result['texture_available']==0 and result['pitch_px'] is None
        assert all(result[k]==0 for k in TEXTURE_KEYS)
        assert result['texture_status']=='unavailable'


def test_invalid_inputs_rejected_without_modifying_source():
    rgb,mask=fixture();before=rgb.copy()
    with pytest.raises(ValueError):extract_bundle_texture(rgb,mask[1:])
    extract_bundle_texture(rgb,mask)
    np.testing.assert_array_equal(before,rgb)
