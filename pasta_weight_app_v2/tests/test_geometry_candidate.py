import cv2
import numpy as np
import pytest
from pasta_weight_app_v2.geometry_candidate import ProductCalibration,estimate,geometry,rectify_plane
from pasta_weight_app_v2.experiments.geometry_benchmark import scene,CALIBRATION
from pasta_weight_app_v2.pasta_vision import detect_spread


def test_crossed_strands_not_counted_as_single_region():
    rgb=scene(20,True,78)
    old=detect_spread(rgb)
    new=estimate(rgb,'spread',CALIBRATION,single_layer=True,perpendicular=True)
    assert old.count==10
    assert new['success'],new['reason']
    assert abs(new['predicted_g']-20)<4
    assert new['features']['endpoint_count']==20


def test_scale_and_rotation_preserve_count_within_experimental_tolerance():
    values=[]
    for angle,scale in [(0,1),(90,1),(15,.75),(0,1.1)]:
        result=estimate(scene(20,True,79,angle,scale),'spread',CALIBRATION,single_layer=True,perpendicular=True)
        assert result['success'],result['reason']
        values.append(result['predicted_g'])
    assert max(values)-min(values)<5


def test_dense_side_bundle_and_unconfirmed_view_are_rejected():
    rgb=scene(20,False,79)
    for mode,kwargs in [('bundle_side',dict(single_layer=True,perpendicular=True)),('spread',dict()),('bundle_end',dict(perpendicular=True))]:
        result=estimate(rgb,mode,CALIBRATION,**kwargs)
        assert not result['success'] and result['predicted_g'] is None


def test_missing_calibration_does_not_turn_pixels_into_grams():
    result=estimate(scene(20,False,79),'spread',single_layer=True,perpendicular=True)
    assert not result['success'] and result['predicted_g'] is None


@pytest.mark.parametrize('diameter,length,count,mass',[(0,250,50,20),(1.6,float('nan'),50,20),(1.6,250,0,20),(1.6,250,50,-1)])
def test_calibration_requires_measured_positive_dimensions(diameter,length,count,mass):
    with pytest.raises(ValueError):ProductCalibration('fixture',diameter,length,count,mass).validate()


def test_cropped_mask_has_border_warning():
    mask=np.zeros((100,100),np.uint8);mask[40:47,:]=255
    assert geometry(mask)['border']


def test_reference_plane_rectification_requires_real_corners():
    rgb=scene(20,False,78)
    corrected,matrix=rectify_plane(rgb,[[10,10],[1000,10],[1000,1000],[10,1000]],200,200)
    assert corrected.shape==(400,400,3) and np.isfinite(matrix).all()
    with pytest.raises(ValueError):rectify_plane(rgb,[[10,10]]*4,200,200)


def test_adaptive_end_faces_at_different_scales():
    for radius in (4,8,16):
        rgb=np.full((240,320,3),35,np.uint8)
        for i in range(12):cv2.circle(rgb,(50+i%4*60,50+i//4*60),radius,(215,175,65),-1)
        result=estimate(rgb,'bundle_end',CALIBRATION,perpendicular=True,all_tips_visible=True)
        assert result['success'],result['reason']
        assert result['predicted_g']==12


def test_structural_background_filter_preserves_visible_long_strand():
    from pasta_weight_app_v2.geometry_candidate import candidate_mask,structural_mask
    rgb=np.full((200,500,3),190,np.uint8)
    cv2.rectangle(rgb,(40,70),(400,76),(215,175,65),-1)
    cv2.circle(rgb,(100,140),15,(215,175,65),-1)
    mask=candidate_mask(rgb);filtered=structural_mask(mask)
    assert filtered[73,100]==255
    assert filtered[140,100]==0
    assert not np.any((filtered>0)&(mask==0))


def test_neutral_shadow_is_not_pasta_foreground():
    from pasta_weight_app_v2.geometry_candidate import candidate_mask
    rgb=np.full((100,100,3),190,np.uint8);rgb[:,30:70]=40
    assert not candidate_mask(rgb).any()
