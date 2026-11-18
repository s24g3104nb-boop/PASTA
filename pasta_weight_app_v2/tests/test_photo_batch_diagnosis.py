"""Mechanism tests use generated data. Actual photos are replayed privately."""
import numpy as np
import pytest
from pasta_weight_app_v2.pasta_vision import detect_spread,analyze_spread_mask,_yellow_mask
from pasta_weight_app_v2.experiments.photo_batch_diagnosis import image_features,regression_features,fit_models


def image():
    rgb=np.full((240,400,3),30,np.uint8)
    for i in range(6):rgb[25+30*i:28+30*i,40:360]=(220,180,40)
    return rgb


def test_common_mask_analyzer_preserves_normal_detector_exactly():
    rgb=image();before=rgb.copy()
    normal=detect_spread(rgb)
    shared=analyze_spread_mask(rgb,_yellow_mask(rgb,20,75))
    assert normal.count==shared.count==6
    assert normal.evidence==shared.evidence
    assert np.array_equal(normal.mask,shared.mask)
    assert np.array_equal(normal.overlay_rgb,shared.overlay_rgb)
    assert np.array_equal(rgb,before)


@pytest.mark.parametrize('mask',[np.zeros((1,1),np.uint8),np.full((240,400),128,np.uint8),np.zeros((240,400),float)])
def test_invalid_experimental_mask_is_rejected(mask):
    with pytest.raises(ValueError):analyze_spread_mask(image(),mask)


def test_feature_extraction_has_no_labels_and_rotation_preserves_geometry():
    rgb=image();mask=_yellow_mask(rgb,20,75)
    first=image_features(rgb,mask)
    rotated=image_features(np.rot90(rgb).copy(),np.rot90(mask).copy())
    for key in ('area_fraction','length_px','width_px','width_over_length','rect_density','components'):
        assert rotated[key]==pytest.approx(first[key],abs=1e-5)
    assert all(np.isfinite(list(first.values())))
    assert regression_features(rgb)==regression_features(rgb.copy())


def test_count_only_images_do_not_supply_a_mass_to_regression(tmp_path):
    records=[dict(image_id=str(i),reported_g=mass) for i,mass in enumerate([20.,50.,100.,None])]
    images={str(i):image() for i in range(4)}
    fits,predictions=fit_models(records,images,tmp_path)
    assert all(row['original_images']==3 and row['independent_mae_g'] is None for row in fits)
    assert {r['image_id'] for r in predictions}=={'0','1','2'}
    assert all(not r['production_eligible'] for r in predictions)
