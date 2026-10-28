import csv
import json
import numpy as np
from PIL import Image
import pytest
from pasta_weight_app_v2.experiments.compare_dataset import run,validate_research_data
from pasta_weight_app_v2.pasta_ml.dataset import load_dataset
from pasta_weight_app_v2.pasta_ml.model import load_model


def dataset(tmp_path):
    rows=[]
    for i,split in enumerate(['train','train','train','val','test']):
        Image.fromarray(np.random.default_rng(i).integers(0,256,(40,40,3),dtype=np.uint8)).save(tmp_path/f'{i}.png')
        rows.append(dict(image_path=f'{i}.png',batch_id=f'b{i}',session_id=f's{i}',physical_sample_id=f'p{i}',
            product_id='fixture',mode='spread',source='synthetic',actual_weight_g=[20,40,60,30,50][i],split=split,
            weight_verified='YES',group_verified='YES',qc_status='APPROVED',leakage_checked='YES'))
    path=tmp_path/'labels.csv'
    def write():
        with path.open('w') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    write()
    calibration=tmp_path/'calibration.json'
    calibration.write_text(json.dumps({'fixture':dict(product_id='fixture',diameter_mm=1.6,length_mm=250,weighed_strands=50,weighed_mass_g=25)}))
    return path,calibration,rows,write


def test_unconfirmed_labels_and_groups_are_blocked(tmp_path):
    path,_,rows,write=dataset(tmp_path)
    rows[0]['weight_verified']='NO';write()
    with pytest.raises(ValueError,match='未確定'):validate_research_data(load_dataset(path),True)
    rows[0]['weight_verified']='YES';rows[0]['group_verified']='NO';write()
    with pytest.raises(ValueError,match='物理試料'):validate_research_data(load_dataset(path),True)


def test_same_physical_sample_must_not_cross_split(tmp_path):
    path,_,rows,write=dataset(tmp_path)
    rows[4]['physical_sample_id']='p0';write()
    with pytest.raises(ValueError):validate_research_data(load_dataset(path),True)


def test_fixed_split_comparison_and_no_test_label_training(tmp_path):
    path,cal,rows,write=dataset(tmp_path)
    r=run(path,cal,tmp_path/'first',True)
    assert r['production_eligible'] is False
    assert len(r['reports']['test']['paired'])==15
    assert r['reports']['test']['geometry']['overall']['failed']==1
    first=json.loads((tmp_path/'first/models.json').read_text())
    rows[4]['actual_weight_g']=55;write()
    run(path,cal,tmp_path/'second',True)
    second=json.loads((tmp_path/'second/models.json').read_text())
    assert first['models']==second['models']
    assert first['dataset_sha256']!=second['dataset_sha256']
    with pytest.raises(ValueError):load_model(tmp_path/'first/models.json')
