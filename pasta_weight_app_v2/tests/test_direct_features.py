"""Image-stage contracts without private files or research model dependencies."""
import csv,hashlib,io,sys,zipfile
from pathlib import Path
import cv2
import numpy as np
import pytest
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from pasta_weight_app_v2.experiments.direct_features import extract
from pasta_weight_app_v2.experiments.direct_intake import receive,visual_signature
from pasta_weight_app_v2.experiments.intake_receipt import verify_protection


def test_rotation_preserves_area_and_major_minor_size():
    rgb=np.zeros((100,140,3),np.uint8);mask=np.zeros((100,140),np.uint8);mask[30:60,20:120]=255;rgb[mask>0]=(240,190,60)
    f,_=extract(rgb,mask);g,_=extract(np.rot90(rgb).copy(),np.rot90(mask).copy())
    for k in ['mask_area_px','mask_fraction','long_axis_px','short_axis_px','width_p50_relative']:
        assert f[k]==pytest.approx(g[k])
    assert f['public_prediction_g'] is None and not f['mask_approved']


def test_empty_and_clipped_masks_are_flagged_not_approved():
    rgb=np.zeros((100,100,3),np.uint8);mask=np.zeros((100,100),np.uint8)
    f,_=extract(rgb,mask);assert 'too_little_foreground' in f['qc_flags']
    mask[:40,:40]=255;f,_=extract(rgb,mask)
    assert 'foreground_at_image_border' in f['qc_flags'] and not f['mask_approved']


def archive(tmp_path,bad_hash=False):
    rows=[];payload={}
    for w in (20,40,60,80,100):
        for i in range(3):
            name=f'synthetic_{w}_{i}.jpeg';path=f'photos/{w}/{name}';b=io.BytesIO();Image.new('RGB',(40,30),(w,i*50,0)).save(b,format='JPEG');raw=b.getvalue();payload[path]=raw
            rows.append(dict(filename=name,relative_path=path,weight_g_reported=str(w),capture_group_id=f'group{w}',sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),width_px=40,height_px=30,approved_for_training='no_pending_review',approved_for_independent_test='no_pending_lineage_check'))
    if bad_hash:rows[0]['sha256']='0'*64
    csvtext=io.StringIO();wr=csv.DictWriter(csvtext,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    payload['labels.csv']=csvtext.getvalue().encode();payload['sha256.txt']='\n'.join(r['sha256']+'  '+r['relative_path'] for r in rows).encode()
    for n in ('README_JA.md','DATASET_NOTES_JA.md','CODEX_PROMPT_JA.md'):payload[n]=b'Synthetic fixture'
    path=tmp_path/'test.zip'
    with zipfile.ZipFile(path,'w') as z:
        for n,b in payload.items():z.writestr(n,b)
    return path


def test_receipt_preserves_existing_and_never_approves_or_overwrites(tmp_path):
    old=tmp_path/'existing';old.mkdir();(old/'sentinel.db').write_bytes(b'unchanged')
    z=archive(tmp_path);output=old/'new';r=receive(z,old,output)
    assert r['images']==15 and r['capture_groups']==5 and r['independent_test_images']==0
    assert all(x['split']=='UNASSIGNED' and not x['production_training_approved'] for x in r['rows'])
    assert verify_protection(output)['unchanged']
    with pytest.raises(FileExistsError):receive(z,old,output)


def test_bad_original_hash_rejected_before_output_created(tmp_path):
    old=tmp_path/'existing';old.mkdir();out=old/'new'
    with pytest.raises(ValueError):receive(archive(tmp_path,True),old,out)
    assert not out.exists()


def test_visual_similarity_is_separate_from_byte_identity(tmp_path):
    im=Image.new('RGB',(40,30),'black');im.paste('yellow',(5,10,30,15));a=tmp_path/'a.png';b=tmp_path/'b.png';im.save(a);im.rotate(90,expand=True).save(b)
    x,px=visual_signature(a);y,py=visual_signature(b)
    assert px!=py and min(np.count_nonzero(i!=j) for i in x for j in y)<=2
