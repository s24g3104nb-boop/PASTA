"""An image displayed in chat is never silently treated as a verified original."""
import importlib.util
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

spec=importlib.util.spec_from_file_location('photo_manifest',Path(__file__).parents[1]/'diagnose_photo_manifest.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_missing_mismatched_and_verified_photos(tmp_path):
    images=tmp_path/'images';images.mkdir()
    rgb=np.full((480,640,3),42,np.uint8)
    rgb[120:124,100:530]=(220,180,40)
    image=images/'ok.png';Image.fromarray(rgb).save(image)
    (images/'changed.png').write_bytes(image.read_bytes())
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'photos':[
        dict(filename='missing.jpeg',sha256='0'*64,user_reported_weight_g=50),
        dict(filename='changed.png',sha256='0'*64,user_reported_weight_g=50),
        dict(filename='ok.png',sha256=hashlib.sha256(image.read_bytes()).hexdigest(),user_reported_weight_g=50)]}))
    result=module.run(manifest,images,tmp_path/'out')
    assert [r['status'] for r in result['rows']]==['missing_original','hash_mismatch','diagnosed']
    assert result['originals_verified']==1
    assert result['rows'][-1]['count']==1
    assert result['rows'][-1]['displayed_weight_g'] is None
    assert result['rows'][-1]['hypothetical_weight_g'] is None
    assert result['independent_mae_g'] is None
    assert (tmp_path/'out/ok.png.mask.png').is_file()


def test_nested_count_and_weight_labels_do_not_change_detector(tmp_path):
    root=tmp_path/'input';group=root/'photos/set_B';group.mkdir(parents=True)
    rgb=np.full((480,640,3),42,np.uint8)
    for i in range(10):rgb[40+i*35:44+i*35,100:530]=(220,180,40)
    path=group/'ten(1).jpeg';Image.fromarray(rgb).save(path,quality=95)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    item=dict(filename=path.name,zip_path='photos/set_B/'+path.name,sha256=digest,
              label=dict(kind='strand_count',value=10,weight_g=None),size_pixels=[640,480])
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(n_images=1,photos=[item])))
    before=path.read_bytes()
    first=module.run(manifest,root,tmp_path/'first')
    row=first['rows'][0]
    assert row['count']==10 and row['count_absolute_error']==0
    assert row['reported_count']==10 and row['reported_g'] is None
    assert row['displayed_weight_g'] is None and row['absolute_error_g'] is None
    item['label']=dict(kind='measured_weight_g',value=100.)
    manifest.write_text(json.dumps(dict(n_images=1,photos=[item])))
    second=module.run(manifest,root,tmp_path/'second')
    assert second['rows'][0]['count']==row['count']
    assert second['rows'][0]['reported_count'] is None
    assert second['rows'][0]['reported_g']==100
    assert path.read_bytes()==before
    assert first['independent_mae_g'] is None


def test_broken_jpeg_does_not_prevent_reporting_other_missing_images(tmp_path):
    source=tmp_path/'input';source.mkdir()
    raw=b'broken jpeg';(source/'bad.jpeg').write_bytes(raw)
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(photos=[dict(filename='bad.jpeg',sha256=hashlib.sha256(raw).hexdigest()),
                                              dict(filename='missing.jpeg',sha256='0'*64)])))
    result=module.run(manifest,source,tmp_path/'result')
    assert [r['status'] for r in result['rows']]==['decode_failed','missing_original']
    assert all(r['count'] is None for r in result['rows'])


def test_manifest_traversal_is_rejected_before_output(tmp_path):
    import pytest
    manifest=tmp_path/'manifest.json';source=tmp_path/'input';source.mkdir()
    manifest.write_text(json.dumps(dict(photos=[dict(filename='x.jpeg',zip_path='../x.jpeg',sha256='0'*64)])))
    with pytest.raises(ValueError,match='Unsafe'):
        module.run(manifest,source,tmp_path/'result')
    assert not (tmp_path/'result').exists()
