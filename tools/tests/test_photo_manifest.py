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
