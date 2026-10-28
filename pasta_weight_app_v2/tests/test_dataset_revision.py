"""実写真を使用しない訂正・受領・漏洩の回帰検証。"""
import csv
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import zipfile

import numpy as np
from PIL import Image
import pytest

from pasta_weight_app_v2.dataset_revision import (canonical_id,read_manifest,receive_zip,
    inventory,revision_plan,apply_revision)
from pasta_weight_app_v2.photo_intake import load_manifest,list_intake,promote_photo
from pasta_weight_app_v2.storage import init_db
from pasta_weight_app_v2.training_manager import init_training_manager


@pytest.fixture
def revision(tmp_path):
    root=tmp_path/'Pasta_Lens_Private_Dataset';(root/'docs').mkdir(parents=True)
    legacy=tmp_path/'Pasta_Lens_42_Photos';md=legacy/'dataset/metadata';md.mkdir(parents=True)
    db=tmp_path/'measurements.sqlite3';init_db(db);init_training_manager(db)
    original=[]; latest=[]; blobs=[]
    for i,status in enumerate(['confirmed_by_user','confirmed_latest_conflicts_older_note','tentative_chronological']):
        ident=f'IMG_{1000+i}'
        f=io.BytesIO();Image.fromarray(np.random.default_rng(i).integers(0,255,(70,80,3),dtype=np.uint8)).save(f,'JPEG')
        blob=f.getvalue();blobs.append(blob);sha=hashlib.sha256(blob).hexdigest()
        latest.append(dict(image_id=ident,canonical_filename=ident+'.jpeg',original_attachment_filename=ident+'(2).jpeg',
            sha256=sha,subgroup='group'+str(i),nominal_weight_g=20*(i+1),weight_label_status=status,
            provenance_ja='利用者確認' if i<2 else '撮影順候補',old42_or_extra16='old42' if i<2 else 'extra16',file_size_bytes=len(blob)))
        if i<2:
            original.append(dict(filename=ident+'.jpeg',original_sha256=sha,capture_group_id='old',session_id='pending'))
            path=legacy/'dataset/previews'/original[-1]['filename'];path.parent.mkdir(exist_ok=True);path.write_bytes(blob)
        else:
            path=root/'photos'/latest[-1]['canonical_filename'];path.parent.mkdir();path.write_bytes(blob)
    (legacy/'dataset/previews/receipts.json').write_text(json.dumps({r['filename']:dict(sha256=r['original_sha256']) for r in original}))
    with (md/'labels_NEED_CONFIRMATION.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=original[0]);w.writeheader();w.writerows(original)
    load_manifest(legacy,db,expected_count=2)
    manifest=root/'docs/ALL_58_PHOTO_MANIFEST.csv'
    with manifest.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=latest[0]);w.writeheader();w.writerows(latest)
    return root,legacy,manifest,db,blobs


def test_revision_separates_confirmation_from_group_and_retains_audit(revision):
    root,old,manifest,db,_=revision
    plan=revision_plan(root,old,manifest,db)
    assert plan['inventory']['valid']==3
    assert all(r['weight_g'] is None for r in list_intake(db))  # dry-runは無変更
    result=apply_revision(root,old,manifest,plan['plan_sha256'],db)
    assert Path(result['backup'],db.name).exists()
    rows=list_intake(db)
    assert [r['weight_g'] for r in rows]==[20,40,None]
    assert [r['weight_verified'] for r in rows]==[1,1,0]
    assert not any(r['group_verified'] for r in rows)
    assert json.loads(rows[1]['metadata_json'])['latest_label']['conflict_history'][0]['reported_older_weight_g']==60
    with sqlite3.connect(db) as con:
        assert con.execute("SELECT count(*) FROM photo_intake_audit WHERE action='latest_label_revision'").fetchone()[0]==3
    with pytest.raises(ValueError,match='未確認'): promote_photo(rows[0]['id'],db)
    assert len(list((root/'photos').glob('*.jpeg')))==3
    p2=revision_plan(root,old,manifest,db)
    assert apply_revision(root,old,manifest,p2['plan_sha256'],db)['changed']==0
    assert len(list_intake(db))==3


def test_stale_plan_cannot_override_another_operation(revision):
    root,old,manifest,db,_=revision
    plan=revision_plan(root,old,manifest,db)
    with sqlite3.connect(db) as con: con.execute('UPDATE photo_intake SET weight_g=123 WHERE id=1')
    with pytest.raises(ValueError,match='dry-run'):apply_revision(root,old,manifest,plan['plan_sha256'],db)


def test_missing_photos_block_apply_but_report_inventory(revision):
    root,old,manifest,db,_=revision
    (root/'photos/IMG_1002.jpeg').unlink()
    plan=revision_plan(root,old,manifest,db)
    assert plan['inventory']['missing']==['IMG_1002.jpeg']
    with pytest.raises(ValueError,match='不足'):apply_revision(root,old,manifest,plan['plan_sha256'],db)


def test_running_training_and_metadata_rollback(revision):
    root,old,manifest,db,_=revision
    previous=(old/'dataset/metadata/labels_NEED_CONFIRMATION.csv').read_bytes()
    with sqlite3.connect(db) as con:
        con.execute("INSERT INTO training_jobs(id,status,progress,message,created_at,alpha,allow_synthetic) VALUES('x','running',0,'','2026-10-28',1,0)")
    plan=revision_plan(root,old,manifest,db)
    with pytest.raises(ValueError,match='実行中'):apply_revision(root,old,manifest,plan['plan_sha256'],db)
    assert len(list_intake(db))==2
    assert previous==(old/'dataset/metadata/labels_NEED_CONFIRMATION.csv').read_bytes()


@pytest.mark.parametrize('name',['../IMG_1002.jpeg','Pasta_Lens_Private_Dataset/photos/../IMG_1002.jpeg','Pasta_Lens_Private_Dataset/photos/IMG_9999.jpeg'])
def test_zip_paths_and_unlisted_ids_refused(revision,name):
    root,_,manifest,_,blobs=revision
    path=root/'bad.zip'
    with zipfile.ZipFile(path,'w') as z:z.writestr(name,blobs[-1])
    with pytest.raises(ValueError):receive_zip(path,root,manifest)


def test_zip_hash_and_internal_duplicate_conflict(revision):
    root,_,manifest,_,blobs=revision
    path=root/'good.zip';name='Pasta_Lens_Private_Dataset/photos/IMG_1002.jpeg'
    with zipfile.ZipFile(path,'w') as z:z.writestr(name,blobs[-1])
    assert receive_zip(path,root,manifest)['jpeg_names']==['IMG_1002.jpeg']
    (root/'photos/IMG_1002.jpeg').write_bytes(blobs[0])
    with pytest.raises(ValueError,match='上書き'):receive_zip(path,root,manifest)


@pytest.mark.parametrize('filename,expected',[('IMG_7242(3).jpeg','IMG_7242'),('img_7280.jpg','IMG_7280')])
def test_canonical_aliases(filename,expected):assert canonical_id(filename)==expected


def test_unknown_label_status_not_assumed_confirmed(revision):
    root,_,manifest,_,_=revision
    manifest.write_text(manifest.read_text().replace('confirmed_by_user','probably_confirmed'))
    with pytest.raises(ValueError,match='未知'):read_manifest(manifest)


def test_pixel_duplicates_distinguished_from_near_photos(revision):
    root,old,manifest,db,blobs=revision
    original=read_manifest(manifest)
    original[-1].update(sha256=original[0]['sha256'],file_size_bytes=original[0]['file_size_bytes'])
    with manifest.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=original[0]);w.writeheader();w.writerows(original)
    (root/'photos/IMG_1002.jpeg').write_bytes(blobs[0])
    report=inventory(root,old,manifest)
    assert report['photos'][-1]['duplicate_pixels']=='IMG_1000'
