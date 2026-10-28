"""訂正マニフェストの再表示で、後から確認した情報を失わない。"""
import csv
import json
import sqlite3

import pytest

from test_dataset_revision import revision
from pasta_weight_app_v2.dataset_revision import revision_plan,apply_revision,read_manifest
from pasta_weight_app_v2.photo_intake import confirm_group,list_intake
from pasta_weight_app_v2.storage import save_product,list_products
from pasta_weight_app_v2.training_data import register_sample,list_samples


def apply_fixture(revision):
    root,old,manifest,db,_=revision
    plan=revision_plan(root,old,manifest,db)
    apply_revision(root,old,manifest,plan['plan_sha256'],db)
    return root,old,manifest,db


def confirm_fixture(old,db):
    save_product('確認した実製品',.9,350,12,2,70,db_path=db)
    product=str(list_products(db)[-1]['id'])
    confirm_group(old,'group1',40,batch_id='verified-weighing',session_id='verified-shoot',
                  product_id=product,confirmed_same_batch=True,confirmed_by='利用者',
                  physical_sample_id='same-physical-pasta',db_path=db)


def test_unchanged_manifest_preserves_later_user_group_confirmation(revision):
    root,old,manifest,db=apply_fixture(revision)
    confirm_fixture(old,db)
    before=list_intake(db)
    plan=revision_plan(root,old,manifest,db)
    assert not any(c['changed'] for c in plan['changes'])
    assert apply_revision(root,old,manifest,plan['plan_sha256'],db)['changed']==0
    assert list_intake(db)==before
    row=before[1]
    assert row['group_verified']==1 and row['batch_id']=='verified-weighing'
    assert json.loads(row['metadata_json'])['latest_label']['physical_sample_id']=='same-physical-pasta'


def test_first_revision_metadata_migration_preserves_runtime_and_pointer(revision):
    root,old,manifest,db=apply_fixture(revision)
    confirm_fixture(old,db)
    # 初期版にはlatest_labelだけがあり、immutableなCSV記録がなかった。
    with sqlite3.connect(db) as con:
        raw=con.execute('SELECT metadata_json FROM photo_intake WHERE id=2').fetchone()[0]
        metadata=json.loads(raw);metadata.pop('label_manifest_evidence')
        con.execute('UPDATE photo_intake SET metadata_json=? WHERE id=2',(json.dumps(metadata,ensure_ascii=False),))
    before=list_intake(db)[1]
    plan=revision_plan(root,old,manifest,db)
    change=next(c for c in plan['changes'] if c['image_id']=='IMG_1001')
    assert change['changed'] and change['preserve_runtime']
    apply_revision(root,old,manifest,plan['plan_sha256'],db)
    after=list_intake(db)[1]
    for key in ('weight_g','weight_verified','group_verified','batch_id','session_id','confirmed_by'):
        assert before[key]==after[key]
    assert json.loads(after['metadata_json'])['label_manifest_evidence']['record']['nominal_weight_g']=='40'


def test_qc_change_invalidates_full_state_dry_run(revision):
    root,old,manifest,db,_=revision
    plan=revision_plan(root,old,manifest,db)
    with sqlite3.connect(db) as con:
        con.execute("UPDATE photo_intake SET qc_status='APPROVED',mask_sha256='new-hash' WHERE id=1")
    with pytest.raises(ValueError,match='dry-run'):
        apply_revision(root,old,manifest,plan['plan_sha256'],db)


def test_corrected_sample_keeps_explicit_quarantine_restore_pointer(revision):
    root,old,manifest,db=apply_fixture(revision)
    blob=revision[-1][0]
    sample_id=register_sample(blob,dict(actual_weight_g=40,mode='spread',source='real',
        product_id='2',batch_id='old-weighing',session_id='old-shoot'),db_path=db)
    with sqlite3.connect(db) as con:
        con.execute('UPDATE photo_intake SET training_sample_id=? WHERE id=2',(sample_id,))
    rows=read_manifest(manifest);rows[1]['nominal_weight_g']='35'
    with manifest.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    plan=revision_plan(root,old,manifest,db)
    assert [c['image_id'] for c in plan['changes'] if c['changed'] and not c['preserve_runtime']]==['IMG_1001']
    apply_revision(root,old,manifest,plan['plan_sha256'],db)
    intake=list_intake(db)[1];metadata=json.loads(intake['metadata_json'])
    assert intake['weight_g']==35 and intake['training_sample_id'] is None
    assert metadata['quarantined_training_sample_id']==sample_id
    assert not list_samples(db)
    historical=list_samples(db,include_deleted=True)[0]
    assert historical['deleted_at'] is not None and historical['actual_weight_g']==40
    with sqlite3.connect(db) as con:
        action=con.execute('SELECT action FROM training_sample_revisions WHERE sample_id=? ORDER BY version DESC',(sample_id,)).fetchone()[0]
    assert action=='label_revision_quarantined'


def test_same_canonical_id_in_legacy_ledger_is_not_silently_overwritten(revision):
    root,old,manifest,db,_=revision
    with sqlite3.connect(db) as con:
        columns=[r[1] for r in con.execute('PRAGMA table_info(photo_intake)') if r[1]!='id']
        con.execute('INSERT INTO photo_intake('+','.join(columns)+') SELECT '+','.join(
            "'IMG_1000(2).jpeg'" if c=='filename' else c for c in columns)+' FROM photo_intake WHERE id=1')
    with pytest.raises(ValueError,match='同じ正規写真ID'):
        revision_plan(root,old,manifest,db)


@pytest.mark.parametrize('change_mask',[False,True])
def test_revised_photo_can_be_readopted_after_group_and_qc_confirmation(revision,change_mask):
    import numpy as np
    from pasta_weight_app_v2.photo_intake import load_preview,save_review,set_photo_mode,promote_photo
    root,old,manifest,db=apply_fixture(revision)
    confirm_fixture(old,db)
    set_photo_mode(2,'spread',db)
    row=list_intake(db)[1]
    mask=np.full(load_preview(row).shape[:2],255,dtype=np.uint8)
    save_review(2,mask,reviewer='fixture QC',shape_confirmed=True,leakage_checked=True,
                source_variant='preview',db_path=db)
    sample_id=promote_photo(2,db)
    records=read_manifest(manifest);records[1]['nominal_weight_g']='35'
    with manifest.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=records[0]);writer.writeheader();writer.writerows(records)
    plan=revision_plan(root,old,manifest,db)
    apply_revision(root,old,manifest,plan['plan_sha256'],db)
    assert not list_samples(db)
    product=list_intake(db)[1]['product_id']
    confirm_group(old,'group1',35,batch_id='verified-weighing',session_id='verified-shoot',
        product_id=product,confirmed_same_batch=True,confirmed_by='利用者再確認',
        physical_sample_id='same-physical-pasta',db_path=db)
    if change_mask:
        mask=mask.copy();mask[:3]=0
    save_review(2,mask,reviewer='fixture QC again',shape_confirmed=True,leakage_checked=True,
                source_variant='preview',db_path=db)
    restored_id=promote_photo(2,db)
    assert (restored_id!=sample_id) is change_mask
    samples=list_samples(db)
    assert len(samples)==1 and samples[0]['actual_weight_g']==35
    metadata=json.loads(list_intake(db)[1]['metadata_json'])
    assert 'quarantined_training_sample_id' not in metadata
    with sqlite3.connect(db) as con:
        snapshots=[json.loads(r[0]) for r in con.execute('SELECT snapshot_json FROM training_sample_revisions WHERE sample_id=? ORDER BY version',(sample_id,))]
    assert snapshots[0]['actual_weight_g']==40
    if change_mask:
        historical=list_samples(db,include_deleted=True)
        assert len(historical)==2 and historical[0]['deleted_at'] is not None
    else:
        assert snapshots[-1]['actual_weight_g']==35
    assert not any(c['changed'] for c in revision_plan(root,old,manifest,db)['changes'])


def test_promotion_audit_failure_rolls_back_and_retry_reuses_identical_png(revision):
    import numpy as np
    from pasta_weight_app_v2.photo_intake import load_preview,save_review,set_photo_mode,promote_photo
    root,old,manifest,db=apply_fixture(revision)
    confirm_fixture(old,db)
    row=list_intake(db)[1];mask=np.full(load_preview(row).shape[:2],255,dtype=np.uint8)
    set_photo_mode(row['id'],'spread',db)
    save_review(row['id'],mask,reviewer='fixture',shape_confirmed=True,leakage_checked=True,
                source_variant='preview',db_path=db)
    with sqlite3.connect(db) as con:
        con.execute("CREATE TRIGGER fail_registration BEFORE INSERT ON photo_intake_audit WHEN NEW.action='registered_for_training' BEGIN SELECT RAISE(ABORT,'test failure'); END")
    with pytest.raises(sqlite3.IntegrityError):promote_photo(row['id'],db)
    assert not list_samples(db)
    assert list_intake(db)[1]['training_sample_id'] is None
    with sqlite3.connect(db) as con:con.execute('DROP TRIGGER fail_registration')
    ident=promote_photo(row['id'],db)
    assert ident==list_samples(db)[0]['id']
    assert list_intake(db)[1]['training_sample_id']==ident
