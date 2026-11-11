import sys
from pathlib import Path
from io import BytesIO
import pytest
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research_collection import CollectionStore, DuplicateConflictError

def test_atomic_idempotent_private_registration(tmp_path,monkeypatch):
    store=CollectionStore(tmp_path/'research',duplicate_roots=[])
    stream=BytesIO();Image.new('RGB',(300,300),'gold').save(stream,format='PNG');photo=stream.getvalue()
    meta=dict(measured_weight_g=34.,product_name='synthetic',sample_id='s1',capture_session_id='c1',captured_at='2026-11-10T12:00:00+09:00',angle='overhead',scale_reference='none',scale_verified=False,relation='unknown',parent_sample_ids=[])
    def fail(conn):raise OSError('synthetic disk failure')
    original=store._commit;monkeypatch.setattr(store,'_commit',fail)
    with pytest.raises(OSError):store.register(photo,meta,submission_token='a')
    assert not store.list_records() and not list((store.root/'originals').iterdir())
    monkeypatch.setattr(store,'_commit',original)
    saved=store.register(photo,meta,submission_token='a')
    assert not saved['training_approved']
    assert store.register(photo,meta,submission_token='a')['id']==saved['id']
    with pytest.raises(DuplicateConflictError):store.register(photo,dict(meta,measured_weight_g=50.),submission_token='b')
    assert len(store.list_records())==1
    assert (store.root/saved['original_key']).read_bytes()==photo
    assert store.read_preview(saved['id'])

def test_quality_never_claims_whole_pasta_or_loses_inherited_crop():
    from collection_quality import inspect_photo
    stream=BytesIO();Image.new('RGB',(300,300),'white').save(stream,format='PNG')
    quality,_=inspect_photo(stream.getvalue(),reference=True,inherited_flags=['foreground_touches_frame'])
    assert quality['status']=='撮り直し推奨'
    assert 'foreground_touches_frame' in quality['flags']
    assert not quality['whole_pasta_verified']
    assert not quality['reference']['usable_for_metric_scale']

def test_commit_acknowledgement_failure_preserves_committed_photos(tmp_path,monkeypatch):
    store=CollectionStore(tmp_path/'research',duplicate_roots=[])
    stream=BytesIO();Image.new('RGB',(300,300),'gold').save(stream,format='PNG');photo=stream.getvalue()
    meta=dict(measured_weight_g=34.,product_name='synthetic',sample_id='s1',capture_session_id='c1',captured_at='2026-11-10T12:00:00+09:00',angle='overhead',scale_reference='none',scale_verified=False,relation='unknown',parent_sample_ids=[])
    def fail_after_commit(conn):conn.commit();raise OSError('ack lost')
    monkeypatch.setattr(store,'_commit',fail_after_commit)
    with pytest.raises(OSError):store.register(photo,meta,submission_token='a')
    row=store.list_records()[0]
    assert (store.root/row['original_key']).read_bytes()==photo
    assert store.read_preview(row['id'])
    assert store.register(photo,meta,submission_token='a')['id']==row['id']
