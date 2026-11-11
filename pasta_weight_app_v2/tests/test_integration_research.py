"""Synthetic intake boundary/recovery checks; never uses private photographs."""
from io import BytesIO
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collection_quality import inspect_photo
from research_collection_ui import _photo_diagnostic, _preview


def png(size=(320, 320), color='gold'):
    stream = BytesIO()
    Image.new('RGB', size, color).save(stream, format='PNG')
    return stream.getvalue()


@pytest.mark.parametrize('data', [b'not-an-image', png((8, 8))])
def test_invalid_original_rejected_before_quality_or_preview(data):
    with pytest.raises((ValueError, OSError)):
        _preview(data)
    with pytest.raises(ValueError):
        inspect_photo(data)


def test_quality_cache_tracks_original_and_reference_conditions(monkeypatch):
    calls = []
    def inspect(data, **options):
        calls.append((data, options))
        return {'training_approved': False}, np.zeros((16, 16, 3), np.uint8)
    monkeypatch.setattr('collection_quality.inspect_photo', inspect)
    draft = dict(photo_bytes=png(), scale_reference='none', scale_verified=False, coplanar=False)
    first = _photo_diagnostic(draft)
    assert _photo_diagnostic(draft) is first
    draft['measured_weight_g'] = 100
    assert _photo_diagnostic(draft) is first  # metadata-only changes need no CV rerun
    draft['scale_reference'] = 'a4_markers'
    _photo_diagnostic(draft)
    draft['scale_verified'] = True
    _photo_diagnostic(draft)
    draft['coplanar'] = True
    _photo_diagnostic(draft)
    draft['photo_bytes'] = png(color='black')
    _photo_diagnostic(draft)
    assert len(calls) == 5
    assert len(draft['quality_cache']) == 2  # one current result, no image history
    assert draft['quality_cache']['result'][0]['training_approved'] is False


def test_quality_failure_keeps_draft_and_allows_photo_reselection(tmp_path):
    script = '''
from io import BytesIO
import sys
sys.path.insert(0, 'pasta_weight_app_v2')
from PIL import Image
from streamlit.testing.v1 import AppTest
import collection_quality
from research_collection import CollectionStore
at=AppTest.from_file('pasta_weight_app_v2/app.py', default_timeout=20).run()
at.radio(key='main_nav').set_value('設定').run()
at.selectbox(key='settings_section').set_value('開発・検証ツール').run()
at.selectbox(key='developer_section').set_value('独立試料・撮影データ登録').run()
b=BytesIO();Image.new('RGB',(320,320),'gold').save(b,format='PNG')
d=at.session_state['research_collection_draft'];d['photo_bytes']=b.getvalue();d['photo_name']='synthetic.png';d['measured_weight_g']=100.
def fail(*args, **kwargs):raise RuntimeError('synthetic diagnostic failure')
collection_quality.inspect_photo=fail
at.run()
assert not at.exception,at.exception
assert any('写真の状態を確認できません' in e.value for e in at.error)
assert at.button(key='collection_save').disabled
assert at.session_state['research_collection_draft']['photo_bytes']==b.getvalue()
assert at.session_state['research_collection_draft']['measured_weight_g']==100.
assert not CollectionStore().list_records()
at.button(key='collection_clear_photo').click().run()
assert at.session_state['research_collection_draft']['photo_bytes'] is None
assert at.session_state['research_collection_draft']['measured_weight_g']==100.
assert not at.exception,at.exception
'''
    result = subprocess.run([sys.executable, '-c', script],
        cwd=Path(__file__).resolve().parents[2],
        env=dict(os.environ, PASTA_DATA_DIR=str(tmp_path/'app'), PASTA_RESEARCH_DATA_DIR=str(tmp_path/'research')),
        capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


def test_failed_save_never_deletes_preview_reused_by_next_registration(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from PIL.PngImagePlugin import PngInfo
    from research_collection import CollectionStore
    store = CollectionStore(tmp_path/'research', duplicate_roots=[])
    other = CollectionStore(tmp_path/'research', duplicate_roots=[])
    original = png()
    stream = BytesIO()
    info = PngInfo(); info.add_text('description', 'synthetic alternate original metadata')
    Image.new('RGB', (320, 320), 'gold').save(stream, format='PNG', pnginfo=info)
    second = stream.getvalue()  # Different originals with identical clean previews.
    assert original != second
    meta = dict(measured_weight_g=40., product_name='synthetic', sample_id='first',
        capture_session_id='session', captured_at='2026-11-11T12:00:00+09:00',
        angle='overhead', scale_reference='none', scale_verified=False,
        relation='unknown', parent_sample_ids=[])
    connect = store._connection
    entries = []
    calls = 0
    @contextmanager
    def interleaved_connection():
        nonlocal calls
        calls += 1
        if calls == 2:
            # Deterministic interleaving: rollback released the failed writer;
            # a second browser saves before the first starts orphan cleanup.
            entries.append(other.register(second, dict(meta, sample_id='second'), submission_token='second'))
        with connect() as conn:
            yield conn
    def fail(conn):
        raise OSError('synthetic first writer failure')
    monkeypatch.setattr(store, '_connection', interleaved_connection)
    monkeypatch.setattr(store, '_commit', fail)
    with pytest.raises(OSError):
        store.register(original, meta, submission_token='first')
    assert len(entries) == 1
    assert other.read_preview(entries[0]['id'])
    assert (other.root/entries[0]['original_key']).read_bytes() == second
    assert len(list((other.root/'originals').iterdir())) == 1
