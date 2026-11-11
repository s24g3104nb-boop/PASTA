"""Launcher behavior without opening a real browser or touching user data."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import sys
import pytest

SUPPORT = Path(__file__).resolve().parents[2] / 'packaging/windows'
sys.path.insert(0, str(SUPPORT))
import launch_windows as launcher
import verify_install as verifier


def test_source_mode_is_explicit_and_pack_cannot_bypass_manifest(tmp_path, monkeypatch):
    assert launcher.application_root(source=True) == SUPPORT.parents[1]
    monkeypatch.setattr(launcher, '__file__', str(tmp_path / 'launch_windows.py'))
    assert launcher.application_root() == tmp_path
    with pytest.raises(ValueError):
        launcher.application_root(source=True)
    with pytest.raises(ValueError, match='MANIFEST'):
        verifier.verify_files(tmp_path)
    verifier.verify_files(SUPPORT.parents[1], source=True)


def test_environment_is_local_respects_existing_and_does_not_create_data(tmp_path, monkeypatch):
    monkeypatch.delenv('PASTA_DATA_DIR', raising=False)
    monkeypatch.delenv('PASTA_RESEARCH_DATA_DIR', raising=False)
    assert launcher.launch_environment(tmp_path)['PASTA_DATA_DIR'] == str(tmp_path / 'user_data')
    assert launcher.launch_environment(tmp_path, True)['PASTA_DATA_DIR'] == str(tmp_path / 'pasta_weight_app_v2/data')
    monkeypatch.setenv('PASTA_DATA_DIR', str(tmp_path / 'personal'))
    assert launcher.launch_environment(tmp_path)['PASTA_RESEARCH_DATA_DIR'] == str(tmp_path / 'personal/research_collection')
    assert not list(tmp_path.iterdir())
    assert '--server.address=127.0.0.1' in launcher.launch_command(tmp_path)
    assert '--server.address=0.0.0.0' in launcher.launch_command(tmp_path, True)


def test_health_check_must_be_running_and_healthy():
    class Response:
        status = 200
        def read(self): return b'ok'
        def __enter__(self): return self
        def __exit__(self, *args): pass
    healthy = SimpleNamespace(open=lambda *a, **k: Response())
    assert launcher.wait_ready(SimpleNamespace(poll=lambda: None), opener=healthy)
    assert not launcher.wait_ready(SimpleNamespace(poll=lambda: 1), opener=healthy)
    times = iter([0, 0, 2])
    Response.status = 503
    assert not launcher.wait_ready(SimpleNamespace(poll=lambda: None), timeout=1,
                                   clock=lambda: next(times), sleep=lambda _: None, opener=healthy)


def test_browser_opens_after_ready_and_failure_terminates_child(tmp_path, monkeypatch):
    (tmp_path / 'pasta_weight_app_v2').mkdir()
    (tmp_path / 'pasta_weight_app_v2/app.py').write_text('')
    events = []
    class Process:
        done = False
        def poll(self): return 0 if self.done else None
        def wait(self, **kwargs): self.done = True; return 0
        def terminate(self): events.append('terminate'); self.done = True
    monkeypatch.setattr(launcher, 'application_root', lambda source: tmp_path)
    monkeypatch.setattr(launcher, 'ensure_port_available', lambda: None)
    monkeypatch.setattr(launcher.subprocess, 'Popen', lambda *a, **k: Process())
    monkeypatch.setattr(launcher, 'wait_ready', lambda p: events.append('ready') or True)
    monkeypatch.setattr(launcher.webbrowser, 'open', lambda url: events.append('browser') or True)
    monkeypatch.setattr(sys, 'argv', ['launch_windows.py'])
    assert launcher.main() == 0
    assert events == ['ready', 'browser']
    events.clear()
    monkeypatch.setattr(launcher, 'wait_ready', lambda p: False)
    assert launcher.main() == 1
    assert events == ['terminate']


def test_source_wrappers_never_install_global_dependencies():
    root = SUPPORT.parents[1]
    for name in ('setup_windows.bat', 'start_windows.bat', 'start_lan.bat'):
        raw = (root / name).read_bytes()
        assert b'\r\n' in raw and b'--source' in raw
        assert b'pip install' not in raw
    legacy = (root / 'pasta_weight_app_v2/start_windows.bat').read_text()
    assert 'pip install' not in legacy and '..\\start_windows.bat' in legacy
