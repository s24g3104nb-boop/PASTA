from pathlib import Path
import sys,zipfile,hashlib,subprocess
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from build_windows_pack import inputs,build

ROOT=Path(__file__).resolve().parents[2]


def test_allowlist_excludes_private_data_and_only_synthetic_images():
    names=[p.as_posix() for p in inputs(ROOT)]
    assert not any('/data/' in p or '/models/' in p or p.endswith(('.sqlite3','.db','.jpg','.csv','.zip')) for p in names)
    assert {p for p in names if p.endswith('.png')}=={'pasta_weight_app_v2/sample_images/demo_spread_12.png','pasta_weight_app_v2/sample_images/demo_bundle_16.png'}
    for required in ('manual_weight_ui.py','manual_measurement_storage.py','capture_guide_ui.py','research_collection_ui.py'):
        assert 'pasta_weight_app_v2/'+required in names


def test_archive_hashes_assets_and_bat_contracts(tmp_path):
    archive=build(ROOT,tmp_path/'Pasta_Lens_test')
    with zipfile.ZipFile(archive) as z:
        z.extractall(tmp_path/'unpack')
    pack=tmp_path/'unpack/Pasta_Lens_test'
    for line in (pack/'MANIFEST_SHA256.txt').read_text().splitlines():
        digest,relative=line.split('  ',1)
        assert hashlib.sha256((pack/relative).read_bytes()).hexdigest()==digest
    result=subprocess.run([sys.executable,str(pack/'verify_install.py')],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    for name in ('setup_windows.bat','start_windows.bat','start_lan.bat'):
        raw=(pack/name).read_bytes()
        assert b'\r\n' in raw and not raw.startswith(b'\xef\xbb\xbf')
        assert 'setlocal' in raw.decode('utf8')
    setup=(pack/'setup_windows.bat').read_text()
    assert '-m venv ".venv"' in setup and '".venv\\Scripts\\python.exe" -m pip install' in setup
    start=(pack/'start_windows.bat').read_text()
    assert '127.0.0.1' in start and 'choice /c YN' in start
    assert 'user_data' in start and 'verify_install.py' in start
    assert not (pack/'user_data').exists()
    # Use pip's real recursive reader so both layouts resolve nested -r paths.
    from pip._internal.req.req_file import parse_requirements
    from pip._internal.network.session import PipSession
    with PipSession() as session:
        source=[r.requirement for r in parse_requirements(str(ROOT/'packaging/windows/requirements.txt'),session=session)]
        packaged=[r.requirement for r in parse_requirements(str(pack/'requirements.txt'),session=session)]
    assert source == packaged
    assert 'streamlit>=1.49,<2' in source and 'streamlit==1.67.0' in source
