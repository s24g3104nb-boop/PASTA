import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('check_public_tree',Path(__file__).parents[1]/'check_public_tree.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_private_paths_and_synthetic_exceptions():
    for path in ('user_data/a.txt','photos/user.heic','research/model.joblib','a.sqlite3-wal','private.env/.env'):
        assert module.unsafe(path)
    assert not module.unsafe('pasta_weight_app_v2/sample_images/demo_spread_12.png')
    assert not module.unsafe('docs/assets/capture_sheet/Pasta_Lens_A4_reference_v1.pdf')


def test_credential_content_without_printing_values():
    assert module.has_secret(b'ghp_'+b'a'*36)
    assert module.has_secret(b'AKIA'+b'0'*16)
    assert not module.has_secret(b'--token YOUR_TOKEN; hashlib.sha256(image)')
