"""Read-only installation check. Never creates or migrates user databases."""
from pathlib import Path
import argparse
import hashlib
import importlib
import sys


def verify_files(root, *, source=False):
    if source:
        required = ('pasta_weight_app_v2/app.py', 'pasta_weight_app_v2/ui/design.css',
                    'pasta_weight_app_v2/manual_weight_ui.py', 'pasta_weight_app_v2/research_collection_ui.py',
                    'docs/assets/capture_sheet/Pasta_Lens_A4_reference_v1.pdf', '.streamlit/config.toml')
        for relative in required:
            if not (root / relative).is_file():
                raise ValueError('必要なファイルがありません: ' + relative)
        return
    manifest = root / 'MANIFEST_SHA256.txt'
    if not manifest.is_file():
        raise ValueError('MANIFEST_SHA256.txt がありません')
    lines = manifest.read_text(encoding='utf-8').splitlines()
    if not lines:
        raise ValueError('配布ファイルの一覧が空です')
    for line in lines:
        digest, relative = line.split('  ', 1)
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('必要なファイルがありません: ' + relative)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('ファイルが変更されています: ' + relative)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', action='store_true')
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        print('Python 3.12 が必要です。setup_windows.bat を確認してください。')
        return 1
    try:
        from launch_windows import application_root
        root = application_root(args.source)
        verify_files(root, source=args.source)
        for name in ('streamlit', 'numpy', 'pandas', 'PIL', 'cv2'):
            importlib.import_module(name)
        import cv2
        if not hasattr(cv2, 'aruco') or not hasattr(cv2.aruco, 'ArucoDetector'):
            raise ValueError('OpenCVのマーカー機能がありません')
    except (ValueError, ImportError, OSError) as exc:
        print('起動前の確認に失敗しました:', exc)
        print('新しく展開し直すか、setup_windows.bat を実行してください。')
        return 1
    print('必要ファイル・Python・依存ライブラリを確認しました。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
