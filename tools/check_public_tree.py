"""公開Gitの追跡予定と過去コミットのパスを検査する。私有データの内容は出力しない。"""
from pathlib import PurePosixPath
import subprocess
import sys

ALLOWED_CSV={'dataset/dataset_template.csv','dataset/ml_template.csv'}
ALLOWED_IMAGES={'pasta_weight_app_v2/sample_images/demo_bundle_16.png',
                'pasta_weight_app_v2/sample_images/demo_spread_12.png'}
DENIED_SUFFIX={'.jpg','.jpeg','.mpo','.png','.zip','.sqlite3','.sqlite','.db','.pt','.pth','.onnx','.pem','.key'}


def unsafe(path):
    p=PurePosixPath(path)
    if any(part.lower() in {'data','pasta_lens_private_dataset','pasta_lens_42_photos','user_photos','models'} for part in p.parts):return True
    if p.name.startswith('.env') or p.name in {'secrets.toml','id_rsa','id_ed25519'}:return True
    if p.suffix.lower()=='.csv':return path not in ALLOWED_CSV
    return p.suffix.lower() in DENIED_SUFFIX and path not in ALLOWED_IMAGES


def main():
    paths=set(subprocess.check_output(['git','ls-files','-z']).decode().split('\0'))
    if '--history' in sys.argv:
        paths.update(subprocess.check_output(['git','log','--all','--pretty=format:','--name-only']).decode().splitlines())
    bad=sorted(p for p in paths if p and unsafe(p))
    if bad:
        print('公開対象に不適切なパス:',*bad,sep='\n');return 1
    print('追跡対象'+('・全履歴' if '--history' in sys.argv else '')+'の私有データパス検査: 問題なし')
    return 0


if __name__=='__main__':raise SystemExit(main())
