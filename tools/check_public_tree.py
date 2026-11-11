"""公開Gitの追跡予定と過去コミットのパスを検査する。私有データの内容は出力しない。"""
from pathlib import PurePosixPath
import subprocess
import sys
import re

ALLOWED_CSV={'dataset/dataset_template.csv','dataset/ml_template.csv'}
ALLOWED_IMAGES={'pasta_weight_app_v2/sample_images/demo_bundle_16.png',
                'pasta_weight_app_v2/sample_images/demo_spread_12.png'}
DENIED_SUFFIX={'.jpg','.jpeg','.heic','.heif','.webp','.mpo','.png','.zip','.sqlite3','.sqlite','.db','.backup','.pt','.pth','.onnx','.joblib','.pkl','.pem','.key'}
SECRET_PATTERNS = [
    rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    rb'\bgh[pousr]_[A-Za-z0-9]{30,}\b',
    rb'\bgithub_pat_[A-Za-z0-9_]{50,}\b',
    rb'\bAKIA[A-Z0-9]{16}\b',
]


def has_secret(content):
    """High-confidence patterns only; never emit matching credential values."""
    return any(re.search(pattern, content) for pattern in SECRET_PATTERNS)


def unsafe(path):
    p=PurePosixPath(path)
    if any(part.lower() in {'data','user_data','pasta_lens_private_dataset','pasta_lens_42_photos','user_photos','meal_photos','backups','models'} for part in p.parts):return True
    if '.sqlite3-' in p.name:return True
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
    objects=set()
    # Inspect indexed blobs, so unstaged removals cannot conceal staged secrets.
    for line in subprocess.check_output(['git','ls-files','-s']).decode().splitlines():
        meta,path=line.split('\t',1)
        objects.add((meta.split()[1],path))
    if '--history' in sys.argv:
        for line in subprocess.check_output(['git','rev-list','--objects','--all']).decode().splitlines():
            if ' ' in line:
                oid,path=line.split(' ',1);objects.add((oid,path))
    secret_paths=set()
    for oid,path in objects:
        if subprocess.check_output(['git','cat-file','-t',oid]).strip()!=b'blob':continue
        content=subprocess.check_output(['git','cat-file','blob',oid])
        if has_secret(content):secret_paths.add(path)
    if secret_paths:
        print('秘密情報候補を含む公開ファイル（値は非表示）:',*sorted(secret_paths),sep='\n');return 1
    print('追跡対象'+('・全履歴' if '--history' in sys.argv else '')+'の私有データパス検査: 問題なし')
    print('公開内容の秘密鍵・認証トークン検査: 問題なし（一般的な認証形式を検査）')
    return 0


if __name__=='__main__':raise SystemExit(main())
