"""私有写真CSVの検証、訂正のdry-runと履歴付き適用。学習は開始しない。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import re
import shutil
import sqlite3
import stat
import zipfile
from pathlib import Path, PurePosixPath
from uuid import uuid4

from PIL import Image

from .photo_intake import _init, _connection, _audit, _now, _manifest, list_intake
from .pasta_ml.dataset import image_rgb, hashes
from .storage import DEFAULT_DB

CONFIRMED = {'confirmed_by_user', 'confirmed_latest_conflicts_older_note'}
TENTATIVE = {'tentative_user_says_probably', 'tentative_chronological'}


@contextmanager
def _revision_transaction(db_path, metadata_dir, backup):
    try:
        with _connection(db_path) as con:
            con.execute('BEGIN IMMEDIATE')
            yield con
    except Exception:
        # DBは_connectionがrollback。公開台帳CSVも訂正前へ戻す。
        for name in ('labels_NEED_CONFIRMATION.csv','originals_SHA256.txt'):
            previous=backup/'metadata'/name
            if previous.exists(): shutil.copyfile(previous,metadata_dir/name)
            else: (metadata_dir/name).unlink(missing_ok=True)
        raise


def canonical_id(filename):
    match = re.fullmatch(r'(IMG_\d{4})(?:\(\d+\))?\.jpe?g', filename, re.IGNORECASE)
    if not match:
        raise ValueError('写真IDをIMG_XXXX.jpeg形式で指定してください。')
    return match[1].upper()


def read_manifest(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required = {'image_id','canonical_filename','sha256','subgroup','nominal_weight_g',
                    'weight_label_status','provenance_ja','old42_or_extra16','file_size_bytes'}
        if not required <= set(reader.fieldnames or []) or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError('最新CSVの必須列・一意の列名を確認してください。')
        rows = list(reader)
    ids = set()
    for row in rows:
        if None in row or any(v is None for v in row.values()):
            raise ValueError('CSVの列数が一致しません。')
        ident = canonical_id(row['canonical_filename'])
        if row['image_id'] != ident or ident in ids or row['canonical_filename'] != ident+'.jpeg':
            raise ValueError('CSVに重複IDまたは不正な正規名があります。')
        ids.add(ident)
        if not re.fullmatch('[a-f0-9]{64}',row['sha256']):
            raise ValueError('SHA-256が不正です。')
        if row['weight_label_status'] not in CONFIRMED | TENTATIVE:
            raise ValueError('未知の重量確認状態です。')
        weight = float(row['nominal_weight_g'])
        if not 0 < weight < float('inf') or not row['subgroup'].strip() or int(row['file_size_bytes']) <= 0:
            raise ValueError('候補重量・グループ・サイズが不正です。')
    if not rows:
        raise ValueError('CSVが空です。')
    return rows


def receive_zip(path, root, manifest_path):
    """全エントリを検証してから保存。別ハッシュの同一IDとZIPパス攻撃は拒否。"""
    root = Path(root).resolve()
    expected = {r['canonical_filename']:r for r in read_manifest(manifest_path)}
    path = Path(path)
    if path.stat().st_size > 64_000_000:
        raise ValueError('ZIPは64MB以下にしてください。')
    plan, seen = [], set()
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos)>1000 or sum(i.file_size for i in infos)>512_000_000:
            raise ValueError('ZIPの展開上限を超えています。')
        for info in infos:
            parts = PurePosixPath(info.filename).parts
            if info.is_dir():
                continue
            if (len(parts)!=3 or parts[:2]!=('Pasta_Lens_Private_Dataset','photos') or
                    '\\' in info.filename or ':' in info.filename or stat.S_ISLNK(info.external_attr>>16)):
                raise ValueError('ZIPに想定外のパスまたはリンクがあります。')
            name = parts[2]
            if name not in expected or name in seen or info.file_size>30_000_000:
                raise ValueError('ZIPに未登録・重複写真または巨大ファイルがあります。')
            seen.add(name)
            data = archive.read(info)
            row = expected[name]
            if len(data)!=int(row['file_size_bytes']) or hashlib.sha256(data).hexdigest()!=row['sha256']:
                raise ValueError('写真のサイズまたはSHA-256が一致しません。')
            target = root/'photos'/name
            if target.is_symlink() or not target.resolve().is_relative_to(root):
                raise ValueError('写真の保存先が不正です。')
            if target.exists() and target.read_bytes()!=data:
                raise ValueError('同一IDの異なる画像を上書きしません。')
            # image_rgbと同じ制限をメモリ上で検査する。
            from .training_data import _image_rgb
            _image_rgb(data)
            plan.append((target,data))
    created=[]
    try:
        for target,data in plan:
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                with target.open('xb') as f:
                    f.write(data)
                created.append(target)
    except Exception:
        for target in created:
            target.unlink(missing_ok=True)
        raise
    return dict(zip_name=path.name,bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),jpeg_names=sorted(seen))


def inventory(root, legacy_root, manifest_path):
    root, legacy_root = Path(root).resolve(),Path(legacy_root).resolve()
    rows=read_manifest(manifest_path)
    old_receipts=legacy_root/'dataset/previews/receipts.json'
    receipts=json.loads(old_receipts.read_text()) if old_receipts.exists() else {}
    photos=[]; seen_bytes={}; seen_pixels={}; near=[]; previous=[]
    for row in rows:
        name=row['canonical_filename']; ident=row['image_id']
        original=root/'photos'/name
        source=original; variant='manifest_original'
        if row['old42_or_extra16']=='old42':
            candidates=[p for p in (legacy_root/'dataset/previews').glob('*.jpeg') if canonical_id(p.name)==ident]
            if len(candidates)>1:
                raise ValueError('旧写真に同じIDが複数あります。')
            if candidates:
                source=candidates[0]
                variant='legacy_preview'
        item=dict(image_id=ident,filename=name,source_path=str(source),variant=variant,
                  expected_sha256=row['sha256'],subgroup=row['subgroup'],received=False)
        if source.is_file():
            if source.is_symlink() or not (source.resolve().is_relative_to(root) or source.resolve().is_relative_to(legacy_root)):
                raise ValueError('写真が私有領域外を参照しています。')
            data=source.read_bytes(); digest=hashlib.sha256(data).hexdigest()
            item.update(received=True,received_sha256=digest,bytes=len(data),
                        manifest_sha256_match=digest==row['sha256'])
            try:
                if variant=='legacy_preview' and digest!=receipts.get(source.name,{}).get('sha256'):
                    raise ValueError('旧プレビューの受領ハッシュが変わっています。')
                if variant=='manifest_original' and (digest!=row['sha256'] or len(data)!=int(row['file_size_bytes'])):
                    raise ValueError('原本マニフェストのサイズ・ハッシュと一致しません。')
                with Image.open(source) as im:
                    im.verify()
                with Image.open(source) as im:
                    item.update(raw_size=list(im.size),exif_orientation=im.getexif().get(274,1))
                rgb=image_rgb(source); pixel,dhash=hashes(rgb)
                item.update(valid=True,oriented_size=[rgb.shape[1],rgb.shape[0]],pixel_sha256=pixel,
                            perceptual_hash=str(dhash))
                for kind,key,seen in [('bytes',digest,seen_bytes),('pixels',pixel,seen_pixels)]:
                    if key in seen: item['duplicate_'+kind]=seen[key]
                    seen[key]=ident
                for other,ph in previous:
                    if (dhash^ph).bit_count()<=4: near.append([other,ident])
                previous.append((ident,dhash))
            except (ValueError,OSError) as exc:
                item.update(valid=False,error=str(exc))
        photos.append(item)
    return dict(expected=len(rows),received=sum(i['received'] for i in photos),
                valid=sum(i.get('valid',False) for i in photos),
                missing=[i['filename'] for i in photos if not i['received']],
                near_pairs=near,photos=photos)


def revision_plan(root, legacy_root, manifest_path, db_path=DEFAULT_DB):
    rows=read_manifest(manifest_path)
    report=inventory(root,legacy_root,manifest_path)
    manifest_sha256=hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
    existing={}
    for record in list_intake(db_path):
        if Path(record['package_root']).resolve()!=Path(legacy_root).resolve():
            continue
        ident=canonical_id(record['filename'])
        if ident in existing:
            raise ValueError('旧台帳に同じ正規写真IDが複数あります。上書きせず照合してください。')
        existing[ident]=record
    changes=[]
    for supplied,photo in zip(rows,report['photos']):
        old=existing.get(supplied['image_id']); verified=supplied['weight_label_status'] in CONFIRMED
        evidence=dict(supplied,manifest_sha256=manifest_sha256,
                      shoot_id=None,weighing_id=None,physical_sample_id=None,
                      grouping_verified=False,confidence_reason=supplied['provenance_ja'])
        if supplied['weight_label_status']=='confirmed_latest_conflicts_older_note':
            evidence['conflict_history']=[dict(reported_older_weight_g=60,latest_weight_g=40,
                source='latest supplied manifest; older assertion is not proof of local DB registration')]
        # QC・マスク・計量対応の並行更新もdry-runの後に見逃さない。
        before=dict(old) if old else None
        metadata=json.loads(old['metadata_json']) if old else {}
        source=metadata.get('label_manifest_evidence') or {}
        previous_latest=metadata.get('latest_label') or {}
        same_source=bool(old and previous_latest and (
            source.get('manifest_sha256')==manifest_sha256 or source.get('record')==supplied or
            # 最初の実装で適用済みの台帳も、利用者の確認を失わず移行する。
            not source and previous_latest.get('manifest_sha256')==manifest_sha256))
        if not same_source:
            metadata['latest_label']=evidence
            if old and old['training_sample_id'] is not None:
                metadata['quarantined_training_sample_id']=old['training_sample_id']
        metadata.update(label_manifest_evidence=dict(manifest_sha256=manifest_sha256,record=dict(supplied)),
                        canonical_id=supplied['image_id'],
                        canonical_filename=supplied['canonical_filename'],
                        original_attachment_filename=supplied.get('original_attachment_filename',''))
        encoded=json.dumps(metadata,ensure_ascii=False,sort_keys=True)
        if old and metadata==json.loads(old['metadata_json']):
            encoded=old['metadata_json']
        if same_source:
            after={key:old[key] for key in ('capture_group_id','weight_g','weight_verified')}
            after['metadata_json']=encoded
        else:
            after=dict(capture_group_id=supplied['subgroup'],weight_g=float(supplied['nominal_weight_g']) if verified else None,
                       weight_verified=int(verified),metadata_json=encoded)
        unchanged=old and all(old[k]==v for k,v in after.items())
        changes.append(dict(image_id=supplied['image_id'],before=before,after=after,
                            photo=photo,changed=not unchanged,preserve_runtime=same_source))
    body=dict(manifest_sha256=manifest_sha256,
              inventory=report,changes=changes)
    body['plan_sha256']=hashlib.sha256(json.dumps(body,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    return body


def apply_revision(root, legacy_root, manifest_path, expected_plan_sha256, db_path=DEFAULT_DB):
    """dry-runと同じ状態のみ適用。バックアップから復元可。旧版は削除しない。"""
    plan=revision_plan(root,legacy_root,manifest_path,db_path)
    if plan['plan_sha256']!=expected_plan_sha256:
        raise ValueError('dry-run以後に写真・CSV・台帳が変わっています。差分を再確認してください。')
    if plan['inventory']['missing'] or plan['inventory']['valid']!=plan['inventory']['expected']:
        raise ValueError('写真が不足または検証に失敗しています。訂正の一括適用は行いません。')
    if not any(c['changed'] for c in plan['changes']):
        return dict(changed=0,backup=None)
    db_path=Path(db_path); root=Path(root).resolve(); legacy_root=Path(legacy_root).resolve()
    backup=db_path.parent/'backups'/('revision_'+uuid4().hex)
    backup.mkdir(parents=True)
    with sqlite3.connect(db_path) as con,sqlite3.connect(backup/db_path.name) as dest:
        con.backup(dest)
    metadata_dir=legacy_root/'dataset/metadata'
    shutil.copytree(metadata_dir,backup/'metadata')
    shutil.copyfile(manifest_path,backup/'supplied_manifest.csv')
    (backup/'dry_run.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2))
    # ファイルは追加だけ。DB失敗時も未採用の受領物として保持される。
    old_manifest=_manifest(legacy_root)
    names={canonical_id(r['filename']) for r in old_manifest}
    for change in plan['changes']:
        photo=change['photo']; source=Path(photo['source_path'])
        canonical=root/'photos'/photo['filename']
        if canonical.exists() and canonical.read_bytes()!=source.read_bytes():
            raise ValueError('統合先の同一IDを異なるハッシュで上書きしません。')
        canonical.parent.mkdir(exist_ok=True)
        if not canonical.exists(): shutil.copyfile(source,canonical)
        if photo['variant']=='manifest_original':
            target=legacy_root/'dataset/originals'/photo['filename']
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and target.read_bytes()!=source.read_bytes():
                raise ValueError('旧領域の同一ID原本を上書きしません。')
            if not target.exists(): shutil.copyfile(source,target)
        if change['image_id'] not in names:
            old_manifest.append(dict(filename=photo['filename'],original_sha256=photo['expected_sha256'],
                capture_group_id=change['after']['capture_group_id'],session_id='UNCONFIRMED',
                roi_xyxy='',diameter_mm=''))
    _init(db_path)
    with _revision_transaction(db_path,metadata_dir,backup) as con:
        if con.execute("SELECT 1 FROM sqlite_master WHERE name='training_jobs'").fetchone():
            if con.execute("SELECT 1 FROM training_jobs WHERE status IN ('queued','running')").fetchone():
                raise ValueError('学習ジョブの実行中は訂正できません。先に停止してください。')
        for change in plan['changes']:
            if not change['changed']: continue
            old=change['before']; after=change['after']; photo=change['photo']
            if old:
                current=con.execute('SELECT * FROM photo_intake WHERE id=?',(old['id'],)).fetchone()
                if any(current[k]!=v for k,v in old.items()):
                    raise ValueError('適用中に台帳が変わりました。再度dry-runしてください。')
                ident=old['id']
                # 過去の学習行と派生は監査可能なまま採用を撤回する。
                if old['training_sample_id'] is not None and not change['preserve_runtime']:
                    pending=[old['training_sample_id']]; seen=set()
                    from .training_data import _journal
                    while pending:
                        sample_id=pending.pop()
                        if sample_id in seen: continue
                        seen.add(sample_id)
                        pending.extend(r[0] for r in con.execute('SELECT id FROM training_samples WHERE parent_sample_id=?',(sample_id,)))
                        con.execute('UPDATE training_samples SET deleted_at=?,updated_at=?,version=version+1 WHERE id=?',(_now(),_now(),sample_id))
                        _journal(con,sample_id,'label_revision_quarantined')
                    con.execute('UPDATE photo_intake SET training_sample_id=NULL WHERE id=?',(ident,))
            else:
                cur=con.execute('''INSERT INTO photo_intake(package_root,filename,expected_sha256,capture_group_id,metadata_json,created_at)
                    VALUES(?,?,?,?,?,?)''',(str(legacy_root),photo['filename'],photo['expected_sha256'],after['capture_group_id'],'{}',_now()))
                ident=cur.lastrowid
            if change['preserve_runtime']:
                con.execute('UPDATE photo_intake SET metadata_json=? WHERE id=?',(after['metadata_json'],ident))
            else:
                con.execute('''UPDATE photo_intake SET capture_group_id=?,weight_g=?,weight_verified=?,metadata_json=?,
                    group_verified=0,batch_id=NULL,session_id=NULL,confirmed_by=NULL WHERE id=?''',
                    (after['capture_group_id'],after['weight_g'],after['weight_verified'],after['metadata_json'],ident))
            _audit(con,ident,'latest_label_revision',dict(before=old,after=after,plan_sha256=plan['plan_sha256']))
        # 学習成果物は履歴保存したまま利用を止める。旧snapshotは再学習しない。
        if (any(c['changed'] and not c['preserve_runtime'] for c in plan['changes']) and
                con.execute("SELECT 1 FROM sqlite_master WHERE name='training_models'").fetchone()):
            con.execute('CREATE TABLE IF NOT EXISTS model_revocations(model_id TEXT PRIMARY KEY,reason TEXT NOT NULL,at TEXT NOT NULL)')
            con.execute('INSERT OR IGNORE INTO model_revocations SELECT id,?,? FROM training_models',
                        ('label_revision: '+plan['plan_sha256'],_now()))
            con.execute('UPDATE training_models SET active=0')
        if con.execute('PRAGMA integrity_check').fetchone()[0]!='ok':
            raise ValueError('DB整合性検査に失敗しました。')
        columns=list(dict.fromkeys(k for r in old_manifest for k in r))
        temporary=metadata_dir/'labels_NEED_CONFIRMATION.csv.tmp'
        with temporary.open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(old_manifest)
        temporary.replace(metadata_dir/'labels_NEED_CONFIRMATION.csv')
        sha=metadata_dir/'originals_SHA256.txt'
        sha.write_text(''.join(r['original_sha256']+'  '+r['filename']+'\n' for r in old_manifest))
    return dict(changed=sum(c['changed'] for c in plan['changes']),backup=str(backup),plan_sha256=plan['plan_sha256'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root');parser.add_argument('legacy_root');parser.add_argument('--db',default=str(DEFAULT_DB))
    parser.add_argument('--apply-plan',help='先にdry-runで出したplan_sha256を指定')
    args=parser.parse_args(); manifest=Path(args.root)/'docs/ALL_58_PHOTO_MANIFEST.csv'
    result=apply_revision(args.root,args.legacy_root,manifest,args.apply_plan,args.db) if args.apply_plan else revision_plan(args.root,args.legacy_root,manifest,args.db)
    reports=Path(args.root)/'reports';reports.mkdir(exist_ok=True)
    dest=reports/('revision_applied.json' if args.apply_plan else 'revision_dry_run.json')
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(dict(report=str(dest),plan_sha256=result.get('plan_sha256'),changed=result.get('changed'),inventory={k:v for k,v in result.get('inventory',{}).items() if k!='photos'}),ensure_ascii=False))
