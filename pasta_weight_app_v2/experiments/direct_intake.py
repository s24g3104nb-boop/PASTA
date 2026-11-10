"""Private receipt and similarity triage for direct-weight research; no DB access."""
import argparse,csv,hashlib,io,json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageOps
from .intake_receipt import _archive,_snapshot


def sha(data):return hashlib.sha256(data).hexdigest()


def visual_signature(path):
    with Image.open(path) as im:
        im=ImageOps.exif_transpose(im).convert('RGB');pixels=np.asarray(im)
        gray=np.array(im.convert('L').resize((32,32)))
    hashes=[]
    for k in range(4):
        d=cv2.dct(np.rot90(gray,k).astype(np.float32))[:8,:8].flatten()[1:]
        hashes.append(d>np.median(d))
    return hashes,sha(str(pixels.shape).encode()+pixels.tobytes())


def receive(archive,existing,output):
    archive=Path(archive);existing=Path(existing).resolve();output=Path(output)
    if output.exists():raise FileExistsError('No overwrites')
    payload=_archive(archive)
    for name in ('README_JA.md','DATASET_NOTES_JA.md','CODEX_PROMPT_JA.md','labels.csv','sha256.txt'):
        if name not in payload:raise ValueError('Missing required document')
    rows=list(csv.DictReader(io.StringIO(payload['labels.csv'].decode('utf-8-sig'))))
    if Counter(r['weight_g_reported'] for r in rows)!=Counter({str(w):3 for w in (20,40,60,80,100)}):raise ValueError('Expected 5 weights x 3')
    sums=dict((line.split(maxsplit=1)[1].strip().lstrip('*'),line.split()[0]) for line in payload['sha256.txt'].decode().splitlines() if line.strip())
    paths=set();groups={}
    for r in rows:
        name=r['relative_path'];data=payload.get(name)
        if not name.startswith('photos/') or Path(name).name!=r['filename'] or name in paths or data is None:raise ValueError('Bad original path')
        if sha(data)!=r['sha256'] or sums.get(name)!=r['sha256'] or len(data)!=int(r['bytes']):raise ValueError('Hash/size mismatch')
        with Image.open(io.BytesIO(data)) as im:
            if im.format!='JPEG' or im.size!=(int(r['width_px']),int(r['height_px'])):raise ValueError('JPEG dimensions mismatch')
            im.load()
        g=r['capture_group_id'];groups.setdefault(g,set()).add(r['weight_g_reported'])
        if not g or r['approved_for_training']!='no_pending_review' or r['approved_for_independent_test']!='no_pending_lineage_check':raise ValueError('Review approval metadata')
        paths.add(name)
    if len(groups)!=5 or any(len(v)!=1 for v in groups.values()):raise ValueError('Invalid capture groups')
    if {n for n in payload if n.startswith('photos/')}!=paths:raise ValueError('Unlabelled photo')
    before=_snapshot(existing);byhash={}
    for item in before['files']:byhash.setdefault(item['sha256'],[]).append(item['path'])
    prior=[]
    for batch,root in [('old58',existing/'Pasta_Lens_Private_Dataset/photos'),('previous15',existing/'intake_series_nov05/originals/photos'),('calibration3',existing/'calibration_50_34_nov08/originals/photos')]:
        # Previous receipt uses images/, not photos/ in some releases.
        if not root.exists() and batch=='previous15':root=existing/'intake_series_nov05/originals/images'
        for f in root.rglob('*'):
            if f.suffix.lower() in ('.jpg','.jpeg','.png'):
                sig,pixel=visual_signature(f);prior.append((batch,str(f.relative_to(existing)),sig,pixel))
    output.mkdir(parents=True)
    (output/'protection_before.json').write_text(json.dumps(before))
    for name,data in payload.items():
        p=output/'originals'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    new=[];ledger=[]
    for r in rows:
        sig,pixel=visual_signature(output/'originals'/r['relative_path']);new.append((r['filename'],sig,pixel))
        nearest=[]
        for batch,path,other,otherpixel in prior:
            distance=min(int(np.count_nonzero(a!=b)) for a in sig for b in other)
            nearest.append(dict(batch=batch,path=path,phash_hamming=distance,pixel_equal=pixel==otherpixel,review_candidate=distance<=10))
        nearest.sort(key=lambda x:x['phash_hamming'])
        ledger.append(dict(r,exact_duplicate_paths=byhash.get(r['sha256'],[]),decoded_pixel_sha256=pixel,nearest_prior=nearest[:5],near_prior_candidates=[x for x in nearest if x['review_candidate']],
            conservative_lineage_block='UNKNOWN_RELATIONSHIP_ALL_WEIGHTS_AND_OLDER_BATCHES',split='UNASSIGNED',research_use='explicitly_requested_exploratory_fit_only',production_training_approved=False,mask_approved=False))
    within=[dict(a=a[0],b=b[0],phash_hamming=min(int(np.count_nonzero(x!=y)) for x in a[1] for y in b[1]),pixel_equal=a[2]==b[2]) for i,a in enumerate(new) for b in new[i+1:]]
    report=dict(archive_sha256=sha(archive.read_bytes()),images=15,capture_groups=5,independent_test_images=0,approval_status='UNCHANGED_PENDING',prior_image_counts=dict(Counter(x[0] for x in prior)),
        exact_duplicates=sum(bool(x['exact_duplicate_paths']) for x in ledger),unique_original_sha256=len({r['sha256'] for r in rows}),within_batch_similarity=within,similarity_note='63-bit rotation-aware pHash <=10 is a review flag, not proof of duplicate or physical identity; may miss crop/recompression differences.',rows=ledger)
    (output/'receipt.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--zip',required=True);p.add_argument('--existing',required=True);p.add_argument('--output',required=True);a=p.parse_args();r=receive(a.zip,a.existing,a.output);print(r['prior_image_counts'],r['exact_duplicates'])
