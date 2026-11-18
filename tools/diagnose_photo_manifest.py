"""Read-only intake and OpenCV diagnosis. Missing originals remain unmeasured."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import sys


def run(manifest_path, images, output, calibration=None):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pasta_weight_app_v2'))
    from pasta_vision import image_to_rgb, detect_spread, image_quality_warnings
    from measurement_diagnostics import decide_count_weight
    from PIL import Image
    if calibration is not None and not (0 < calibration < float('inf')):
        raise ValueError('Calibration must be finite and positive')
    manifest_path, images, output = map(lambda p: Path(p).resolve(), (manifest_path, images, output))
    if output.exists() or output.is_relative_to(images):
        raise ValueError('Use a new output folder outside the original image folder')
    records = json.loads(manifest_path.read_text(encoding='utf-8'))['photos']
    names = [r['filename'] for r in records]
    if len(names) != len(set(names)) or any(Path(n).name != n or n in ('.','..') for n in names):
        raise ValueError('Unsafe or duplicate filename')
    for record in records:
        digest = record['sha256']
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid SHA-256')
    output.mkdir(parents=True)
    rows, cards = [], []
    for record in records:
        filename = record['filename']
        path = images / filename
        row = dict(filename=filename, expected_sha256=record['sha256'], status='missing_original',
                   reported_g=record.get('user_reported_weight_g'), count=None,
                   hypothetical_weight_g=None, displayed_weight_g=None, reasons=[],
                   independent_evaluation_eligible=False, training_approved=False)
        if path.is_file() and not path.is_symlink():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            row.update(actual_sha256=digest, status='hash_mismatch')
            if digest == record['sha256']:
                rgb = image_to_rgb(path)
                result = detect_spread(rgb)
                value = decide_count_weight(dict(name='診断対象製品',g_per_strand=calibration),result,mode='広げた状態')
                quality = image_quality_warnings(rgb)
                row.update(status='diagnosed',count=result.count,foreground_pixels=value.foreground_pixels,
                           evidence=result.evidence, reasons=list(value.reasons)+quality,
                           hypothetical_weight_g=result.count*calibration if calibration else None,
                           displayed_weight_g=value.weight_g if not quality else None,
                           calibration_status='operator_supplied_product_match_required' if calibration else 'not_provided')
                for suffix, array in [('mask',result.mask),('detections',result.overlay_rgb)]:
                    name = filename + '.' + suffix + '.png'
                    Image.fromarray(array).save(output/name)
                    cards.append('<img loading="lazy" width="360" src="'+html.escape(name)+'" alt="'+html.escape(filename+' '+suffix)+'">')
                if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise RuntimeError('Original changed during diagnosis')
        rows.append(row)
        cards.append('<pre>'+html.escape(json.dumps(row,ensure_ascii=False,indent=2))+'</pre>')
    report=dict(rows=rows, originals_expected=len(rows), originals_verified=sum(r['status']=='diagnosed' for r in rows),
                independent_mae_g=None, independent_rmse_g=None, model_training_performed=False)
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'index.html').write_text('<meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        '<title>Pasta Lens 写真診断</title><h1>写真ごとの診断</h1><p>原本の欠落・ハッシュ不一致は未測定です。'
        'hypothetical_weight_gは入力した校正値での暫定換算です。実製品との一致を別途確認してください。</p>'+''.join(cards),encoding='utf-8')
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True,type=Path)
    parser.add_argument('--images',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--calibration',type=float)
    args=parser.parse_args()
    result=run(args.manifest,args.images,args.output,args.calibration)
    print(f"Verified originals: {result['originals_verified']}/{result['originals_expected']}")
