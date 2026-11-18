"""Read-only intake and OpenCV diagnosis. Missing originals remain unmeasured."""
import argparse
import hashlib
import html
import json
import math
from pathlib import Path
from pathlib import PurePosixPath
import sys


def photo_spec(record):
    """Read labels for later comparison, never supply them to detection."""
    name = record['filename']
    relative = record.get('zip_path', name)
    path = PurePosixPath(relative)
    if (not isinstance(name, str) or Path(name).name != name or '\\' in name
            or name in ('.', '..') or path.is_absolute() or '..' in path.parts
            or '\\' in relative or ':' in relative or path.name != name):
        raise ValueError('Unsafe image path')
    label = record.get('label')
    weight, count = record.get('user_reported_weight_g'), None
    if label is not None:
        if label['kind'] == 'measured_weight_g':
            weight = label['value']
        elif label['kind'] == 'strand_count':
            count = label['value']
            weight = label.get('weight_g')
        else:
            raise ValueError('Unknown label kind')
    for value in (weight, count):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value) or value <= 0):
            raise ValueError('Invalid positive label')
    if count is not None and int(count) != count:
        raise ValueError('Count label must be an integer')
    return path, weight, count


def manifest_records(manifest):
    """Normalize the two supplied manifest schemas without inferring labels."""
    if 'photos' in manifest and 'images' in manifest:
        raise ValueError('Ambiguous manifest schemas')
    if 'images' in manifest:
        records = []
        for image in manifest['images']:
            weight = image.get('weight_g_user_reported')
            count = image.get('strand_count_user_reported')
            if weight is None and count is None:
                raise ValueError('A weight or count label is required')
            label = (dict(kind='strand_count', value=count, weight_g=weight) if count is not None else
                     dict(kind='measured_weight_g', value=weight))
            records.append(dict(filename=image['filename'], zip_path=image['file'],
                                sha256=image['image_sha256'], label=label, group=image.get('set'),
                                expected_bytes=image.get('bytes')))
        expected = manifest.get('images_count', len(records))
    else:
        records = manifest['photos']
        expected = manifest.get('n_images', len(records))
    if expected != len(records):
        raise ValueError('Image count does not match manifest')
    return records


def run(manifest_path, images, output, calibration=None):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pasta_weight_app_v2'))
    from pasta_vision import image_to_rgb, detect_spread, image_quality_warnings
    from measurement_diagnostics import decide_count_weight
    from PIL import Image
    if calibration is not None and (isinstance(calibration, bool) or not (0 < calibration < float('inf'))):
        raise ValueError('Calibration must be finite and positive')
    manifest_path, images, output = map(lambda p: Path(p).resolve(), (manifest_path, images, output))
    if output.exists() or output.is_relative_to(images):
        raise ValueError('Use a new output folder outside the original image folder')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    records = manifest_records(manifest)
    names = [r['filename'] for r in records]
    if len(names) != len(set(names)) or any(Path(n).name != n or n in ('.','..') for n in names):
        raise ValueError('Unsafe or duplicate filename')
    for record in records:
        photo_spec(record)
        digest = record['sha256']
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid SHA-256')
    output.mkdir(parents=True)
    rows, cards = [], []
    for record in records:
        filename = record['filename']
        relative, weight_label, count_label = photo_spec(record)
        path = images.joinpath(*relative.parts)
        row = dict(filename=filename, expected_sha256=record['sha256'], status='missing_original',
                   submitted_path=relative.as_posix(), reported_g=weight_label, reported_count=count_label,
                   count=None, count_absolute_error=None, absolute_error_g=None,
                   hypothetical_weight_g=None, displayed_weight_g=None, reasons=[],
                   mask_evaluation='not_measured', features=None,
                   independent_evaluation_eligible=False, training_approved=False)
        if path.is_symlink() or not path.resolve().is_relative_to(images):
            raise ValueError('Image path resolves outside input folder or is a symlink')
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            row.update(actual_sha256=digest, status='hash_mismatch')
            if digest == record['sha256']:
                try:
                    with Image.open(path) as image:
                        dimensions = list(image.size)
                    expected_dimensions = record.get('size_pixels')
                    if expected_dimensions and expected_dimensions != dimensions:
                        raise ValueError('Dimensions do not match manifest')
                    rgb = image_to_rgb(path)
                except (OSError, ValueError, Image.DecompressionBombError) as exc:
                    row.update(status='decode_failed', reasons=[str(exc)])
                    rows.append(row)
                    cards.append('<pre>'+html.escape(json.dumps(row,ensure_ascii=False,indent=2))+'</pre>')
                    continue
                result = detect_spread(rgb)
                value = decide_count_weight(dict(name='診断対象製品',g_per_strand=calibration),result,mode='広げた状態')
                quality = image_quality_warnings(rgb)
                row.update(status='diagnosed',count=result.count,foreground_pixels=value.foreground_pixels,
                           count_absolute_error=abs(result.count-count_label) if count_label is not None else None,
                           evidence=result.evidence, reasons=list(value.reasons)+quality,
                           quality_warnings=quality, image_shape=list(rgb.shape),
                           mask_evaluation='manual_review_required',
                           hypothetical_weight_g=result.count*calibration if calibration else None,
                           displayed_weight_g=value.weight_g if not quality else None,
                           calibration_status='operator_supplied_product_match_required' if calibration else 'not_provided')
                if row['displayed_weight_g'] is not None and weight_label is not None:
                    row['absolute_error_g'] = abs(row['displayed_weight_g']-weight_label)
                for suffix, array in [('mask',result.mask),('detections',result.overlay_rgb)]:
                    name = filename + '.' + suffix + '.png'
                    Image.fromarray(array).save(output/name)
                    cards.append('<img loading="lazy" width="360" src="'+html.escape(name)+'" alt="'+html.escape(filename+' '+suffix)+'">')
                if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise RuntimeError('Original changed during diagnosis')
        rows.append(row)
        cards.append('<pre>'+html.escape(json.dumps(row,ensure_ascii=False,indent=2))+'</pre>')
    report=dict(rows=rows, originals_expected=len(rows), originals_verified=sum(r['status']=='diagnosed' for r in rows),
                manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                unresolved_lineage_policy='one_unresolved_series_no_train_test_split',
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
