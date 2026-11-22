"""Run a private, local foundation-model comparison on verified original photos."""
import argparse
import json
import os
from pathlib import Path
import platform
from statistics import mean, median
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageOps

from ..pasta_vision import image_to_rgb
from .area_calibration import product_geometry_grams
from .foundation_backends import MAX_SIDE, SEED, DinoV3, embedding_image, sha256_file, start_segmenter
from .foundation_masks import a4_projection, annotate_mask, mask_scores, read_a4_reviews, read_annotations, run_masks
from .foundation_resources import device_sync, memory_report, torch_runtime
from .lineage_revision import verify_revision
from .photo_batch_diagnosis import FEATURE_GROUPS, image_features, load_diagnostic_entries
from .stability_pipeline import shape_features
from .weight_evaluation import summarize_attempts

SEGMENT_SCHEMA = 'pasta-foundation-segmentation-v1'
GEOMETRY = ('mask_fraction', 'area_over_length_squared', 'width_over_length',
            'quantile_density', 'profile_width_cv')
BACKGROUND = ('Lab_L_mean', 'Lab_L_sd', 'Lab_a_mean', 'Lab_b_mean')
A4 = ('sheet_projection_mm2', 'sheet_projection_length_mm', 'sheet_projection_width_mm')
ALPHA = 10.0


def cohort(datasets, revision_paths, output):
    entries = load_diagnostic_entries(datasets, output)
    if not entries:
        raise ValueError('An original cohort is required')
    cumulative = {}
    for path in revision_paths:
        content = json.loads(Path(path).read_text(encoding='utf-8'))
        roots = {e['dataset_root'] for e in entries if e['manifest_sha256'] == content['source_manifest_sha256']}
        if len(roots) != 1:
            raise ValueError('Each cumulative revision must match exactly one original manifest')
        verified = verify_revision(path, roots.pop())
        for stage in verified['stages']:
            for photo in stage['photos']:
                digest = photo['original_sha256']
                if digest in cumulative:
                    raise ValueError('An original cannot belong to two cumulative revisions')
                cumulative[digest] = verified['physical_family_id'], stage['material_id']
    for entry in entries:
        digest = entry['sha256']
        entry['physical_family_id'], entry['material_stage_id'] = cumulative.get(digest, (None, None))
    return entries


def decode(entry):
    if sha256_file(entry['path']) != entry['sha256']:
        raise ValueError('Original bytes changed during the comparison')
    with Image.open(entry['path']) as image:
        shape = tuple(reversed(ImageOps.exif_transpose(image).size))
    return image_to_rgb(entry['path'], max_side=MAX_SIDE), shape


def _new_output(out):
    out.mkdir(parents=True, exist_ok=False)
    return out


def _write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def runtime_info(torch):
    return dict(os=platform.system(), machine=platform.machine(), python=platform.python_version(),
                opencv=cv2.__version__, torch=torch.__version__ if torch is not None else None,
                cv_threads=1, torch_cpu_threads=1 if torch is not None else None)


def segment(entries, output, backend, *, annotations=None, a4_reviews=None, device='cpu', checkpoint=None):
    cv2.setNumThreads(1)
    torch = torch_runtime(not backend.startswith('opencv'), device)
    start = perf_counter()
    segmenter = start_segmenter(backend, checkpoint, device, torch)
    device_sync(torch, device)
    load_ms = (perf_counter()-start)*1000
    digests = [e['sha256'] for e in entries]
    references = read_annotations(annotations, digests)
    reviews = read_a4_reviews(a4_reviews, digests)
    out = _new_output(output)
    rows = []
    for entry in entries:
        rgb, native_shape = decode(entry)
        digest = entry['sha256']
        start = perf_counter()
        mask = segmenter(rgb)
        device_sync(torch, device)
        elapsed = (perf_counter()-start)*1000
        if mask.dtype != np.bool_ or mask.shape != rgb.shape[:2]:
            raise ValueError('Backend mask must align with the oriented input')
        path = out/f'{digest}.png'
        Image.fromarray(mask.astype(np.uint8)*255).save(path)
        overlay = rgb.copy()
        overlay[mask] = np.rint(.58*rgb[mask] + .42*np.array([0, 200, 210])).astype(np.uint8)
        Image.fromarray(overlay).save(out/f'{digest}.overlay.jpg', quality=88)
        reference_scores = None
        if digest in references:
            reference = annotate_mask(references[digest], native_shape, mask.shape)
            reference_scores = mask_scores(mask, reference)
            Image.fromarray(reference.astype(np.uint8)*255).save(out/f'{digest}.reference.png')
        rows.append(dict(original_sha256=digest, shape=list(mask.shape), mask_sha256=sha256_file(path),
                         physical_family_id=entry['physical_family_id'], material_stage_id=entry['material_stage_id'],
                         candidate_fraction=float(mask.mean()), has_candidate=bool(mask.any()),
                         segmentation_ms=elapsed, manual_mask_scores=reference_scores,
                         a4=a4_projection(rgb, mask, reviews.get(digest))))
    times = [row['segmentation_ms'] for row in rows]
    scored = [row['manual_mask_scores'] for row in rows if row['manual_mask_scores'] is not None]
    summary = {}
    for field in ('iou', 'background_in_predicted_fraction', 'missed_visible_pasta_fraction'):
        available = [score[field] for score in scored if score[field] is not None]
        summary[field] = dict(mean=mean(available) if available else None, scored_originals=len(available))
    report = dict(schema=SEGMENT_SCHEMA, backend=segmenter.metadata, originals=len(rows),
                  annotated_originals=len(references), manual_mask_summary=summary,
                  max_input_side=MAX_SIDE, load_ms=load_ms,
                  first_photo_ms=times[0], median_after_first_ms=median(times[1:]) if len(times) > 1 else None,
                  max_photo_ms=max(times), runtime=runtime_info(torch), memory=memory_report(torch, device),
                  images=rows)
    _write_json(out/'segmentation.json', report)
    return report


def feature_vectors(rgb, masks, encoded, projections):
    """Only pixels, masks and aligned sheet geometry enter the feature extractor."""
    result = {}
    if encoded is not None:
        result['dinov3_full'] = encoded(embedding_image(rgb))
    for method, mask in masks.items():
        if not np.any(mask):
            continue
        shape = shape_features(mask, SEED)
        geom = [shape[field] for field in GEOMETRY]
        result[method+'_geometry'] = geom
        result[method+'_image_area_fraction'] = geom[:1]
        if projections[method] is not None:
            result[method+'_a4_geometry'] = geom+projections[method]
            result[method+'_a4_area_only'] = projections[method][:1]
        if method == 'opencv_hsv':
            traditional = image_features(rgb, mask.astype(np.uint8)*255)
            keys = list(dict.fromkeys(FEATURE_GROUPS['scene'] + FEATURE_GROUPS['shape_texture']))
            result['opencv_features'] = [traditional[field] for field in keys]
            # Fix the reference background mask across all SAM comparisons and
            # exclude a 9x9 margin to reduce direct leakage from pasta pixels.
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
            safe = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)) == 0
            if safe.any():
                pixels = lab[safe]
                result['background_only'] = [float(pixels[:, 0].mean()), float(pixels[:, 0].std()),
                                              float(pixels[:, 1].mean()), float(pixels[:, 2].mean())]
        if encoded is not None and method in ('mobile_sam', 'sam2', 'sam31'):
            result[method+'_dinov3'] = encoded(embedding_image(rgb, mask))
    return result


def expect_variants(methods, use_dino):
    result = ['opencv_features', 'background_only']
    if use_dino:
        result.append('dinov3_full')
    for method in methods:
        result.extend((method+'_geometry', method+'_image_area_fraction', method+'_a4_geometry', method+'_a4_area_only'))
        if use_dino and method in ('mobile_sam', 'sam2', 'sam31'):
            result.append(method+'_dinov3')
    return result


def exploratory_fits(rows, variants):
    """Same fixed Ridge. Count-only originals contribute masks, never weight targets."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    labelled = [row for row in rows if row['measured_g'] is not None]

    def fit(method, selected):
        fitted = [row for row in selected if method in row['features']]
        predictions = {}
        if len(fitted) >= 2:
            regressor = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA))
            x = [r['features'][method] for r in fitted]
            regressor.fit(x, [r['measured_g'] for r in fitted])
            predictions = {row['original_sha256']: float(pred)
                           for row, pred in zip(fitted, regressor.predict(x)) if pred > 0 and np.isfinite(pred)}
        attempts = [dict(sample_id=row['original_sha256'], actual_g=row['measured_g'],
                         predicted_g=predictions.get(row['original_sha256'])) for row in selected]
        scores = summarize_attempts(attempts)
        return dict(labelled_originals=len(selected), feature_vectors=len(fitted), prediction_count=scores['accepted'],
                    fitted_fraction=scores['success_rate'], training_mae_g=scores['mae_g'],
                    training_rmse_g=scores['rmse_g'], training_max_absolute_error_g=scores['max_absolute_error_g'],
                    fitted_predictions=predictions)

    # Score available cases and the shared cohort. A4 is a separate paired
    # comparison, so absent markers cannot empty all the main comparisons.
    main = [v for v in variants if '_a4_' not in v]
    shared = [row for row in labelled if set(main) <= row['features'].keys()]
    a4_paired = {}
    for method in variants:
        if method.endswith('_a4_geometry'):
            backend = method.removesuffix('_a4_geometry')
            pair = (backend+'_geometry', backend+'_a4_area_only', method)
            matched = [row for row in labelled if set(pair) <= row['features'].keys()]
            a4_paired[method] = {v: fit(v, matched) for v in pair}
    return dict(available={v: fit(v, labelled) for v in variants},
                shared_unscaled={v: fit(v, shared) for v in main},
                a4_paired=a4_paired)


def regress(entries, folders, output, *, dino=None, device='cpu', product_geometry=None):
    cv2.setNumThreads(1)
    torch = torch_runtime(dino is not None, device)
    start = perf_counter()
    encoder = DinoV3(dino, device, torch) if dino is not None else None
    device_sync(torch, device)
    load_ms = (perf_counter()-start)*1000 if encoder is not None else None
    digests = [entry['sha256'] for entry in entries]
    # Decode one image at a time in inference. Here dimensions alone are needed.
    shapes = {entry['sha256']: decode(entry)[0].shape[:2] for entry in entries}
    geometry = json.loads(Path(product_geometry).read_text(encoding='utf-8-sig')) if product_geometry else {}
    if not isinstance(geometry, dict) or not set(geometry) <= set(digests):
        raise ValueError('Product calibration must be indexed by original SHA-256')
    methods = {}
    for folder in folders:
        kind, result = run_masks(folder, digests, shapes)
        if kind in methods or kind not in ('opencv_hsv', 'opencv_lab', 'mobile_sam', 'sam2', 'sam31'):
            raise ValueError('Each segmentation method needs one unique run')
        methods[kind] = (folder, result)
    if 'opencv_hsv' not in methods:
        raise ValueError('Provide the baseline opencv_hsv segmentation folder')
    out = _new_output(output)
    rows = []
    direct_trials = {kind: [] for kind in methods}
    for entry in entries:
        digest = entry['sha256']
        rgb, _ = decode(entry)
        masks, projections = {}, {}
        for kind, (folder, result) in methods.items():
            row = result[digest]
            if (row['physical_family_id'], row['material_stage_id']) != (entry['physical_family_id'], entry['material_stage_id']):
                raise ValueError('Segmentation lineage does not match current cohort')
            path = Path(folder)/f'{digest}.png'
            if sha256_file(path) != row['mask_sha256']:
                raise ValueError('Segmentation mask hash mismatch')
            with Image.open(path) as image:
                mask = np.asarray(image)
            if mask.shape != rgb.shape[:2] or not np.isin(mask, [0, 255]).all():
                raise ValueError('Expected source-aligned binary mask PNG')
            masks[kind], projections[kind] = mask != 0, row['a4']['features']
        if entry['reported_g'] is not None:
            for kind, projected in projections.items():
                calibration = geometry.get(digest)
                predicted = None
                if projected is not None and calibration is not None:
                    if calibration.get('measured_product_confirmed') is not True:
                        raise ValueError('Only confirmed product measurements may be used')
                    predicted = product_geometry_grams(projected[0],
                        strand_length_mm=calibration['strand_length_mm'],
                        strand_diameter_mm=calibration['strand_diameter_mm'],
                        grams_per_strand=calibration['grams_per_strand'])
                direct_trials[kind].append(dict(sample_id=digest, actual_g=entry['reported_g'], predicted_g=predicted))
        start = perf_counter()
        features = feature_vectors(rgb, masks, encoder, projections)
        device_sync(torch, device)
        rows.append(dict(original_sha256=digest, physical_family_id=entry['physical_family_id'],
                         material_stage_id=entry['material_stage_id'], measured_g=entry['reported_g'],
                         feature_ms=(perf_counter()-start)*1000, features=features))
    report = dict(scope='exploratory_training_on_originals', model='StandardScaler + Ridge', alpha=ALPHA,
                  independent_test_status='not_run', independent_mae_g=None, production_eligible=False,
                  originals=len(rows), measured_originals=sum(r['measured_g'] is not None for r in rows),
                  embeddings=encoder.metadata if encoder is not None else None, embedding_load_ms=load_ms,
                  geometry_columns=GEOMETRY, background_columns=BACKGROUND, a4_additional_columns=A4,
                  runtime=runtime_info(torch), memory=memory_report(torch, device),
                  product_geometry_diagnostic={kind: summarize_attempts(trials) for kind, trials in direct_trials.items()},
                  geometry_product_profiles_present=len(geometry),
                  fits=exploratory_fits(rows, expect_variants(methods, encoder is not None)))
    _write_json(out/'features.json', rows)
    _write_json(out/'regression.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument('--dataset', type=Path, action='append', required=True)
    shared.add_argument('--lineage-revision', type=Path, action='append', default=[])
    shared.add_argument('--output', type=Path, required=True)
    shared.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    sub = parser.add_subparsers(dest='command', required=True)
    masks = sub.add_parser('segment', parents=[shared])
    masks.add_argument('--backend', choices=('opencv_hsv', 'opencv_lab', 'mobile_sam', 'sam2', 'sam31'), required=True)
    masks.add_argument('--checkpoint', type=Path)
    masks.add_argument('--annotations', type=Path)
    masks.add_argument('--a4-reviews', type=Path)
    fits = sub.add_parser('regress', parents=[shared])
    fits.add_argument('--segmentation', type=Path, action='append', required=True)
    fits.add_argument('--dinov3', type=Path)
    fits.add_argument('--product-geometry', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('出力先には新しいフォルダーを指定してください。')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    try:
        entries = cohort(args.dataset, args.lineage_revision, args.output)
        if args.command == 'segment':
            if not args.backend.startswith('opencv') and (args.checkpoint is None or not args.checkpoint.is_file()):
                parser.error('公式のローカル重みを --checkpoint に指定してください。')
            segment(entries, args.output, args.backend, annotations=args.annotations,
                    a4_reviews=args.a4_reviews, checkpoint=args.checkpoint, device=args.device)
        else:
            regress(entries, args.segmentation, args.output, dino=args.dinov3, device=args.device, product_geometry=args.product_geometry)
    except (ValueError, OSError, ImportError) as exc:
        parser.error(str(exc))
    print(f'{len(entries)} verified originals. Outputs: {args.output}')


if __name__ == '__main__':
    main()
