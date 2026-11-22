"""Compare scale-free dry-pasta features on private originals without a sheet."""
import argparse
from importlib.metadata import version
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np

from .foundation_backends import DinoV3, MAX_SIDE, SEED, embedding_image, start_segmenter
from .foundation_benchmark import ALPHA, _write_json, cohort, decode
from .foundation_resources import torch_runtime
from .product_calibration import load_profile
from .stability_experiment import transformations
from .stability_pipeline import normalize_size, shape_features
from .weight_evaluation import prediction_drift, summarize_attempts

FEATURES = {
    'frame_geometry': ('mask_fraction', 'quantile_length_relative', 'quantile_width_relative',
                       'quantile_density', 'profile_width_cv'),
    'object_geometry': ('area_over_length_squared', 'width_over_length',
                        'quantile_density', 'profile_width_cv'),
}
DEFAULT_VIEWS = ('resize50', 'canvas_scale75', 'rotate90', 'rotate3', 'brightness110')


def product_features(geometry, measurements, mass_per_strand_g):
    """Scale proxy for full-length parallel pasta viewed nearly from above.

    A mask can merge touching strands. Projected length is the visible long
    axis of the arrangement. It equals nominal strand length only under the
    assumed setup. This proxy is a feature, not a measurement or a gram result.
    """
    length_to_diameter = measurements['length_mm'] / measurements['diameter_mm']
    proxy = geometry['area_over_length_squared'] * length_to_diameter * mass_per_strand_g
    values = [length_to_diameter, mass_per_strand_g / measurements['length_mm'], proxy]
    if not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError('Invalid product reference features')
    return values


def features(rgb, mask, encoder=None, product=None):
    if mask.dtype != np.bool_ or mask.shape != rgb.shape[:2]:
        raise ValueError('Segmentation mask must be binary and image-aligned')
    output = {}
    if encoder is not None:
        output['full_image_embedding'] = encoder(embedding_image(rgb))
    if not mask.any():
        return output
    geometry = shape_features(mask, SEED)
    for label, columns in FEATURES.items():
        output[label] = [geometry[column] for column in columns]
    safe_mask = cv2.dilate(mask.astype(np.uint8), np.ones((9,9), np.uint8)) == 0
    if safe_mask.any():
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[safe_mask]
        output['background_control'] = [float(lab[:, 0].mean()), float(lab[:, 0].std()),
                                        float(lab[:, 1].mean()), float(lab[:, 2].mean())]
    if encoder is not None:
        encoded = encoder(embedding_image(rgb, mask))
        output['foreground_embedding'] = encoded
        output['object_with_embedding'] = output['object_geometry'] + encoded
    if product is not None and geometry['quantile_length_px'] > 1 and not (
            mask[[0, -1], :].any() or mask[:, [0, -1]].any()):
        output['object_with_product'] = output['object_geometry'] + product_features(
            geometry, product['measurements'], product['mean_weight_per_strand_g'])
    return output


def _load_encoder(clip_checkpoint, dino, torch, device):
    if clip_checkpoint:
        from ..local_dish_vision import CHECKPOINT_SHA256, MODEL, OpenClipVision
        model = OpenClipVision(clip_checkpoint, device)

        def encode(image):
            vector = model.encode(image)[0].float().cpu().numpy()
            if vector.shape != (512,) or not np.isfinite(vector).all():
                raise ValueError('Invalid OpenCLIP image embedding')
            return vector.tolist()

        return encode, dict(model=MODEL, open_clip_version=version('open-clip-torch'),
                            checkpoint_sha256=CHECKPOINT_SHA256)
    if dino:
        model = DinoV3(dino, device, torch)
        return model, model.metadata
    return None, None


def _load_matches(profiles, match_path, originals):
    by_name = {}
    for path in profiles:
        profile = load_profile(path)
        name = profile['measurements']['product_name']
        if name in by_name:
            raise ValueError('Use only one reviewed calibration profile per product')
        by_name[name] = profile
    if match_path is None:
        return {}
    matches = json.loads(match_path.read_text(encoding='utf-8-sig'))
    if not isinstance(matches, dict) or not set(matches) <= set(originals):
        raise ValueError('Product matches must be indexed by verified original SHA-256')
    result = {}
    for digest, match in matches.items():
        name = match['product_name']
        if name not in by_name or match.get('product_and_intact_single_layer_reviewed') is not True:
            raise ValueError('Each product match requires a reviewed intact, aligned single layer and a named profile')
        result[digest] = by_name[name]
    return result


def compare(rows, variants):
    """Fit only originals, apply the same fixed estimator to each derivative."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    labelled = [r for r in rows if r['measured_g'] is not None]

    def fit(variant, selected):
        train = [r for r in selected if variant in r['features']]
        model = None
        if len(train) >= 2:
            x = [r['features'][variant] for r in train]
            model = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA)).fit(x, [r['measured_g'] for r in train])
        def predict(vectors):
            if model is None or variant not in vectors:
                return None
            grams = float(model.predict([vectors[variant]])[0])
            return grams if grams > 0 and math.isfinite(grams) else None

        attempts = []
        for row in selected:
            attempts.append(dict(sample_id=row['original_sha256'], actual_g=row['measured_g'],
                                 predicted_g=predict(row['features']),
                                 transformed_predictions={name: predict(vectors) for name, vectors in row['views'].items()}))
        metrics = summarize_attempts(attempts)
        # Hash-based bookkeeping scores images. It cannot certify physical independence.
        return dict(fit_images=len(train) if model is not None else 0, predictions_returned=metrics['accepted'],
                    training_mae_g=metrics['mae_g'], training_rmse_g=metrics['rmse_g'],
                    training_max_abs_error_g=metrics['max_absolute_error_g'],
                    derivatives=prediction_drift(attempts))

    active = [name for name in variants if any(name in row['features'] for row in labelled)]
    common = [row for row in labelled if set(active) <= row['features'].keys()]
    return dict(available={v: fit(v, labelled) for v in variants}, common_originals=len(common),
                same_images={v: fit(v, common) for v in active})


def run(entries, args):
    cv2.setNumThreads(1)
    torch = torch_runtime(not args.backend.startswith('opencv') or args.openclip or args.dinov3, args.device)
    segment = start_segmenter(args.backend, args.checkpoint, 'cpu' if args.backend.startswith('opencv') else args.device, torch)
    encoder, encoder_config = _load_encoder(args.openclip, args.dinov3, torch, args.device)
    matches = _load_matches(args.product_profile, args.product_matches, [r['sha256'] for r in entries])
    def extract(image, product):
        if args.canonicalize:
            image = normalize_size(image, MAX_SIDE)
        return features(image, segment(image), encoder, product)

    rows = []
    for entry in entries:
        rgb, _ = decode(entry)
        digest = entry['sha256']
        product = matches.get(digest)
        original = extract(rgb, product)
        # A padded derivative cannot recover the original's clipped ends.
        if 'object_with_product' not in original:
            product = None
        views = {name: extract(image, product) for name, image in transformations(rgb) if name in args.view}
        rows.append(dict(original_sha256=digest, physical_family_id=entry['physical_family_id'],
                         material_stage_id=entry['material_stage_id'], measured_g=entry['reported_g'],
                         features=original, views=views))

    variants = [*FEATURES, 'background_control', 'object_with_product']
    if encoder is not None:
        variants += ['full_image_embedding', 'foreground_embedding', 'object_with_embedding']
    report = dict(scope='exploratory_fits_on_originals', originals=len(rows),
                  weight_labelled_originals=sum(r['measured_g'] is not None for r in rows),
                  cumulative_originals=sum(r['physical_family_id'] is not None for r in rows),
                  independent_test_attempts=0, independent_mae_g=None,
                  physical_distance_evaluation='not_run', production_eligible=False,
                  model='StandardScaler + Ridge', alpha=ALPHA, backend=segment.metadata, encoder=encoder_config,
                  geometry_fields=FEATURES, canonical_long_side_px=MAX_SIDE if args.canonicalize else None,
                  reviewed_product_matches=len(matches),
                  product_profile_sha256=sorted({p['source_sha256'] for p in matches.values()}),
                  transforms=list(args.view), results=compare(rows, variants))
    args.output.mkdir(parents=True, exist_ok=False)
    _write_json(args.output/'features.json', rows)
    _write_json(args.output/'report.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, action='append', required=True)
    parser.add_argument('--lineage-revision', type=Path, action='append', default=[])
    parser.add_argument('--backend', choices=('opencv_hsv', 'opencv_lab', 'mobile_sam', 'sam2', 'sam31'), required=True)
    parser.add_argument('--checkpoint', type=Path)
    encoder = parser.add_mutually_exclusive_group()
    encoder.add_argument('--openclip', type=Path)
    encoder.add_argument('--dinov3', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--canonicalize', action='store_true', help='Normalize input long side before extraction')
    parser.add_argument('--view', choices=DEFAULT_VIEWS, action='append', help='Repeat to select derivative checks')
    parser.add_argument('--product-profile', type=Path, action='append', default=[])
    parser.add_argument('--product-matches', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Provide a new output directory outside the original dataset')
    args.view = tuple(dict.fromkeys(args.view or DEFAULT_VIEWS))
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    try:
        run(cohort(args.dataset, args.lineage_revision, args.output), args)
    except (OSError, ValueError, ImportError, KeyError) as exc:
        parser.error(str(exc))
    print(args.output/'report.json')


if __name__ == '__main__':
    main()
