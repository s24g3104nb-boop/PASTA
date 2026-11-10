"""Reproducible image perturbations and preprocessing ablations, offline only."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform

import cv2
import numpy as np
from PIL import Image

from ..pasta_vision import image_to_rgb
from .stability_pipeline import configurations, process, raw_mask, VERSION
from .direct_features import extract

TRANSFORM_SETTINGS = dict(resize_factors=[.5, .25], down='INTER_AREA', up='INTER_CUBIC',
                          jpeg_quality=[85, 60], jpeg_subsampling='OpenCV default',
                          rotation_degrees=[3, -3], rotation_border='expanded black canvas; no original cropping',
                          brightness_gain=[.9, 1.1], scale_stress=.75)


def rotated(rgb, degrees):
    h, w = rgb.shape[:2]
    theta = np.deg2rad(degrees)
    nw = int(np.ceil(abs(w*np.cos(theta))+abs(h*np.sin(theta))))
    nh = int(np.ceil(abs(h*np.cos(theta))+abs(w*np.sin(theta))))
    mat = cv2.getRotationMatrix2D(((w-1)/2, (h-1)/2), degrees, 1.)
    mat[:, 2] += [(nw-w)/2, (nh-h)/2]
    return cv2.warpAffine(rgb, mat, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def transformations(rgb):
    yield 'original', rgb.copy()
    for factor, name in [(.5, '50'), (.25, '25')]:
        small = cv2.resize(rgb, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        yield 'resize'+name, small
        yield 'roundtrip'+name, cv2.resize(small, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
    for name, interpolation in [('linear', cv2.INTER_LINEAR), ('nearest', cv2.INTER_NEAREST)]:
        yield 'resize50_'+name, cv2.resize(rgb, None, fx=.5, fy=.5, interpolation=interpolation)
    for quality in (85, 60):
        ok, buf = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok: raise ValueError('JPEG encoder failed')
        yield 'jpeg'+str(quality), cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    for angle in (3, -3):
        yield 'rotate'+str(angle), rotated(rgb, angle)
    for gain, name in [(.9, 'brightness90'), (1.1, 'brightness110')]:
        yield name, np.clip(np.rint(rgb.astype(float)*gain), 0, 255).astype(np.uint8)
    yield 'rotate90', np.rot90(rgb).copy()
    # Digital canvas test is NOT a measured camera-distance or perspective experiment.
    small = cv2.resize(rgb, None, fx=.75, fy=.75, interpolation=cv2.INTER_AREA)
    canvas = np.zeros_like(rgb)
    y = (rgb.shape[0]-small.shape[0])//2; x = (rgb.shape[1]-small.shape[1])//2
    canvas[y:y+small.shape[0], x:x+small.shape[1]] = small
    yield 'canvas_scale75', canvas


def pixel_sha(rgb):
    return hashlib.sha256(str(rgb.shape).encode()+rgb.tobytes()).hexdigest()


def write_preview(rgb, path):
    im = Image.fromarray(rgb); im.thumbnail((480, 480)); im.save(path, quality=88)


def mask_agreement(candidate, original):
    # Diagnostic self-consistency, not IoU against a ground-truth pasta mask.
    aligned = cv2.resize(candidate, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    ref = original > 0
    return float(np.count_nonzero(aligned & ref)/max(1, np.count_nonzero(aligned | ref)))


def run(previous, output):
    previous, output = Path(previous), Path(output)
    if output.exists(): raise FileExistsError('No overwrite of prior extraction')
    receipt = json.loads((previous/'receipt.json').read_text())
    old = json.loads((previous/'features/features.json').read_text())
    for row in receipt['rows']:
        p = previous/'originals'/row['relative_path']
        if not p.resolve().is_relative_to((previous/'originals').resolve()) or hashlib.sha256(p.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Original path/hash mismatch')
    output.mkdir(parents=True)
    cv2.setNumThreads(2)
    configs = configurations()
    rows, inputs, ablations, reproduced = [], [], [], []
    for record in receipt['rows']:
        ident = Path(record['filename']).stem
        rgb = image_to_rgb(previous/'originals'/record['relative_path'])
        image_dir = output/ident; image_dir.mkdir()
        references = {}
        for transform, scene in transformations(rgb):
            folder = image_dir/transform; folder.mkdir()
            Image.fromarray(scene).save(folder/'input.png')
            write_preview(scene, folder/'input_preview.jpg')
            inputs.append(dict(image_id=ident, transform=transform, shape=list(scene.shape), pixel_sha256=pixel_sha(scene),
                               path=str((folder/'input.png').relative_to(output)), original_sha256=record['sha256'],
                               capture_group=record['capture_group_id'], lineage_block=record['conservative_lineage_block']))
            for config in configs:
                f, processed, mask = process(scene, config)
                if transform == 'original': references[config.name] = mask.copy()
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                overlay = processed.copy(); cv2.drawContours(overlay, contours, -1, (255, 0, 255), 1)
                prefix = folder/config.name
                Image.fromarray(mask).save(str(prefix)+'_mask.png')
                write_preview(overlay, Path(str(prefix)+'_overlay.jpg'))
                match = None if transform.startswith('rotate') or transform == 'canvas_scale75' else mask_agreement(mask, references[config.name])
                row = dict(f, image_id=ident, reported_g=int(record['weight_g_reported']), capture_group=record['capture_group_id'],
                           lineage_block=record['conservative_lineage_block'], original_sha256=record['sha256'], transform=transform,
                           pipeline=config.name, config_sha256=config.digest, mask_sha256=pixel_sha(mask),
                           processed_shape=list(processed.shape), self_mask_iou=match, mask_approved=False,
                           mask_path=str(prefix.relative_to(output))+'_mask.png', overlay_path=str(prefix.relative_to(output))+'_overlay.jpg')
                rows.append(row)
                if config.name.startswith('A_') and transform in ('original', 'resize50'):
                    method = 'hsv_baseline' if config.segmentation == 'hsv' else 'hsv_lab'
                    ref = next(r for r in old['rows'] if r['image_id'] == ident and r['method'] == method and r['transform'] == ('original' if transform == 'original' else 'resize_half'))
                    difference = max(abs(row[k]-ref[k]) for k in config.keys)
                    reproduced.append(dict(image_id=ident, pipeline=config.name, transform=transform, maximum_feature_difference=difference))
                    if difference > 1e-8: raise ValueError('Legacy feature reproduction failed')
            # Fixed-pixel morphology and interpolation audit, separate from fitted pipelines.
            if transform in ('original', 'resize50', 'resize25', 'resize50_linear', 'resize50_nearest'):
                raw = raw_mask(scene, configs[1])
                for kernel in (1, 3, 5):
                    mask = raw if kernel == 1 else cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((kernel, kernel), np.uint8))
                    f, _ = extract(scene, mask)
                    Image.fromarray(mask).save(folder/f'ablation_close{kernel}.png')
                    ablations.append(dict(image_id=ident, transform=transform, closing_kernel=kernel,
                                          area_fraction=f['mask_fraction'], area_px=f['mask_area_px'], components=f['components'],
                                          contour_length=f['contour_perimeter_px'], skeleton_density=f['skeleton_density_512'],
                                          removed_or_added_pixels=int(np.count_nonzero(mask != raw))))
        print(ident, 'completed', flush=True)
    report = dict(version=VERSION, configuration=[asdict(c) for c in configs], feature_keys={c.name:c.keys for c in configs},
                  transform_settings=TRANSFORM_SETTINGS, source_images=15, independent_test_images=0, production_approved=False,
                  transform_is_independent_evaluation=False, input_receipt_sha256=hashlib.sha256((previous/'receipt.json').read_bytes()).hexdigest(),
                  versions=dict(python=platform.python_version(), numpy=np.__version__, opencv=cv2.__version__),
                  source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), Path(__file__).with_name('stability_pipeline.py')]},
                  rows=rows, inputs=inputs, morphology_ablations=ablations, legacy_reproduction=reproduced)
    (output/'features.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--previous', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); run(args.previous, args.output)
