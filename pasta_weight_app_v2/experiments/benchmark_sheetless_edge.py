"""Compare two edge policies on deterministic synthetic images.

Run from the repository root using the project's virtual environment:

    python -m pasta_weight_app_v2.experiments.benchmark_sheetless_edge --width

The 1 g per drawn strand label is synthetic. The experiment leaves the shipped
calibration fixed and separately measures change against the paired clean image.
"""
import argparse
from collections import defaultdict
from dataclasses import dataclass
import importlib.util
from io import BytesIO
import json
from pathlib import Path
from statistics import mean
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image

from .. import sheetless_area
from ..pasta_vision import image_to_rgb
from .area_proportion import analyze
from .practical_capture import WidthReference, decide_capture
from .strand_scale import AGREED, COLOR_EDGE, calibrate_widths, measure_for_area, width_outcomes


STRAND = (220, 180, 40)
REFLECTION = (175, 158, 121)
HALO = (155, 147, 125)


@dataclass
class Case:
    label: str
    clipped: bool
    photo: np.ndarray
    grams: int


def spread(strands, right=576, top=40):
    image = np.full((480, 640, 3), 42, np.uint8)
    for y in range(top, top + strands * 10, 10):
        image[y:y + 5, right - 512:right] = STRAND
    return image


def glint(photo, start=103, left=False):
    edge = photo[:, ::-1] if left else photo
    edge[start:start + 35, -8:-3] = HALO
    edge[start:start + 35, -3:] = REFLECTION


def scenes(strands):
    photo = spread(strands)
    yield Case('clean', False, photo, strands)
    yield Case('contained_near_edge', False, spread(strands, right=636), strands)

    small = photo.copy()
    small[15:25, -4:] = REFLECTION
    yield Case('small_reflection', False, small, strands)

    broad = photo.copy()
    broad[:17, 230:259] = REFLECTION
    yield Case('broad_reflection', False, broad, strands)

    edge_line = photo.copy()
    edge_line[103:138, -3:] = REFLECTION
    yield Case('plain_weak_streak', False, edge_line, strands)

    halo = photo.copy()
    glint(halo)
    yield Case('reflection_with_halo', False, halo, strands)

    strong_edge_line = photo.copy()
    strong_edge_line[103:138, -3:] = STRAND
    yield Case('strong_tangential_crop', True, strong_edge_line, strands + 1)

    near = spread(strands, right=630)
    glint(near)
    yield Case('near_reflection', False, near, strands)

    multi = halo.copy()
    glint(multi, start=125, left=True)
    yield Case('two_edge_reflections', False, multi, strands)

    bright = np.where(photo[:, :, :1] == 42, 233, photo).astype(np.uint8)
    glint(bright)
    yield Case('bright_with_reflection', False, bright, strands)

    # A whole strand of the same color and width remains in the photo up to
    # the crop. The remainder is outside the frame.
    clipped = photo.copy()
    clipped[100:105, 576:] = STRAND
    yield Case('connected_crop', True, clipped, strands)

    fragmented = spread(strands, right=619)
    fragmented[100:105, 623:] = STRAND
    yield Case('fragmented_crop', True, fragmented, strands)

    # A real strand outside the field of view can leave just a short visible
    # tip. Keep this color-matched counterexample in the safety denominator.
    fragment = photo.copy()
    fragment[92:97, -10:] = STRAND
    yield Case('isolated_crop_tip', True, fragment, strands + 1)

    same_color = photo.copy()
    same_color[92:104, -12:] = STRAND
    yield Case('ambiguous_edge_patch', True, same_color, strands + 1)

    # Weak color alone is ambiguous if a clipped strand is in shade.
    shadow = photo.copy()
    shadow[92:97, -25:] = REFLECTION
    yield Case('shadowed_crop_tip', True, shadow, strands + 1)

    dim_cross = photo.copy()
    dim_cross[92:106, -11:] = REFLECTION
    yield Case('ambiguous_dim_patch', True, dim_cross, strands + 2)

    shallow = photo.copy()
    cv2.line(shallow, (646, 91), (638, 138), REFLECTION, thickness=3, lineType=cv2.LINE_8)
    yield Case('shallow_angle_dim_crop', True, shallow, strands + 1)

    parallel = photo.copy()
    cv2.line(parallel, (640, 103), (640, 137), REFLECTION, thickness=5, lineType=cv2.LINE_8)
    yield Case('parallel_dim_crop', True, parallel, strands + 1)


def dense_scene(grams):
    photo = spread(32)
    # Filled foreground simulates overlap with nominal 65/100/120g labels.
    photo[75:75 + grams * 2, 64:576] = STRAND
    glint(photo)
    return Case(f'dense_{grams}g_with_reflection', False, photo, grams)


def load_baseline(ref):
    path = Path(sheetless_area.__file__).resolve()
    root = path.parents[1]
    model_path = sheetless_area.MODEL_PATH
    model = subprocess.check_output(['git', 'show', f'{ref}:{model_path.relative_to(root).as_posix()}'], cwd=root)
    if json.loads(model) != json.loads(model_path.read_bytes()):
        raise ValueError('The baseline must use the same calibration as the working version.')
    code = subprocess.check_output(['git', 'show', f'{ref}:{path.relative_to(root).as_posix()}'], cwd=root)
    name = 'pasta_weight_app_v2._edge_benchmark_baseline'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(code, str(path), 'exec'), module.__dict__)
    return module


def width_reference():
    pairs = []
    for shift in (0, 3, 6):
        rgb = spread(20, top=40 + shift)
        evidence = analyze(rgb)
        pairs.append((evidence, measure_for_area(rgb, evidence)))
    reference = calibrate_widths(pairs)
    return reference, WidthReference(reference[COLOR_EDGE], reference[AGREED])


def target_error(grams):
    for upper, tolerance in ((40, 5), (60, 10), (100, 15)):
        if grams <= upper:
            return tolerance
    return 20


def picture_variants(photo, *, jpeg):
    if jpeg:
        for factor in (1, 3):
            source = cv2.resize(photo, None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
            for quality in (95, 85):
                with BytesIO() as encoded:
                    Image.fromarray(source).save(encoded, 'JPEG', quality=quality)
                    yield f'jpeg_{source.shape[1]}px_{quality}', image_to_rgb(encoded)
        return
    yield '640px', photo
    enlarged = cv2.resize(photo, (960, 720), interpolation=cv2.INTER_LINEAR)
    yield '960px_rotated', np.rot90(enlarged)


def summarize(rows, method):
    full = [row for row in rows if not row['clipped']]
    clips = [row for row in rows if row['clipped']]
    shown = [row for row in full if row[method] is not None]
    errors = [abs(row[method] - row['grams']) for row in shown]
    result = dict(
        shown=len(shown),
        available=len(full),
        abs_error_mean=mean(errors) if errors else None,
        abs_error_max=max(errors, default=None),
        shown_within_target=sum(error <= target_error(row['grams']) for error, row in zip(errors, shown)),
        cropped_shown=sum(row[method] is not None for row in clips),
        cropped_total=len(clips),
    )
    if method == 'after':
        changed = [row for row in shown if row['before'] is None]
        result['new_shown'] = len(changed)
        result['new_shown_within_target'] = sum(
            abs(row[method] - row['grams']) <= target_error(row['grams']) for row in changed
        )
        result['new_vs_clean_max'] = max(
            (abs(row[method] - row['clean_reference']) for row in changed if row['clean_reference'] is not None),
            default=None,
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='5e903a24d2a4dcda5f6580b24fa6b6152072e16a',
                        help='Trusted Git revision of the previous estimator.')
    parser.add_argument('--width', action='store_true', help='Also run the research estimators.')
    parser.add_argument('--jpeg', action='store_true', help='Use 640px and 1920px JPEG uploads at qualities 95 and 85.')
    args = parser.parse_args()
    cv2.setNumThreads(1)
    baseline = load_baseline(args.baseline)
    variants = ('before', 'after')
    if args.width:
        reference, profile = width_reference()
        variants += (COLOR_EDGE, AGREED, 'research_policy')

    rows = []
    masses = (24, 32, 40)
    clean_weights = {(grams, transform): baseline.estimate(rgb).weight_g
                     for grams in masses for transform, rgb in picture_variants(spread(grams), jpeg=args.jpeg)}
    cases = [case for grams in masses for case in scenes(grams)]
    cases.extend(dense_scene(grams) for grams in (65, 100, 120))
    for case in cases:
        for transform, rgb in picture_variants(case.photo, jpeg=args.jpeg):
            old, new = baseline.estimate(rgb), sheetless_area.estimate(rgb)
            row = dict(case=case.label, transform=transform,
                       grams=case.grams, clipped=case.clipped,
                       before=old.weight_g, after=new.weight_g,
                       clean_reference=clean_weights.get((case.grams, transform)) if not case.clipped else None,
                       before_reason=old.reason, after_reason=new.reason)
            if args.width:
                evidence = analyze(rgb)
                scale = measure_for_area(rgb, evidence)
                comparison = width_outcomes(evidence, scale, reference)
                row[COLOR_EDGE] = comparison[COLOR_EDGE]['estimated_g']
                row[AGREED] = comparison[AGREED]['estimated_g']
                row['research_policy'] = decide_capture(evidence, scale, profile).estimated_g
            rows.append(row)

    grouped = defaultdict(list)
    for row in rows:
        grouped[row['case']].append(row)
    print(json.dumps({
        'synthetic': True,
        'baseline': args.baseline,
        'summary': {method: summarize(rows, method) for method in variants},
        'by_case': {label: {method: sum(row[method] is not None for row in group)
                            for method in variants} for label, group in grouped.items()},
        'rows': rows,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
