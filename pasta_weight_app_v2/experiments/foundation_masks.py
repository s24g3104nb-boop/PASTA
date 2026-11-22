"""Reference annotations, segmentation scores, and A4 sheet-plane geometry."""

import json
from pathlib import Path
import re

import cv2
import numpy as np

from .capture_sheet import HEIGHT_MM, PIXELS_PER_MM, WIDTH_MM
from .marker_reference import analyze_marker_reference
from .stability_pipeline import shape_features
from .foundation_backends import SEED


def read_annotations(directory, expected):
    if directory is None:
        return {}
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError('Annotations must be an existing directory')
    result = {}
    for path in sorted(directory.glob('*.annotation.json')):
        document = json.loads(path.read_text(encoding='utf-8-sig'))
        digest = document['original_sha256']
        if digest not in expected or digest in result:
            raise ValueError('Each reference must match one original SHA-256 exactly once')
        result[digest] = document
    if not result:
        raise ValueError('No .annotation.json files found')
    return result


def annotate_mask(document, native_shape, scoring_shape):
    """Rasterize manual polygons and variable-width polylines at scoring size."""
    if document['schema'] != 'pasta-visible-mask-v1' or document['image_shape'] != list(native_shape):
        raise ValueError('Annotation schema or EXIF-oriented dimensions do not match')
    strokes = document['strokes']
    if not isinstance(strokes, list) or len(strokes) > 20000:
        raise ValueError('Invalid annotation strokes')
    height, width = scoring_shape
    source_h, source_w = native_shape
    factors = np.array([width / source_w, height / source_h])
    mask = np.zeros(scoring_shape, np.uint8)
    for stroke in strokes:
        mode = stroke['mode']
        points = np.asarray(stroke['points'])
        if (mode not in ('add', 'erase') or points.ndim != 2 or points.shape[1] != 2
                or points.dtype.kind not in 'iuf' or not np.isfinite(points).all()
                or np.any(points < 0) or np.any(points >= [source_w, source_h])):
            raise ValueError('Invalid reference-mask points')
        color = int(mode == 'add')
        pixels = np.minimum(np.rint(points*factors), [width-1, height-1]).astype(np.int32)
        if stroke['tool'] == 'polygon' and len(points) >= 3:
            cv2.fillPoly(mask, [pixels], color)
        elif stroke['tool'] == 'line' and len(points) >= 1:
            brush = stroke['width_px']
            if type(brush) not in (int, float) or not 0 < brush <= max(native_shape):
                raise ValueError('Invalid reference brush width')
            thickness = max(1, round(brush * min(factors)))
            if len(points) > 1:
                cv2.polylines(mask, [pixels], False, color, thickness)
            else:
                cv2.circle(mask, tuple(pixels[0]), thickness//2, color, -1)
        else:
            raise ValueError('Invalid reference-mask tool')
    return mask != 0


def mask_scores(candidate, reference):
    if candidate.shape != reference.shape:
        raise ValueError('Mask and reference must use the same image coordinates')
    tp = int(np.count_nonzero(candidate & reference))
    fp = int(np.count_nonzero(candidate & ~reference))
    fn = int(np.count_nonzero(~candidate & reference))
    return dict(true_positive_px=tp, false_positive_px=fp, false_negative_px=fn,
                iou=tp/(tp+fp+fn) if tp+fp+fn else None,
                background_in_predicted_fraction=fp/(tp+fp) if tp+fp else None,
                missed_visible_pasta_fraction=fn/(tp+fn) if tp+fn else None)


def read_a4_reviews(path, expected):
    if path is None:
        return {}
    reviews = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(reviews, dict) or not set(reviews) <= set(expected):
        raise ValueError('A4 reviews must be indexed by original SHA-256')
    for review in reviews.values():
        if type(review.get('flat_sheet_confirmed')) is not bool:
            raise ValueError('Explicit flat-sheet confirmation is required')
        line = review.get('printed_line_mm')
        if type(line) not in (int, float) or not np.isfinite(line) or line <= 0:
            raise ValueError('Measure the printed line in mm')
    return reviews


def a4_projection(rgb, mask, review):
    verified_print = bool(review and review['flat_sheet_confirmed'] and abs(review['printed_line_mm']-100) <= 1)
    info, _, _ = analyze_marker_reference(rgb)
    result = dict(geometry_detected=info['geometry_valid'], print_verified=verified_print,
                  status='markers_invalid' if not info['geometry_valid'] else 'print_not_verified', features=None)
    if not info['geometry_valid'] or not verified_print:
        return result
    if not np.any(mask):
        return dict(result, status='empty_foreground')
    if mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any():
        return dict(result, status='foreground_at_image_edge')
    polygon = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(polygon, np.rint(info['projected_sheet_corners']).astype(np.int32), 1)
    if np.count_nonzero(mask & (polygon == 0)) / np.count_nonzero(mask) > .01:
        return dict(result, status='foreground_outside_sheet')
    paper = np.array([[0, 0], [WIDTH_MM, 0], [WIDTH_MM, HEIGHT_MM], [0, HEIGHT_MM]], np.float32)
    corners = np.asarray(info['projected_sheet_corners'], np.float32)
    projection = cv2.getPerspectiveTransform(corners, paper * PIXELS_PER_MM)
    projected = cv2.warpPerspective(mask.astype(np.uint8), projection,
                                    (round(WIDTH_MM*PIXELS_PER_MM), round(HEIGHT_MM*PIXELS_PER_MM)),
                                    flags=cv2.INTER_NEAREST)
    shape = shape_features(projected, SEED)
    if not shape['mask_area_px']:
        return dict(result, status='empty_projection')
    return dict(result, status='sheet_plane_projection',
                features=[shape['mask_area_px']/PIXELS_PER_MM**2,
                          shape['quantile_length_px']/PIXELS_PER_MM, shape['quantile_width_px']/PIXELS_PER_MM])


def run_masks(folder, digests, shapes):
    """Join model masks to the same immutable byte hashes, never to filenames."""
    folder = Path(folder)
    report = json.loads((folder/'segmentation.json').read_text(encoding='utf-8'))
    rows = {row['original_sha256']: row for row in report['images']}
    if (report['schema'] != 'pasta-foundation-segmentation-v1' or set(rows) != set(digests)
            or len(rows) != len(report['images']) or not all(re.fullmatch('[0-9a-f]{64}', d) for d in rows)):
        raise ValueError('Segmentation runs must describe exactly the same verified originals')
    if any(rows[d]['shape'] != list(shapes[d]) for d in rows):
        raise ValueError('Segmentation image dimensions differ')
    return report['backend']['method'], rows
