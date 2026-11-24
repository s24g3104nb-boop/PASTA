"""Analyze a local photo with the same decoder and area gates used by Pasta Lens."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from ..pasta_vision import image_quality_warnings, image_to_rgb
from ..sheetless_area import EDGE_NAMES, MAX_MASK_FRACTION, MAX_THICKNESS_RATIO, diagnose, estimate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('photo', type=Path, help='Path to an original, unmarked photo.')
    photo = parser.parse_args().photo
    rgb = image_to_rgb(photo)
    diagnostic = diagnose(rgb)
    height, width = diagnostic.overlay.shape[:2]
    result = estimate(rgb)
    warnings = image_quality_warnings(rgb)
    by_side = {}
    for side in EDGE_NAMES:
        by_side[side] = {}
        for retained, status in ((True, 'retained'), (False, 'excluded')):
            runs = [run for run in diagnostic.contacts if run.side == side and run.retained == retained]
            by_side[side][status] = {
                'components': len({run.component for run in runs}),
                'pixels': sum(run.end - run.start + 1 for run in runs),
            }

    print(json.dumps({
        'analysis_size': {'width': width, 'height': height},
        'coordinates': 'Zero-based, inclusive. Top/bottom use x, left/right use y. Corners count on both sides.',
        'by_side': by_side,
        'contact_runs': [asdict(run) for run in diagnostic.contacts],
        'mask_fraction': diagnostic.mask_fraction,
        'max_mask_fraction': MAX_MASK_FRACTION,
        'thickness_ratio': diagnostic.thickness_ratio,
        'max_thickness_ratio': MAX_THICKNESS_RATIO,
        'density_rejected': diagnostic.density_rejected,
        'area_reason': result.reason,
        'quality_warnings': warnings,
        'area_weight_g': None if warnings else result.weight_g,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
