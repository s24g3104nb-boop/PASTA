"""Compare count masks with evidence from the uncropped source frame."""
from dataclasses import replace
import math

import cv2
import numpy as np

from ..pasta_vision import analyze_spread_mask, detect_spread
from .stability_pipeline import PipelineConfig, raw_mask

LAB_MASK = PipelineConfig('spread_lab', None, 'hsv_lab', False, 'shape5')


def compare_spread_counts(rgb, **parameters):
    """Return production HSV and a research HSV+Lab candidate.

    A weak-color border region can be removed when it is detached from the
    Lab foreground. Keep the original clipping warning when the border region
    connects to that foreground, even if Lab removes the border pixels.
    """
    baseline = detect_spread(rgb, **parameters)
    min_area = parameters.get('min_area', 160)
    lab = analyze_spread_mask(rgb, raw_mask(rgb, LAB_MASK), close=False,
                             min_area=min_area, min_elongation=parameters.get('min_elongation', 3.0))
    evidence = dict(lab.evidence, linked_source_border_regions=0,
                    excluded_source_border_regions=0)
    if baseline.evidence['cropped_regions']:
        _, labels, stats, _ = cv2.connectedComponentsWithStats(baseline.mask)
        edge_ids = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
        foreground_ids = set(np.unique(labels[lab.mask > 0]))
        for index in edge_ids[edge_ids > 0]:
            _, _, width, height, area = stats[index]
            if area < min_area and max(width, height) < math.sqrt(min_area):
                continue
            key = ('linked_source_border_regions' if index in foreground_ids else
                   'excluded_source_border_regions')
            evidence[key] += 1
        # These counts may refer to the same object. Preserve the warning once.
        evidence['cropped_regions'] = max(evidence['cropped_regions'], evidence['linked_source_border_regions'])
    return baseline, replace(lab, evidence=evidence)
