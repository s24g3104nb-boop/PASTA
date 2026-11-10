"""Read-only continuation: observed tray reference feasibility and inherited QC.

All derivatives remain attached to their original. Reference rejection is never
replaced by image dimensions, and an original rejection survives later padding.
"""
import argparse
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import platform

import cv2
import numpy as np
from PIL import Image

from .stability_pipeline import configurations, VERSION
from .stability_experiment import pixel_sha
from .reference_scale import analyze_reference
from .research_contract import propagate_capture_qc, validate_research_artifact

E_KEYS = ['foreground_over_reference_area', 'length_over_reference_sqrt_area',
          'width_over_reference_sqrt_area', 'quantile_density', 'profile_width_cv']


class FrozenTableContract:
    """Cache provenance sentinel: never synthesizes predictions."""
    def predict(self, values):
        raise RuntimeError('Read the frozen prediction table; no inference available')


def reference_features(features, area):
    if area is None or not np.isfinite(area) or area <= 0:
        return None
    return dict(foreground_over_reference_area=features['mask_area_px']/area,
                length_over_reference_sqrt_area=features['quantile_length_px']/np.sqrt(area),
                width_over_reference_sqrt_area=features['quantile_width_px']/np.sqrt(area),
                quantile_density=features['quantile_density'], profile_width_cv=features['profile_width_cv'])


def run(previous, output):
    previous, output = Path(previous).resolve(), Path(output)
    if output.exists(): raise FileExistsError('No overwrite')
    data = json.loads((previous/'extraction/features.json').read_text())
    config = configurations()[3]
    source = Path(__file__).parent
    actual_hashes = {n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in data['source_hashes']}
    image_versions = dict(numpy=np.__version__, opencv=cv2.__version__)
    recorded_versions = {k:data['versions'][k] for k in image_versions}
    recorded_config = next(c for c in data['configuration'] if c['name']=='B_normalized5')
    cached = {(r['image_id'],r['transform']):r for r in data['rows'] if r['pipeline']=='B_normalized5'}
    # This is a cache provenance contract, not approval or a model promotion.
    contract = dict(model=FrozenTableContract(), pipeline_version=data['version'], pipeline_config=recorded_config,
                    config_sha256=next(iter(cached.values()))['config_sha256'], feature_keys=data['feature_keys']['B_normalized5'],
                    source_hashes=data['source_hashes'], image_versions=recorded_versions,
                    approval='EXPLORATORY_ONLY', production_eligible=False, independent_validation=False)
    validate_research_artifact(contract, expected_version=VERSION, expected_config=asdict(config),
                              expected_feature_keys=config.keys, expected_source_hashes=actual_hashes,
                              expected_image_versions=image_versions)
    contract['model'] = 'frozen_B_prediction_table; provenance checked, no predictor loaded'
    if data['independent_test_images'] != 0 or data['production_approved']:
        raise ValueError('Unexpected independent evaluation/approval')
    with (previous/'regression/predictions.csv').open() as stream:
        old_predictions = [r for r in csv.DictReader(stream) if r['pipeline']=='B_normalized5']
    frozen = {(r['image_id'],r['transform'],r['model']):r for r in old_predictions}
    output.mkdir(parents=True)
    (output/'images').mkdir()
    cv2.setNumThreads(2)
    original_reference = {}
    rows = []
    # Never depend on a transform sorting convention for the original QC context.
    ordered = sorted(data['inputs'], key=lambda r:(r['transform']!='original',r['image_id'],r['transform']))
    for item in ordered:
        path = (previous/'extraction'/item['path']).resolve()
        if not path.is_relative_to((previous/'extraction').resolve()): raise ValueError('Unsafe derived-image path')
        with Image.open(path) as im: rgb = np.array(im.convert('RGB'))
        if pixel_sha(rgb) != item['pixel_sha256']: raise ValueError('Input pixels changed')
        meta, scene, ref_mask, overlay = analyze_reference(rgb)
        old = cached[(item['image_id'],item['transform'])]
        if list(scene.shape) != old['processed_shape']: raise ValueError('Coordinate system mismatch')
        if item['transform']=='original': original_reference[item['image_id']] = meta
        original = cached[(item['image_id'],'original')]
        inherited = propagate_capture_qc(original['qc_flags'], old['qc_flags'])
        parent_reference = original_reference[item['image_id']]
        original_reference_ok = bool(parent_reference['usable_for_relative_scale'])
        current_reference_ok = bool(meta['usable_for_relative_scale'])
        scale_valid = original_reference_ok and current_reference_ok
        values = reference_features(old, meta['area_px']) if scale_valid else None
        # Arithmetic eligibility is not supervised approval or physical independence.
        blockers = {'foreground_touches_frame','insufficient_foreground','low_detail_review','low_source_resolution'}
        quality_eligible = not blockers.intersection(inherited['qc_flags'])
        eligible = scale_valid and quality_eligible
        key = item['image_id']+'_'+item['transform']
        before = np.array(Image.open(previous/'extraction'/old['mask_path']))
        contours,_ = cv2.findContours(before,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay,contours,-1,(30,230,80),1)
        Image.fromarray(ref_mask).save(output/'images'/(key+'_reference_mask.png'))
        Image.fromarray(overlay).save(output/'images'/(key+'_overlay.jpg'),quality=88)
        reasons = list(meta['reasons'])
        if not original_reference_ok: reasons.append('original_reference_unusable')
        if not quality_eligible: reasons.append('capture_quality_review_required')
        rows.append(dict(image_id=item['image_id'],transform=item['transform'],original_sha256=item['original_sha256'],
                         capture_group=item['capture_group'],lineage_block=item['lineage_block'],reported_g=old['reported_g'],
                         reference=meta,original_reference_usable=original_reference_ok,reference_computable=scale_valid,
                         arithmetic_eligible=eligible,qc_flags=inherited['qc_flags'],capture_reasons=inherited['capture_reasons'],
                         reasons=sorted(set(reasons)),B_features={k:old[k] for k in config.keys},E_features=values,
                         frozen_B={name:dict(prediction_g=float(frozen[(item['image_id'],item['transform'],name)]['research_prediction_g']),
                                            delta_g=float(frozen[(item['image_id'],item['transform'],name)]['delta_from_original_g']))
                                   for name in ['mean','linear','ridge','random_forest','gradient_boosting']},
                         overlay_path='images/'+key+'_overlay.jpg',reference_mask_path='images/'+key+'_reference_mask.png',
                         public_prediction_g=None,reference_approved=False,physical_calibration=False))
    report = dict(purpose='same-tray-relative-reference-feasibility',source_originals=15,derived_inputs=len(rows),
                  independent_test_images=0,production_approved=False,reference_approved=False,
                  B_keys=config.keys,E_keys=E_KEYS,source_provenance_contract=contract,
                  reference_config=rows[0]['reference'].get('config'),reference_version=rows[0]['reference'].get('version'),
                  source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),source/'reference_scale.py',source/'research_contract.py']},
                  versions=dict(python=platform.python_version(),numpy=np.__version__,opencv=cv2.__version__),
                  frozen_prediction_sha256=hashlib.sha256((previous/'regression/predictions.csv').read_bytes()).hexdigest(),rows=rows)
    (output/'reference_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--previous',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.previous,args.output)
