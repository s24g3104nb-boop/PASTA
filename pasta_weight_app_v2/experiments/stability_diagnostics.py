"""Counterfactual mask audit and deterministic end-to-end inference replay.

Requires both image and sklearn research dependencies. Never changes prior runs.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image, ImageDraw

from .direct_features import extract
from .stability_pipeline import PipelineConfig, process, research_inference
from .stability_experiment import pixel_sha
from ..geometry_candidate import skeletonize


def run(root):
    root = Path(root); out = root/'diagnostics'
    if out.exists(): raise FileExistsError('No overwrite')
    data = json.loads((root/'extraction/features.json').read_text())
    if data['versions']['numpy'] != np.__version__ or data['versions']['opencv'] != cv2.__version__:
        raise ValueError('Use the same NumPy/OpenCV versions as feature extraction for exact replay')
    out.mkdir()
    artifact = joblib.load(root/'regression/models/A_lab_legacy12__linear.joblib')
    source = root/'extraction/IMG_7509'
    original = np.array(Image.open(source/'original/input.png'))
    half = np.array(Image.open(source/'resize50/input.png'))
    original_mask = np.array(Image.open(source/'original/A_lab_legacy12_mask.png'))
    actual_mask = np.array(Image.open(source/'resize50/A_lab_legacy12_mask.png'))
    frozen_mask = cv2.resize(original_mask, (half.shape[1], half.shape[0]), interpolation=cv2.INTER_NEAREST)
    records, panels = [], []
    for name, rgb, mask in [('original',original,original_mask),('half_rethreshold',half,actual_mask),('half_frozen_mask_nearest',half,frozen_mask)]:
        f, _ = extract(rgb, mask)
        records.append(dict(condition=name, **{k:f[k] for k in artifact['feature_keys']}))
        small = cv2.resize(mask, (round(mask.shape[1]*512/max(mask.shape)), round(mask.shape[0]*512/max(mask.shape))), interpolation=cv2.INTER_NEAREST)
        skeleton = skeletonize(small)*255
        Image.fromarray(small).save(out/(name+'_mask512.png'))
        Image.fromarray(skeleton).save(out/(name+'_skeleton512.png'))
        panels.append((name, small, skeleton))
    X = np.array([[r[k] for k in artifact['feature_keys']] for r in records])
    predicted = artifact['model'].predict(X)
    for r, pred in zip(records, predicted):
        r['research_prediction_g'] = float(pred)
        r['delta_from_original_g'] = float(pred-predicted[0])
    changes = (artifact['model'][0].transform(X)[1]-artifact['model'][0].transform(X)[0])*artifact['model'][1].coef_
    parts = [dict(feature=k, contribution_g=float(v)) for k,v in zip(artifact['feature_keys'],changes)]
    assert np.isclose(sum(changes),predicted[1]-predicted[0])
    assert abs(predicted[1]-predicted[0]+34.12505233193249) < 1e-6
    canvas = Image.new('RGB',(1200,600),'white');draw = ImageDraw.Draw(canvas)
    for i,(name,m,sk) in enumerate(panels):
        draw.text((400*i+5,5),name,fill='black')
        for j,array in enumerate([m,sk]):
            im = Image.fromarray(array).convert('RGB');im.thumbnail((390,275));canvas.paste(im,(400*i,30+j*285))
    canvas.save(out/'topology_comparison.jpg')
    # Re-extract all 225 inputs with C_clean5, independent of cached feature vectors.
    config = PipelineConfig(**next(c for c in data['configuration'] if c['name']=='C_clean5'))
    expected = {(r['image_id'],r['transform']):r for r in data['rows'] if r['pipeline']=='C_clean5'}
    maxima = []
    cv2.setNumThreads(2)
    for image in data['inputs']:
        rgb = np.array(Image.open(root/'extraction'/image['path']))
        f, _, mask = process(rgb,config);ref = expected[(image['image_id'],image['transform'])]
        if pixel_sha(mask) != ref['mask_sha256']: raise ValueError('Mask replay mismatch')
        difference = max(abs(f[k]-ref[k]) for k in config.keys)
        if difference > 1e-7: raise ValueError('Feature replay mismatch')
        maxima.append(difference)
    # Exercise the stored-model adapter on all 15 original images.
    artifact_c = joblib.load(root/'regression/models/C_clean5__ridge.joblib')
    adapter_differences = []
    for row in [r for r in data['rows'] if r['pipeline']=='C_clean5' and r['transform']=='original']:
        rgb = np.array(Image.open(root/'extraction'/row['image_id']/'original/input.png'))
        inferred = research_inference(rgb,artifact_c)
        cached = float(artifact_c['model'].predict([[row[k] for k in config.keys]])[0])
        adapter_differences.append(abs(cached-inferred['research_prediction_g']))
        assert inferred['predicted_weight_g'] is None and not inferred['independent_validation']
    assert max(adapter_differences) < 1e-7
    result = dict(counterfactual_records=records, linear_contributions=sorted(parts,key=lambda r:abs(r['contribution_g']),reverse=True),
                  mask_replays=len(maxima),mask_hashes_identical=True,maximum_feature_replay_difference=max(maxima),
                  end_to_end_originals=15,maximum_adapter_prediction_difference_g=max(adapter_differences),
                  independent_evaluation=False,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--root',required=True);run(parser.parse_args().root)
