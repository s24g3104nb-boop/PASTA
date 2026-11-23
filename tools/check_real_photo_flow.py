"""Private AppTest replay with actual photos and a separate temporary database.

Products/calibrations in this script are artificial UI fixtures. Any calculated
weight is counterfactual, and is never scored as the photo's true gram weight.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

SCRIPT = r'''
import sys,json
from pathlib import Path
from io import BytesIO
from unittest.mock import patch
sys.path.insert(0,'pasta_weight_app_v2')
from streamlit.testing.v1 import AppTest
import capture_ui
from storage import save_product,list_measurements
from pasta_weight_app_v2.experiments.photo_batch_diagnosis import load_diagnostic_entries
from pasta_weight_app_v2.experiments.spread_count_candidate import compare_spread_counts
records=[]
for entry in load_diagnostic_entries(json.loads(sys.argv[1]), Path(sys.argv[2])):
    records.append((entry['path'],entry['record']['filename'],entry['reported_count']))
research_lab=sys.argv[3]=='lab'
# These numeric values only test product propagation, not calibration of the supplied product.
at=AppTest.from_file('pasta_weight_app_v2/app.py',default_timeout=30).run()
fixture=save_product('隔離したUI検査用・仮校正',0,350,12,2,70)
results=[]
for path,name,known_count in records:
    at=AppTest.from_file('pasta_weight_app_v2/app.py',default_timeout=30).run()
    at.button(key='home_camera').click().run()
    at.session_state['capture_photo']=BytesIO(path.read_bytes())
    if research_lab:
        def lab_detector(rgb, **opts):
            return compare_spread_counts(rgb, **opts)[1]
        with patch.object(capture_ui,'detect_spread',lab_detector):
            at.run()
    else:
        at.run()
    at.selectbox(key='which_product').set_value(fixture['id']).run()
    assert not at.exception,at.exception
    assert not any(m.label=='推定重量' for m in at.metric)
    cached=at.session_state['result_request']['_detection_result']
    save_product('隔離したUI検査用・仮校正',.7,350,12,2,70,product_id=fixture['id'])
    at.run()
    assert not at.exception,at.exception
    assert at.session_state['result_request']['product']['g_per_strand']==.7
    assert at.session_state['result_request']['_detection_result'] is cached
    displayed=[m.value for m in at.metric if m.label=='推定重量']
    if displayed:
        assert displayed == [f'{cached.count*.7:.1f} g']
    else:
        assert at.button(key='manual_weight_save')
    save_product('隔離したUI検査用・仮校正',0,350,12,2,70,product_id=fixture['id'])
    results.append(dict(filename=name,automatic_count=cached.count,known_count=known_count,
                        weight_display=bool(displayed),displayed_fixture_weight=displayed[0] if displayed else None,
                        photo_weight_estimated_g=None,
                        product_sync_verified=True,photo_and_detection_preserved=True))
assert list_measurements()==[]
Path(sys.argv[2]).write_text(json.dumps(dict(checks=results,test_database_only=True,
 research_lab=research_lab,artificial_calibration=True,no_photo_accuracy_claim=True),ensure_ascii=False,indent=2),encoding='utf-8')
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--research-lab', action='store_true',
                        help='Use the Lab candidate only in this isolated test process')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Choose a new output file')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='pasta-real-flow-') as tmp:
        env=dict(os.environ,PASTA_DATA_DIR=str(Path(tmp)/'data'),PASTA_RESEARCH_DATA_DIR=str(Path(tmp)/'research'))
        result=subprocess.run([sys.executable,'-c',SCRIPT,json.dumps([str(p.resolve()) for p in args.dataset]),
                               str(args.output.resolve()),'lab' if args.research_lab else 'hsv'],
            cwd=Path(__file__).resolve().parents[1],env=env,capture_output=True,text=True,timeout=300)
        if result.returncode:raise RuntimeError(result.stdout+result.stderr)
    print('Real-photo UI replay passed. Artificial calibration only.')


if __name__=='__main__':main()
