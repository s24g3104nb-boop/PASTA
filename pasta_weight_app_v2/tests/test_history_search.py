"""アプリ全体を隔離プロセス・一時DBで起動し、検索0件と復帰を確認する。"""
from pathlib import Path
import os
import subprocess
import sys


def test_history_search_empty_results_and_recovery(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    script = r'''
import sys
from pathlib import Path
import numpy as np
from streamlit.testing.v1 import AppTest
sys.path.insert(0, str(Path('pasta_weight_app_v2').resolve()))
from storage import init_db, save_measurement, list_measurements
init_db()
save_measurement(product_name='検索試験パスタ', mode='spread', detected_count=20,
    estimated_g=20, actual_g=20,
    nutrition=dict(kcal=70,protein_g=2,fat_g=.4,carb_g=14),
    image_rgb=np.full((32,32,3), 150, dtype=np.uint8), source_kind='real')
at = AppTest.from_file('pasta_weight_app_v2/app.py').run(timeout=20)
assert not at.exception, [item.message for item in at.exception]
at.radio(key='main_nav').set_value('記録').run(timeout=20)
at.selectbox(key='record_kind').set_value('乾麺の測定履歴').run()
at.text_input(key='history_search').set_value('存在しない検索語').run(timeout=20)
assert not at.exception, [item.message for item in at.exception]
assert any('一致する履歴はありません' in item.value for item in at.info)
at.text_input(key='history_search').set_value('検索試験').run(timeout=20)
assert not at.exception, [item.message for item in at.exception]
assert len(list_measurements()) == 1
'''
    env = dict(os.environ, PASTA_DATA_DIR=str(tmp_path / 'isolated_data'))
    result = subprocess.run([sys.executable, '-c', script], cwd=repo, env=env,
                            capture_output=True, text=True, timeout=65)
    assert result.returncode == 0, result.stdout + result.stderr


def test_history_preserves_original_photo_estimate_for_accuracy(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    script = r'''
import sys
from pathlib import Path
import numpy as np
from streamlit.testing.v1 import AppTest
sys.path.insert(0, str(Path('pasta_weight_app_v2').resolve()))
from sheetless_area import ESTIMATOR
from storage import init_db, save_measurement, list_measurements

init_db()
photo = np.full((32, 32, 3), 150, dtype=np.uint8)
common = dict(mode='spread', detected_count=0, image_rgb=photo, source_kind='real',
              estimator=ESTIMATOR, nutrition=dict(kcal=70, protein_g=2, fat_g=.4, carb_g=14))
legacy = save_measurement(product_name='旧記録', estimated_g=22, actual_g=20, **common)
edited = save_measurement(product_name='修正後に実測', estimated_g=42, actual_g=50,
                          diagnostics={'image_estimated_g': 64.3}, **common)
save_measurement(product_name='修正済み・実測なし', estimated_g=40, actual_g=None,
                 diagnostics={'image_estimated_g': 55.2}, **common)

rows = {row['id']: row for row in list_measurements(include_image_estimate=True)}
assert rows[legacy]['image_estimated_g'] is None
assert rows[edited]['image_estimated_g'] == 64.3 and rows[edited]['estimated_g'] == 42
assert 'image_estimated_g' not in list_measurements()[0]

at = AppTest.from_string("""
import streamlit as st
from history_ui import render_history
st.session_state['_app_admin'] = True
render_history()
""").run(timeout=20)
assert not at.exception, [item.message for item in at.exception]
assert any('修正した概算です。写真の推定は約55.2 g' in item.value for item in at.caption)
assert any('元の推定と実測の差 14.3 g' in item.value for item in at.markdown)
assert at.metric[0].value == '8.15 g'
'''
    env = dict(os.environ, PASTA_DATA_DIR=str(tmp_path / 'isolated_data'))
    result = subprocess.run([sys.executable, '-c', script], cwd=repo, env=env,
                            capture_output=True, text=True, timeout=65)
    assert result.returncode == 0, result.stdout + result.stderr
