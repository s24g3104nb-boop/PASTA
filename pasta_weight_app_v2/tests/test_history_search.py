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
    image_rgb=np.full((32,32,3), 150, dtype=np.uint8), source_kind='synthetic')
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
