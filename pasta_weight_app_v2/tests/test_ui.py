import os
from pathlib import Path
import subprocess
import sys


def test_streamlit_measurement_and_cooking_selection(tmp_path):
    app_dir = Path(__file__).resolve().parents[1]
    runner = tmp_path / 'ui_smoke.py'
    runner.write_text(SMOKE, encoding='utf-8')
    task_env = dict(os.environ, PASTA_APP_DIR=str(app_dir), PASTA_DATA_DIR=str(tmp_path / 'data'))
    result = subprocess.run([sys.executable, str(runner)], env=task_env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


SMOKE = r'''
import sys
from pathlib import Path
import os
sys.path.insert(0, os.environ['PASTA_APP_DIR'])
from streamlit.testing.v1 import AppTest

at = AppTest.from_file(str(Path(os.environ['PASTA_APP_DIR']) / 'app.py'), default_timeout=15).run()
assert not at.exception, at.exception
assert at.metric[0].value == '12 個'
at.checkbox(key='override_count').check().run()
at.number_input(key='manual_count').set_value(99).run()
assert any(m.value == '99.0 g' for m in at.metric)
at.radio[0].set_value('束の断面').run()
assert not at.exception, at.exception
assert at.metric[0].value == '16 個'
assert not at.checkbox(key='override_count').value
assert any(m.value == '16.0 g' for m in at.metric)
next(b for b in at.button if b.label == '💾 この測定結果を保存').click().run()
assert not at.exception, at.exception
assert at.selectbox(key='cooking_measurement').options[0] == '履歴を選択してください'
from storage import list_measurements
rows = list_measurements()
assert len(rows) == 1 and rows[0]['source_kind'] == 'synthetic'
assert rows[0]['estimator'] == 'opencv'
at.selectbox(key='cooking_measurement').set_value(rows[0]['id']).run()
assert not at.exception, at.exception
assert any('CGまたはデモ' in w.value for w in at.warning)
at.radio(key='estimate_method').set_value('学習モデル（実験用回帰）').run()
assert not at.exception, at.exception
assert any('有効なモデルがありません' in w.value for w in at.warning)
print('UI smoke passed: spread/bundle, manual reset, save, cooking selection and demo warning')
'''
