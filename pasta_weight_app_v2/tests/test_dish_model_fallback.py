"""Photo diary remains usable when the optional cooking model cannot load."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('model_state', ['unconfigured', 'missing_file', 'wrong_sha256'])
def test_dish_model_failure_can_save_manual_recipe_with_photo(tmp_path, model_state):
    data_dir = tmp_path / 'data'
    env = dict(os.environ, PASTA_DATA_DIR=str(data_dir))
    env.pop('PASTA_OPENCLIP_CHECKPOINT', None)
    if model_state != 'unconfigured':
        model_file = tmp_path / 'openclip.pt'
        env['PASTA_OPENCLIP_CHECKPOINT'] = str(model_file)
        if model_state == 'wrong_sha256':
            model_file.write_bytes(b'invalid checkpoint for the local test')
    script = """
from io import BytesIO
import sys

from PIL import Image
from streamlit.testing.v1 import AppTest

sys.path.insert(0, 'pasta_weight_app_v2')
from meal_diary import entries, photo_path

state = sys.argv[1]
app = AppTest.from_file('pasta_weight_app_v2/app.py', default_timeout=20).run()
app.button(key='home_meal').click().run()
with BytesIO() as fixture:
    Image.new('RGB', (24, 24), '#c18051').save(fixture, format='PNG')
    app.file_uploader[0].upload('test.png', fixture.getvalue(), 'image/png').run()
assert not app.exception, app.exception
status = ('写真認識を利用できません' if state == 'unconfigured'
          else '写真認識を実行できませんでした')
assert any(status in entry.value for entry in app.info)
selector = next(widget for widget in app.selectbox if widget.label == '標準レシピの料理')
assert selector.value is None
selector.set_value('ジェノベーゼ').run()
app.button(key='meal_use_recipe').click().run()
assert any('料理全体' in entry.value for entry in app.markdown)
app.button(key='meal_save').click().run()
assert not app.exception, app.exception
meal, = entries()
assert meal['dish_name'] == 'ジェノベーゼ'
assert meal['measurement_id'] is None
assert meal['nutrition_status'] == 'template_estimate'
assert meal['nutrition_snapshot']['kcal'] > 0
assert photo_path(meal['photo']).is_file()
"""
    completed = subprocess.run([sys.executable, '-c', script, model_state], cwd=ROOT, env=env,
                               capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    log = (data_dir / 'logs' / 'pasta_lens.log').read_text(encoding='utf-8')
    assert ('完成料理の画像認識 失敗' in log) == (model_state != 'unconfigured')
