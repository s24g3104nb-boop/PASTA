"""The public photo diary ranks dishes without an optional OpenCLIP model."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('model_state', ['unconfigured', 'missing_file', 'wrong_sha256'])
def test_dish_photo_selects_recipe_without_openclip(tmp_path, model_state):
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

from PIL import Image, ImageDraw
from streamlit.testing.v1 import AppTest

sys.path.insert(0, 'pasta_weight_app_v2')
from meal_diary import entries, photo_path

app = AppTest.from_file('pasta_weight_app_v2/app.py', default_timeout=20).run()
app.button(key='home_meal').click().run()
with BytesIO() as fixture:
    image = Image.new('RGB', (48, 48), '#ececea')
    ImageDraw.Draw(image).ellipse((6, 6, 41, 41), fill='#cd372d')
    image.save(fixture, format='PNG')
    app.file_uploader[0].upload('test.png', fixture.getvalue(), 'image/png').run()
assert not app.exception, app.exception
assert 'open_clip' not in sys.modules
assert not any('判別できません' in entry.value for entry in app.info)
selector = next(widget for widget in app.selectbox if widget.label == '標準レシピの料理')
assert selector.value == 'トマトパスタ'
assert any('トマトパスタの標準レシピ' in entry.value for entry in app.caption)
selector.set_value('ジェノベーゼ').run()
assert any('ジェノベーゼの標準レシピ' in entry.value for entry in app.caption)
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
    completed = subprocess.run([sys.executable, '-c', script], cwd=ROOT, env=env,
                               capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    log = (data_dir / 'logs' / 'pasta_lens.log').read_text(encoding='utf-8')
    assert '完成料理の画像認識 失敗' not in log
