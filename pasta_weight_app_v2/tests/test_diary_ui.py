"""End-to-end Streamlit state test with a separate process/database."""
import os
from pathlib import Path
import subprocess
import sys


def test_diary_capture_draft_failure_calendar_edit_delete(tmp_path):
    script=r'''
from streamlit.testing.v1 import AppTest
import sys,sqlite3
from datetime import date
sys.path.insert(0,'pasta_weight_app_v2')
from storage import DEFAULT_DB,list_measurements
from meal_diary import entries
at=AppTest.from_string("import sys; from pathlib import Path; sys.path.insert(0, str(Path.cwd()/'pasta_weight_app_v2')); from app_runtime import render_application; render_application(admin=True)", default_timeout=20).run()
assert not at.exception
at.button(key='home_demo').click().run()
at.button(key='analyze_photo').click().run()
at.checkbox(key='actual_known').check().run()
at.number_input(key='actual_weight').set_value(40.).run()
at.button(key='measurement_save').click().run()
assert len(list_measurements())==1 and at.button(key='measurement_save').disabled
at.button(key='measurement_to_diary').click().run()
assert not at.exception
name=next(w for w in at.text_input if w.label=='料理名');name.set_value('カルボナーラ').run()
day=next(w for w in at.date_input if w.label=='食べた日');day.set_value(date(2020,1,2)).run()
at.radio(key='main_nav').set_value('ホーム').run()
at.button(key='home_meal').click().run()
assert next(w for w in at.text_input if w.label=='料理名').value=='カルボナーラ'
assert next(w for w in at.date_input if w.label=='食べた日').value==date(2020,1,2)
with sqlite3.connect(DEFAULT_DB) as c:c.execute("CREATE TRIGGER reject_meal BEFORE INSERT ON meal_entries BEGIN SELECT RAISE(ABORT,'test failure'); END")
at.button(key='meal_save').click().run()
assert at.error and not entries()
assert next(w for w in at.text_input if w.label=='料理名').value=='カルボナーラ'
with sqlite3.connect(DEFAULT_DB) as c:c.execute('DROP TRIGGER reject_meal')
at.button(key='meal_save').click().run()
assert not at.exception and len(entries())==1
row=entries()[0]
assert row['measurement_snapshot']['actual_g']==40 and row['measurement_snapshot']['estimated_g']==12
assert row['nutrition_status']=='dry_only'
at.button(key='meal_edit').click().run()
next(w for w in at.text_input if w.label=='料理名').set_value('編集後').run()
at.button(key='meal_save').click().run()
assert entries()[0]['dish_name']=='編集後'
at.button(key='meal_detail_cooking').click().run()
assert not at.exception
mid=row['measurement_id']
at.checkbox(key=f'cooking_batch_{mid}').check().run()
at.checkbox(key=f'cooking_complete_{mid}').check().run()
at.button(key=f'cooking_save_{mid}').click().run()
assert not at.exception
at.button(key='cooking_saved_to_diary').click().run()
assert next(w for w in at.text_input if w.label=='料理名').value=='編集後'
at.button(key='meal_save').click().run()
assert entries()[0]['cooking_session_id'] is not None
assert entries()[0]['nutrition_status']=='confirmed_amounts'
at.button(key='meal_detail_back').click().run()
at.text_input(key='diary_filter_query').set_value('なし').run()
assert any('0件' in x.value for x in at.caption)
at.text_input(key='diary_filter_query').set_value('').run()
at.radio(key='diary_view').set_value('カレンダー').run()
assert at.button(key='diary_add_selected_day')
at.radio(key='diary_view').set_value('写真付き一覧').run()
at.button(key='meal_detail_'+row['id']).click().run()
at.checkbox(key='meal_delete_confirm_'+row['id']).check().run()
at.button(key='meal_delete').click().run()
assert not at.exception and not entries() and len(list_measurements())==1
at.button(key='diary_add').click().run()
next(w for w in at.text_input if w.label=='料理名').set_value('測定なし').run()
at.button(key='meal_save').click().run()
assert not at.exception and entries()[0]['nutrition_status']=='unknown'
'''
    result=subprocess.run([sys.executable,'-c',script],cwd=Path(__file__).resolve().parents[2],
       env=dict(os.environ,PASTA_DATA_DIR=str(tmp_path/'data')),capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr


def test_finished_recipe_manual_edit_portion_and_calendar(tmp_path):
    script = r'''
from datetime import datetime
import sys
sys.path.insert(0,'pasta_weight_app_v2')
from streamlit.testing.v1 import AppTest
from meal_diary import entries
from diary_ui import JST
at=AppTest.from_string("from app_runtime import render_application; render_application(admin=False)", default_timeout=20).run()
at.button(key='home_meal').click().run()
select=lambda label: next(w for w in at.selectbox if w.label==label)
number=lambda label: next(w for w in at.number_input if w.label==label)
caption=lambda text: any(text in w.value for w in at.caption)
select('標準レシピの料理').set_value('カルボナーラ').run()
at.button(key='meal_use_recipe').click().run()
assert not at.exception and number('乾麺の参考重量（g）').value==80
assert next(w for w in at.text_input if w.label=='料理名').value=='カルボナーラ'
assert caption('実測値やAI推定値ではありません')
number('乾麺の参考重量（g）').set_value(100).run()
number('食べた割合（%）').set_value(50).run()
next(w for w in at.checkbox if w.label=='実際に使った材料と分量を確認しました').check().run()
assert any('料理全体' in w.value and '638.3 kcal' in w.value for w in at.markdown)
assert any('食べた分' in w.value and '319.' in w.value for w in at.markdown)
assert next(w for w in at.text_input if w.label=='料理名').value=='カルボナーラ'
at.button(key='meal_save').click().run()
assert not at.exception and len(entries())==1
row=entries()[0]
assert row['dish_name']=='カルボナーラ' and row['measurement_id'] is None
assert row['nutrition_snapshot']['calculation']['recipe_input']['ingredients_confirmed']
assert abs(row['nutrition_snapshot']['kcal']-319.15)<1e-6
at.button(key='meal_detail_cooking').click().run()
assert not at.exception and number('食べた割合（%）').value==50
assert next(w for w in at.checkbox if w.label=='実際に使った材料と分量を確認しました').value
number('食べた割合（%）').set_value(25).run()
at.button(key='meal_save').click().run()
assert not at.exception and len(entries())==1 and abs(entries()[0]['nutrition_snapshot']['kcal']-159.575)<1e-6
at.button(key='meal_detail_back').click().run()
at.radio(key='diary_view').set_value('カレンダー').run()
day=datetime.now(JST).date().isoformat()
assert '•' in at.button(key='diary_day_'+day).label
'''
    result = subprocess.run([sys.executable, '-c', script], cwd=Path(__file__).resolve().parents[2],
                           env=dict(os.environ, PASTA_DATA_DIR=str(tmp_path/'data')),
                           capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


def test_vision_suggestion_runs_once_and_accepts_manual_override(tmp_path):
    script = r'''
import sys
sys.path.insert(0,'pasta_weight_app_v2')
from streamlit.testing.v1 import AppTest
at=AppTest.from_string("""
import numpy as np
import streamlit as st
from types import SimpleNamespace
import dish_vision_ui
class MockVision:
    def predict(self, rgb):
        st.session_state['recognition_runs']=st.session_state.get('recognition_runs',0)+1
        return SimpleNamespace(name='トマトパスタ', visible_options=())
dish_vision_ui._classifier=lambda checkpoint, device: MockVision()
draft=st.session_state.setdefault('meal', {'id':'capture-fixture'})
# Streamlit may deliver a callback for a selector already cleared by a photo change.
dish_vision_ui._remember_dish_choice(draft, 'already-removed')
rgb=np.zeros((30,30,3), np.uint8) if st.checkbox('test-photo') else None
selected=dish_vision_ui.render_dish_choice(rgb, draft)
st.write(selected)
""").run()
at.toggle[0].set_value(False).run()
at.checkbox[0].check().run()
assert not at.exception and 'recognition_runs' not in at.session_state
at.toggle[0].set_value(True).run()
assert not at.exception and at.session_state['recognition_runs']==1
assert at.selectbox[0].value=='トマトパスタ'
assert at.selectbox[0].options==['料理を選ぶ（任意）', 'カルボナーラ', 'トマトパスタ', 'ジェノベーゼ']
at.selectbox[0].set_value('ジェノベーゼ').run()
assert not at.exception and at.selectbox[0].value=='ジェノベーゼ'
at.toggle[0].set_value(False).run()
assert at.selectbox[0].value=='ジェノベーゼ' and not at.info
assert at.session_state['recognition_runs']==1
'''
    result = subprocess.run([sys.executable, '-c', script], cwd=Path(__file__).resolve().parents[2],
                            env=dict(os.environ, PASTA_DATA_DIR=str(tmp_path/'data'),
                                     PASTA_OPENCLIP_CHECKPOINT=str(__file__),
                                     PASTA_DISH_HEAD_PATH=str(tmp_path/'invalid_research_artifact.json')),
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_meal_steps_preserve_photo_and_manual_selection_until_save(tmp_path):
    script = r'''
from io import BytesIO
import sys
from PIL import Image
from streamlit.testing.v1 import AppTest

sys.path.insert(0, 'pasta_weight_app_v2')
from meal_diary import entries, photo_path

def photo(color):
    image = BytesIO()
    Image.new('RGB', (32, 24), color).save(image, format='PNG')
    return image.getvalue()

def field(elements, label):
    return next(item for item in elements if item.label == label)

at = AppTest.from_file('pasta_weight_app_v2/app.py', default_timeout=20).run()
at.button(key='home_meal').click().run()
assert [heading.value for heading in at.subheader if heading.value.startswith('STEP ')] == [
    'STEP 1 · 料理の写真', 'STEP 2 · 料理名を確認',
    'STEP 3 · 材料と栄養を確認', 'STEP 4 · 食事日記に保存',
]
assert at.button(key='meal_save').disabled
at.file_uploader[0].upload('first.png', photo('red'), 'image/png').run()
field(at.selectbox, '標準レシピの料理').set_value('ジェノベーゼ').run()
field(at.text_input, '料理名').set_value('夕食のパスタ').run()
day = field(at.date_input, '食べた日').value
field(at.date_input, '食べた日').set_value(None).run()
assert at.button(key='meal_save').disabled
assert any('日付を入力してください' in notice.value for notice in at.caption)
field(at.date_input, '食べた日').set_value(day).run()
at.radio(key='main_nav').set_value('ホーム').run()
at.button(key='home_calendar').click().run()
assert at.radio(key='diary_view').value == 'カレンダー'
assert at.button(key='diary_add_selected_day')
resume = next(button for button in at.button if button.key and button.key.startswith('resume_'))
resume.click().run()
assert field(at.selectbox, '標準レシピの料理').value == 'ジェノベーゼ'
assert field(at.text_input, '料理名').value == '夕食のパスタ'
assert at.button(key='meal_remove_photo')
at.file_uploader[0].upload('replacement.png', photo('green'), 'image/png').run()
assert field(at.selectbox, '標準レシピの料理').value == 'ジェノベーゼ'
assert field(at.text_input, '料理名').value == '夕食のパスタ'
at.button(key='meal_save').click().run()
assert not at.exception
assert any('食事記録を保存しました' in notice.value for notice in at.success)
row, = entries()
with Image.open(photo_path(row['photo'])) as saved_photo:
    assert saved_photo.getpixel((0, 0)) == (0, 128, 0)

at.button(key='meal_detail_back').click().run()
assert at.radio(key='diary_view').value == 'カレンダー'
fresh = AppTest.from_file('pasta_weight_app_v2/app.py', default_timeout=20).run()
fresh.button(key='home_history').click().run()
assert fresh.radio(key='diary_view').value == '写真付き一覧'
assert any(heading.value == '夕食のパスタ' for heading in fresh.subheader)
assert len(entries()) == 1
'''
    env = dict(os.environ, PASTA_DATA_DIR=str(tmp_path/'data'))
    env.pop('PASTA_OPENCLIP_CHECKPOINT', None)
    result = subprocess.run([sys.executable, '-c', script], cwd=Path(__file__).resolve().parents[2],
                            env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
