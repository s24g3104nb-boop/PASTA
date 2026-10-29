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
at=AppTest.from_file('pasta_weight_app_v2/app.py',default_timeout=15).run()
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
