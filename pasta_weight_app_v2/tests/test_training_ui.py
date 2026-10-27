"""管理画面を一時DBで検証し、利用者の写真・履歴を使用しない。"""
from pathlib import Path

from streamlit.testing.v1 import AppTest


def renderer_script(db, *, seed=False):
    app_dir = str(Path(__file__).resolve().parents[1])
    return f'''
import sys
sys.path.insert(0, {app_dir!r})
import io
import numpy as np
from PIL import Image
from storage import init_db
from training_data import register_sample, list_samples
from training_ui import render_training_tab
db = {str(db)!r}
init_db(db)
if {seed!r} and not list_samples(db, include_deleted=True):
    image = io.BytesIO()
    Image.fromarray(np.random.default_rng(1).integers(0,255,(32,32,3),dtype=np.uint8)).save(image,'PNG')
    register_sample(image.getvalue(), dict(actual_weight_g=20, mode='spread',
        product_id='1', batch_id='fixture-b1',session_id='fixture-s1',source='real'),db_path=db)
render_training_tab(db)
'''


def test_no_data_and_photo_required(tmp_path):
    at = AppTest.from_string(renderer_script(tmp_path / 'ui.sqlite3')).run(timeout=15)
    assert not at.exception
    assert at.button(key='training_start').disabled
    assert any('学習データはまだありません' in row.value for row in at.info)
    next(row for row in at.button if row.label == '学習データを登録').click().run(timeout=15)
    assert not at.exception
    assert any('写真をアップロード' in row.value for row in at.error)


def test_edit_and_delete_registered_sample(tmp_path):
    db = tmp_path / 'ui.sqlite3'
    at = AppTest.from_string(renderer_script(db, seed=True)).run(timeout=15)
    assert not at.exception
    at.selectbox(key='training_selected_sample').set_value(1).run(timeout=15)
    assert not at.exception
    at.number_input(key='training_edit_1_1_weight').set_value(35)
    next(row for row in at.button if row.label == 'ラベル変更を保存').click().run(timeout=15)
    assert not at.exception
    assert any('ラベルを更新しました' in row.value for row in at.success)
    import sqlite3
    with sqlite3.connect(db) as con:
        weight, version = con.execute('SELECT actual_weight_g,version FROM training_samples').fetchone()
    assert weight == 35 and version == 2
    at.checkbox(key='training_delete_confirm_1').check().run(timeout=15)
    at.button(key='training_delete_1').click().run(timeout=15)
    assert not at.exception
    assert any('学習対象から削除しました' in row.value for row in at.success)
    with sqlite3.connect(db) as con:
        assert con.execute('SELECT deleted_at FROM training_samples').fetchone()[0] is not None


def test_reports_and_failed_job_status_are_visible(tmp_path):
    script = renderer_script(tmp_path / 'status.sqlite3')
    script = script[:script.rindex('render_training_tab(db)')] + '''
import training_ui
report = dict(split_counts={'train':9,'val':3,'test':3}, notice='valのみのテスト用表示')
for key in ('regression','opencv'):
    report[key] = dict(overall={'total':1,'succeeded':1,'failed':0,'mae_g':2.5,'mape_percent':5},
        rows=[dict(actual_weight_g=50,predicted_g=52.5,mode='spread',success=True,reason='表示テスト')],
        by={'mode': {'spread': {'total':1,'mae_g':2.5}},'actual_weight_g':{'50':{'total':1,'mae_g':2.5}}})
training_ui.list_jobs = lambda *a, **k: [dict(id='job1',status='failed',progress=20,
    message='検証中止',error='独立グループが不足しています',sample_count=1)]
training_ui.list_models = lambda *a, **k: [dict(id='model1',job_id='job0',source='synthetic',
    active=False,created_at='fixture',report=report)]
def broken_active(*a, **k):
    raise ValueError('モデルのチェックサムが不一致です')
training_ui.get_active_model = broken_active
render_training_tab(db)
'''
    at = AppTest.from_string(script).run(timeout=15)
    assert not at.exception
    assert any('独立グループが不足' in row.value for row in at.error)
    assert any('選択中のモデルを読み込めません' in row.value for row in at.error)
    at.selectbox(key='training_selected_model').set_value('model1').run(timeout=15)
    assert not at.exception
    assert at.button(key='training_activate').disabled
    assert any('合成画像テスト用' in row.value for row in at.warning)
    assert any('MAE/RMSE/MAPE' in row.value for row in at.caption)
    assert len(at.dataframe) >= 4
