"""定数を返すテストモデルで画面と履歴を検証する。実写精度の試験ではない。"""
from pathlib import Path
import sqlite3

from streamlit.testing.v1 import AppTest


def script(db, *, product_id=2, demo=False):
    return f'''
import sys
sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
import numpy as np
from storage import init_db
from measurement_ui import render_regression_measurement
db = {str(db)!r}
init_db(db)
# Controlled fixture exercises the real-only branch, without real-world accuracy claims.
artifact = dict(source='real', model_version='test-fixture', entries={{'spread:2': dict(
    mean=[0]*36, scale=[1]*36, coef=[0]*36, intercept=35.0,
    min_weight=20, max_weight=100, max_feature_distance=999)}})
active = dict(id='fixture-model', artifact=artifact)
product = dict(id={product_id}, name='test product', g_per_strand=0.9,
               kcal_100=350, protein_100=12, fat_100=2, carb_100=70)
render_regression_measurement(active, rgb=np.random.default_rng(1).integers(0,255,(64,64,3),dtype=np.uint8),
    product=product, mode='広げた状態', using_demo={demo}, db_path=db)
'''


def test_continuous_model_measurement_and_history(tmp_path):
    db = tmp_path / 'measurement.sqlite3'
    at = AppTest.from_string(script(db)).run(timeout=15)
    assert not at.exception
    assert at.metric[0].value == '35.0 g'
    assert at.metric[1].value == '122.5 kcal'
    at.button[0].click().run(timeout=15)
    assert not at.exception
    with sqlite3.connect(db) as con:
        row = con.execute('SELECT estimated_g,detected_count,estimator,source_kind FROM measurements').fetchone()
    assert row == (35, 0, 'ridge:fixture-model', 'real')


def test_untrained_product_has_no_weight_or_save(tmp_path):
    at = AppTest.from_string(script(tmp_path / 'unknown.sqlite3', product_id=99)).run(timeout=15)
    assert not at.exception
    assert not at.metric and not at.button
    assert any('未学習' in row.value for row in at.warning)


def test_cg_cannot_be_measured_with_real_model(tmp_path):
    at = AppTest.from_string(script(tmp_path / 'cg.sqlite3', demo=True)).run(timeout=15)
    assert not at.exception
    assert not at.metric and not at.button
    assert any('実物の写真' in row.value for row in at.warning)
