"""Isolated storage contracts including failure rollback and source snapshots."""
import sys
from pathlib import Path
from datetime import date,time
from io import BytesIO
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import meal_diary as diary
from storage import init_db,save_measurement
from cooking_storage import save_cooking_session
from meal_photos import prepare_photo
from meal_image_ai import UnavailableMealImageEstimator

@pytest.fixture
def db(tmp_path):
    db=tmp_path/'test.sqlite3';init_db(db);return db

def save(db,ident='one',**kw):
    return diary.save(ident,date(2020,1,2),time(12,30),'トマトパスタ',db=db,**kw)

def measure(db):
    return save_measurement(product_name='試験',mode='spread',detected_count=20,estimated_g=20,actual_g=25,
     nutrition=dict(kcal=70,protein_g=2,fat_g=1,carb_g=14),image_rgb=np.zeros((30,30,3),dtype=np.uint8),
     product_snapshot=dict(g_per_strand=1,kcal_100=350,protein_100=12,fat_100=2,carb_100=70),db_path=db,source_kind='synthetic',submission_token='dry')

def test_optional_photo_unknown_and_multiple_same_day(db):
    a=save(db);save(db,'two')
    assert len(diary.entries(db))==2 and a['nutrition_snapshot'] is None
    assert a['nutrition_status']=='unknown' and a['eaten_at'].endswith('+09:00')
    assert a['created_at']!=a['eaten_at']

def test_backup_before_upgrade_preserves_old_rows(db):
    with sqlite3.connect(db) as c:
        c.execute('CREATE TABLE meal_entries(id TEXT PRIMARY KEY,eaten_date TEXT,eaten_time TEXT,dish_name TEXT,memo TEXT,photo TEXT,created_at TEXT,updated_at TEXT,deleted INTEGER DEFAULT 0)')
        c.execute("INSERT INTO meal_entries VALUES('old','2020-01-01',NULL,'古い料理','',NULL,'stamp','stamp',0)")
    diary.initialize(db)
    assert diary.get('old',db)['dish_name']=='古い料理'
    backups=list(db.parent.glob('*.backup'));assert backups
    with sqlite3.connect(backups[0]) as c:assert len(list(c.execute('PRAGMA table_info(meal_entries)')))==9

def test_idempotency_and_explicit_edit_conflict(db):
    a=save(db);assert save(db)['updated_at']==a['updated_at']
    b=diary.save('one',date(2020,2,2),None,'変更',db=db,edit=True,expected_updated_at=a['updated_at'])
    assert b['dish_name']=='変更' and b['eaten_at'] is None and b['created_at']==a['created_at']
    with pytest.raises(ValueError):diary.save('one',date.today(),None,'競合',db=db,edit=True,expected_updated_at=a['updated_at'])
    diary.delete('one',db,expected_updated_at=b['updated_at']);assert not diary.entries(db)
    assert diary.get('one',db) is None

def test_concurrent_double_save(db):
    diary.initialize(db)
    with ThreadPoolExecutor(2) as pool:rows=list(pool.map(lambda _:save(db),range(2)))
    assert rows[0]['id']==rows[1]['id'] and len(diary.entries(db))==1

def test_measurement_snapshot_and_actual_nutrition(db):
    mid=measure(db);assert measure(db)==mid
    a=save(db,measurement_id=mid)
    assert a['measurement_snapshot']['estimated_g']==20 and a['measurement_snapshot']['actual_g']==25
    assert a['nutrition_snapshot']['kcal']==87.5 and a['nutrition_status']=='dry_only'
    with sqlite3.connect(db) as c:c.execute('UPDATE measurements SET actual_g=40 WHERE id=?',(mid,))
    a=diary.save('one',date.today(),None,'メモ編集',db=db,edit=True,measurement_id=mid)
    assert a['nutrition_snapshot']['kcal']==87.5

def test_cooking_link_and_invalid_reference(db):
    mid=measure(db)
    from dish_classifier import DISH_NAMES
    sid=save_cooking_session(measurement_id=mid,selected_dish=DISH_NAMES[0],confirmed_same_batch=True,ingredients=[],servings=1,db_path=db)
    a=save(db,cooking_session_id=sid)
    assert a['measurement_id']==mid and a['nutrition_status']=='confirmed_amounts'
    assert 'calculation' in a['nutrition_snapshot']
    with pytest.raises(ValueError):save(db,'bad',measurement_id=999,cooking_session_id=sid)

def test_photo_metadata_validation_failure_cleanup(db,monkeypatch):
    image=Image.new('RGB',(48,48),'orange');buf=BytesIO();image.save(buf,'PNG')
    a=save(db,photo=buf.getvalue());assert diary.photo_path(a['photo'],db).is_file()
    with pytest.raises(ValueError):diary.photo_path('../escape.png',db)
    with pytest.raises(ValueError):prepare_photo(b'broken')
    prediction=UnavailableMealImageEstimator().predict(prepare_photo(buf.getvalue()))
    assert prediction.nutrition is None and prediction.status=='unavailable'
    with sqlite3.connect(db) as c:c.execute("CREATE TRIGGER fail_meal BEFORE INSERT ON meal_entries BEGIN SELECT RAISE(ABORT,'test'); END")
    count=len(list((db.parent/'meal_photos').iterdir()))
    with pytest.raises(sqlite3.Error):save(db,'failed',photo=buf.getvalue())
    assert len(list((db.parent/'meal_photos').iterdir()))==count

def test_backup_failure_leaves_schema_untouched(db,monkeypatch):
    monkeypatch.setattr(diary,'backup_database',lambda *a,**kw:(_ for _ in ()).throw(OSError('disk full')))
    with pytest.raises(OSError):diary.initialize(db)
    with sqlite3.connect(db) as c:assert not c.execute("SELECT 1 FROM sqlite_master WHERE name='meal_entries'").fetchone()


def test_finished_recipe_reference_and_portion_without_measurement(db):
    from finished_recipe import standard_rows
    rows = standard_rows('トマトパスタ', 80)
    assert [row['amount_g'] for row in rows] == pytest.approx([80, 6.4, 2.4])
    rows[0]['amount_g'] = 60
    rows.pop()
    rows.append(dict(name='追加食材', amount_g=25, kcal_100=100, protein_100=4, fat_100=2, carb_100=10))
    recipe = dict(dish='トマトパスタ', reference_dry_g=80, ingredients=rows,
                  consumed_percent=50, ingredients_confirmed=True)
    result = save(db, finished_recipe=recipe)
    snapshot = result['nutrition_snapshot']
    calc = snapshot['calculation']
    assert result['nutrition_status'] == 'template_estimate'
    assert result['measurement_id'] is None and result['cooking_session_id'] is None
    assert calc['dry_weight_source'] == 'reference' and calc['dry_weight_g'] == 80
    assert calc['ingredient_origins'] == ['user_input', 'standard_recipe', 'user_input']
    assert calc['total']['kcal'] == pytest.approx(296.8 + 14.4 + 56.576 + 25)
    assert snapshot['kcal'] == pytest.approx(calc['total']['kcal'] / 2)
    assert all(snapshot[nutrient] == pytest.approx(value/2) for nutrient, value in calc['total'].items())
    assert diary.get('one', db)['nutrition_snapshot'] == snapshot
    assert save(db, finished_recipe=recipe)['updated_at'] == result['updated_at']


def test_finished_recipe_with_saved_dry_and_frozen_snapshot(db):
    from finished_recipe import standard_rows
    mid = measure(db)
    recipe = dict(dish='ジェノベーゼ', reference_dry_g=80, ingredients=standard_rows('ジェノベーゼ', 25),
                  consumed_percent=25, ingredients_confirmed=True)
    first = save(db, measurement_id=mid, finished_recipe=recipe)
    with sqlite3.connect(db) as c:
        c.execute('UPDATE measurements SET actual_g=40 WHERE id=?', (mid,))
    assert first['nutrition_status'] == 'confirmed_amounts'
    assert first['nutrition_snapshot']['calculation']['dry_weight_source'] == 'actual'
    assert first['nutrition_snapshot']['calculation']['dry_weight_g'] == 25
    unchanged = save(db, measurement_id=mid, edit=True, expected_updated_at=first['updated_at'])
    assert unchanged['nutrition_snapshot'] == first['nutrition_snapshot']
    recipe['consumed_percent'] = 50
    edited = save(db, measurement_id=mid, edit=True, expected_updated_at=unchanged['updated_at'],
                  finished_recipe=recipe)
    assert edited['nutrition_snapshot']['kcal'] == pytest.approx(2 * first['nutrition_snapshot']['kcal'])
    assert edited['nutrition_snapshot']['calculation']['dry_weight_g'] == 25


@pytest.mark.parametrize(('dish', 'amounts'), [
    ('カルボナーラ', [50, 25, 15]), ('トマトパスタ', [100, 8, 3]), ('ジェノベーゼ', [15, 15, 8, 12, 3]),
])
def test_finished_standard_recipe_amounts(dish, amounts):
    from finished_recipe import standard_rows
    assert [row['amount_g'] for row in standard_rows(dish, 100)] == amounts


def test_finished_recipe_can_use_older_weight_without_saved_nutrition():
    from finished_recipe import recipe_dry_basis
    dry = recipe_dry_basis({'reference_dry_g': 80}, {'estimated_g': 55, 'actual_g': 50})
    assert dry['weight_source'] == 'actual' and dry['dry_weight_g'] == 50
    assert dry['nutrition']['kcal'] == 185.5
    assert 'USDA' in dry['nutrition_source']
