"""Optional developer check; writes only to a new temporary directory."""
import os
from pathlib import Path
import tempfile
import sys

root=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix='pasta-pack-test-') as temp:
    os.environ['PASTA_DATA_DIR']=str(Path(temp)/'data')
    os.environ['PASTA_RESEARCH_DATA_DIR']=str(Path(temp)/'research')
    sys.path.insert(0,str(root/'pasta_weight_app_v2'))
    from streamlit.testing.v1 import AppTest
    from storage import save_product,list_measurements
    from meal_diary import entries
    app=AppTest.from_file(str(root/'pasta_weight_app_v2/app.py'),default_timeout=20).run()
    assert not app.exception,app.exception
    save_product('配布検証用',0,350,12,2,70)
    app.button(key='home_manual_weight').click().run()
    app.selectbox(key='manual_only_product').set_value(2).run()
    next(n for n in app.number_input if n.label=='はかりで測った実測重量（g）').set_value(100.).run()
    app.button(key='manual_weight_save').click().run()
    assert not app.exception,app.exception
    row=list_measurements()[0]
    assert row['actual_g']==100 and row['estimated_g'] is None and row['kcal']==350 and row['image_filename'] is None
    app.button(key='manual_weight_to_diary').click().run()
    next(n for n in app.text_input if n.label=='料理名').set_value('配布検証用パスタ').run()
    app.button(key='meal_save').click().run()
    assert not app.exception and entries()[0]['nutrition_snapshot']['kcal']==350
    app.radio(key='main_nav').set_value('設定').run()
    assert app.get('download_button')
    app.selectbox(key='settings_section').set_value('開発・検証ツール').run()
    app.selectbox(key='developer_section').set_value('独立試料・撮影データ登録').run()
    assert not app.exception and app.button(key='collection_save').disabled
    from experiments.capture_sheet import render_sheet
    from experiments.marker_reference import analyze_marker_reference
    assert analyze_marker_reference(render_sheet())[0]['detected_marker_count']==4
print('PASS: first launch, 100g manual/no photo, 350kcal, diary, PDF, research registration and synthetic markers')
