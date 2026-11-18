"""Run the upload-to-weight flow in Chromium with a temporary DB and generated test images.

python tools/check_weight_flow.py --app-python .venv/bin/python --output /tmp/weight-check
The driver requires Playwright and Pillow. --source can select an older checkout for comparison.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import shutil
import socket
import sqlite3
import subprocess
import tempfile
import time
from urllib.request import build_opener, ProxyHandler

from PIL import Image, ImageDraw
from playwright.sync_api import expect, sync_playwright


CALIBRATED = '検証用・校正済み'
UNCALIBRATED = '検証用・校正未登録'
DEMO = 'デモ用（仮の栄養・重量データ）'
NEW = 'ブラウザで新規登録した合成テスト製品'
UPDATED = 'ブラウザで校正を更新する製品'


def fixtures(work):
    # Achromatic detail keeps the no-pasta control in the normal quality range.
    background = Image.new('RGB', (680, 470), (42, 42, 42))
    for x in range(0, background.width, 40):
        ImageDraw.Draw(background).line((x, 0, x, 469), fill=(65, 65, 65))
    background.save(work / 'empty.png')
    yellow = (213, 174, 70)
    separated = background.copy()
    for i in range(12):
        y = 80 + i * 24
        ImageDraw.Draw(separated).rectangle((75, y, 605, y + 5), fill=yellow)
    separated.save(work / 'spread.png')
    ImageDraw.Draw(separated).point((0, 0), fill=yellow)
    separated.save(work / 'edge-speck.png')
    dense = background.copy()
    ImageDraw.Draw(dense).rectangle((75, 150, 605, 240), fill=yellow)
    dense.save(work / 'dense.png')


@contextmanager
def test_server(source, app_python, work, output):
    env = dict(os.environ, PASTA_DATA_DIR=str(work / 'data'), PASTA_RESEARCH_DATA_DIR=str(work / 'research'))
    subprocess.run([app_python, '-c',
                    'from storage import init_db, save_product\ninit_db()\n'
                    'import sys\nfor name, grams in zip(sys.argv[1:], (.68, 0, .68)):\n'
                    '    save_product(name, grams, 350, 12, 2, 70)\n',
                    CALIBRATED, UNCALIBRATED, UPDATED],
                   cwd=source / 'pasta_weight_app_v2', env=env, check=True)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    with (output / 'server.log').open('w', encoding='utf-8') as log:
        proc = subprocess.Popen([app_python, '-m', 'streamlit', 'run', 'pasta_weight_app_v2/app.py',
                                 f'--server.port={port}', '--server.address=127.0.0.1', '--server.headless=true',
                                 '--browser.gatherUsageStats=false', '--server.fileWatcherType=none'],
                                cwd=source, env=env, stdout=log, stderr=log)
        try:
            opener = build_opener(ProxyHandler({}))
            for _ in range(100):
                if proc.poll() is not None:
                    raise RuntimeError('Isolated test server stopped before startup')
                try:
                    with opener.open(url + '/_stcore/health', timeout=1) as response:
                        if response.read().strip() == b'ok':
                            break
                except OSError:
                    time.sleep(.2)
            else:
                raise RuntimeError('Isolated test server did not start')
            yield url
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def button(page, key):
    return page.locator(f'.st-key-{key}').get_by_role('button')


def choose(page, key, name):
    select = page.locator('.st-key-' + key)
    combo = select.get_by_role('combobox')
    combo.click()
    combo.press('ArrowDown')
    page.get_by_role('option', name=name, exact=True).click()
    expect(combo).to_have_value(name)


def navigate(page, name):
    page.locator('.st-key-main_nav label').filter(has_text=name).click()


def open_capture(page, url, work, filename, product):
    page.goto(url)
    page.get_by_text('写真を選ぶ・はかりの数値を記録する', exact=True).click()
    button(page, 'home_upload').click()
    page.locator('input[type=file]').set_input_files(work / filename)
    choose(page, 'which_product', product)
    button(page, 'analyze_photo').click()


def read_result(page):
    hero = page.locator('.st-key-weight_hero')
    hero.or_(button(page, 'manual_weight_save')).first.wait_for()
    if hero.count():
        return hero.get_by_test_id('stMetricValue').inner_text().strip()
    return None


def show_diagnostics(page, require):
    label = page.get_by_text('写真と重量の診断', exact=True)
    if require:
        expect(label).to_be_visible()
    if label.count():
        label.click()
        return label.locator('xpath=ancestor::details[1]').inner_text()
    return None


def fill_product(page, name, grams, *, new=False):
    choose(page, 'product_edit_mode', '＋ 新しい製品を登録' if new else name)
    editor = page.get_by_test_id('stForm')
    if new:
        editor.get_by_role('textbox', name='製品名', exact=True).fill(name)
    else:
        expect(editor.get_by_role('textbox', name='製品名', exact=True)).to_have_value(name)
    weight = editor.get_by_role('spinbutton', name='1本あたりの重量（g）', exact=True)
    weight.fill(str(grams))
    weight.press('Tab')
    editor.get_by_role('button', name='製品データを保存', exact=True).click()
    page.get_by_text(re.compile('製品データを保存しました')).wait_for()


def run_checks(browser, url, work, output, baseline):
    results = []
    cases = [
        ('calibrated', 'spread.png', CALIBRATED, '8.2 g'),
        ('uncalibrated', 'spread.png', UNCALIBRATED, None),
        ('demo', 'spread.png', DEMO, None),
        ('new_product', 'spread.png', DEMO, '8.2 g'),
        ('recalibrate', 'spread.png', UPDATED, '9.0 g'),
        ('dense', 'dense.png', CALIBRATED, None),
        ('no_pasta', 'empty.png', CALIBRATED, None),
        ('border_speck', 'edge-speck.png', CALIBRATED, '8.2 g'),
    ]
    for index, (case, file, product, expected) in enumerate(cases):
        mobile = index % 2 == 1
        width, height = (390, 844) if mobile else (1280, 900)
        page = browser.new_page(viewport=dict(width=width, height=height), color_scheme='dark' if mobile else 'light')
        page.set_default_timeout(10000)
        check = dict(case=case, viewport=[width, height], expected=expected)
        try:
            open_capture(page, url, work, file, product)
            observed = read_result(page)
            if case == 'new_product':
                assert observed is None, 'real upload with demo product yielded a weight'
                button(page, 'recovery_calibration').click()
                fill_product(page, NEW, .68, new=True)
                button(page, 'calibration_return').click()
                observed = read_result(page)
            elif case == 'recalibrate':
                assert observed == '8.2 g'
                editor = browser.new_page(viewport=dict(width=1280, height=900))
                try:
                    editor.goto(url)
                    navigate(editor, '設定')
                    choose(editor, 'settings_section', '製品・校正')
                    fill_product(editor, UPDATED, .75)
                finally:
                    editor.close()
                navigate(page, '記録')
                navigate(page, 'ホーム')
                button(page, 'home_resume_capture').click()
                observed = read_result(page)
            check['observed'] = observed
            check['messages'] = page.get_by_test_id('stAlert').all_inner_texts()
            diagnostic = show_diagnostics(page, require=not baseline)
            if diagnostic:
                check['diagnosis'] = diagnostic
            target = page.get_by_text('写真と重量の診断', exact=True) if diagnostic else (
                page.locator('.st-key-weight_hero') if observed else page.get_by_test_id('stAlert').first)
            target.scroll_into_view_if_needed()
            page.wait_for_timeout(400)
            page.screenshot(path=output / (case + '.png'))
            assert not page.get_by_test_id('stException').count(), 'Streamlit exception'
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1'), 'horizontal overflow'
            assert observed == expected, f'expected {expected!r}, observed {observed!r}'
            if observed is None:
                expect(button(page, 'manual_weight_save')).to_be_visible()
            if case == 'new_product':
                button(page, 'measurement_save').click()
                button(page, 'measurement_to_diary').wait_for()
                with sqlite3.connect(work / 'data' / 'measurements.sqlite3') as db:
                    row = db.execute('SELECT product_name, g_per_strand, estimated_g, detected_count FROM measurements').fetchone()
                assert row[:2] == (NEW, .68) and abs(row[2] - 8.16) < 1e-6 and row[3] == 12
                check['saved_count'] = row[3]
            check['passed'] = True
        except Exception as exc:
            check.update(passed=False, error=str(exc))
            page.screenshot(path=output / (case + '_failure.png'))
            (output / (case + '_failure.txt')).write_text(page.locator('body').inner_text(), encoding='utf-8')
        finally:
            page.close()
        print(check['case'], check['passed'], check.get('observed'), check.get('error', ''), flush=True)
        results.append(check)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--app-python', type=Path, default=Path('.venv/bin/python'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', action='store_true', help='Record the pre-fix behavior; diagnostics may be absent.')
    args = parser.parse_args()
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='pasta-browser-weight-') as tmp:
        work = Path(tmp)
        fixtures(work)
        with test_server(args.source.resolve(), args.app_python.absolute(), work, output) as url, sync_playwright() as pw:
            browser = pw.chromium.launch(executable_path=shutil.which('chromium'), args=['--disable-dev-shm-usage'])
            try:
                results = run_checks(browser, url, work, output, args.baseline)
            finally:
                browser.close()
    (output / 'results.json').write_text(json.dumps(dict(synthetic_fixtures=True, physical_device=False,
        baseline=args.baseline, checks=results), ensure_ascii=False, indent=2), encoding='utf-8')
    return 0 if all(check['passed'] for check in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
