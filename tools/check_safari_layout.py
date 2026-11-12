"""Check alert contrast and phone navigation using disposable Streamlit UI only.

Run with a Python environment containing Playwright:
  python tools/check_safari_layout.py --app-python .venv/bin/python --output /tmp/layout-check
Requires Chromium, or --engine webkit with a separately installed Playwright engine.
An engine run is not verification on a physical iPhone.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
from urllib.request import ProxyHandler, build_opener


def contrast_ratio(foreground, background):
    def luminance(rgb):
        components = [c / 255 for c in rgb[:3]]
        values = [c / 12.92 if c <= .04045 else ((c + .055) / 1.055) ** 2.4 for c in components]
        return sum(c * w for c, w in zip(values, (.2126, .7152, .0722)))
    low, high = sorted([luminance(foreground), luminance(background)])
    return (high + .05) / (low + .05)


def rectangles_overlap(left, right):
    return left['width'] > 0 and right['width'] > 0 and min(left['x'] + left['width'], right['x'] + right['width']) > max(left['x'], right['x']) and min(left['y'] + left['height'], right['y'] + right['height']) > max(left['y'], right['y'])


def fixture(app_dir):
    return f'''
import sys
sys.path.insert(0,{str(app_dir)!r})
import streamlit as st
from ui_shell import shell
st.set_page_config(page_title='Pasta Lens layout verification',layout='centered')
st.session_state['theme_preference']='ダーク' if st.query_params.get('theme')=='dark' else 'ライト'
shell()
st.title('重量推定を確認する')
st.warning('重量を計算できません：**接触・重なり**を検出しました。製品の校正値 `0.68g/本` だけでは本数は分かりません。[撮影の案内](#guide)を確認してください。')
st.error('写真から重量を推定できませんでした。**写真は保持しています。** 撮り直すか、はかりで測った値を入力してください。')
st.info('実測重量がある場合は、推定結果と区別して保存できます。')
for n in range(3):
 with st.container(border=True): st.write('写真の確認 '+str(n+1));st.caption('原本の変更は行いません。')
st.subheader('実際の重さを入力する')
st.number_input('はかりで測った重さ（g）',min_value=0.1,value=100.,key='verification_actual')
st.button('実測重量を保存する',key='verification_save',width='stretch')
st.warning('保存に失敗しました。**入力した100gは保持しています。** 空き容量を確認して再試行してください。')
st.caption('末尾の案内もナビゲーションに隠れないことを確認します。')
'''


COLOR_PROBE = r'''(root) => {
 const rgba = value => {const m=value.match(/[\d.]+/g).map(Number); return m.length===3 ? [...m,1] : m;};
 const background = node => {
  const lineage=[];for(let p=node;p;p=p.parentElement)lineage.unshift(p);
  let result=[255,255,255];for(const p of lineage){const c=rgba(getComputedStyle(p).backgroundColor);result=result.map((v,i)=>c[i]*c[3]+v*(1-c[3]));}return result;
 };
 return [...root.querySelectorAll('p,strong,code,a')].map(n=>({tag:n.tagName, foreground:rgba(getComputedStyle(n).color).slice(0,3), background:background(n)}));
}'''


@contextmanager
def server(app_python, root, output, app_dir):
    with tempfile.TemporaryDirectory(prefix='pasta-layout-') as tmp:
        path = Path(tmp)
        app = path / 'app.py'
        app.write_text(fixture(app_dir))
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        env = dict(os.environ, PASTA_DATA_DIR=str(path / 'data'), PASTA_RESEARCH_DATA_DIR=str(path / 'research'))
        with (output / 'server.log').open('w') as log:
            process = subprocess.Popen([str(app_python), '-m', 'streamlit', 'run', str(app), '--server.address=127.0.0.1', f'--server.port={port}', '--server.headless=true', '--browser.gatherUsageStats=false'], cwd=root, env=env, stdout=log, stderr=log)
            try:
                opener = build_opener(ProxyHandler({}))
                for _ in range(100):
                    if process.poll() is not None:
                        raise RuntimeError('Test Streamlit server exited')
                    try:
                        with opener.open(f'http://127.0.0.1:{port}/_stcore/health', timeout=1) as response:
                            if response.read().strip() == b'ok':break
                    except OSError:time.sleep(.2)
                else:raise RuntimeError('Test Streamlit server did not become ready')
                yield f'http://127.0.0.1:{port}'
            finally:
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app-python', type=Path, default=Path('.venv/bin/python'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ui-source', type=Path, help='Optional old ui_shell.py directory for before/after recording')
    parser.add_argument('--engine', choices=('chromium', 'webkit'), default='chromium')
    parser.add_argument('--baseline', action='store_true', help='Record failures without treating the old layout as passing')
    args = parser.parse_args()
    from playwright.sync_api import sync_playwright
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve();output.mkdir(parents=True, exist_ok=True)
    checks = []
    with server(args.app_python.absolute(), root, output, args.ui_source or root / 'pasta_weight_app_v2') as url, sync_playwright() as playwright:
        browser = getattr(playwright, args.engine).launch(**({'executable_path':'/usr/bin/chromium', 'args':['--no-sandbox']} if args.engine == 'chromium' else {}))
        try:
            for width, height in [(1280,900), (390,844), (320,568), (844,390)]:
                for theme in ('light', 'dark'):
                    page = browser.new_page(viewport=dict(width=width,height=height), color_scheme=theme)
                    page.goto(url + '/?theme=' + theme)
                    page.get_by_text('重量推定を確認する',exact=True).wait_for()
                    page.wait_for_timeout(200)
                    page.screenshot(path=str(output / f'{theme}_{width}x{height}_top.png'), full_page=True)
                    colors = [color for alert in page.get_by_test_id('stAlert').all() for color in alert.evaluate(COLOR_PROBE)]
                    for color in colors:color['contrast']=contrast_ratio(color['foreground'],color['background'])
                    minimum = min(color['contrast'] for color in colors)
                    nav=page.locator('.st-key-main_nav')
                    position=nav.evaluate('(e)=>getComputedStyle(e).position')
                    targets = [page.get_by_role('spinbutton'),page.get_by_role('button',name='実測重量を保存する',exact=True),page.get_by_test_id('stAlert').last]
                    overlaps=[]
                    for target in targets:
                        target.scroll_into_view_if_needed()
                        overlaps.append(rectangles_overlap(target.bounding_box(),nav.bounding_box()))
                    page.get_by_role('spinbutton').focus()
                    horizontal=page.evaluate('document.documentElement.scrollWidth>innerWidth+2')
                    page.screenshot(path=str(output / f'{theme}_{width}x{height}.png'), full_page=True)
                    # Native controls must still receive input/click after scrolling.
                    page.get_by_role('spinbutton').fill('100')
                    page.get_by_role('button',name='実測重量を保存する',exact=True).click()
                    checks.append(dict(width=width,height=height,theme=theme,minimum_text_contrast=minimum,colors=colors,navigation_position=position,critical_content_overlaps=overlaps,horizontal_overflow=horizontal,passed=minimum>=4.5 and not horizontal and position=='static' and not any(overlaps)))
                    page.close()
        finally:browser.close()
    result=dict(engine=args.engine,physical_iphone=False,all_passed=all(c['passed'] for c in checks),checks=checks)
    (output/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    (output/'index.html').write_text('<!doctype html><html lang="ja"><meta charset="utf-8"><title>警告表示・重なり検証</title><style>body{font:16px sans-serif;max-width:1000px;margin:auto;background:#faf8f2;color:#29352e}img{max-width:100%}figure{border:1px solid #ddd;padding:12px}</style><h1>警告・ナビゲーション検証</h1><p>専用合成画面による'+args.engine+'検証。iPhone実機ではありません。</p>'+''.join(f'<figure><figcaption>{p.name}</figcaption><img src="{p.name}"></figure>' for p in sorted(output.glob('*.png'))))
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}))
    if not args.baseline and not result['all_passed']:raise SystemExit(1)


if __name__=='__main__':main()
