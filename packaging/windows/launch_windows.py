"""Start only this app, then open its browser after the health check succeeds."""
from pathlib import Path
import argparse
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser


def application_root(source=False):
    support = Path(__file__).resolve().parent
    if source:
        if support.name != 'windows' or support.parent.name != 'packaging':
            raise ValueError('ソース起動はリポジトリ内の起動ファイルで実行してください。')
        return support.parents[1]
    return support


def launch_command(root, lan=False):
    return [sys.executable, '-m', 'streamlit', 'run', str(root / 'pasta_weight_app_v2/app.py'),
            '--server.address=' + ('0.0.0.0' if lan else '127.0.0.1'),
            '--server.port=8501', '--server.headless=true', '--browser.gatherUsageStats=false']


def launch_environment(root, source=False):
    env = dict(os.environ)
    # A checkout keeps the established app/data location. Packs isolate user_data.
    # Explicitly configured existing data paths are respected in either mode.
    default = root / ('pasta_weight_app_v2/data' if source else 'user_data')
    env.setdefault('PASTA_DATA_DIR', str(default))
    env.setdefault('PASTA_RESEARCH_DATA_DIR', str(Path(env['PASTA_DATA_DIR']) / 'research_collection'))
    return env


def ensure_port_available():
    # Do not mistake an older app already on 8501 for this new process.
    with socket.socket() as sock:
        if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        sock.bind(('127.0.0.1', 8501))


def wait_ready(process, timeout=40, *, clock=time.monotonic, sleep=time.sleep, opener=None):
    # Local health requests should not use a corporate/system HTTP proxy.
    opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = clock() + timeout
    while clock() < deadline:
        if process.poll() is not None:
            return False
        try:
            with opener.open('http://127.0.0.1:8501/_stcore/health', timeout=1) as response:
                if response.status == 200 and response.read().strip() == b'ok':
                    return process.poll() is None
        except (OSError, urllib.error.URLError):
            pass
        sleep(.25)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', action='store_true')
    parser.add_argument('--lan', action='store_true')
    parser.add_argument('--no-browser', action='store_true', help='Diagnostic/headless execution')
    args = parser.parse_args()
    process = None
    try:
        root = application_root(args.source)
        if not (root / 'pasta_weight_app_v2/app.py').is_file():
            raise ValueError('アプリのファイルがありません。ZIPをすべて展開してください。')
        ensure_port_available()
        process = subprocess.Popen(launch_command(root, args.lan), cwd=root, env=launch_environment(root, args.source))
        if not wait_ready(process):
            print('起動を確認できませんでした。上のメッセージを確認し、再起動してください。')
            return 1
        print('Pasta Lensを起動しました: http://localhost:8501', flush=True)
        if not args.no_browser:
            try:
                if not webbrowser.open('http://localhost:8501'):
                    print('ブラウザで http://localhost:8501 を開いてください。')
            except webbrowser.Error:
                print('ブラウザで http://localhost:8501 を開いてください。')
        return process.wait()
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError) as exc:
        print('起動できませんでした:', exc)
        print('8501番が使用中の場合は、旧版のPasta Lensを終了してください。')
        return 1
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == '__main__':
    raise SystemExit(main())
