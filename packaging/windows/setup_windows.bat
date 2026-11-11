@echo off
chcp 65001 >nul
setlocal
set "SUPPORT_DIR=%~dp0"
set "SOURCE_FLAG="
cd /d "%~dp0"
if /i "%~1"=="--source" (
 set "SOURCE_FLAG=--source"
 cd /d "%~dp0..\.."
) else if not "%~1"=="" (
 goto missing
)
if not exist "%SUPPORT_DIR%requirements.txt" goto missing
if not exist "%SUPPORT_DIR%verify_install.py" goto missing
if not exist "%SUPPORT_DIR%launch_windows.py" goto missing
if not exist "pasta_weight_app_v2\app.py" goto missing
if exist ".venv\Scripts\python.exe" goto install
py -3.12 -c "import sys; sys.exit(sys.version_info[:2] != (3,12))" >nul 2>&1
if errorlevel 1 goto try_python
py -3.12 -m venv ".venv"
if errorlevel 1 goto failed
goto install
:try_python
python -c "import sys; sys.exit(sys.version_info[:2] != (3,12))" >nul 2>&1
if errorlevel 1 goto no_python
python -m venv ".venv"
if errorlevel 1 goto failed
:install
".venv\Scripts\python.exe" -c "import sys; sys.exit(sys.version_info[:2] != (3,12))"
if errorlevel 1 goto no_python
".venv\Scripts\python.exe" -m pip install -r "%SUPPORT_DIR%requirements.txt"
if errorlevel 1 goto failed
".venv\Scripts\python.exe" "%SUPPORT_DIR%verify_install.py" %SOURCE_FLAG%
if errorlevel 1 goto failed
echo セットアップ完了。start_windows.bat を実行してください。
pause
exit /b 0
:missing
echo 配布ファイルが不足しています。ZIPを右クリックして、すべて展開してください。
goto fail_exit
:no_python
echo Python 3.12 が見つからない、または仮想環境のバージョンが違います。
echo Python 3.12 をインストールし、新しい展開先でセットアップしてください。
goto fail_exit
:failed
echo セットアップに失敗しました。上のエラーとネット接続、空き容量を確認してください。
:fail_exit
pause
exit /b 1
