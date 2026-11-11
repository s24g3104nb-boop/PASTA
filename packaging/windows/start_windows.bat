@echo off
chcp 65001 >nul
setlocal
set "SUPPORT_DIR=%~dp0"
set "SOURCE_FLAG="
cd /d "%~dp0"
if /i "%~1"=="--source" (
 set "SOURCE_FLAG=--source"
 cd /d "%~dp0..\.."
 shift
)
if not exist ".venv\Scripts\python.exe" goto setup_needed
if not exist "%SUPPORT_DIR%verify_install.py" goto missing
if not exist "%SUPPORT_DIR%launch_windows.py" goto missing
".venv\Scripts\python.exe" "%SUPPORT_DIR%verify_install.py" %SOURCE_FLAG%
if errorlevel 1 goto failed
rem Default: 127.0.0.1 only. Pack data: user_data. Source: pasta_weight_app_v2/data.
set "LAN_FLAG="
if /i "%~1"=="--lan" goto lan
if not "%~1"=="" goto failed
goto run
:lan
echo 家庭内の信頼できるプライベートネットワーク専用です。
echo 認証機能はありません。同じネットワークの人に記録が見える可能性があります。
echo 外部公開、ルーターのポート開放、公衆Wi-Fiでの利用はしないでください。
choice /c YN /n /m "LAN接続を有効にしますか？ [Y/N]: "
if errorlevel 2 exit /b 0
set "LAN_FLAG=--lan"
:run
echo 起動後にPCのブラウザを開きます。開かない場合は http://localhost:8501 を開いてください。
echo 終了するときは、この画面で Ctrl+C を押してください。
".venv\Scripts\python.exe" "%SUPPORT_DIR%launch_windows.py" %SOURCE_FLAG% %LAN_FLAG%
if errorlevel 1 goto failed
exit /b 0
:setup_needed
echo 先に setup_windows.bat を実行してください。
goto fail_exit
:missing
echo 配布ファイルが不足しています。ZIPを新しいフォルダへ再展開してください。
goto fail_exit
:failed
echo 起動できませんでした。上のエラーを確認してください。
echo 8501番ポート使用中の場合は、以前のPasta Lensを終了してください。
:fail_exit
pause
exit /b 1
