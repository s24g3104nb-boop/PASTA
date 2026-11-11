@echo off
chcp 65001 >nul
setlocal
if not exist "%~dp0..\start_windows.bat" (
 echo 上位フォルダの起動ファイルがありません。最新版ZIPをすべて展開してください。
 pause
 exit /b 1
)
call "%~dp0..\start_windows.bat"
exit /b %errorlevel%
