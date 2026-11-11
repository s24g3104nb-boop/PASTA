@echo off
chcp 65001 >nul
setlocal
if not exist "%~dp0packaging\windows\start_windows.bat" (
 echo 必要な起動ファイルがありません。GitHubのZIPをすべて展開してください。
 pause
 exit /b 1
)
call "%~dp0packaging\windows\start_windows.bat" --source --lan
exit /b %errorlevel%
