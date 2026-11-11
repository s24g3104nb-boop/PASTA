@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if not exist "start_windows.bat" (
 echo start_windows.bat がありません。ZIPをすべて展開してください。
 pause
 exit /b 1
)
call "%~dp0start_windows.bat" --lan
exit /b %errorlevel%
