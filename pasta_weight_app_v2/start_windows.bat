@echo off
cd /d "%~dp0"
where py >nul 2>nul
if %ERRORLEVEL%==0 (
  set "PY=py"
) else (
  set "PY=python"
)
%PY% -m pip install -r requirements.txt
if errorlevel 1 (
  echo Failed to install requirements. Please check Python and Internet access.
  pause
  exit /b 1
)
%PY% -m streamlit run app.py
pause
