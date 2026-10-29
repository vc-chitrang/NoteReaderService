@echo off
rem One-time install (needs internet once). Afterwards everything runs offline on this PC.
setlocal
cd /d "%~dp0"

echo [1/2] Creating Python environment (.venv) ...
if not exist ".venv\Scripts\python.exe" (
    py -3.13 -m venv .venv 2>nul || python -m venv .venv
)
if not exist ".venv\Scripts\python.exe" (
    echo Python 3.11+ is required. Install it from python.org and run this again.
    pause
    exit /b 1
)

echo [2/2] Installing Python packages ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Package install failed.
    pause
    exit /b 1
)

echo.
echo Done. Double-click start.bat to run the note scanner.
pause
