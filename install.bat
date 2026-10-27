@echo off
rem One-time install (needs internet once). Afterwards everything runs offline on this PC.
setlocal
cd /d "%~dp0"

set "OLLAMA=D:\AI\Ollama\ollama.exe"

echo [1/3] Creating Python environment (.venv) ...
if not exist ".venv\Scripts\python.exe" (
    py -3.13 -m venv .venv 2>nul || python -m venv .venv
)
if not exist ".venv\Scripts\python.exe" (
    echo Python 3.11+ is required. Install it from python.org and run this again.
    pause
    exit /b 1
)

echo [2/3] Installing Python packages ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Package install failed.
    pause
    exit /b 1
)

echo [3/3] Checking AI models ...
if not exist "%OLLAMA%" (
    echo Ollama not found at %OLLAMA%. Run Install-NoteReader.ps1 from the wall project first.
    pause
    exit /b 1
)
"%OLLAMA%" list | findstr /c:"gemma3:4b" >nul || "%OLLAMA%" pull gemma3:4b
"%OLLAMA%" list | findstr /c:"gemma3:12b" >nul || "%OLLAMA%" pull gemma3:12b

echo.
echo Done. Double-click start.bat to run the note reader.
pause
