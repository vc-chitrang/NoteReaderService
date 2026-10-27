@echo off
rem Starts the note reader desktop console. Close the console window to stop it.
setlocal
cd /d "%~dp0"
title Good Deed Tree - Note Reader

if not exist ".venv\Scripts\python.exe" (
    echo Not installed yet: run install.bat first.
    pause
    exit /b 1
)

rem The AI model runs in Ollama; start it if it is not running already.
tasklist /fi "imagename eq ollama.exe" | find /i "ollama.exe" >nul
if errorlevel 1 (
    echo Starting Ollama ...
    start "" /min "D:\AI\Ollama\ollama.exe" serve
    timeout /t 3 /nobreak >nul
)

".venv\Scripts\python.exe" -m app.desktop
echo.
echo Note reader stopped.
pause
