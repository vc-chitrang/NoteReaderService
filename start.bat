@echo off
rem Starts the note scanner desktop console. Close the console window to stop it.
setlocal
cd /d "%~dp0"
title Good Deed Tree - Note Scanner

if not exist ".venv\Scripts\python.exe" (
    echo Not installed yet: run install.bat first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m app.desktop
echo.
echo Note scanner stopped.
pause
