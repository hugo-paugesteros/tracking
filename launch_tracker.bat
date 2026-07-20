@echo off
REM Double-click this file to launch the tracker.
cd /d "%~dp0"

if not exist "venv" (
    echo No 'venv' folder found here.
    echo Please follow the setup steps in README.md first ^(creating the
    echo virtual environment and running 'pip install -e .'^).
    echo.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
launch-tracker

echo.
pause
