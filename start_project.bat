@echo off
echo ====================================================
echo   Starting CrisisLens Web Application...
echo   Please wait 60-90 seconds for models to load.
echo ====================================================
echo.

:: Change to the correct directory automatically
cd /d "%~dp0"

:: Set text encoding to prevent crashes
set PYTHONIOENCODING=utf-8

:: Activate the virtual environment
call .venv\Scripts\activate.bat

:: Run the Flask Application
python app.py

pause
