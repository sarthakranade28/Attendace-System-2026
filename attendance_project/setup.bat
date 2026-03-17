@echo off
echo ============================================
echo Face Attendance System - Setup Script
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org
    pause
    exit /b 1
)

echo [1/4] Checking Python version...
python --version

echo.
echo [2/4] Creating virtual environment...
if exist venv (
    echo Virtual environment already exists
) else (
    python -m venv venv
    echo Virtual environment created
)

echo.
echo [3/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo [4/4] Installing dependencies...
echo This may take 5-10 minutes on first installation due to large ML models...
echo.

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo.
echo ============================================
echo Setup Complete!
echo ============================================
echo.
echo To start the application:
echo   1. Run: venv\Scripts\activate.bat
echo   2. Run: python app.py
echo   3. Open browser: http://127.0.0.1:5000
echo.
pause
