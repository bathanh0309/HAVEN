@echo off
setlocal

echo ========================================
echo HAVEN - Smart Surveillance System
echo ========================================
echo.

cd /d %~dp0

:: 0. Check for .env file
if exist ".env" goto :env_exists

echo [ERROR] .env file not found!
if not exist ".env.example" goto :no_example

echo Creating .env from template...
copy .env.example .env >nul
echo [INFO] Created .env file.
echo [WARN] PLEASE EDIT .env WITH YOUR CAMERA CREDENTIALS BEFORE RUNNING!
echo.
echo Opening .env for editing...
start notepad .env
echo Press any key to start anyway (or Ctrl+C to stop)...
pause >nul
goto :env_exists

:no_example
echo [ERROR] .env.example not found. Please pull latest code.
pause
exit /b 1

:env_exists

:: 1. Activate Environment
echo [1/2] Activating Virtual Environment...
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo [ERROR] Virtual environment not found! Please run 'python -m venv .venv' first.
    pause
    exit /b 1
)

:: 2. Start Unified Server
echo [2/2] Starting Unified Server (Backend + Frontend)...
echo.
echo    Backend API: http://localhost:8000
echo    Dashboard:   http://localhost:8000
echo.

:: Run using module syntax to fix import errors
python -m backend.src.main

if errorlevel 1 (
    echo.
    echo [ERROR] Server crashed. Check error logs above.
    pause
)