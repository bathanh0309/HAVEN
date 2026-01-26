@echo off
chcp 65001 >nul
setlocal

echo ========================================
echo HAVEN - Pose + ADL Combined Test
echo ========================================
echo.

cd /d %~dp0

:: Check .env
if not exist ".env" (
    echo [ERROR] .env file not found!
    pause
    exit /b 1
)

:: Check video
if not exist "data\video\walking.mp4" (
    echo [ERROR] Video not found: data\video\walking.mp4
    pause
    exit /b 1
)

:: Activate venv
echo [1/2] Activating Virtual Environment...
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo [ERROR] Virtual environment not found!
    pause
    exit /b 1
)

:: Run test
echo.
echo [2/2] Running Pose + ADL Test...
echo.
echo    Controls:
echo    - Q: Quit
echo    - Space: Pause/Resume
echo    - L: Toggle Loop mode
echo.
echo ========================================

python backend\test_pose_adl.py

if errorlevel 1 (
    echo.
    echo [ERROR] Script crashed. Check logs above.
    pause
)
