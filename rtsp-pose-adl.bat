@echo off
chcp 65001 >nul
title HAVEN - RTSP Pose ADL Test

echo ============================================================
echo    HAVEN - RTSP Camera Pose + ADL Test
echo ============================================================
echo.

cd /d "%~dp0"

REM Activate virtual environment if exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

echo [INFO] Starting RTSP Pose + ADL Test...
echo.

cd backend\test-rtsp-pose
python pose-adl.py

echo.
pause
