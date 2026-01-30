@echo off
REM ============================================================
REM HAVEN Multi-Camera Sequential Processing
REM ============================================================
REM Features: Pose + ADL + ReID (Color Histogram)
REM Videos: Cam1 -> Cam2 -> Cam3 -> Cam4
REM ============================================================

cd /d %~dp0

echo ============================================================
echo HAVEN Multi-Camera Sequential Processing
echo ============================================================
echo.

REM Check venv
if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
    echo.
)

REM Run
echo Starting...
echo Features: Pose + ADL + ReID
echo Cameras: Cam1 -^> Cam2 -^> Cam3 -^> Cam4
echo.
python backend\multi\run.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Script failed!
    echo.
)

echo.
echo Done!
pause
