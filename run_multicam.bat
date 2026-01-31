@echo off
REM HAVEN Multi-Camera ReID System Runner
REM Double-click  chy!

echo ============================================================
echo HAVEN Multi-Camera Person Re-Identification System
echo ============================================================
echo.

REM Activate virtual environment if exists
if exist .venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo Warning: Virtual environment not found at .venv\
    echo Running with system Python...
)

echo.
echo Starting multi-camera ReID system...
echo.

REM Run Python script
python backend\src\run_multicam_reid.py --config configs\multicam.yaml

echo.
echo ============================================================
echo System stopped
echo ============================================================
pause

