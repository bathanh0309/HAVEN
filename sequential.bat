@echo off
REM HAVEN Sequential ReID - Full Features
REM ADL + Colorful Pose + Object Detection + Hierarchical ReID
REM cam1 -> cam2 -> cam3 -> cam4

echo ============================================================
echo HAVEN Sequential ReID (Full Features)
echo ============================================================
echo.
echo Features:
echo   - Pose Skeleton (Colorful)
echo   - ADL Detection (SITTING, STANDING, FALL_DOWN)
echo   - Dangerous Objects (knife, bat, racket)
echo   - Hierarchical ReID (cam1=MASTER, cam2-4=SLAVE)
echo.

REM Activate virtual environment if exists
if exist .venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo Warning: Virtual environment not found
)

echo.
echo Processing: cam1 then cam2 then cam3 then cam4
echo.
echo Controls:
echo   SPACE = Pause/Resume
echo   N     = Skip to next camera
echo   Q     = Quit
echo   G     = Record MP4
echo.

REM Run the full-feature runner from backend/multi
python backend\multi\run.py

echo.
echo ============================================================
echo Sequential processing complete!
echo ============================================================
echo.
echo Output: D:\HAVEN\backend\outputs\log.csv
echo.
pause

