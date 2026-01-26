@echo off
setlocal

echo ========================================
echo HAVEN - Smart Surveillance System
echo ========================================
echo.

cd /d %~dp0

:: Kiem tra file .env
if not exist ".env" (
    echo [ERROR] Khong tim thay file .env!
    if exist ".env.example" (
        echo Dang tao .env tu template...
        copy .env.example .env >nul
        echo [WARN] VUI LONG CHINH SUA .env VOI THONG TIN CAMERA CUA BAN!
        start notepad .env
        pause
    ) else (
        echo [ERROR] Khong tim thay .env.example. Vui long pull code moi nhat.
        pause
        exit /b 1
    )
)

:: Kich hoat moi truong ao
echo [1/2] Dang kich hoat Virtual Environment...
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo [ERROR] Khong tim thay moi truong ao!
    echo Vui long chay: python -m venv .venv
    echo Sau do: pip install -r requirements.txt
    pause
    exit /b 1
)

:: Khoi dong Backend AI
echo.
echo [2/2] Dang khoi dong HAVEN Backend (YOLO AI)...
echo.
echo    Giao dien:   http://localhost:8000
echo    WebSocket:   ws://localhost:8000/ws/stream
echo    API Docs:    http://localhost:8000/docs
echo.
echo ========================================
python -m backend.src.main

if errorlevel 1 (
    echo.
    echo [ERROR] Server gap loi. Kiem tra log phia tren.
    pause
)