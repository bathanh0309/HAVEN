@echo off
echo ========================================
echo HAVEN Camera Stream Server (FastAPI)
echo ========================================
echo.

cd /d %~dp0

echo [1/3] Activating virtual environment...
call .venv\Scripts\activate.bat

echo [2/3] Installing dependencies...
pip install fastapi uvicorn[standard] opencv-python websockets -q

echo [3/3] Starting FastAPI server...
echo.
echo Backend: http://localhost:8000
echo Video:   http://localhost:8000/video_feed
echo WS:      ws://localhost:8000/ws/stream
echo Frontend: Open frontend\index.html
echo.
echo Press Ctrl+C to stop
echo.

python backend\stream_server.py

pause
