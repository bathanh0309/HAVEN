@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
echo Environment activated! (type 'deactivate' to exit)
cmd /k
