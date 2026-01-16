@echo off
echo ================================
echo  Git Pull from GitHub
echo ================================
echo.

REM Navigate to project directory
cd /d "%~dp0"

REM Configure Git identity (if not already set)
echo Configuring Git identity...
git config --global user.name "bathanh0309"
git config --global user.email "bathanh1234asd@gmail.com"

REM Check if we're in a git repository
git rev-parse --git-dir >nul 2>&1
if errorlevel 1 (
    echo Initializing Git repository...
    git init
    git remote add origin https://github.com/bathanh0309/HAVEN.git
)

REM Pull latest changes from GitHub
echo Pulling latest changes from https://github.com/bathanh0309/HAVEN...
git pull origin main

echo.
echo ================================
echo  Pull Complete!
echo ================================
pause
