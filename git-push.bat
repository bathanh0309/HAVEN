@echo off
echo ================================
echo  Git Push to GitHub
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

REM Add all changes
echo Adding all changes...
git add .

REM Request commit message
set /p commit_msg="Enter commit message: "

REM Commit changes
echo Committing changes...
git commit -m "%commit_msg%"

REM Check if main branch exists, if not create it
git show-ref --verify --quiet refs/heads/main
if errorlevel 1 (
    echo Creating main branch...
    git branch -M main
)

REM Push to GitHub
echo Pushing to https://github.com/bathanh0309/HAVEN...
git push -u origin main

echo.
echo ================================
echo  Push Complete!
echo ================================
pause
