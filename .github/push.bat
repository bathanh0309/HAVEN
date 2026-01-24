@echo off
setlocal
title HAVEN - Git Push

echo ===================================================
echo HAVEN Security Push
echo ===================================================
echo.
echo Reviewing files to be committed...
echo.

:: Check for .env files in staging area just in case
git diff --cached --name-only | findstr /i ".env"
if %errorlevel% equ 0 (
    echo [CRITICAL ERROR] A .env file is staged for commit!
    echo Please run 'git reset' and check your .gitignore.
    pause
    exit /b 1
)

:: Show status
git status
echo.
echo.
set /p commit_msg="Enter commit message (or press Enter for 'Update'): "
if "%commit_msg%"=="" set commit_msg=Update

echo.
echo Committing with message: "%commit_msg%"
echo.

git add .
git commit -m "%commit_msg%"
git push

echo.
if %errorlevel% equ 0 (
    echo [SUCCESS] Code pushed successfully.
) else (
    echo [ERROR] Push failed. Check your network or git configuration.
)
pause
