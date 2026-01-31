@echo off
setlocal
title HAVEN - Git Push
chcp 65001 >nul

echo ===================================================
echo HAVEN - Hệ thống Đẩy Code (GitHub)
echo ===================================================
echo.

:: Di chuyển ra thư mục gốc project
cd /d "%~dp0.."



echo [1/4] Đang thêm các thay đổi vào staging area...
git add .
echo Loại trừ thư mục .agent và deployment...
git reset .agent
git reset deployment

echo.
echo [2/4] Kiểm tra an ninh (Security Check)...
:: Chỉ tìm các file .env "thật", không tính .env.example
:: Lọc bỏ các file có chữ "example" trong tên rồi mới tìm ".env"
git diff --cached --name-only | findstr /i /v "example" | findstr /i ".env" >nul
if %errorlevel% equ 0 (
    echo [NGUY HIỂM] Phát hiện file .env thực sự đang được chuẩn bị đưa lên!
    git diff --cached --name-only | findstr /i /v "example" | findstr /i ".env"
    echo.
    echo Vui lòng chạy lệnh: git reset .env
    echo để gỡ bỏ file nhạy cảm trước khi push.
    pause
    exit /b 1
)
echo  An toàn: Không tìm thấy file thông tin nhạy cảm (.env).

echo.
echo [3/4] Trạng thái hiện tại:
git status -s
echo.

set /p commit_msg="Nhập commit message (Nhấn Enter để dùng 'Update'): "
if "%commit_msg%"=="" set commit_msg=Update

echo.
echo Đang thực hiện commit: "%commit_msg%"...
git commit -m "%commit_msg%"

echo.
echo [4/4] Đang đẩy code lên GitHub (Push)...
git push

echo.
if %errorlevel% equ 0 (
    echo ===================================================
    echo [THÀNH CÔNG] Code đã được đẩy lên GitHub!
    echo ===================================================
) else (
    echo [THẤT BẠI] Không thể đẩy code. Kiểm tra mạng hoặc config git.
)

pause
