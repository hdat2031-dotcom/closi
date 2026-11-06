@echo off
chcp 65001 >nul
echo ========================================================
echo   DANG DAY BAI LAB DAY 1 LEN GITHUB: hdat2031-dotcom/closi
echo ========================================================
echo.

cd /d "%~dp0"
"C:\Program Files\Git\cmd\git.exe" branch -M main
"C:\Program Files\Git\cmd\git.exe" push -u origin main

if %ERRORLEVEL% equ 0 (
    echo.
    echo ========================================================
    echo   [THANH CONG] Da day bao cao len GitHub thanh cong!
    echo   Link nộp LMS: https://github.com/hdat2031-dotcom/closi
    echo ========================================================
) else (
    echo.
    echo [THONG BAO] Neu GitHub yeu cau dang nhap, ban hay dang nhap tren trinh duyet nhe!
)
pause

