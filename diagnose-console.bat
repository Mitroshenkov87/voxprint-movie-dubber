@echo off
rem Diagnostics in the console window (no GUI) - the fallback if the window does not open.  Report -> Desktop.
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (echo Run setup.bat first. & pause & exit /b 1)
".venv\Scripts\python.exe" main.py --diagnose-cli %*
echo.
pause
