@echo off
rem Starts Voxprint AI Movie Dubber from the portable bundle (run setup.bat once before).
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (echo Run setup.bat first. & pause & exit /b 1)
start "" ".venv\Scripts\pythonw.exe" main.py %*
