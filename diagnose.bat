@echo off
rem Starts the program and runs the diagnostics at once; the report is saved to the Desktop
rem (Voxprint-MovieDubber-Diagnostics-<date>.txt).  Add --quick for a short run, --no-download to forbid model downloads.
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (echo Run setup.bat first. & pause & exit /b 1)
start "" ".venv\Scripts\pythonw.exe" main.py --diagnose %*
