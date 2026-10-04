@echo off
rem ==========================================================================
rem  Voxprint AI Movie Dubber - one-time setup of the portable bundle (Windows 10/11 x64, NVIDIA GPU recommended).
rem  Creates .venv next to this file: Python 3.11 (downloaded by uv, nothing is installed system-wide),
rem  PyTorch with the CUDA build matching your driver (uv --torch-backend=auto) and the other packages.
rem  Needs internet (about 3-4 GB the first time) and ~8 GB of free disk space.  Safe to run again (it just updates).
rem  Then:  run.bat  (the program)   or   diagnose.bat  (the diagnostics -> a report file on your Desktop)
rem ==========================================================================
setlocal enableextensions
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1

rem --- uv (a single exe by Astral; taken from its GitHub release, no scripts are executed)
set UV=
where uv >nul 2>&1 && set UV=uv
if not defined UV if exist "tools\uv\uv.exe" set UV=tools\uv\uv.exe
if not defined UV (
    echo [1/4] Downloading uv ^(package manager^)...
    if not exist "tools\uv" mkdir "tools\uv"
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -UseBasicParsing 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' -OutFile 'tools\uv\uv.zip'; Expand-Archive -Force 'tools\uv\uv.zip' 'tools\uv'; Remove-Item 'tools\uv\uv.zip'"
    if exist "tools\uv\uv.exe" set UV=tools\uv\uv.exe
)
if not defined UV (echo [ERROR] Could not get uv. Check the internet connection, or install uv manually: https://docs.astral.sh/uv/ & exit /b 1)

echo [2/4] Python 3.11 + virtual environment (.venv)...
if not exist ".venv\Scripts\python.exe" (
    %UV% venv --python 3.11 --seed .venv || (echo [ERROR] venv creation failed & exit /b 1)
)
set VPY=%~dp0.venv\Scripts\python.exe

echo [3/4] PyTorch (CUDA build chosen for your driver)...
if not defined VOX_TORCH_BACKEND set VOX_TORCH_BACKEND=auto
%UV% pip install --python "%VPY%" torch torchaudio --torch-backend=%VOX_TORCH_BACKEND%
if errorlevel 1 (
    echo [i] automatic selection failed - trying the CUDA 12.8 index
    %UV% pip install --python "%VPY%" torch torchaudio --index-url https://download.pytorch.org/whl/cu128 || (echo [ERROR] PyTorch install failed & exit /b 1)
)

echo [4/4] Other packages...
%UV% pip install --python "%VPY%" -r requirements.txt || (echo [ERROR] package install failed & exit /b 1)

echo.
echo === Self-test ===
"%VPY%" main.py --selftest
if errorlevel 1 (echo [WARN] the self-test reported a problem - run diagnose.bat and send the report) else (echo OK.)
echo.
echo Done. Start the program with run.bat; for the diagnostics run diagnose.bat (the report is saved to your Desktop).
endlocal
