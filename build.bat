@echo off
rem ==========================================================================
rem  Voxprint AI Movie Dubber - developer build on Windows 10/11 x64 (same flow as the Audiobook Builder's build.bat).
rem  UNTESTED on Windows so far (the skeleton was written on Linux) - see docs\BUILDING.md.
rem
rem  build.bat            -> tests + portable bundle zip   (dist\Voxprint-MovieDubber-<ver>-portable.zip)   [the reliable path]
rem  build.bat exe        -> additionally PyInstaller --onedir (dist\VoxprintMovieDubber\)  and the Inno Setup installer
rem                          (installer\Output\VoxprintMovieDubber-Setup.exe) if Inno Setup 6 is installed.
rem  Requires Python 3.11 (py launcher).  VOX_SKIP_TESTS=1 skips pytest.
rem  Rules carried over from the Audiobook Builder: PyTorch only via "uv pip install torch --torch-backend=auto";
rem  never "uv run" without --no-sync (it would swap CUDA torch for the CPU build).
rem ==========================================================================
setlocal enableextensions
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1

if not exist ".venv\Scripts\python.exe" (
    py -3.11 -m venv .venv || (echo [ERROR] Python 3.11 not found ^(py -3.11^). & exit /b 1)
)
call ".venv\Scripts\activate.bat"
python -m pip install -U pip wheel uv || exit /b 1

echo === PyTorch (CUDA, automatic selection) ===
if not defined VOX_TORCH_BACKEND set VOX_TORCH_BACKEND=auto
uv pip install torch torchaudio --torch-backend=%VOX_TORCH_BACKEND% || exit /b 1

echo === Dependencies ===
uv pip install -r requirements-dev.txt || exit /b 1

echo === Tests ===
if not defined VOX_SKIP_TESTS ( python -m pytest -q || (echo [ERROR] Tests failed. & exit /b 1) )

echo === Portable bundle ===
python tools\make_portable_zip.py || exit /b 1

if /I not "%~1"=="exe" goto :finish

echo === PyInstaller (onedir) ===
pyinstaller --onedir --windowed --noconfirm --clean --name VoxprintMovieDubber ^
  --paths . ^
  --add-data "dubber\locales;dubber\locales" --add-data "assets;assets" ^
  --hidden-import dubber.workers.w_gpu --hidden-import dubber.workers.w_vad --hidden-import dubber.workers.w_sep ^
  --hidden-import dubber.workers.w_asr --hidden-import dubber.workers.w_diar --hidden-import dubber.workers.w_mt ^
  --hidden-import dubber.workers.w_tts --hidden-import dubber.workers.w_fetch ^
  --hidden-import dubber.third_party.look2hear.models.tiger_dnr ^
  --collect-all qwen_tts --collect-all faster_qwen3_tts --collect-all faster_whisper --collect-all silero_vad ^
  --collect-all imageio_ffmpeg --collect-all sentencepiece --collect-all certifi --collect-data librosa ^
  --copy-metadata transformers --copy-metadata tokenizers --copy-metadata huggingface_hub --copy-metadata safetensors ^
  --copy-metadata accelerate --copy-metadata torch --copy-metadata numpy --copy-metadata tqdm --copy-metadata regex ^
  --copy-metadata requests --copy-metadata packaging --copy-metadata filelock --copy-metadata qwen-tts-hf --copy-metadata faster-qwen3-tts ^
  --exclude-module gradio --exclude-module tkinter --exclude-module matplotlib ^
  main.py
if errorlevel 1 (echo [ERROR] PyInstaller failed. & exit /b 1)
rem Workers are started as "VoxprintMovieDubber.exe --worker NAME args.json" (see dubber\diag\procs.py), so the one exe serves both roles.

set ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe
if not exist "%ISCC%" set ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe
if exist "%ISCC%" (
    "%ISCC%" installer\VoxprintMovieDubber.iss
    echo Installer: installer\Output\VoxprintMovieDubber-Setup.exe
) else (
    echo [i] Inno Setup 6 not found - installer not built. The folder dist\VoxprintMovieDubber can be zipped instead.
)
:finish
endlocal
