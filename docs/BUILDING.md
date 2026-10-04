# Building

## Run from source
Python 3.11 is required; Windows x64 for the installer.
```
python -m venv .venv && .venv\Scripts\activate          (Linux: source .venv/bin/activate)
pip install uv
uv pip install torch torchaudio --torch-backend=auto
uv pip install -r tools/requirements-dev.txt
python -m pytest tests                                   (no GPU needed)
python main.py
```
Never run `uv run` without `--no-sync`: it can replace the CUDA build of PyTorch with the CPU one.

## Online installer
`installer/VoxprintMovieDubber.iss` (Inno Setup 6) compiles into `installer\Output\VoxprintMovieDubber-Setup.exe` (about 2 MB):
```
ISCC installer\VoxprintMovieDubber.iss
```
The installer contains the program's own files only. During setup `installer/install-runtime.ps1` downloads `uv`, a private Python 3.11,
PyTorch (the build matching the NVIDIA driver) and the dependencies from `requirements.txt` into `<install folder>\runtime`, and an optional
final step downloads the AI models (`main.py --fetch-models`). Nothing is installed system-wide.

Setup switches: `/TORCH=auto|cpu|cu128` (PyTorch build), `/TASKS=""` (skip the model download), `/VERYSILENT /SUPPRESSMSGBOXES /DIR=<folder>`.
The setup adds a Start menu folder (program and "Run diagnostics"), an Apps & features entry and a full uninstaller; the uninstaller
asks whether to delete the downloaded models and logs in `%LOCALAPPDATA%\VoxprintMovieDubber`.

## GitHub Actions
`.github/workflows/build-installer.yml` (manual run or a `v*` tag) builds the installer on a free `windows-latest` runner, installs it
silently with the CPU build of PyTorch, runs the installed program and the tests, uninstalls it and uploads the installer as an artifact.
A `v*` tag also attaches it to a pre-release.
