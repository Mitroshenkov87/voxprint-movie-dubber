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
The installer contains the program's own files only. During setup `installer/install-runtime.ps1` downloads `uv` and installs the
runtime: Python 3.11 and PyTorch 2.11.0 pinned by `dubber/infra/runtime_lock.json` (a verbatim copy of the Voxprint AI Audiobook
Builder lock; flavor cu128 / cu126 / cpu chosen from the driver), the dependencies from `requirements.txt` and the pins in
`installer/runtime-constraints.txt`. The runtime lives in `%LOCALAPPDATA%\Voxprint\runtime-<key>` (key = hash of all pins, see
`dubber/infra/runtime.py`): an existing folder with the same key is reused, a different set of versions gets its own folder side by
side, and a shared runtime is never upgraded in place. `<install folder>\runtime` is a junction to it. `runtime-<key>\.users.json`
counts the programs using it. An optional final step downloads the AI models (`main.py --fetch-models`).

Setup switches: `/TORCH=auto|cpu|cu126|cu128` (PyTorch flavor), `/TASKS=""` (skip the model download), `/VERYSILENT /SUPPRESSMSGBOXES /DIR=<folder>`.
The setup puts its shortcuts into the Start menu folder "Voxprint" (shared with the Audiobook Builder), adds an Apps & features entry
and a full uninstaller. The uninstaller never touches the Audiobook Builder; it removes the shared runtime only when no other program
uses it, and the shared models only when no other program uses them and you answer Yes (default: keep).

Shared settings: `%LOCALAPPDATA%\Voxprint\state\suite.json` (schema 1: `ui_language`, `theme`, `models_dir`, `gpu`), read and written by
both programs (`dubber/infra/suite.py`).

## Version and codename

`APP_VERSION`, `APP_BUILD` and `CODENAME` in `dubber/appinfo.py` are the only copy of the release identity. The installer reads that file when it is compiled. The window, the splash, diagnostics and the GitHub release title use `version_label()` and `release_title()`.

A codename is one Biblical Hebrew word in ASCII transliteration. It describes the state of that build (readiness, a milestone), not a theme of the product. It is shown only when it is not empty, in quotes: `1.0.0 RC · build 999 "Hineni"`. Build 999 is Hineni (Genesis 22:1, "here I am": readiness).

## GitHub Actions
`.github/workflows/build-installer.yml` (manual run or a `v*` tag) builds the installer on a free `windows-latest` runner, installs it
silently with the CPU build of PyTorch, runs the installed program and the tests, uninstalls it and uploads the installer as an artifact.
A `v*` tag also attaches it to a pre-release.
