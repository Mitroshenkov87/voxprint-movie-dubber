# Building

## Run from source
Python 3.14 is required; Windows x64 for the installer.
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
runtime: Python 3.14 and PyTorch 2.11.0+cu130 pinned by `dubber/infra/runtime_lock.json` (the Audiobook Builder lock schema;
the sibling main lock is still 3.11/cu128 and is not copied). There is no torchaudio wheel for torch 2.14.1 on the cu130
index, so the pin follows the newest official pair: torch 2.11.0+cu130 and torchaudio 2.11.0+cu130. torchcodec is 0.17.0+cu130.
`nvidia-cublas-cu12` and `nvidia-cudnn-cu12` supply the CUDA 12 libraries CTranslate2 needs. The dependencies come from
`requirements.txt` and the pins in `installer/runtime-constraints.txt`. The runtime lives in `%LOCALAPPDATA%\Voxprint\runtime-<key>` (key = hash of all pins, see
`dubber/infra/runtime.py`): an existing folder with the same key is reused, a different set of versions gets its own folder side by
side, and a shared runtime is never upgraded in place. `<install folder>\runtime` is a junction to it. `runtime-<key>\.users.json`
counts the programs using it. An optional final step downloads the AI models (`main.py --fetch-models`).

Setup switches: `/TORCH=auto|cu130` (PyTorch flavor). `/TORCH=cpu` installs the CPU build and is only for the CI smoke run on a GPU-less runner; a user install stops instead of offering that build. `/TASKS=""` skips the model download. `/VERYSILENT /SUPPRESSMSGBOXES /DIR=<folder>`. cu130 needs NVIDIA driver branch 580 or newer; the hardware gate already requires branch 600.

The setup stops, and installs nothing, when no NVIDIA GPU has compute capability 8.9 or higher (GeForce RTX 40 / Ada or newer) or the driver branch is below 600. The window does the same check with PyTorch before it opens (`VOXPRINT_SKIP_GPU_GATE=1` skips that check for tests and CI).
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
