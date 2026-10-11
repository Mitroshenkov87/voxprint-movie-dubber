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
`requirements.txt` and the pins in `installer/runtime-constraints.txt`. The runtime lives in
`%LOCALAPPDATA%\Voxprint\shared\runtimes\py3.14-torch2.11-cu130` (see [SHARED-RESOURCES.md](SHARED-RESOURCES.md)). A folder
for that version is reused. A different Python or torch line gets its own directory, and a shared runtime is never upgraded
in place. `<install folder>\runtime` is a junction to its `env`. `shared\manifest.json` counts the programs using each
resource. The last setup step downloads the AI models (`main.py --fetch-models`) into `shared\models`. `/SKIPMODELS=1`
skips that download and is only for CI.

## Full (offline) installer
The full installer carries everything the online one downloads, so setup needs no internet (only the optional AI model download does):
```
powershell -File installer\make-full-payload.ps1 -Out build\payload -Requirements requirements.txt `
    -Constraints installer\runtime-constraints.txt -Lock dubber\infra\runtime_lock.json -Python <python 3.14> -Flavor cu130
ISCC /DFull installer\VoxprintMovieDubber.iss
```
`make-full-payload.ps1` puts into `build\payload`: `uv.exe`, the python-build-standalone archive this uv installs for Python 3.14,
every wheel of the runtime (`pip wheel` with the same pins as the online setup, including torch 2.11.0+cu130, torchaudio and
torchcodec) and the LGPL ffmpeg archive. The result is `VoxprintMovieDubber-Full-Setup.exe` plus `VoxprintMovieDubber-Full-Setup-N.bin`
parts, each below 2 GB (the GitHub release asset limit); keep all of them in one folder. During setup `install-runtime.ps1 -Payload`
installs the same runtime key as the online setup from those files, so both installers share one runtime folder.
The AI models are not bundled; setup downloads them into the shared models folder. `/SKIPMODELS=1` skips that download and is only for CI.

Setup switches: `/TORCH=auto|cu130` (PyTorch flavor). `/TORCH=cpu` installs the CPU build and is only for the CI smoke run on a GPU-less runner; a user install stops instead of offering that build. `/SKIPGPUGATE=1` (full installer only) installs its cu130 runtime without the GPU check and is likewise only for CI. `/SKIPMODELS=1` skips the model download and is only for CI. `/VERYSILENT /SUPPRESSMSGBOXES /DIR=<folder>`. cu130 needs NVIDIA driver branch 580 or newer; the hardware gate already requires branch 600.

The setup stops, and installs nothing, when no NVIDIA GPU has compute capability 8.9 or higher (GeForce RTX 40 / Ada or newer) or the driver branch is below 600. The window does the same check with PyTorch before it opens (`VOXPRINT_SKIP_GPU_GATE=1` skips that check for tests and CI).
The setup puts its shortcuts into the Start menu folder "Voxprint" (shared with the Audiobook Builder), adds an Apps & features entry
and a full uninstaller. The uninstaller never touches the Audiobook Builder. It drops this program's references in
`shared\manifest.json` and deletes a shared runtime, ffmpeg build, or models folder only when no program still references it.
Projects, logs and reports are removed only when you answer Yes (default: keep).

Shared settings: `%LOCALAPPDATA%\Voxprint\state\suite.json` (schema 1: `ui_language`, `theme`, `models_dir`, `gpu`), read and written by
both programs (`dubber/infra/suite.py`).

## Version and codename

`BUILD.json` at the repository root is the only copy of the build offset and the codename. `dubber/appinfo.py` reads it. `APP_VERSION` in that module is the version string (`1.0.0-rc`). The build number is not stored as a constant.

On the Build installers workflow the build number is `GITHUB_RUN_NUMBER` plus `offset`. The workflow writes that number and the codename into `build_info.json`. The installed program reads the stamp, so it does not need the git checkout. A local run, with no stamp and no `GITHUB_RUN_NUMBER`, uses the offset alone. `VOXPRINT_BUILD=dev` (or a missing `BUILD.json` and no stamp) shows the marker `dev`. `VOXPRINT_BUILD=<digits>` forces that number.

The offset was chosen from the Build installers workflow: run_number 17 on 2026-10-10 (workflow run 38057799983). The next run is 18, and 18 + 983 = 1001.

The window, the splash, About, `--version`, the installer and the GitHub release title all show `1.0.0 RC · build N "Bochan"` from that calculation. A codename is one Biblical Hebrew word in ASCII transliteration. It describes the state of that build (readiness, a milestone), not a theme of the product. It is shown only when it is not empty, in quotes. The codename of this line is Bochan.

Headless commands for other programs are documented in [CLI.md](CLI.md).

## GitHub Actions
`.github/workflows/build-installer.yml` (manual run or a `v*` tag) builds both installers on free `windows-latest` runners (a matrix
of `online` and `full`), installs each silently (online with the CPU build of PyTorch, full with its own cu130 runtime and no
downloads), runs the installed program and the tests, uninstalls it and uploads the installers as artifacts. A `v*` tag then
publishes both on one pre-release; the release job checks every `.sha256` file and appends a table of file names, sizes and
SHA-256 to the release notes (`docs/release-notes/<tag>.md`).
