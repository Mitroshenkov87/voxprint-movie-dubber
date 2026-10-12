# Shared resources

Every Voxprint program installs the heavy pieces once, into one versioned folder, and records who uses them.
Movie Dubber does this during setup, checks it again on every launch and with `selftest`, and removes a piece
on uninstall only when no program still references it.

Linux uses the same layout under `~/.local/share/voxprint/shared/` (`$XDG_DATA_HOME/voxprint/shared` when that
variable is set). Windows uses `%LOCALAPPDATA%\Voxprint\shared\`.

## Layout

```
shared/
  manifest.json
  manifest.lock
  runtimes/py3.14-torch2.11-cu130/     Python 3.14, torch 2.11.0+cu130, torchaudio 2.11.0, torchcodec 0.17.0
    python/                            uv-managed CPython
    env/                               virtual environment (the program folder's runtime junction points here)
    runtime-key.json
    .install-complete
  runtimes/py3.14-torch2.11-cpu/       CI only. A user install does not create this folder.
  ffmpeg/n8.1/                         LGPL ffmpeg and ffprobe (BtbN FFmpeg-Builds, release 8.1)
    ffmpeg or ffmpeg.exe
    ffprobe or ffprobe.exe
```

Model weights live at `<home>/models` (`%LOCALAPPDATA%\Voxprint\models`), not inside `shared/`.
`VOXPRINT_MODELS_DIR`, `suite.json` `models_dir`, and the older `state/models_dir.txt` still override that
folder. The models directory itself is never a component. VOXPrint MDL records one ref per file, and
that path stays under the models root (or `VOXPRINT_MODEL_CACHE`). Other component paths stay relative to `shared/`.

The pin hash is still written into `runtime-key.json`. The directory name is the version, so a different Python
or torch line is a different directory. A broken copy of the same version is repaired in place. It is never
upgraded in place to another line.

An older `runtime-<12 hex digits>` directory, and ffmpeg that still sits in the program's `bin` or `tools`
folder, is moved into this layout on install and on the next launch. After that move the old path is removed
and is not searched again. `%LOCALAPPDATA%\Voxprint\runtime` belongs to Voxprint AI Audiobook Builder and is
not moved.

## manifest.json (schema 1)

The file format is [formats/SHARED-RUNTIME.md](formats/SHARED-RUNTIME.md). `voxprint_common.runtime_manifest`
reads and writes it. `dubber/infra/components.py` chooses this program's ids, turns the older list-shaped
file into the object the library can migrate, and deletes a directory when no component still points at it.
The app id in `refs` is `movie-dubber`.

```json
{
  "schema": 1,
  "components": {
    "python-3.14": {
      "kind": "runtime",
      "version": "py3.14-torch2.11-cu130",
      "path": "runtimes/py3.14-torch2.11-cu130",
      "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "installed_at": "2026-10-11T12:00:00",
      "refs": ["movie-dubber"]
    },
    "torch-2.11.0-cu130": {
      "kind": "wheels",
      "version": "2.11.0+cu130",
      "path": "runtimes/py3.14-torch2.11-cu130",
      "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "installed_at": "2026-10-11T12:00:00",
      "refs": ["movie-dubber"]
    },
    "ctranslate2-cu12": {
      "kind": "wheels",
      "version": "cu12",
      "path": "runtimes/py3.14-torch2.11-cu130",
      "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "installed_at": "2026-10-11T12:00:00",
      "refs": ["movie-dubber"]
    },
    "ffmpeg-8": {
      "kind": "tool",
      "version": "n8.1",
      "path": "ffmpeg/n8.1",
      "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "installed_at": "2026-10-11T12:00:00",
      "refs": ["movie-dubber", "audiobook-builder"]
    }
  }
}
```

A component id matches `[a-z0-9][a-z0-9._-]*`. The same id, version, and sha256 is reused. A different
version or digest is written only when no other app still references the component. Writes take
`manifest.lock` for at most 30 seconds and publish through a temporary file, fsync, and rename.
Paths are relative to `shared/`, except a `model` component, whose path is absolute under the shared models directory. `faster-whisper` pulls in the CPU package `onnxruntime`. The installer uninstalls that distribution and installs `onnxruntime-gpu==1.30.0` (the version in `runtime_lock.json`). The CPU distribution must not remain.

`torch` version `2.11.0+cu130` stands for torch 2.11.0, torchaudio 2.11.0 and torchcodec 0.17.0, all `+cu130`.
`ctranslate2` version `cu12` means CTranslate2 can import and the CUDA 12 libraries are `nvidia-cublas-cu12==12.9.2.10`
and `nvidia-cudnn-cu12==9.27.0.42`. The CPU runtime only requires CTranslate2 to import. Its component id is
`python-3.14-cpu` and its torch id is `torch-2.11.0-cpu`.

### Reference counting

* Install adds `movie-dubber` to each component. An entry with the same id, version and checksum is reused
  and is not downloaded again.
* Uninstall removes `movie-dubber`. When `refs` becomes empty, the component is deleted. The directory is
  deleted only when no remaining component still uses that path. A relative path must stay inside `shared/`.
  An absolute model path must stay under the shared models directory. The models directory itself is not
  removed while another program's weights are inside it. Movie Dubber deletes a model folder only when it
  recorded that folder in `<models>/.installed.json` and it is the only program in `<models>/.users.json`.
  `.users.json` and `manifest.json` are updated under a file lock. Unknown keys and other programs' entries stay.
  A checksum failure re-downloads the bad files. It does not remove a folder another program shares.
* A directory that this process is running from, or that cannot be deleted yet, is left on disk. The Windows
  uninstaller deletes those pending paths after the process exits.
* When no components remain, `manifest.json` is deleted. `manifest.lock` may stay.

The installers run:

```
python components.py add --home <voxprint home> --id ffmpeg-8 --kind tool --version n8.1 --path <dir> --app movie-dubber --sha256 <hex>
python components.py release --home <voxprint home> --app movie-dubber
```

## Install

The Windows setup (online and full) and `installer/linux/install.sh` install anything that is missing before
the first launch: Python 3.14, the torch wheels, the CUDA 12 libraries, ffmpeg, and the models the default
pipeline needs. Progress is numbered. A download that stops can be continued: the partial file is kept and
the next run sends an HTTP Range request.

`/TORCH=cpu`, `/SKIPGPUGATE=1` and `/SKIPMODELS=1` (Linux: `--skip-gpu-check`, `--skip-models`) are CI-only.
A user install always uses the cu130 runtime and always downloads the models.

An NVIDIA driver older than branch 600, or a GPU that is not RTX 40-series or newer (compute capability 8.9),
stops setup. The message links to <https://www.nvidia.com/Download/index.aspx>.

## selftest

Every window launch runs the same check. `python main.py selftest` and `python main.py --selftest` do too.
A missing or broken piece is repaired. Progress lines on the command line are JSON objects on stderr:

```json
{"progress": true, "resource": "ffmpeg", "status": "repairing", "message": "Repairing ffmpeg"}
```

`status` is `checking`, `repairing`, `ready` or `failed`. The window shows a dialog only while a piece is
being repaired or has failed.

The result is one JSON object on stdout when `--json` is set:

```json
{"ok": true, "exit_code": 0, "command": "selftest", "outcome": "repair", "repaired": ["ffmpeg"], "resources": [{"id": "runtime", "version": "py3.14-torch2.11-cu130", "path": "...", "ok": true}]}
```

`outcome` is `ok` (nothing to do), `repair` (something was restored and the second check passed), `gpu`,
`models` or `failed`. Exit codes are the table in [CLI.md](CLI.md): `gpu` is 3, `models` is 4, any other
repair failure is 6. A successful repair is exit 0.

A driver older than branch 600, or a GPU below RTX 40-series, exits 3 with the driver download link and
does not download anything.
