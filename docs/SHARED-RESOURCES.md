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
  models/                              the model store. VOXPRINT_MODELS_DIR, suite.json and models_dir.txt still override it
```

The pin hash is still written into `runtime-key.json`. The directory name is the version, so a different Python
or torch line is a different directory. A broken copy of the same version is repaired in place. It is never
upgraded in place to another line.

An older `runtime-<12 hex digits>` directory, and ffmpeg that still sits in the program's `bin` or `tools`
folder, is moved into this layout on install and on the next launch. After that move the old path is removed
and is not searched again. `%LOCALAPPDATA%\Voxprint\runtime` belongs to Voxprint AI Audiobook Builder and is
not moved.

## manifest.json (schema 1)

Other suite programs implement this file the same way. Unknown keys are copied through unchanged.
The write is atomic (a temporary file, then replace). Writers take `manifest.lock` (an OS file lock) and,
inside one process, a thread lock.

```json
{
  "schema": 1,
  "resources": [
    {
      "id": "runtime",
      "version": "py3.14-torch2.11-cu130",
      "path": "C:\\Users\\me\\AppData\\Local\\Voxprint\\shared\\runtimes\\py3.14-torch2.11-cu130",
      "sha256": null,
      "size": null,
      "apps": ["movie-dubber"]
    },
    {
      "id": "torch",
      "version": "2.11.0+cu130",
      "path": "C:\\Users\\me\\AppData\\Local\\Voxprint\\shared\\runtimes\\py3.14-torch2.11-cu130",
      "sha256": null,
      "size": null,
      "apps": ["movie-dubber"]
    },
    {
      "id": "ctranslate2",
      "version": "cu12",
      "path": "C:\\Users\\me\\AppData\\Local\\Voxprint\\shared\\runtimes\\py3.14-torch2.11-cu130",
      "sha256": null,
      "size": null,
      "apps": ["movie-dubber"]
    },
    {
      "id": "ffmpeg",
      "version": "n8.1",
      "path": "C:\\Users\\me\\AppData\\Local\\Voxprint\\shared\\ffmpeg\\n8.1",
      "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "size": 80000000,
      "apps": ["movie-dubber", "audiobook-builder"]
    },
    {
      "id": "models",
      "version": "store",
      "path": "C:\\Users\\me\\AppData\\Local\\Voxprint\\shared\\models",
      "sha256": null,
      "size": null,
      "apps": ["movie-dubber"]
    }
  ]
}
```

| Field | Meaning |
| --- | --- |
| `schema` | Always `1` for this document. |
| `resources` | List of resource objects. A list keeps unknown per-resource keys intact. |
| `id` | `runtime`, `torch`, `ctranslate2`, `ffmpeg`, `models`, or an id another app adds. |
| `version` | The version string above. Same id and version are one resource. |
| `path` | Absolute directory. Models follow `VOXPRINT_MODELS_DIR` when that variable is set. |
| `sha256` | Hex digest of the file when the resource is one file (ffmpeg). `null` when it does not apply. An empty digest does not erase a digest already stored. |
| `size` | Size in bytes when it applies, otherwise `null`. |
| `apps` | App ids that hold a reference. This program's id is `movie-dubber`. |

`torch` version `2.11.0+cu130` stands for torch 2.11.0, torchaudio 2.11.0 and torchcodec 0.17.0, all `+cu130`.
`ctranslate2` version `cu12` means CTranslate2 can import and the CUDA 12 libraries are `nvidia-cublas-cu12==12.9.2.10`
and `nvidia-cudnn-cu12==9.27.0.42`. The CPU runtime only requires CTranslate2 to import.

### Reference counting

* Install adds `movie-dubber` to each resource. An entry with the same id, version and checksum, whose path
  exists, is reused and is not downloaded again.
* Uninstall removes `movie-dubber`. When `apps` becomes empty, the directory is deleted if it is inside
  `shared/`. A path outside `shared/` (a models folder chosen with `VOXPRINT_MODELS_DIR`) is dropped from
  the manifest and is not deleted.
* A directory that this process is running from, or that cannot be deleted yet, is left on disk. The Windows
  uninstaller deletes those pending paths after the process exits. They are still removed only because the
  reference list was already empty.
* When no resources remain, `manifest.json` is deleted. `manifest.lock` may stay.

The stdlib script `dubber/infra/shared_manifest.py` is what the installers run:

```
python shared_manifest.py add --home <voxprint home> --id ffmpeg --version n8.1 --path <dir> --app movie-dubber --sha256 <hex> --size <bytes>
python shared_manifest.py release --home <voxprint home> --app movie-dubber
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
