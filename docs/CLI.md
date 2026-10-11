# Command line

Headless entry points for other Voxprint programs, scripts and agents. The same process as `python main.py`. The window is not opened. Nothing waits for a prompt.

`python -m dubber.cli` takes the same arguments. On an installed copy the program is `python main.py` inside the installation folder.

The JSON object always has `ok`, `exit_code`, `command`, `dry_run`, `version`, `label`, `build` and `codename`. `label` is the same line as About. A failure also has `error`. A failed job also has `debug` with the traceback.

```
python main.py version
python main.py diagnose
python main.py fetch-models [--models KEY,KEY]
python main.py run-project PATH [--languages SRC,TGT] [--stages NAME,NAME]
python main.py info PATH
python main.py --dry-run --json
```

## Global flags

| Flag | Meaning |
| --- | --- |
| `--json` | Print one JSON object on stdout and exit. Logs go to stderr. |
| `--dry-run` | Discovery only. Implies --json. No model is loaded and nothing is downloaded. Works with no NVIDIA GPU: the JSON reports the GPU and the process exits 3. VOXPRINT_SKIP_GPU_GATE does not change --dry-run. |
| `--help`, `-h` | Print this command list and the exit codes, then exit 0. With --json or --dry-run, that list is one JSON object. |

## Flag forms

| Flag | Meaning |
| --- | --- |
| `--version` | Same as the version command. |
| `--diagnose-cli` | Same as the diagnose command. |
| `--fetch-models` | Same as the fetch-models command. |
| `--run-project PATH` | Same as run-project PATH. |
| `--project-info PATH` | Same as info PATH. |

## Diagnostic flags

| Flag | Meaning |
| --- | --- |
| `--out FILE` | Write the diagnostics report to this file. |
| `--quick` | Shorter diagnostics: one TTS phrase and the SDPA modes only. |
| `--no-download` | Do not download models during diagnostics. |
| `--target LANG` | Diagnostics dub language: ru, en, or de. Default: ru. |
| `--skip LIST` | Skip these diagnostic groups: system, gpu, network, models, tts, stages. |
| `--tts-model KEY` | TTS model key: tts_1_7b or tts_0_6b. |
| `--tts-modes LIST` | Comma-separated TTS modes. Known: standard_sdpa, standard_fa2, graphs_sdpa, graphs_fa2. |
| `--cpu-tts` | Run the diagnostics TTS check on the CPU. |
| `--cpu-sep` | Run the diagnostics separation check on the CPU. |
| `--clip FILE` | Use this clip instead of the bundled diagnostics clip. |
| `--asr-repo REPO` | Hugging Face repo for the speech-recognition model. |
| `--models KEY,KEY` | Model keys to download. Unknown keys exit 2. |
| `--report FILE` | Write the fetch-models log to this file. The installer reads it after the download. |
| `--languages SRC,TGT` | Source and target, for example en,ru. Source: auto, en, ru, de. Target: en, ru, de. |
| `--source-lang LANG` | Source language. Default: auto. Kept when the project already has one. |
| `--target-lang LANG` | Target language. Default: ru. |
| `--stages NAME,NAME` | Run from the start of the pipeline through the last named stage. Unknown names exit 2. |

## Exit codes

| Code | Name | When |
| --- | --- | --- |
| 0 | ok | The command finished. |
| 2 | usage | Unknown command or flag, or a missing value, language, stage, or model key. |
| 3 | gpu | No supported NVIDIA GPU (compute capability 8.9 or newer, RTX 40-series or newer). |
| 4 | models | A model this command needs is not on disk. Dry-run does not download it. |
| 5 | input | The video, project folder, or .vxdub file is missing or cannot be read. |
| 6 | job | The dubbing job failed. |
| 7 | cancelled | The job was cancelled. |

Code 1 is not used. When several of these apply, usage is reported first, then a missing or unreadable input, then an unsupported GPU, then missing models, then a job failure, then cancellation.

## version

Prints the version line (build and codename included). Exit 0. With --dry-run, the same discovery object as a bare --dry-run is added, and an unsupported GPU exits 3.

```
python main.py version
```

## diagnose

Runs the same checks as --diagnose-cli. --dry-run lists the check groups and does not run them. The diagnostic flags still apply. A real run on a machine below the GPU minimum exits 3 unless VOXPRINT_SKIP_GPU_GATE is set. --dry-run still exits 3 on that machine.

```
python main.py diagnose
```

## fetch-models

Downloads the models the installer would fetch, or the keys passed to --models. An unknown key is exit 2. A download that does not finish is exit 4. --dry-run reports which of those keys are already on disk and does not download. The real download does not refuse to start because of the GPU.

```
python main.py fetch-models
```

## run-project

PATH is a video (.mkv, .mp4, .avi, .mov, .webm, .m4v, .ts, .m2ts), a .vxdub project, or a project folder that already contains project.json. Languages default to auto to ru. Values already stored in a project or a .vxdub file are kept when the flags are omitted. --dry-run describes the file, the languages, the stages and the models that would be needed. It does not create a project folder, load a model, or download. A missing file is exit 5. An unsupported GPU is exit 3. Missing models, once the GPU is acceptable, are exit 4. A real run exits 3 on an unsupported GPU (unless VOXPRINT_SKIP_GPU_GATE is set), 6 when the dub fails, and 7 when it is cancelled.

```
python main.py run-project PATH
```

## info (project-info)

Prints the kind, the languages and the line count. A video that is not yet a project is reported as video. A missing file is exit 5. This command does not need a GPU. --dry-run adds the discovery object and then uses the same GPU exit code as the other dry-runs.

```
python main.py info PATH
python main.py project-info PATH
```

## --dry-run

```
python main.py --dry-run --json
```

With no other command, --dry-run is discovery only:

- `label`, `build`, `codename` from BUILD.json (and from build_info.json when the installer stamped one)
- `runtime`: Python version, platform, and the installed PyTorch version (null when PyTorch is not installed)
- `gpu`: whether the card meets the minimum, the reason when it does not (`no_cuda` or `low_compute`), VRAM, and the VRAM tier (`16gb` or `24gb`; `16gb` is also the tier when the size is unknown)
- `models`: every known model key and whether its files are already on disk
- `would_run.stages`: the dub stages that would run if a project were given

No model is loaded. Nothing is downloaded. On a GPU-less machine the process exits 3 and `gpu.ok` is false.

## Window and installer

Open the window, or run an installer helper. These forms do not use the headless exit codes.

The desktop window returns 1 when the graphics card is below the minimum. That code is not one of the headless codes above.

| Flag | Meaning |
| --- | --- |
| `--diagnose` | Open the window and run the diagnostics at once. The report goes to the Desktop. |
| `--worker NAME ARGS` | Internal. Run one heavy step in its own process. NAME is the worker; ARGS is its JSON file. |
| `--selftest` | Create the window, process events briefly, and exit 0. Smoke test. |
| `--register-models-user` | Installer. Add this program to the shared models folder's .users.json. |
| `--unregister-models-user` | Uninstaller. Remove this program from .users.json. Exit 0 unless the write failed. |
| `--register-runtime-user` | Installer. Add this program to the shared runtime's .users.json. |
| `--unregister-runtime-user` | Uninstaller. Remove this program from the shared runtime's .users.json. |
| `--sync-suite-settings` | Installer. Write the shared suite.json (models folder, UI language) when it has none yet. |
| `--out FILE` | With an unregister command, write '<other users>\n<folder>' to FILE. |

Positional `PROJECT.vxdub`: Open this project file in the window.
