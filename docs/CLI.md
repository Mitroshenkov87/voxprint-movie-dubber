# Command line

Headless entry points for other Voxprint programs, scripts and agents. The same process as `python main.py`. The window is not opened. Nothing waits for a prompt.

```
python main.py version
python main.py diagnose
python main.py fetch-models [--models KEY,KEY]
python main.py run-project PATH [--languages SRC,TGT] [--stages NAME,NAME]
python main.py info PATH
python main.py --dry-run --json
```

`python -m dubber.cli` takes the same arguments. On an installed copy the program is `python main.py` inside the installation folder.

## Global flags

| Flag | Meaning |
| --- | --- |
| `--json` | Print one JSON object on stdout and exit. Logs go to stderr. |
| `--dry-run` | Discovery only. Implies `--json`. No model is loaded and nothing is downloaded. |
| `--help`, `-h` | Print this command list and the exit codes, then exit 0. With `--json` or `--dry-run`, that list is one JSON object. |

`--dry-run` works on a machine with no NVIDIA GPU. It reports that GPU in the JSON and exits with code 3. It does not crash and it does not open a window. `VOXPRINT_SKIP_GPU_GATE` does not change `--dry-run` (that variable only lets the window and a real diagnose or dub start on a test runner).

The JSON object always has `ok`, `exit_code`, `command`, `dry_run`, `version`, `label`, `build` and `codename`. `label` is the same line as About: `1.0.0 RC · build N "Bochan"`. A failure also has `error`.

## Exit codes

| Code | Name | When |
| --- | --- | --- |
| 0 | ok | The command finished. |
| 2 | usage | Unknown command or flag, or a missing value, language, stage, or model key. |
| 3 | gpu | No supported NVIDIA GPU (compute capability 8.9 or newer, RTX 40-series or newer). |
| 4 | models | A model this command needs is not on disk. Dry-run does not download it. |
| 5 | input | The video, project folder, or `.vxdub` file is missing or cannot be read. |
| 6 | job | The dubbing job failed. |
| 7 | cancelled | The job was cancelled. |

Code 1 is not used. When several of these apply, usage is reported first, then a missing or unreadable input, then an unsupported GPU, then missing models, then a job failure, then cancellation.

## version

```
python main.py version
python main.py --version
python main.py version --json
```

Prints the version line (build and codename included). Exit 0. With `--dry-run`, the same discovery object as below is added, and an unsupported GPU exits 3.

## diagnose

```
python main.py diagnose --json
python main.py --diagnose-cli --quick --no-download --json
python main.py diagnose --dry-run --json
```

Runs the same checks as `--diagnose-cli`. `--dry-run` lists the check groups and does not run them. Flags accepted by the diagnostics still work: `--out`, `--quick`, `--no-download`, `--target`, `--skip`, `--tts-model`, `--tts-modes`, `--cpu-tts`, `--cpu-sep`, `--clip`, `--asr-repo`.

A real run on a machine below the GPU minimum exits 3 unless `VOXPRINT_SKIP_GPU_GATE` is set (the installer's own smoke test). `--dry-run` still exits 3 on that machine.

## fetch-models

```
python main.py fetch-models --json
python main.py --fetch-models --models sep
python main.py fetch-models --models sep --dry-run --json
```

Downloads the models the installer would fetch, or the keys passed to `--models`. An unknown key is exit 2. A download that does not finish is exit 4. `--dry-run` reports which of those keys are already on disk and does not download. The real download does not refuse to start because of the GPU: the installer uses it on a CPU test runner.

## run-project

```
python main.py run-project MOVIE.mkv --languages en,ru --json
python main.py run-project PROJECT.vxdub --target-lang de
python main.py run-project PROJECT_FOLDER --stages mux --dry-run --json
python main.py --run-project PROJECT_FOLDER --stages asr,translation
```

`PATH` is a video (`.mkv`, `.mp4`, `.avi`, `.mov`, `.webm`, `.m4v`, `.ts`, `.m2ts`), a `.vxdub` project, or a project folder that already contains `project.json`.

Languages: `--languages SRC,TGT`, or `--source-lang` and `--target-lang`. Source may be `auto`, `en`, `ru` or `de`. Target may be `en`, `ru` or `de`. The default is `auto` to `ru`. Values already stored in a project or a `.vxdub` are kept when the flags are omitted.

`--stages` names pipeline stages. The job runs from the start through the last named stage. Unknown names are exit 2.

`--dry-run` describes the file, the languages, the stages and the models that would be needed. It does not create a project folder, load a model, or download. A missing file is exit 5. An unsupported GPU is exit 3. Missing models, once the GPU is acceptable, are exit 4.

A real run exits 3 on an unsupported GPU (unless `VOXPRINT_SKIP_GPU_GATE` is set), 6 when the dub fails, and 7 when it is cancelled.

## info

```
python main.py info PATH --json
python main.py project-info PATH --json
python main.py --project-info PATH
```

Reads a project folder or a `.vxdub` file and prints its kind, languages and line count. A video that is not yet a project is reported as `video`. A missing file is exit 5. This command does not need a GPU. `--dry-run` adds the discovery object and then uses the same GPU exit code as the other dry-runs.

## --dry-run

```
python main.py --dry-run --json
```

With no other command, this is discovery only:

- `label`, `build`, `codename` from `BUILD.json` (and from `build_info.json` when the installer stamped one)
- `runtime`: Python version, platform, and the installed PyTorch version (`null` when PyTorch is not installed)
- `gpu`: whether the card meets the minimum, the reason when it does not (`no_cuda` or `low_compute`), VRAM, and the VRAM tier (`16gb` or `24gb`; `16gb` is also the tier when the size is unknown)
- `models`: every known model key and whether its files are already on disk
- `would_run.stages`: the dub stages that would run if a project were given

No model is loaded. Nothing is downloaded. On a GPU-less Linux runner the process exits 3 and the JSON `gpu.ok` field is false.
