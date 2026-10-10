# Architecture

Voxprint AI Movie Dubber is a Windows desktop program. It adds one dubbed audio track to a movie and leaves the picture and the original audio untouched. The window is Qt (PySide6). The heavy steps run in worker processes so a model crash does not take the window with it. Everything in this overview is local: there is no account and no server.

## Pipeline

A dub is a project folder. Each stage reads and writes files in that folder, so a stopped dub resumes at the first stage whose inputs changed. The order is:

1. **Probe** the file (duration, tracks, subtitle streams).
2. **Extract** the chosen audio track.
3. **Subtitles.** Use a track in the file, a file next to it, or a download. With none of those, the later speech step is the source of the lines.
4. **VAD.** Find the speech regions.
5. **Separation.** Split dialogue from music and effects (TIGER-DnR, or Mel-Band RoFormer when the card can run it). Songs stay in the original language.
6. **ASR.** Recognise the dialogue (faster-whisper).
7. **Script.** Turn the recognition into timed lines.
8. **Diarization.** Assign lines to speakers when the film uses more than one voice. Without a Hugging Face token the simpler clustering path is used.
9. **Translation.** Marian / Opus-MT into English, Russian, or German.
10. **Voices.** Pick a cloned voice or a library voice for each speaker.
11. **TTS, with time-fit.** Speak the lines in film order, in blocks of about a minute. A line that does not fit its slot is shortened and spoken again once, then placed, sped up, or moved into a pause.
12. **Mix** the new dialogue with the music and effects, and with as much of the original dialogue as the user left in.
13. **Mux** the new audio onto a copy of the file. The video is not re-encoded.

Model stages of one dub share a worker process and keep a model loaded while it fits the video-memory budget. The window never imports PyTorch.

## Module map

| Path | Role |
| --- | --- |
| `main.py` | Process entry: window, headless commands, worker, installer helpers. |
| `dubber/cli.py` | Headless commands and the argparse definition the command-line reference is generated from. |
| `dubber/core` | Project, subtitles, timing, mixing, voices, and the `.vxdub` file. No Qt and no model imports. |
| `dubber/engines` | One engine per model stage, each with a CPU stand-in used by the tests. |
| `dubber/pipeline` | The stage functions and the runner (cache, worker, GPU lock, progress, preview). |
| `dubber/workers` | One heavy step per process. `main.py --worker NAME ARGS.json`. |
| `dubber/infra` | Shared folders, model store, GPU lock, VRAM tiers, thermal pauses. |
| `dubber/diag` | The diagnostics report. |
| `dubber/ui` | The window, in the Voxprint glass theme. |
| `dubber/third_party` | Vendored Look2Hear (TIGER). Upstream code, not part of the API reference. |

## GPU lock

Before a GPU job the program creates `voxprint-gpu.lock` in the temporary directory (`VOXPRINT_GPU_LOCK` overrides the path). The file names the owner (`movie-dubber`), the job, when it started, and when it expects to finish. Another Voxprint program that finds the lock waits. A lock is stale only when its expected finish is more than two hours ago and nvidia-smi shows no Python or Voxprint process using GPU memory. A lock this program itself left behind after a crash is taken over.

## VRAM tiers and the thermal rule

The card is measured at install and at every start. There is no processor-only mode.

* **16 GB tier** for cards below 24 GB (a card that reports about 22 GB or more is the 24 GB tier).
* **24 GB tier** for cards with 24 GB or more.

A tier is a cap, not a different pipeline: how many lines one speech call may take (6 on the 16 GB tier, 12 on the 24 GB tier) and which models are offered. A model is offered only when it fits with headroom left free. The headroom is `max(2 GB, 8% of the card)`, measured again before each speech batch. The budget can also be capped with `VOXPRINT_VRAM_FRACTION`. Nothing here is a user mode; Settings can only pick a tier the card supports.

The thermal rule is also automatic. The job runs at full speed for the first 2.5 hours, counted from the start of the whole dub. After that it pauses between blocks while the five-minute median GPU temperature is 83 C or higher, or while the card keeps throttling for more than 60 seconds, and it resumes at 75 C. A missing sensor never pauses the job.

## Model store

Weights live in one shared Voxprint models folder (the Audiobook Builder uses the same folder). Each model is a plain directory, not the Hugging Face cache. A download lands in a `.partial` directory and is renamed only after the size and SHA-256 checks in `model_manifest.json`. The installer records this program in `<models>/.users.json` and the uninstaller removes it; the folder is deleted only when no other Voxprint program remains and the user agrees.

## `.vxdub`

A `.vxdub` file is a ZIP of one dub without the video. The first member is an uncompressed `mimetype` whose bytes are `application/vnd.voxprint.dub+zip`. The other members are JSON (manifest, transcript, translation, voices, job) plus optional voice-reference clips. The format is specified in [formats/VXDUB.md](formats/VXDUB.md).

## Command line

`python main.py` with no arguments opens the window. `python main.py --dry-run --json` prints one JSON object and does not load or download a model. The commands, flags, and exit codes are generated from the parsers in `dubber/cli` and written to [CLI.md](CLI.md).
