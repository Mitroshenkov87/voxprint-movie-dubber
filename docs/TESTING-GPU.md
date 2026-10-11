# GPU tests on a virtual machine

`scripts/vm_test.sh` (Linux) and `scripts/vm_test.ps1` (Windows) run the dubber without a window.
They are for a machine that already has the shared runtime, the models, and an NVIDIA GPU.
They do not install those pieces.

```
scripts/vm_test.sh --smoke --json ./clips
scripts/vm_test.sh --full --json ./clips
powershell -File scripts\vm_test.ps1 -Smoke -Json -Clips .\clips
powershell -File scripts\vm_test.ps1 -Full -Json -Clips .\clips
```

The clips folder defaults to `./clips`. Any `.mp4`, `.mkv`, `.webm`, or `.mov` under that folder is a clip.
`--smoke` is the short run, about ten minutes when the models are already on disk:

1. `python main.py --dry-run --json`
2. `python main.py selftest --json`
3. `python main.py fetch-models --dry-run --json` (a check; it does not download)
4. one dub of the smallest clip

`--full` dubs every clip, then dubs the smallest clip again as a one-voice actor blend
(`single_voice.kind` = `actor`, `actor_weight` 0.5), then dubs that clip with `multi_voice`
turned on. Multi-voice does not need a Hugging Face token.

`--json` prints one object. Each case has a name, an exit code, and the wall-clock seconds.
`peak_vram_mib` is the highest `memory.used` nvidia-smi reported during a dub, sampled once
a second. `tts_backend` is the `TTS backend:` line from the dub log. CUDA Graphs is the
expected backend (`graphs`). An actor blend, or a machine where Graphs cannot start, must
log why: a `TTS backends skipped:` line, which includes the CUDA Graphs reason. A dub that
names neither the backend nor a reason fails the run.

Exit codes:

| Code | Meaning |
| --- | --- |
| 0 | Every case passed, and the TTS log names CUDA Graphs or says why not. |
| 1 | A case failed, or the TTS log is silent about the backend. |
| 2 | Python is missing, or the clips folder has no video. |
| 3 | `nvidia-smi` is missing or reports no GPU. |

`PYTHON` selects the interpreter. The default is `python3` on Linux and `python` on Windows.
