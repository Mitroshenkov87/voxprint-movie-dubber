# Built-in diagnostics

The window has a **Run diagnostics** button (Start menu: *Voxprint AI Movie Dubber -> Run diagnostics*, or `--diagnose`).
It runs every check in a separate process, so a failure or crash of one check never stops the report, and writes
`Voxprint-MovieDubber-Diagnostics-<date>.txt` to the Desktop (a copy goes to the app's data folder). The report is plain English text.

It covers: GPU name, compute capability, driver branch, VRAM, RAM and CUDA; FlashAttention and CUDA Graphs availability and whether they really work; model load times

The GPU check fails when there is no NVIDIA GPU, when compute capability is below 8.9 (RTX 40 / Ada or newer), or when the driver branch is below 600. A separate check fails when CTranslate2 sees no CUDA device (it needs the CUDA 12 libraries from nvidia-cublas-cu12 and nvidia-cudnn-cu12).
(VAD, separation, speech recognition, diarization, translation, TTS); TTS real-time factor on short phrases in four modes
(SDPA, FlashAttention-2, CUDA Graphs, both); timings of each stage on a short bundled test clip; ffmpeg; errors with tracebacks.
Tokens and the Windows user name are masked in the report.

Command line: `--diagnose` (window, starts at once), `--diagnose-cli` (console), `--quick`, `--no-download`, `--out <file>`.
