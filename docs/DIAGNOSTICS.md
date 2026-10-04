# Built-in diagnostics

The window has a **Run diagnostics** button (Start menu: *Voxprint AI Movie Dubber -> Run diagnostics*, or `--diagnose`).
It runs every check in a separate process, so a failure or crash of one check never stops the report, and writes
`Voxprint-MovieDubber-Diagnostics-<date>.txt` to the Desktop (a copy goes to the app's data folder). The report is plain English text.

It covers: GPU, VRAM, RAM, driver and CUDA; FlashAttention and CUDA Graphs availability and whether they really work; model load times
(VAD, separation, speech recognition, diarization, translation, TTS); TTS real-time factor on short phrases in four modes
(SDPA, FlashAttention-2, CUDA Graphs, both); timings of each stage on a short bundled test clip; ffmpeg; errors with tracebacks.
Tokens and the Windows user name are masked in the report.

Command line: `--diagnose` (window, starts at once), `--diagnose-cli` (console), `--quick`, `--no-download`, `--out <file>`.
