# Third-party components

| Component | Use | License |
|---|---|---|
| TIGER (look2hear) code, vendored and trimmed in `dubber/third_party/look2hear` | dialogue / effects / music separation | MIT (see `dubber/third_party/look2hear/LICENSE`) |
| TIGER-DnR weights (`JusperLee/TIGER-DnR`) | downloaded on demand | Apache-2.0 |
| faster-qwen3-tts, qwen-tts-hf | Qwen3-TTS inference, CUDA Graphs | MIT / Apache-2.0 |
| Qwen3-TTS-12Hz Base models (Qwen) | TTS with voice cloning, downloaded on demand | Apache-2.0 |
| faster-whisper, CTranslate2, Whisper models | speech recognition | MIT |
| Silero VAD | voice activity detection | MIT |
| Opus-MT (Helsinki-NLP) | translation | CC-BY-4.0 |
| pyannote speaker-diarization community-1 | speaker diarization (gated, optional) | CC-BY-4.0; needs a Hugging Face token and accepted terms |
| PySide6 / Qt | user interface | LGPL-3.0 |
| PyTorch, transformers, numpy, scipy, librosa, soundfile | runtime | BSD / Apache-2.0 / ISC |
| ffmpeg (from PATH or imageio-ffmpeg) | audio extraction and muxing, called as a separate program | LGPL/GPL, depending on the build |
| espeak-ng (only to generate the synthetic test clip, `tools/make_test_clip.py`) | not shipped | GPL-3.0 (the generated audio is synthetic) |

Licenses of the project itself: not decided yet (see `docs/BUILDING.md`, "Open decisions").
