"""Speech recognition with faster-whisper on the separated speech stem (word timestamps)."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional


def transcribe_faster_whisper(wav16: str, language: Optional[str], repo: str, device: str, allow_download: bool,
                              log: Callable[[str], None]) -> Dict[str, Any]:
    from faster_whisper import WhisperModel

    from dubber import models
    from dubber.core import audio

    folder, _ = models.ensure(repo, allow_download, log=log)
    model = None
    if device == "cuda":
        try:
            model = WhisperModel(str(folder), device="cuda", compute_type="float16")
        except Exception as exc:  # noqa: BLE001 - missing cuBLAS/cuDNN DLLs: CPU still works
            log(f"ASR on GPU failed ({type(exc).__name__}); using the CPU")
    if model is None:
        model = WhisperModel(str(folder), device="cpu", compute_type="int8", cpu_threads=0)
    x, _ = audio.read(wav16, 16000)
    segs, info = model.transcribe(x, beam_size=5, word_timestamps=True, vad_filter=True, language=language or None,
                                  condition_on_previous_text=False)
    out = []
    for s in segs:
        out.append({"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip(),
                    "words": [{"w": w.word, "start": round(w.start, 3), "end": round(w.end, 3)} for w in (s.words or [])]})
        if len(out) % 50 == 0:
            log(f"ASR: {s.end:.0f} s recognised")
    return {"language": info.language, "segments": out}
