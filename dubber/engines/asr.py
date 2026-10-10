"""Speech recognition with faster-whisper on the separated speech stem (word timestamps).

Punctuation matters: lines are rebuilt from sentence ends (:mod:`dubber.core.segment`).  Without context Whisper sometimes writes
a lower-case stream without any punctuation (real case: a 93 s dialogue); a short punctuated prompt in the film's language and
``condition_on_previous_text`` keep the punctuation for the whole film.  The language is detected first so the prompt is in
the film's language, and the temperature fallback stops at 0.4 (higher temperatures produced stray tokens such as "Wellсем"
or "Choi443buz").  Measured on both user clips (CPU int8): every sentence punctuated, no stray tokens, no lost sentence, and
not slower than the old settings.  If the text still comes back unpunctuated, one more pass with a different prompt is tried
and the better-punctuated result is kept.

Holes: with ``condition_on_previous_text`` Whisper can jump over a stretch of speech (real case, clip 2 without separation:
nothing between 9.3 s and 24.2 s).  Speech found by the VAD but not covered by any recognised word for 2 s or more is a hole;
then one pass without the previous-text condition is made and its words inside the holes are taken.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

#: short, punctuated sample sentences per language (the style Whisper copies: casing, commas, ?, !, ...)
PROMPTS = {
    "en": "Hello, everyone. Well, I don't know... Yes! What do you think? It's fine, thanks.",
    "ru": "Привет всем. Ну, я не знаю... Да! Что ты думаешь? Всё хорошо, спасибо.",
    "de": "Hallo zusammen. Na ja, ich weiß nicht... Ja! Was denkst du? Alles gut, danke.",
    "fr": "Bonjour à tous. Eh bien, je ne sais pas... Oui ! Qu'en penses-tu ? Ça va, merci.",
    "es": "Hola a todos. Bueno, no lo sé... ¡Sí! ¿Qué piensas? Está bien, gracias.",
    "it": "Ciao a tutti. Beh, non lo so... Sì! Cosa ne pensi? Va bene, grazie.",
}
RETRY_PROMPTS = {
    "en": "\"Wait, what?\" \"No, listen to me.\" \"Okay, fine. Let's go!\"",
}

_FALLBACK_TEXT = (
    "Speech recognition would run on the CPU even though an NVIDIA GPU is available. "
    "faster-whisper / CTranslate2 could not use CUDA, so the dub stopped instead of falling back silently. "
    "Install the CUDA build (cuBLAS 12) and run it again."
)


class AsrCudaFallback(RuntimeError):
    """faster-whisper / CTranslate2 would run on the CPU while an NVIDIA GPU is available."""


def cuda_gpu_present() -> bool:
    """True when this machine has an NVIDIA GPU, even if this process's PyTorch build is CPU-only."""
    try:
        import torch

        if torch.cuda.is_available():
            return True
    except Exception:  # noqa: BLE001 - torch missing: ask nvidia-smi
        pass
    try:
        from dubber.infra.resources import gpu_from_smi

        found = gpu_from_smi()
        return bool(found and found[1] > 0)
    except Exception:  # noqa: BLE001
        return False


def ensure_cuda_asr(requested: str, resolved: str) -> None:
    """Stop when recognition would use the CPU while a CUDA GPU is present.

    An explicit ``cpu`` request is kept (the user asked for it).  A resolved CUDA device is kept too.
    """
    if requested == "cpu" or resolved == "cuda":
        return
    if cuda_gpu_present():
        raise AsrCudaFallback(_FALLBACK_TEXT)


def is_cuda_fallback_failure(message: str) -> bool:
    """True when a pipeline error is the hard stop above (the window shows a warning for it)."""
    text = message or ""
    return "AsrCudaFallback" in text or "would run on the CPU even though" in text


def decode_options(language: Optional[str], retry: bool = False) -> Dict[str, Any]:
    """Build one faster-whisper pass: word timestamps, temperatures from 0.0 to 0.4, and a punctuated prompt.

    Args:
        retry: Use the alternate prompt. The first prompt for ``language`` is kept when no alternate exists.
    """
    opts: Dict[str, Any] = dict(beam_size=5, word_timestamps=True, vad_filter=True, language=language or None,
                                condition_on_previous_text=True, temperature=(0.0, 0.2, 0.4))
    prompt = (RETRY_PROMPTS if retry else PROMPTS).get(language or "") or (PROMPTS.get(language or "") if retry else None)
    if prompt:
        opts["initial_prompt"] = prompt
    return opts


HOLE_MIN_S = 2.0
_GRID = 0.05


def find_holes(speech: Sequence[Tuple[float, float]], segments: Sequence[Dict[str, Any]], min_s: float = HOLE_MIN_S,
               pad: float = 0.3, bridge: float = 1.0) -> List[Tuple[float, float]]:
    """Stretches of VAD speech (seconds) that no recognised word covers (words count ``pad`` seconds wider)."""
    if not speech:
        return []
    n = int(max(b for _, b in speech) / _GRID) + 2
    mask = [False] * n
    for a, b in speech:
        for i in range(max(0, int(a / _GRID)), min(n, int(b / _GRID) + 1)):
            mask[i] = True
    for s in segments:
        ws = s.get("words") or [{"start": s["start"], "end": s["end"]}]
        for w in ws:
            for i in range(max(0, int((w["start"] - pad) / _GRID)), min(n, int((w["end"] + pad) / _GRID) + 1)):
                mask[i] = False
    runs: List[List[float]] = []
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            a, b = i * _GRID, (j + 1) * _GRID
            if runs and a - runs[-1][1] < bridge:
                runs[-1][1] = b
            else:
                runs.append([a, b])
            i = j + 1
        else:
            i += 1
    return [(round(a, 2), round(b, 2)) for a, b in runs if b - a >= min_s]


def fill_holes(segments: List[Dict[str, Any]], extra: Sequence[Dict[str, Any]], holes: Sequence[Tuple[float, float]],
               margin: float = 0.3) -> Tuple[List[Dict[str, Any]], int]:
    """Words of ``extra`` (a second pass) whose middle lies in a hole become new segments; returns (segments, words added)."""
    taken: List[Dict[str, Any]] = []
    for a, b in holes:
        ws = [w for s in extra for w in (s.get("words") or []) if a - margin <= (w["start"] + w["end"]) / 2 <= b + margin]
        if ws:
            taken.append({"start": ws[0]["start"], "end": ws[-1]["end"], "text": "".join(w["w"] for w in ws).strip(),
                          "words": ws, "filled": True})
    if not taken:
        return list(segments), 0
    return sorted(list(segments) + taken, key=lambda s: s["start"]), sum(len(t["words"]) for t in taken)


def _speech_regions(x) -> List[Tuple[float, float]]:
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        return [(t["start"] / 16000, t["end"] / 16000) for t in get_speech_timestamps(x, VadOptions())]
    except Exception:  # noqa: BLE001 - an older faster-whisper without the VAD helpers: no hole check
        return []


def _detect(model, x, log) -> Optional[str]:
    try:
        lang, prob, _ = model.detect_language(x)
        log(f"ASR: language {lang} ({prob:.2f})")
        return lang
    except Exception:  # noqa: BLE001 - older faster-whisper: let transcribe() detect it
        return None


def load_whisper(folder: str, device: str):
    """The faster-whisper model itself (no transcription).  CUDA uses float16; the CPU uses int8."""
    from faster_whisper import WhisperModel

    if device == "cuda":
        return WhisperModel(str(folder), device="cuda", compute_type="float16")
    return WhisperModel(str(folder), device="cpu", compute_type="int8", cpu_threads=0)


def load_whisper_cached(device: str, repo: str, allow_download: bool, log: Callable[[str], None]):
    """Load (or reuse) the ASR model.  The resident cache keeps it for the next stage and the next clip."""
    from dubber import models
    from dubber.infra import resident

    folder, _ = models.ensure(repo, allow_download, log=log)
    resolved = device if device in ("cuda", "cpu") else "cpu"

    def build():
        return load_whisper(str(folder), resolved)

    if resident.enabled():
        return resident.slot("asr", 2.2, build)
    return build()


def transcribe_faster_whisper(wav16: str, language: Optional[str], repo: str, device: str, allow_download: bool,
                              log: Callable[[str], None], requested: Optional[str] = None) -> Dict[str, Any]:
    """Transcribe a 16 kHz WAV, recover speech the first pass skipped, and retry once if punctuation is missing.

    Raises:
        AsrCudaFallback: An NVIDIA GPU is present but recognition would run on the CPU.
    """
    from dubber.infra import cuda_dlls

    if device == "cuda":
        cuda_dlls.expose()                   # nvidia-cublas-cu12 / nvidia-cudnn-cu12 bin dirs before CTranslate2 loads cublas64_12.dll
    from dubber.core import audio, segment

    x, _ = audio.read(wav16, 16000)

    def decode(model, **opts):
        segs, info = model.transcribe(x, **opts)
        out = []
        for s in segs:                       # the generator does the work: a missing CUDA DLL only fails here
            out.append({"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip(),
                        "words": [{"w": w.word, "start": round(w.start, 3), "end": round(w.end, 3)} for w in (s.words or [])]})
            if len(out) % 50 == 0:
                log(f"ASR: {s.end:.0f} s recognised")
        return out, info

    def run(model) -> Dict[str, Any]:
        lang = language or _detect(model, x, log)
        best: Optional[Dict[str, Any]] = None
        for retry in (False, True):
            out, info = decode(model, **decode_options(lang, retry))
            holes = find_holes(_speech_regions(x), out)
            if holes:
                log("ASR: speech without text at " + ", ".join(f"{a:.0f}-{b:.0f} s" for a, b in holes) + "; one more pass for it")
                extra, _ = decode(model, **{**decode_options(lang, retry), "condition_on_previous_text": False})
                out, added = fill_holes(out, extra, holes)
                log(f"ASR: {added} words recovered")
            res = {"language": info.language, "segments": out}
            words = segment.words_of(out)
            ok = segment.punctuated(words)
            if best is None or ok:
                best = res
            if ok or len(words) < 40:
                break
            log("ASR: the text came back without punctuation; one more pass with another prompt")
        assert best is not None
        return best

    asked = device if requested is None else requested
    ensure_cuda_asr(asked, device)
    if device == "cuda":
        try:
            return run(load_whisper_cached("cuda", repo, allow_download, log))
        except AsrCudaFallback:
            raise
        except Exception as exc:  # noqa: BLE001 - missing cuBLAS/cuDNN: do not hide it behind a silent CPU run
            if "out of memory" in str(exc).lower():
                raise
            if cuda_gpu_present():
                raise AsrCudaFallback(_FALLBACK_TEXT) from exc
            log(f"ASR on GPU failed ({type(exc).__name__}: {str(exc)[:160]}); no NVIDIA GPU was found, using the CPU")
    return run(load_whisper_cached("cpu", repo, allow_download, log))
