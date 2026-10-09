"""Speech recognition with faster-whisper on the separated speech stem (word timestamps).

Punctuation matters: lines are rebuilt from sentence ends (:mod:`dubber.core.segment`).  Without context Whisper sometimes writes
a lower-case stream without any punctuation (real case: a 93 s dialogue); a short punctuated prompt in the film's language and
``condition_on_previous_text`` keep the punctuation for the whole film.  The language is detected first so the prompt is in
the film's language, and the temperature fallback stops at 0.4 (higher temperatures produced stray tokens such as "Wellсем"
or "Choi443buz").  Measured on both user clips (CPU int8): every sentence punctuated, no stray tokens, no lost sentence, and
not slower than the old settings.  If the text still comes back unpunctuated, one more pass with a different prompt is tried
and the better-punctuated result is kept.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

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


def decode_options(language: Optional[str], retry: bool = False) -> Dict[str, Any]:
    opts: Dict[str, Any] = dict(beam_size=5, word_timestamps=True, vad_filter=True, language=language or None,
                                condition_on_previous_text=True, temperature=(0.0, 0.2, 0.4))
    prompt = (RETRY_PROMPTS if retry else PROMPTS).get(language or "") or (PROMPTS.get(language or "") if retry else None)
    if prompt:
        opts["initial_prompt"] = prompt
    return opts


def _detect(model, x, log) -> Optional[str]:
    try:
        lang, prob, _ = model.detect_language(x)
        log(f"ASR: language {lang} ({prob:.2f})")
        return lang
    except Exception:  # noqa: BLE001 - older faster-whisper: let transcribe() detect it
        return None


def transcribe_faster_whisper(wav16: str, language: Optional[str], repo: str, device: str, allow_download: bool,
                              log: Callable[[str], None]) -> Dict[str, Any]:
    from dubber.infra import cuda_dlls

    if device == "cuda":
        cuda_dlls.expose()                   # cublas64_12.dll from torch\lib (cu128) before CTranslate2 needs it
    from faster_whisper import WhisperModel

    from dubber import models
    from dubber.core import audio, segment

    folder, _ = models.ensure(repo, allow_download, log=log)
    x, _ = audio.read(wav16, 16000)

    def run(model) -> Dict[str, Any]:
        lang = language or _detect(model, x, log)
        best: Optional[Dict[str, Any]] = None
        for retry in (False, True):
            segs, info = model.transcribe(x, **decode_options(lang, retry))
            out = []
            for s in segs:                       # the generator does the work: a missing CUDA DLL only fails here
                out.append({"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip(),
                            "words": [{"w": w.word, "start": round(w.start, 3), "end": round(w.end, 3)} for w in (s.words or [])]})
                if len(out) % 50 == 0:
                    log(f"ASR: {s.end:.0f} s recognised")
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

    if device == "cuda":
        try:
            return run(WhisperModel(str(folder), device="cuda", compute_type="float16"))
        except Exception as exc:  # noqa: BLE001 - missing cuBLAS/cuDNN DLLs (e.g. cublas64_12.dll): the CPU still works
            if "out of memory" in str(exc).lower():
                raise
            log(f"ASR on GPU failed ({type(exc).__name__}: {str(exc)[:160]}); using the CPU")
    return run(WhisperModel(str(folder), device="cpu", compute_type="int8", cpu_threads=0))
