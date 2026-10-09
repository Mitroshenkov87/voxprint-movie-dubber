"""Worker ``asr``: faster-whisper (CTranslate2) speech recognition with word timestamps and language detection.

Input (args): ``wav16`` (preferably the dialogue stem), ``repo`` (default large-v3-turbo CT2), ``expected_text`` (for the WER check),
``out_json``, ``allow_download``, ``device``.  GPU first (float16); if CUDA cannot be used (e.g. cuBLAS/cuDNN DLLs missing) it
falls back to CPU int8 and says so (status WARN) - that fallback is exactly the kind of Windows problem the report should reveal.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List

from dubber import models
from dubber.workers.common import WorkerContext, read_wav_mono, wer


def run(args: Dict[str, Any], ctx: WorkerContext) -> Dict[str, Any]:
    from dubber.infra import cuda_dlls

    cuda_dirs = cuda_dlls.expose() if args.get("device", "auto") != "cpu" else []
    t = time.time()
    from faster_whisper import WhisperModel

    import_s = time.time() - t
    repo = args.get("repo") or models.SPECS["asr"].repo
    ctx.log("ASR: locating the model")
    folder, info = models.ensure(repo, args.get("allow_download", True), log=ctx.log)
    details: List[str] = [f"{'model':<28}: {repo} ({info['source']}" + (f", downloaded in {info['download_s']} s)" if info["download_s"] else ")"),
                          f"{'import faster_whisper':<28}: {import_s:.2f} s"]
    if cuda_dirs:
        from pathlib import Path

        details.append(f"{'CUDA 12 libraries (cuBLAS)':<28}: {'found' if cuda_dlls.has_cublas12([Path(d) for d in cuda_dirs]) else 'NOT found'} in {cuda_dirs[0]}")
    want_gpu = args.get("device", "auto") != "cpu"
    gpu_expected = bool(args.get("expect_gpu", True))
    model, device, ctype, fallback_err = None, "cpu", "int8", ""
    t = time.time()
    if want_gpu:
        try:
            import ctranslate2

            if ctranslate2.get_cuda_device_count() > 0:
                model = WhisperModel(str(folder), device="cuda", compute_type="float16")
                device, ctype = "cuda", "float16"
                # force the CUDA libraries to really load now (a missing cuBLAS/cuDNN DLL fails on the first run, not at construction)
                audio_probe, _ = read_wav_mono(args["wav16"], 16000)
                list(model.transcribe(audio_probe[:16000], beam_size=1, language="en")[0])
            else:
                fallback_err = "ctranslate2 sees no CUDA device"
        except Exception as exc:  # noqa: BLE001
            fallback_err = f"{type(exc).__name__}: {str(exc)[:240]}"
            model = None
    if model is None:
        model = WhisperModel(str(folder), device="cpu", compute_type="int8", cpu_threads=0)
        device, ctype = "cpu", "int8"
    load_s = time.time() - t
    details.append(f"{'device / compute type':<28}: {device} / {ctype}")
    if fallback_err and want_gpu:
        details.append(f"{'GPU not used because':<28}: {fallback_err}" + ("" if gpu_expected else "  (expected: no NVIDIA GPU here)"))
    details.append(f"{'model load':<28}: {load_s:.2f} s")

    audio, sr = read_wav_mono(args["wav16"], 16000)
    dur = len(audio) / sr
    ctx.log("ASR: transcribing")

    def transcribe():
        segs_iter, info_ = model.transcribe(audio, beam_size=int(args.get("beam_size", 5)), word_timestamps=True,
                                            vad_filter=False, language=args.get("language") or None,
                                            condition_on_previous_text=False)
        return list(segs_iter), info_

    t = time.time()
    segs, tinfo = transcribe()
    first_s = time.time() - t
    run_s = first_s
    if device == "cuda" or first_s < 20:          # second pass = steady-state speed (skipped for a slow CPU run)
        t = time.time()
        segs, tinfo = transcribe()
        run_s = time.time() - t
    out = [{"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip(),
            "words": [{"w": w.word, "start": round(w.start, 3), "end": round(w.end, 3)} for w in (s.words or [])]} for s in segs]
    text = " ".join(s["text"] for s in out)
    if args.get("out_json"):
        with open(args["out_json"], "w", encoding="utf-8") as fh:
            json.dump({"language": tinfo.language, "segments": out}, fh, ensure_ascii=False)
    details.append(f"{'processing':<28}: {first_s:.2f} s first call, {run_s:.2f} s second call for {dur:.1f} s of audio (x{dur / max(run_s, 1e-6):.1f} real time)")
    details.append(f"{'language detected':<28}: {tinfo.language} (probability {tinfo.language_probability:.2f})")
    details.append(f"{'segments':<28}: {len(out)}")
    for s in out:
        details.append(f"  {s['start']:6.2f}-{s['end']:6.2f}  {s['text']}")
    problem = bool(fallback_err and want_gpu and gpu_expected)
    status = "WARN" if problem else "OK"
    note = ""
    if args.get("expected_text"):
        w = wer(args["expected_text"], text)
        details.append(f"{'WER vs known text':<28}: {w * 100:.1f} %")
        note = f"; WER {w * 100:.0f}%"
        if w > 0.4:
            status = "WARN"
    summary = f"{len(out)} segments, lang={tinfo.language}, {device}/{ctype}, x{dur / max(run_s, 1e-6):.1f} real time{note}"
    if problem:
        summary += " (GPU NOT USED)"
    return {"status": status, "summary": summary, "details": details,
            "metrics": {"load_s": round(load_s, 3), "run_s": round(run_s, 3), "device": device, "download_s": info["download_s"],
                        "gpu_fallback_reason": fallback_err}}
