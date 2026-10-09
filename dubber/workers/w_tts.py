"""Worker ``tts``: Qwen3-TTS speed and (optionally) synthesis of the clip's lines, in one of four modes.

======================  =========================================  ===============================================
mode                    code path                                  what it measures
======================  =========================================  ===============================================
``standard_sdpa``       ``qwen_tts.Qwen3TTSModel`` + SDPA          the baseline the audiobook program uses (3.7x slower than real time)
``standard_fa2``        same + ``attn_implementation=flash_attention_2``   effect of the flash_attn package
``graphs_sdpa``         ``faster_qwen3_tts.FasterQwen3TTS`` (CUDA Graphs + static KV cache) + SDPA   the candidate fix (research note 03/06)
``graphs_fa2``          same + flash_attention_2                  both speed-ups together
======================  =========================================  ===============================================

Args: ``mode``, ``model_repo``, ``ref_audio``, ``ref_text``, ``language`` ("Russian"), ``phrases`` (list of str, timed),
``runs`` (repeats per phrase), ``lines`` (optional: [{"id","text"}] synthesised into ``out_dir`` as ``line_<id>.wav``),
``allow_download``, ``device`` (auto|cpu).  RTF = generation seconds / audio seconds (< 1 means faster than playback).
Every phase (import, model load, voice prompt, warm-up/graph capture, each run) is timed separately.
"""
from __future__ import annotations

import importlib.util
import os
import statistics
import time
from typing import Any, Dict, List, Tuple

from dubber import models
from dubber.workers.common import (WorkerContext, apply_qwen_tts_compat, cuda_sync, free_gpu, peak_vram_gb, reserved_vram_gb, reset_peak, torch_device)

MODES = {"standard_sdpa": ("standard", "sdpa"), "standard_fa2": ("standard", "flash_attention_2"),
         "graphs_sdpa": ("graphs", "sdpa"), "graphs_fa2": ("graphs", "flash_attention_2")}
FRAMES_PER_SECOND = 12.5


def _gb(x) -> str:
    return "n/a (no CUDA)" if x is None else f"{x} GB"


def max_tokens_for(text: str) -> int:
    """Upper bound of codec frames for ``text`` (same rule as the audiobook program: without it a missing end-of-speech token runs for minutes)."""
    seconds = 3.0 + 0.19 * len(text.strip())
    return max(48, min(2048, int(round(seconds * FRAMES_PER_SECOND))))


def run(args: Dict[str, Any], ctx: WorkerContext) -> Dict[str, Any]:
    mode = args.get("mode", "standard_sdpa")
    if mode not in MODES:
        return {"status": "FAIL", "summary": f"unknown mode {mode}"}
    kind, attn = MODES[mode]
    details: List[str] = []
    metrics: Dict[str, Any] = {"mode": mode}
    if attn == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
        return {"status": "SKIP", "summary": "flash_attn package is not installed (no official Windows wheels) - mode cannot run",
                "metrics": metrics, "details": ["pip package 'flash_attn' is missing, so FlashAttention-2 cannot be used by transformers."]}
    t = time.time()
    import numpy as np
    import torch

    device = torch_device(args.get("device", "auto"))
    use_cuda = device == "cuda"
    if kind == "graphs" and not use_cuda:
        return {"status": "SKIP", "summary": "CUDA Graphs need an NVIDIA GPU with CUDA (none visible to PyTorch)", "metrics": metrics}
    dtype = torch.bfloat16 if use_cuda else torch.float32
    repo = args.get("model_repo") or models.SPECS["tts_1_7b"].repo
    ctx.log(f"TTS {mode}: locating the model")
    folder, info = models.ensure(repo, args.get("allow_download", True), log=ctx.log)
    metrics["download_s"] = info["download_s"]
    details.append(f"{'model':<28}: {repo} ({info['source']}" + (f", downloaded in {info['download_s']} s)" if info["download_s"] else ")"))
    details.append(f"{'device / dtype':<28}: {device} / {str(dtype).replace('torch.', '')}   attention requested: {attn}   path: {kind}")

    reset_peak()
    t = time.time()
    if kind == "graphs":
        from faster_qwen3_tts import FasterQwen3TTS
    else:
        from qwen_tts import Qwen3TTSModel
    shim = apply_qwen_tts_compat()
    import_s = time.time() - t
    details.append(f"{'import':<28}: {import_s:.2f} s   (torch {torch.__version__}, " + _versions() + ")")

    if shim:
        details.append(f"{'compat shim':<28}: {shim}")
    ctx.log(f"TTS {mode}: loading the model")
    t = time.time()
    if kind == "graphs":
        model = FasterQwen3TTS.from_pretrained(str(folder), device="cuda", dtype=dtype, attn_implementation=attn,
                                               max_seq_len=int(args.get("max_seq_len", 2048)))
        inner = model.model
    else:
        kw = dict(dtype=dtype, attn_implementation=attn)
        if use_cuda:
            kw["device_map"] = "cuda:0"
        model = Qwen3TTSModel.from_pretrained(str(folder), **kw)
        if not use_cuda:
            model.model.to("cpu")
        inner = model
    cuda_sync()
    load_s = time.time() - t
    metrics["load_s"] = round(load_s, 2)
    used = _attn_used(inner)
    details.append(f"{'model load':<28}: {load_s:.2f} s   (VRAM allocated {_gb(peak_vram_gb())}; attention implementation reported by the model: {used})")
    metrics["attn_reported"] = used

    ref_audio, ref_text, language = args["ref_audio"], args["ref_text"], args.get("language", "Russian")
    t = time.time()
    prompt = None
    if kind == "standard":
        prompt = inner.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text)
    prompt_s = time.time() - t
    if kind == "standard":
        details.append(f"{'voice prompt (cloning)':<28}: {prompt_s:.2f} s")

    def synth(text: str, model: Any = model, inner: Any = inner, prompt: Any = prompt) -> Tuple[np.ndarray, int]:
        # Defaults bind the loaded objects so a later ``del model`` does not make the nested function look undefined.
        mnt = max_tokens_for(text)
        if kind == "graphs":
            wavs, sr = model.generate_voice_clone(text=text, language=language, ref_audio=ref_audio, ref_text=ref_text, max_new_tokens=mnt)
        else:
            with torch.inference_mode():
                wavs, sr = inner.generate_voice_clone(text=text, language=language, voice_clone_prompt=prompt, max_new_tokens=mnt)
        return np.asarray(wavs[0], dtype=np.float32).reshape(-1), int(sr)

    # ---- warm-up (CUDA graph capture happens here for the graphs path; kernels compile/auto-tune for the standard one)
    warm_text = args.get("warmup_text") or "Hello there."
    ctx.log(f"TTS {mode}: warm-up")
    t = time.time()
    if kind == "graphs":
        model.warmup()
    synth(warm_text)
    cuda_sync()
    warmup_s = time.time() - t
    metrics["warmup_s"] = round(warmup_s, 2)
    details.append(f"{'warm-up (graph capture etc.)':<28}: {warmup_s:.2f} s")

    # ---- timed runs
    rows: List[Tuple[str, int, float, float, float, float]] = []
    gen_total, audio_total = 0.0, 0.0
    for text in args.get("phrases") or []:
        for r in range(int(args.get("runs", 2))):
            ctx.log(f"TTS {mode}: phrase {len(rows) + 1}")
            t = time.time()
            wav, sr = synth(text)
            cuda_sync()
            dt = time.time() - t
            dur = len(wav) / sr
            gen_total += dt
            audio_total += dur
            frames = dur * FRAMES_PER_SECOND
            rows.append((text, r + 1, dt, dur, dt / max(dur, 1e-6), frames / max(dt, 1e-6)))
    if rows:
        details.append("timed runs (text | run | gen s | audio s | RTF | codec frames/s)  [12.5 frames/s = real time]:")
        for text, r, dt, dur, rtf, fps in rows:
            short = (text[:46] + "...") if len(text) > 49 else text
            details.append(f"  {short:<49} #{r}  {dt:6.2f}  {dur:6.2f}  {rtf:6.3f}  {fps:6.1f}")
        rtfs = [x[4] for x in rows]
        metrics.update(rtf=round(gen_total / max(audio_total, 1e-6), 3), rtf_best=round(min(rtfs), 3), rtf_median=round(statistics.median(rtfs), 3),
                       xrt=round(audio_total / max(gen_total, 1e-6), 2), audio_s=round(audio_total, 2), gen_s=round(gen_total, 2))
        details.append(f"{'overall':<28}: {gen_total:.2f} s to make {audio_total:.2f} s of audio -> RTF {metrics['rtf']} "
                       f"(best {metrics['rtf_best']}, median {metrics['rtf_median']}); {metrics['xrt']}x real time")

    # ---- the clip's own lines (pipeline stage)
    made = []
    if args.get("lines"):
        import soundfile as sf

        os.makedirs(args["out_dir"], exist_ok=True)
        t = time.time()
        for ln in args["lines"]:
            wav, sr = synth(ln["text"])
            path = os.path.join(args["out_dir"], f"line_{ln['id']}.wav")
            sf.write(path, wav, sr, subtype="PCM_16")
            made.append({"id": ln["id"], "path": path, "seconds": round(len(wav) / sr, 3), "sr": sr})
        cuda_sync()
        metrics["lines_s"] = round(time.time() - t, 2)
        metrics["run_s"] = metrics["lines_s"]
        details.append(f"{'clip lines synthesised':<28}: {len(made)} lines, {sum(m['seconds'] for m in made):.1f} s of audio in {metrics['lines_s']} s")
        metrics["made"] = made

    metrics["peak_vram_gb"] = peak_vram_gb()
    metrics["reserved_vram_gb"] = reserved_vram_gb()
    metrics["device"] = device
    details.append(f"{'peak VRAM (PyTorch)':<28}: {_gb(metrics['peak_vram_gb'])} allocated, {_gb(metrics['reserved_vram_gb'])} reserved")
    status, summary = "OK", ""
    if rows:
        summary = f"RTF {metrics['rtf']} ({metrics['xrt']}x real time), load {load_s:.1f} s, warm-up {warmup_s:.1f} s"
        if not use_cuda:
            status, summary = "WARN", "CPU run (no GPU): " + summary
        elif metrics["rtf"] > 1.0:
            status, summary = "WARN", "SLOWER than real time: " + summary
    else:
        summary = f"loaded in {load_s:.1f} s, warm-up {warmup_s:.1f} s"
    del model
    free_gpu()
    return {"status": status, "summary": summary, "details": details, "metrics": metrics}


def _versions() -> str:
    from importlib import metadata

    out = []
    for pkg in ("transformers", "qwen-tts-hf", "qwen-tts", "faster-qwen3-tts", "flash-attn"):
        try:
            out.append(f"{pkg} {metadata.version(pkg)}")
        except metadata.PackageNotFoundError:
            pass
    return ", ".join(out)


def _attn_used(inner: Any) -> str:
    """Best-effort: which attention implementation the loaded model really uses."""
    try:
        m = getattr(inner, "model", inner)
        for obj in (getattr(m, "talker", None), m):
            cfg = getattr(obj, "config", None)
            impl = getattr(cfg, "_attn_implementation", None)
            if impl:
                return str(impl)
    except Exception:  # noqa: BLE001
        pass
    return "unknown"
