"""Shared helpers of the worker processes: the stdout protocol, timing, CUDA/VRAM helpers, model location and download.

A worker module ``w_<name>.py`` defines ``run(args: dict, ctx: WorkerContext) -> dict`` returning
``{"status": "OK|WARN|FAIL|SKIP", "summary": str, "details": [str], "metrics": {...}}``.  Exceptions are caught by
:func:`worker_main` and become ``FAIL`` with the traceback, so a worker module can stay simple.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

MARK = "@@VX@@ "
_PROTO = sys.stdout          # the real stdout, kept for the protocol; ordinary prints are redirected to stderr in worker_main


class WorkerContext:
    """Passed to ``run``: ``log(msg)`` reports progress to the parent (shown in the UI), ``t0`` is the start time."""

    def __init__(self) -> None:
        self.t0 = time.time()

    def log(self, msg: str) -> None:
        """Send one progress line to the parent on the worker protocol."""
        _emit({"t": "log", "msg": msg})


def _emit(obj: Dict[str, Any]) -> None:
    try:
        _PROTO.write(MARK + json.dumps(obj, ensure_ascii=False, default=str) + "\n")
        _PROTO.flush()
    except Exception:  # noqa: BLE001 - parent gone
        pass


def worker_main(name: str, args_path: str) -> int:
    """Run worker ``name``.  Always prints exactly one result line; returns 0 unless the process itself is broken."""
    sys.stdout = sys.stderr                       # stray prints of libraries must not corrupt the protocol
    try:
        with open(args_path, encoding="utf-8") as fh:
            args = json.load(fh)
    except Exception:  # noqa: BLE001
        _emit({"t": "result", "status": "FAIL", "summary": "worker could not read its arguments", "traceback": traceback.format_exc()})
        return 0
    _prepare_environment(args)
    ctx = WorkerContext()
    try:
        mod = importlib.import_module(f"dubber.workers.w_{name}")
        res = mod.run(args, ctx)
        res = dict(res or {})
        res.setdefault("status", "OK")
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - includes KeyboardInterrupt/MemoryError: still report
        res = {"status": "FAIL", "summary": f"{type(exc).__name__}: {str(exc)[:300]}", "traceback": traceback.format_exc()}
        if _looks_like_oom(exc):
            res["summary"] = "out of memory: " + res["summary"]
    res["t"] = "result"
    res["total_s"] = round(time.time() - ctx.t0, 3)
    _emit(res)
    return 0


def _looks_like_oom(exc: BaseException) -> bool:
    s = f"{type(exc).__name__} {exc}".lower()
    return "out of memory" in s or "cuda error: out of memory" in s or isinstance(exc, MemoryError)


def _prepare_environment(args: Dict[str, Any]) -> None:
    """Hugging Face settings from the parent's request: offline mode when downloads are not allowed, token via env only."""
    if not args.get("allow_download", True):
        os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))     # run from source: repo root importable
    _add_torch_dll_dirs()


def _add_torch_dll_dirs() -> None:
    """Windows: make CUDA DLLs of the installed PyTorch visible to other native libraries (CTranslate2, onnxruntime)."""
    if sys.platform != "win32":
        return
    try:
        import importlib.util

        spec = importlib.util.find_spec("torch")
        if spec and spec.origin:
            lib = Path(spec.origin).parent / "lib"
            if lib.is_dir():
                os.add_dll_directory(str(lib))
                os.environ["PATH"] = str(lib) + os.pathsep + os.environ.get("PATH", "")
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------------------------- torch helpers
def torch_device(prefer: str = "auto") -> str:
    """``cuda`` when available (and not disabled by ``prefer='cpu'``), else ``cpu``."""
    if prefer == "cpu":
        return "cpu"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def cuda_sync() -> None:
    """Wait until this process's queued CUDA work has finished."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        pass


def reset_peak() -> None:
    """Reset PyTorch's peak CUDA-allocation counter for this process."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001
        pass


def peak_vram_gb() -> Optional[float]:
    """Peak memory allocated by PyTorch in this process (GB); None without CUDA."""
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
    except Exception:  # noqa: BLE001
        pass
    return None


def reserved_vram_gb() -> Optional[float]:
    """GPU memory reserved by PyTorch, in GB, or None when CUDA is unavailable."""
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.memory_reserved() / 1024 ** 3, 2)
    except Exception:  # noqa: BLE001
        pass
    return None


def free_gpu() -> None:
    """Run garbage collection and release PyTorch's cached CUDA blocks."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------------------------- audio helpers
def read_wav_mono(path: str, target_sr: Optional[int] = None):
    """(float32 mono samples, sample rate) from a WAV/FLAC; resampled to ``target_sr`` (linear interpolation or scipy) if given."""
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if target_sr and sr != target_sr:
        mono = resample(mono, sr, target_sr)
        sr = target_sr
    return mono.astype(np.float32), sr


def resample(x, sr_from: int, sr_to: int):
    """Polyphase resampling with scipy (falls back to linear interpolation)."""
    import numpy as np

    if sr_from == sr_to:
        return x
    try:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(sr_from, sr_to)
        return resample_poly(x, sr_to // g, sr_from // g).astype(np.float32)
    except Exception:  # noqa: BLE001
        n = int(len(x) * sr_to / sr_from)
        return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def rms_db(x) -> float:
    """Root-mean-square level of ``x`` in dBFS, or -120 for an empty buffer."""
    import numpy as np

    if len(x) == 0:
        return -120.0
    return float(20 * np.log10(max(1e-9, float(np.sqrt(np.mean(np.square(x.astype("float64"))))))))


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate (lower-case, punctuation removed) via edit distance; 0.0 = identical."""
    import re

    def norm(s: str) -> List[str]:
        return re.sub(r"[^\w\s']", " ", s.lower()).split()

    r, h = norm(reference), norm(hypothesis)
    if not r:
        return 0.0 if not h else 1.0
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / len(r)


def apply_qwen_tts_compat() -> str:
    """Work around qwen-tts-hf 0.1.1 vs transformers >= 5.18: its "default" RoPE initialiser (``qwen_tts._transformers_compat``)
    reads ``config.rope_theta``, which Mimi/other configs no longer have (they keep it in ``config.rope_parameters``) ->
    ``AttributeError: ... 'rope_theta'`` while the speech tokenizer loads.  Installs a tolerant initialiser in the transformers
    registry AND in qwen_tts's own module (faster-qwen3-tts imports qwen_tts lazily inside ``from_pretrained``, and qwen_tts only
    ``setdefault``s its function, so both places must hold the tolerant one).  Returns a note for the report ("" if nothing was needed)."""
    try:
        import torch
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    except Exception:  # noqa: BLE001
        return ""
    cur = ROPE_INIT_FUNCTIONS.get("default")
    if cur is not None and getattr(cur, "_vmd_patched", False):
        return ""

    def default_rope(config, device=None, seq_len=None, layer_type=None):
        base = getattr(config, "rope_theta", None)
        if base is None:
            params = getattr(config, "rope_parameters", None) or {}
            if isinstance(params, dict) and layer_type and isinstance(params.get(layer_type), dict):
                params = params[layer_type]
            base = params.get("rope_theta", 10000.0) if isinstance(params, dict) else 10000.0
        factor = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * factor)
        inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
        return inv, 1.0

    default_rope._vmd_patched = True            # type: ignore[attr-defined]
    ROPE_INIT_FUNCTIONS["default"] = default_rope
    try:                                        # qwen_tts may be imported later (faster-qwen3-tts does it lazily)
        import qwen_tts._transformers_compat as qcompat

        qcompat._default_rope_parameters = default_rope
    except Exception:  # noqa: BLE001 - package layout differs: the registry entry above is enough
        pass
    return "applied RoPE compat shim for qwen-tts-hf on this transformers version"
