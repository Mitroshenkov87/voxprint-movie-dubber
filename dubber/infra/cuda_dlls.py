"""Make the CUDA 12 libraries visible to CTranslate2 (faster-whisper) on Windows.

CTranslate2 4.x is built for CUDA 12 and loads ``cublas64_12.dll`` with a plain ``LoadLibrary``. The cu130 PyTorch build
does not ship those CUDA 12 libraries (it ships CUDA 13). The installer therefore installs the ``nvidia-cublas-cu12`` and
``nvidia-cudnn-cu12`` wheels, and :func:`expose` puts their ``bin`` folders on the DLL search path with
``os.add_dll_directory`` before CTranslate2 is imported.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import List, Optional

_done: List[str] = []


def _add(path: Path, out: List[Path], seen: set[str]) -> None:
    if not path.is_dir():
        return
    try:
        key = str(path.resolve()).lower()
    except OSError:
        key = str(path).lower()
    if key in seen:
        return
    seen.add(key)
    out.append(path)


def _nvidia_bins(root: Path, out: List[Path], seen: set[str]) -> None:
    for rel in ("cublas/bin", "cudnn/bin"):
        _add(root / rel, out, seen)
    if root.is_dir():
        for folder in sorted(root.glob("*/bin")):
            _add(folder, out, seen)


def candidate_dirs() -> List[Path]:
    """Folders that may hold ``cublas64_12.dll`` / cuDNN 9: ``torch\\lib`` and the NVIDIA wheel ``bin`` folders."""
    out: List[Path] = []
    seen: set[str] = set()
    spec = importlib.util.find_spec("torch")
    if spec and spec.origin:
        origin = Path(spec.origin)
        _add(origin.parent / "lib", out, seen)
        _nvidia_bins(origin.parent.parent / "nvidia", out, seen)
    for entry in sys.path:
        if entry:
            _nvidia_bins(Path(entry) / "nvidia", out, seen)
    return out


def has_cublas12(dirs: List[Path]) -> bool:
    """Return whether any directory in ``dirs`` contains ``cublas64_12.dll``."""
    return any((d / "cublas64_12.dll").is_file() for d in dirs)


def expose() -> List[str]:
    """Idempotent. On Windows, register every candidate folder with ``os.add_dll_directory`` and PATH."""
    if sys.platform != "win32" or _done:
        return list(_done)
    for folder in candidate_dirs():
        s = str(folder)
        try:
            os.add_dll_directory(s)
        except (OSError, AttributeError):
            pass
        if s.lower() not in os.environ.get("PATH", "").lower():
            os.environ["PATH"] = s + os.pathsep + os.environ.get("PATH", "")
        _done.append(s)
    return list(_done)


def cuda_device_count() -> Optional[int]:
    """How many CUDA devices CTranslate2 sees, after :func:`expose`. None when the package is not installed."""
    expose()
    try:
        import ctranslate2
    except ImportError:
        return None
    return int(ctranslate2.get_cuda_device_count())


def cuda_device_summary(count: Optional[int]) -> tuple[bool, str]:
    """Pass only when CTranslate2 reports at least one CUDA device."""
    if count is None:
        return False, "ctranslate2 is not installed, so it cannot see a CUDA device."
    if count > 0:
        noun = "device" if count == 1 else "devices"
        return True, f"ctranslate2 sees {count} CUDA {noun}."
    return False, (
        "ctranslate2 sees no CUDA device. faster-whisper needs nvidia-cublas-cu12 and nvidia-cudnn-cu12 "
        "on the DLL search path (cublas64_12.dll)."
    )
