"""Make the CUDA 12 libraries of the PyTorch build visible to CTranslate2 (faster-whisper) on Windows.

CTranslate2 4.x is built for CUDA 12 and loads ``cublas64_12.dll`` at run time with a plain ``LoadLibrary`` (the PATH search).
The PyTorch cu128 build of our runtime ships exactly these files in ``torch\\lib`` (checked in the wheel: cublas64_12.dll,
cublasLt64_12.dll, cudart64_12.dll, cudnn*64_9.dll), but that folder is not on PATH, so speech recognition fell back to the CPU
("Library cublas64_12.dll is not found or cannot be loaded").  :func:`expose` puts ``torch\\lib`` (and the ``nvidia\\*\\bin`` folders
of pip CUDA packages, if any) on PATH and on the DLL search list - without importing torch.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import List

_done: List[str] = []


def candidate_dirs() -> List[Path]:
    out: List[Path] = []
    spec = importlib.util.find_spec("torch")
    if spec and spec.origin:
        lib = Path(spec.origin).parent / "lib"
        if lib.is_dir():
            out.append(lib)
        site = Path(spec.origin).parent.parent
        nv = site / "nvidia"
        if nv.is_dir():
            out += sorted(p for p in nv.glob("*/bin") if p.is_dir())
    return out


def has_cublas12(dirs: List[Path]) -> bool:
    return any((d / "cublas64_12.dll").is_file() for d in dirs)


def expose() -> List[str]:
    """Idempotent; returns the folders added (empty off Windows)."""
    if sys.platform != "win32" or _done:
        return list(_done)
    for d in candidate_dirs():
        s = str(d)
        try:
            os.add_dll_directory(s)
        except (OSError, AttributeError):
            pass
        if s.lower() not in os.environ.get("PATH", "").lower():
            os.environ["PATH"] = s + os.pathsep + os.environ.get("PATH", "")
        _done.append(s)
    return list(_done)
