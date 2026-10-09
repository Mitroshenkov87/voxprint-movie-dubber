"""Resource policy for every model stage: which GPU there is, how much VRAM and RAM is free right now, and how much a run may take.

Suite rule (agreed with Voxprint AI Audiobook Builder) for system RAM: a run uses at most ``RAM_SHARE`` of the free RAM.
VRAM is not a share of free memory. The budget is whatever is free right now minus a headroom of ``max(2 GB, 8 % of the card)``,
measured again before each TTS batch, so the driver is not pushed into system RAM. An optional fraction
(``VOXPRINT_VRAM_FRACTION``, or ``vram_fraction`` on the shared ``gpu`` value) can cap that budget; it is off unless set.

What the budget decides:

* TTS batch size (lines per generate call): what fits in the budget after the model is loaded (measured, not guessed);
* whether several voices (LoRA adapters) stay loaded at once or are swapped one after another;
* per stage: a model that does not fit the GPU budget runs on the CPU (light stages), or the lighter TTS model is used.

Models of different stages never share the GPU: every model stage is its own worker process, which frees all its memory on exit,
so stages are always swapped sequentially.  The GUI process never imports torch: there the numbers come from ``nvidia-smi``.
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dubber.infra import suite

RAM_SHARE = 0.75
#: leave at least this much VRAM unused: max(VRAM_HEADROOM_GB, VRAM_HEADROOM_FRAC * total)
VRAM_HEADROOM_GB = 2.0
VRAM_HEADROOM_FRAC = 0.08
GB = 1024 ** 3

#: VRAM a model needs while it runs (weights + workspace, GB); TTS 1.7B with CUDA Graphs peaked at ~5 GB on the RTX 4090 Laptop test.
MODEL_VRAM_GB: Dict[str, float] = {"tts_1_7b": 5.0, "tts_0_6b": 2.6, "separation": 1.6, "asr": 2.2, "diarization": 1.2,
                                   "translation": 0.8, "vad": 0.2}
#: system RAM a model needs when it runs on the CPU (GB, fp32)
MODEL_RAM_GB: Dict[str, float] = {"tts_1_7b": 8.0, "tts_0_6b": 3.5, "separation": 2.0, "asr": 3.0, "diarization": 1.5,
                                  "translation": 1.2, "vad": 0.3}
TTS_ITEM_VRAM_GB = 1.2          # one more line in a batched TTS generate (measured ~1.2 GB; KV cache + activations, bf16)
TTS_ITEM_RAM_GB = 0.6           # the same on the CPU (fp32)
TTS_ACT_RESERVE_GB = 0.5        # workspace of a single generate call
ADAPTER_VRAM_GB = 0.15          # one loaded LoRA voice adapter (+ its prompt)
MAX_BATCH = 12
MAX_CPU_BATCH = 4


@dataclass
class Snapshot:
    gpu: str = ""
    vram_total_gb: float = 0.0
    vram_free_gb: float = 0.0
    ram_total_gb: float = 0.0
    ram_free_gb: float = 0.0
    source: str = ""                      # torch | nvidia-smi | none

    @property
    def has_gpu(self) -> bool:
        return self.vram_total_gb > 0

    @property
    def vram_budget_gb(self) -> float:
        return round(vram_allowance_gb(self.vram_free_gb, self.vram_total_gb, vram_fraction()), 2)

    @property
    def ram_budget_gb(self) -> float:
        return round(RAM_SHARE * self.ram_free_gb, 2)

    def describe(self) -> str:
        ram = f"RAM {self.ram_free_gb:.1f} of {self.ram_total_gb:.1f} GB free (budget {self.ram_budget_gb:.1f} GB)"
        if not self.has_gpu:
            return f"no NVIDIA GPU found; {ram}"
        room = vram_headroom_gb(self.vram_total_gb)
        return (f"{self.gpu}: VRAM {self.vram_free_gb:.1f} of {self.vram_total_gb:.1f} GB free, budget {self.vram_budget_gb:.1f} GB "
                f"(headroom {room:.1f} GB); {ram}")


# ---------------------------------------------------------------------------------------------- measuring
def ram_gb() -> Tuple[float, float]:
    """(total, available) system RAM in GB; (0, 0) if unknown."""
    try:
        import psutil

        vm = psutil.virtual_memory()
        return vm.total / GB, vm.available / GB
    except Exception:  # noqa: BLE001 - psutil missing: the OS directly
        pass
    if sys.platform == "win32":
        try:
            import ctypes

            class MEMSTAT(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = MEMSTAT()
            st.dwLength = ctypes.sizeof(MEMSTAT)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return st.ullTotalPhys / GB, st.ullAvailPhys / GB
        except Exception:  # noqa: BLE001
            return 0.0, 0.0
    try:
        info = {ln.split(":")[0]: float(ln.split()[1]) for ln in Path("/proc/meminfo").read_text().splitlines() if ln.split()[1:]}
        return info["MemTotal"] / 1024 ** 2, info.get("MemAvailable", info.get("MemFree", 0)) / 1024 ** 2
    except (OSError, KeyError, ValueError, IndexError):
        return 0.0, 0.0


def gpu_from_torch() -> Optional[Tuple[str, float, float]]:
    """(name, total GB, free GB) of cuda:0 when torch is already loaded in this process (never imports it)."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info(0)
        return torch.cuda.get_device_name(0), total / GB, free / GB
    except Exception:  # noqa: BLE001
        return None


def gpu_from_smi() -> Optional[Tuple[str, float, float]]:
    """(name, total GB, free GB) of the first NVIDIA GPU from nvidia-smi (no torch needed)."""
    if os.environ.get("VOXPRINT_FAKE_VRAM"):                   # tests / support: "name,total_gb,free_gb"
        name, total, free = os.environ["VOXPRINT_FAKE_VRAM"].split(",")
        return name, float(total), float(free)
    try:
        from dubber.diag.procs import find_nvidia_smi

        smi = find_nvidia_smi()
        if not smi:
            return None
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        out = subprocess.run([smi, "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, timeout=8, creationflags=flags).stdout
        row = [c.strip() for c in out.strip().splitlines()[0].split(",")]
        return row[0], float(row[1]) / 1024, float(row[2]) / 1024
    except Exception:  # noqa: BLE001 - no driver, odd output: no GPU numbers
        return None


def snapshot(use_torch: bool = True) -> Snapshot:
    """What is there and what is free right now (torch if it is already loaded in this process, else nvidia-smi).

    The window process passes ``use_torch=False``: asking torch for free memory would create a CUDA context there (~0.3 GB held)."""
    s = Snapshot()
    s.ram_total_gb, s.ram_free_gb = (round(x, 2) for x in ram_gb())
    g, src = (gpu_from_torch() if use_torch else None), "torch"
    if g is None:
        g, src = gpu_from_smi(), "nvidia-smi"
    if g is not None:
        s.gpu, s.vram_total_gb, s.vram_free_gb, s.source = g[0], round(g[1], 2), round(g[2], 2), src
    else:
        s.source = "none"
    return s


# ---------------------------------------------------------------------------------------------- decisions
@dataclass
class StagePlan:
    device: str                              # cuda | cpu
    tts_model: str = ""
    notes: List[str] = field(default_factory=list)


def plan_stage(stage: str, snap: Snapshot, device: str, tts_model: str = "tts_1_7b", lighter_ok: bool = True) -> StagePlan:
    """Device (and TTS model) for one model stage, so that it fits the VRAM budget (free minus headroom).

    Light stages that do not fit go to the CPU (a few minutes slower, never a stall).  TTS on the CPU is far too slow, so it first
    switches to the 0.6B model; if even that does not fit it stays on the GPU at batch 1 (the batch logic halves on out-of-memory)."""
    out = StagePlan(device, tts_model)
    if device != "cuda" or not snap.has_gpu:
        if device == "cuda":
            return out
        need = MODEL_RAM_GB.get(tts_model if stage == "tts" else stage, 1.0)
        if snap.ram_free_gb and need > snap.ram_budget_gb:
            out.notes.append(f"{stage}: needs ~{need:.1f} GB RAM, budget {snap.ram_budget_gb:.1f} GB - expect swapping; close other programs")
        return out
    budget = snap.vram_budget_gb
    if stage == "tts":
        need = MODEL_VRAM_GB.get(tts_model, 5.0) + TTS_ACT_RESERVE_GB + TTS_ITEM_VRAM_GB
        if need > budget and tts_model != "tts_0_6b" and lighter_ok:
            light = MODEL_VRAM_GB["tts_0_6b"] + TTS_ACT_RESERVE_GB + TTS_ITEM_VRAM_GB
            if light <= budget:
                out.tts_model = "tts_0_6b"
                out.notes.append(f"tts: {tts_model} needs ~{need:.1f} GB, VRAM budget {budget:.1f} GB - using the lighter tts_0_6b")
                return out
        if need > budget:
            out.notes.append(f"tts: needs ~{need:.1f} GB, VRAM budget only {budget:.1f} GB - batch 1, close other GPU programs")
        return out
    need = MODEL_VRAM_GB.get(stage, 1.0)
    if need > budget:
        out.device = "cpu"
        out.notes.append(f"{stage}: needs ~{need:.1f} GB, VRAM budget {budget:.1f} GB - running on the CPU")
    return out


def tts_batch(device: str, budget_left_gb: float, ram_budget_gb: float = 0.0, graphs: bool = False) -> int:
    """Lines per TTS generate call: what fits in what is left of the budget after the model (and its voices) were loaded."""
    if graphs:
        return 1                                                  # CUDA Graphs: static shapes, one line at a time
    if device == "cuda":
        n = int((budget_left_gb - TTS_ACT_RESERVE_GB) // TTS_ITEM_VRAM_GB)
        cap = MAX_BATCH if not ram_budget_gb or ram_budget_gb >= 2.0 else 2      # almost no free RAM: small batches on the GPU too
    else:
        n = int((ram_budget_gb - TTS_ACT_RESERVE_GB) // TTS_ITEM_RAM_GB) if ram_budget_gb else 1
        cap = MAX_CPU_BATCH
    return max(1, min(cap, n))


def keep_voices_loaded(budget_left_gb: float, batch: int = 1) -> bool:
    """True: one more voice adapter fits next to the loaded ones and a batch; False: swap voices one after another."""
    return budget_left_gb - ADAPTER_VRAM_GB >= TTS_ACT_RESERVE_GB + max(1, batch) * TTS_ITEM_VRAM_GB


class VramBudget:
    """In a worker process: how much VRAM a batch may use, from the memory that is free *now*.

    ``budget = max(0, free_now - max(2 GB, 8 % of total))``, optionally capped by :func:`vram_fraction`. Re-measure with
    :meth:`remeasure` before each TTS batch group. ``free_now`` already includes this process, so the result is not reduced
    a second time by what we allocated."""

    def __init__(self) -> None:
        g = gpu_from_torch()
        self.total = g[1] if g else 0.0
        self.free0 = g[2] if g else 0.0
        self.fraction = vram_fraction()
        self.budget = vram_allowance_gb(self.free0, self.total, self.fraction)
        _total, avail = ram_gb()
        self.ram_budget = RAM_SHARE * avail

    def used_by_us(self) -> float:
        g = gpu_from_torch()
        return max(0.0, self.free0 - g[2]) if g else 0.0

    def remeasure(self) -> float:
        """Set ``budget`` from the VRAM that is free at this moment and return it."""
        g = gpu_from_torch()
        free_now = g[2] if g else 0.0
        if g:
            self.total = g[1]
        self.budget = vram_allowance_gb(free_now, self.total, self.fraction)
        return self.budget

    def left(self) -> float:
        """The budget from a fresh measurement (free memory already reflects what this process holds)."""
        return self.remeasure()

    def describe(self) -> str:
        left = self.remeasure()
        cap = f", cap {self.fraction:.0%} of {self.total:.1f} GB" if self.fraction else ""
        return (f"VRAM budget {self.budget:.1f} GB (free now minus headroom{cap}), left {left:.1f} GB; "
                f"RAM budget {self.ram_budget:.1f} GB")


def vram_headroom_gb(total_gb: float) -> float:
    """VRAM left unused on purpose: at least 2 GB, or 8 % of the card when that is larger."""
    return max(VRAM_HEADROOM_GB, VRAM_HEADROOM_FRAC * max(0.0, total_gb))


def _parse_vram_fraction(value: object) -> Optional[float]:
    """A cap in (0, 1], or None when the value is missing, empty, 0, or not a fraction (the cap stays off)."""
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        frac = float(value)
    except (TypeError, ValueError):
        return None
    if frac <= 0 or frac > 1:
        return None
    return frac


def _fraction_from_mapping(data: Dict[str, Any]) -> Optional[float]:
    gpu = data.get("gpu")
    if isinstance(gpu, dict) and "vram_fraction" in gpu:
        return _parse_vram_fraction(gpu.get("vram_fraction"))
    if "vram_fraction" in data:
        return _parse_vram_fraction(data.get("vram_fraction"))
    return None


def vram_fraction(cfg: Optional[Dict[str, Any]] = None) -> Optional[float]:
    """Optional user cap. ``VOXPRINT_VRAM_FRACTION`` wins, then ``cfg``, then suite.json (``gpu.vram_fraction`` or ``vram_fraction``).

    Missing, empty, and 0 mean off."""
    env = os.environ.get("VOXPRINT_VRAM_FRACTION")
    if env is not None and str(env).strip() != "":
        return _parse_vram_fraction(env)
    if cfg:
        if "vram_fraction" in cfg or (isinstance(cfg.get("gpu"), dict) and "vram_fraction" in cfg["gpu"]):
            return _fraction_from_mapping(cfg)
    return _fraction_from_mapping(suite.read_raw())


def vram_allowance_gb(free_gb: float, total_gb: float, fraction: Optional[float] = None) -> float:
    """``max(0, free_now - max(2 GB, 0.08 * total))``, then ``min`` with ``fraction * total`` when a cap is set."""
    budget = max(0.0, free_gb - vram_headroom_gb(total_gb))
    if fraction is not None and fraction > 0:
        budget = min(budget, fraction * total_gb)
    return budget
