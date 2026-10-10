"""Minimum GPU for Voxprint AI Movie Dubber: NVIDIA compute capability 8.9 (RTX 40 / Ada) and driver branch 600.

The installer asks ``nvidia-smi`` before it downloads anything. The window asks PyTorch before it opens.
Diagnostics report the same facts and fail the check when the machine is below the minimum.
``VOXPRINT_SKIP_GPU_GATE=1`` skips only the window check, so the CI smoke run can open the UI on a GPU-less runner.
The CPU build of PyTorch is not a user install; CI passes ``-Backend cpu`` to the installer on its own.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, Literal, Mapping, Never, Optional, Sequence, Tuple

MIN_COMPUTE: Tuple[int, int] = (8, 9)
MIN_DRIVER_BRANCH = 600
SKIP_ENV = "VOXPRINT_SKIP_GPU_GATE"

INSTALLER_LEAD = (
    "Voxprint AI Movie Dubber needs an NVIDIA GeForce RTX 40-series graphics card or newer "
    "(Ada Lovelace or later, compute capability 8.9 or higher) and an NVIDIA driver from the 600 branch or newer."
)
INSTALLER_LEAD_RU = (
    "Для Voxprint AI Movie Dubber нужна видеокарта NVIDIA GeForce RTX 40 или новее "
    "(Ada Lovelace и новее, вычислительная способность 8.9 или выше) и драйвер NVIDIA ветки 600 или новее."
)
_NEED = (
    "An NVIDIA GPU with compute capability 8.9 or higher (RTX 40 / Ada or newer) "
    "and driver branch 600 or newer is required."
)

_COMPUTE = re.compile(r"^(\d+)\.(\d+)$")
_BRANCH = re.compile(r"^(\d+)")
StartupCode = Literal["no_cuda", "low_compute"]


@dataclass(frozen=True)
class GpuFact:
    """One NVIDIA GPU from nvidia-smi: name, compute capability, driver version, and driver branch."""
    name: str
    compute: Tuple[int, int]
    driver: str
    driver_branch: int


@dataclass(frozen=True)
class GateResult:
    """Minimum-GPU check: whether any card qualifies, which cards were seen, and a one-line summary."""
    ok: bool
    gpus: Tuple[GpuFact, ...]
    qualifying: Tuple[GpuFact, ...]
    summary: str


@dataclass(frozen=True)
class StartupBlock:
    """Why the window stays closed: no CUDA device, or compute capability below 8.9."""
    code: StartupCode
    name: str = ""
    capability: str = ""


def parse_compute(text: str) -> Optional[Tuple[int, int]]:
    """Parse a compute capability such as ``8.9`` into ``(8, 9)``, or return None."""
    match = _COMPUTE.match(text.strip())
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def parse_driver_branch(version: str) -> Optional[int]:
    """Major number of an nvidia-smi driver version (``560.94`` -> 560, ``617.42`` -> 617)."""
    match = _BRANCH.match(version.strip())
    if not match:
        return None
    return int(match.group(1))


def parse_nvidia_smi_query(text: str) -> Tuple[GpuFact, ...]:
    """Parse ``nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader``."""
    found = []
    for raw in text.splitlines():
        line = raw.strip()
        if "," not in line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        driver = parts[-1]
        compute = parse_compute(parts[-2])
        branch = parse_driver_branch(driver)
        name = ",".join(parts[:-2]).strip()
        if not name or compute is None or branch is None:
            continue
        found.append(GpuFact(name, compute, driver, branch))
    return tuple(found)


def meets_policy(gpu: GpuFact) -> bool:
    """Return whether the GPU is compute 8.9 or newer and the driver branch is 600 or newer."""
    return gpu.compute >= MIN_COMPUTE and gpu.driver_branch >= MIN_DRIVER_BRANCH


def found_text(gpus: Sequence[GpuFact]) -> str:
    """Return one sentence describing the NVIDIA GPUs that were found."""
    if not gpus:
        return "No NVIDIA GPU was found."
    text = "; ".join(
        f"{gpu.name}, compute capability {gpu.compute[0]}.{gpu.compute[1]}, "
        f"driver {gpu.driver} (branch {gpu.driver_branch})"
        for gpu in gpus
    )
    return text if text.endswith(".") else text + "."


def failure_summary(gpus: Sequence[GpuFact]) -> str:
    """Return why the machine is below the minimum, including the case of no NVIDIA GPU."""
    if not gpus:
        return f"No NVIDIA GPU found. {_NEED}"
    listed = "; ".join(
        f"{gpu.name}, compute capability {gpu.compute[0]}.{gpu.compute[1]}, driver branch {gpu.driver_branch}"
        for gpu in gpus
    )
    return f"{listed}: below the minimum. {_NEED}"


def _ok_summary(gpu: GpuFact) -> str:
    return (
        f"{gpu.name}, compute capability {gpu.compute[0]}.{gpu.compute[1]}, "
        f"driver {gpu.driver} (branch {gpu.driver_branch})"
    )


def evaluate_gpus(gpus: Sequence[GpuFact]) -> GateResult:
    """Pass when at least one GPU is RTX 40 / Ada or newer and the driver branch is 600 or newer."""
    found = tuple(gpus)
    good = tuple(gpu for gpu in found if meets_policy(gpu))
    if good:
        return GateResult(True, found, good, _ok_summary(good[0]))
    return GateResult(False, found, (), failure_summary(found))


def evaluate_nvidia_smi_query(text: str) -> GateResult:
    """Check nvidia-smi CSV text against the RTX 40 and driver-branch 600 minimum."""
    return evaluate_gpus(parse_nvidia_smi_query(text))


def installer_message(found: str) -> str:
    """English text the setup wizard shows when the PC is below the minimum. Nothing is installed."""
    return (
        f"{INSTALLER_LEAD}\n\n"
        "This PC does not meet that requirement, so setup stopped and nothing was installed. "
        "A processor-only copy is not offered.\n\n"
        f"What we found: {found}\n\n"
        "Install a supported graphics card and driver, then run setup again."
    )


def installer_message_ru(found: str) -> str:
    """Russian text written to the setup log. The wizard itself has no language catalog."""
    return (
        f"{INSTALLER_LEAD_RU}\n\n"
        "Этот компьютер не подходит, поэтому установка остановлена и ничего не установлено. "
        "Вариант только для процессора не предлагается.\n\n"
        f"Что обнаружено: {found}\n\n"
        "Установите подходящую видеокарту и драйвер и запустите установку снова."
    )


def startup_block(available: bool, capability: Optional[Tuple[int, int]], name: str) -> Optional[StartupBlock]:
    """Window gate: ``torch.cuda.is_available()`` and ``get_device_capability() >= (8, 9)``.

    ``None`` means the window may open. Driver branch is enforced by the installer and by diagnostics;
    this process can see the capability PyTorch reports.
    """
    if available and capability is not None and tuple(capability) >= MIN_COMPUTE:
        return None
    if available and capability is not None:
        label = name.strip() or "NVIDIA GPU"
        return StartupBlock("low_compute", label, f"{int(capability[0])}.{int(capability[1])}")
    return StartupBlock("no_cuda")


def startup_detail_key(block: StartupBlock) -> Tuple[str, Dict[str, str]]:
    """Locale key and placeholders for the sentence under the shared requirement."""
    code = block.code
    if code == "low_compute":
        return "gpu.gate_low_compute", {"name": block.name, "capability": block.capability}
    if code == "no_cuda":
        return "gpu.gate_no_cuda", {}
    unexpected: Never = code
    raise AssertionError(unexpected)


def gate_skipped(environ: Optional[Mapping[str, str]] = None) -> bool:
    """True when ``VOXPRINT_SKIP_GPU_GATE`` is set. Tests and the CI smoke run set it; users do not."""
    env = os.environ if environ is None else environ
    return env.get(SKIP_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def probe_torch() -> Tuple[bool, Optional[Tuple[int, int]], str]:
    """``(available, capability, device name)`` from PyTorch.

    Torch is imported here, not at module import: a machine without it is "no CUDA", and the CLI
    commands (version, installer helpers) must stay light.
    """
    try:
        import torch
    except Exception:  # noqa: BLE001 - torch is optional; its absence is a failed gate
        return False, None, ""
    try:
        if not torch.cuda.is_available():
            return False, None, ""
        major, minor = torch.cuda.get_device_capability()
        try:
            device_name = str(torch.cuda.get_device_name(0))
        except Exception:  # noqa: BLE001 - the capability is enough; the name is only for the dialog
            device_name = ""
        return True, (int(major), int(minor)), device_name
    except Exception:  # noqa: BLE001 - a broken CUDA stack is the same as no usable GPU
        return False, None, ""
