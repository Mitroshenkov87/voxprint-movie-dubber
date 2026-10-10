"""Overlap the next stage's model load and CPU work with the GPU stage that is already running.

CPU work is VAD and ffmpeg extraction, and only when that work has not been done yet.  The next model is
recorded (and loaded, when the resident cache is on and a loader was registered) on a side thread.  A failure
there is logged and does not fail the stage.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

from dubber.infra.resources import MODEL_VRAM_GB

# stage -> model to load while it runs.  "mix" is CPU and has no weights.
NEXT_MODEL = {
    "vad": "separation",
    "separation": "asr",
    "asr": "diarization",
    "diarization": "translation",
    "translation": "tts",
}
NEXT_VRAM = {
    "vad": "vad",
    "separation": "separation",
    "asr": "asr",
    "diarization": "diarization",
    "translation": "translation",
    "tts": "tts_1_7b",
}
# CPU work that can run beside a GPU stage.  Each item no-ops once its file exists.
CPU_WORK = {
    "separation": ("ffmpeg",),
    "asr": ("vad",),
    "translation": ("ffmpeg",),
    "tts": ("vad", "ffmpeg"),
    "diarization": ("ffmpeg",),
}


@dataclass(frozen=True)
class Plan:
    """Model load and CPU work that can overlap the GPU stage already running.

    Args:
        next_model: Model to prefetch, or empty when this stage has no successor.
        cpu: CPU jobs, ``vad`` or ``ffmpeg``, that are safe to run beside the GPU stage.
        need_gb: VRAM, in GB, that the next model is expected to need.
    """
    next_model: str
    cpu: Tuple[str, ...]
    need_gb: float


def plan(stage: str) -> Plan:
    """Return the next model, the CPU jobs that can overlap ``stage``, and the VRAM that model needs."""
    nxt = NEXT_MODEL.get(stage, "")
    need = MODEL_VRAM_GB.get(NEXT_VRAM.get(nxt, ""), 0.0) if nxt else 0.0
    return Plan(nxt, CPU_WORK.get(stage, ()), need)


last_error: BaseException | None = None


def overlap(gpu: Callable[[], str], cpu: Callable[[], None]) -> str:
    """Run ``cpu`` on a side thread for the whole of ``gpu``.  ``cpu`` errors are kept off the GPU result."""
    global last_error
    errors: list = []

    def side() -> None:
        try:
            cpu()
        except Exception as exc:  # noqa: BLE001 - prefetch must not take the dub down
            errors.append(exc)

    thread = threading.Thread(target=side, daemon=True)
    thread.start()
    try:
        return gpu()
    finally:
        thread.join()
        last_error = errors[0] if errors else None


def run_cpu(names: Sequence[str], project, cfg, emit: Callable[..., None]) -> None:
    """VAD and ffmpeg for this project, skipped when the files are already there."""
    from dubber.pipeline import stages as S

    folder = project.folder
    for name in names:
        if name == "vad" and not (folder / "analysis" / "windows.json").is_file():
            S.st_vad(project, cfg, emit)
        elif name == "ffmpeg" and not (folder / "audio" / "mix16.wav").is_file():
            S.st_extract(project, cfg, emit)
        elif name not in ("vad", "ffmpeg"):
            continue


def start_model(stage: str) -> None:
    """Ask the resident cache to load the next stage's model.  No loader means the request is only noted."""
    item = plan(stage)
    if not item.next_model:
        return
    from dubber.infra import resident

    resident.prefetch(item.next_model, item.need_gb)
