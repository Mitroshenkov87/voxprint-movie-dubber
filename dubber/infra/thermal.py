"""Cooling pauses between blocks, and the GPU line written for each stage.

Full speed for the first 2.5 hours of a job.  After that, a pause is inserted between blocks only when the GPU
is at 87 C or above, or a thermal / power-brake throttle reason has lasted more than 60 seconds.  Work resumes
at 75 C or below (and when the throttle reason has cleared).  There is no user-facing mode: the rule runs on
its own.  A missing sensor never pauses the job.
"""
from __future__ import annotations

import os
import subprocess
import time
from typing import Callable, Optional, Tuple

FULL_SPEED_S = 2.5 * 3600
HOT_C = 87.0
COOL_C = 75.0
THROTTLE_HOLD_S = 60.0
MAX_PAUSE_S = 600.0
PAUSE_STEP_S = 5.0
#: nvidia-smi clocks throttle bits that mean the card is actually holding back: SW thermal, HW thermal, HW power brake
THROTTLE_BITS = 0x20 | 0x40 | 0x80

Sample = Tuple[Optional[float], Optional[float], int]
Sampler = Callable[[], Sample]


def stage_telemetry_line(stage: str, temp_c: Optional[float], power_w: Optional[float], throttle: str) -> str:
    """One log line: temperature, power and the throttle reason for a stage."""
    temp = "-" if temp_c is None else f"{float(temp_c):.0f} C"
    power = "-" if power_w is None else f"{float(power_w):.0f} W"
    return f"{stage}: GPU temperature {temp}, power {power}, throttle {throttle or 'none'}"


def telemetry_from_summary(stage: str, gpu: Optional[dict]) -> str:
    """The same line from a :class:`dubber.diag.procs.GpuSampler` summary.  Empty when nothing was sampled."""
    if not gpu or not gpu.get("n"):
        return ""
    return stage_telemetry_line(stage, gpu.get("temp_max_c"), gpu.get("power_max_w"), str(gpu.get("throttle") or "none"))


def read_nvidia() -> Sample:
    """``(temp C, power W, throttle mask)`` from nvidia-smi, or ``(None, None, 0)`` when it is not there."""
    try:
        from dubber.diag.procs import find_nvidia_smi

        smi = find_nvidia_smi()
        if not smi:
            return None, None, 0
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        fields = "temperature.gpu,power.draw,clocks_throttle_reasons.active"
        proc = subprocess.run([smi, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=8, creationflags=flags)
        if proc.returncode != 0 or not proc.stdout.strip():
            fields = "temperature.gpu,power.draw,clocks_event_reasons.active"
            proc = subprocess.run([smi, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, timeout=8, creationflags=flags)
        row = [c.strip() for c in proc.stdout.strip().splitlines()[0].split(",")]
        temp = float(row[0])
        power = float(row[1].split()[0])
        mask = int(row[2], 16)
        return temp, power, mask
    except Exception:  # noqa: BLE001 - no driver, odd output: the job keeps going
        return None, None, 0


class Guard:
    """Tracks one job from its start and decides the pause between blocks."""

    def __init__(self, now: Callable[[], float] = time.monotonic) -> None:
        self._now = now
        self.t0 = now()
        self.temp: Optional[float] = None
        self.power: Optional[float] = None
        self.mask = 0
        self._throttle_since: Optional[float] = None

    def elapsed(self, now: Optional[float] = None) -> float:
        t = self._now() if now is None else now
        return t - self.t0

    def update(self, temp_c: Optional[float], power_w: Optional[float], mask: int, now: Optional[float] = None) -> None:
        t = self._now() if now is None else now
        self.temp = None if temp_c is None else float(temp_c)
        self.power = None if power_w is None else float(power_w)
        self.mask = int(mask or 0)
        if self.mask & THROTTLE_BITS:
            if self._throttle_since is None:
                self._throttle_since = t
        else:
            self._throttle_since = None

    def throttle_seconds(self, now: Optional[float] = None) -> float:
        if self._throttle_since is None:
            return 0.0
        t = self._now() if now is None else now
        return max(0.0, t - self._throttle_since)

    def should_pause(self, now: Optional[float] = None) -> bool:
        """True only after the full-speed window, and only for heat or a throttle that has lasted."""
        t = self._now() if now is None else now
        if self.elapsed(t) < FULL_SPEED_S:
            return False
        if self.temp is not None and self.temp >= HOT_C:
            return True
        return self.throttle_seconds(t) > THROTTLE_HOLD_S

    def cool_enough(self) -> bool:
        """Resume point: 75 C or below, and the throttle reason gone.  Unknown temperature follows the throttle bit."""
        throttling = self._throttle_since is not None
        if self.temp is None:
            return not throttling
        return self.temp <= COOL_C and not throttling

    def pause_after_block(self, sample: Sampler = read_nvidia, sleep: Callable[[float], None] = time.sleep,
                          log: Callable[[str], None] = lambda _m: None, stage: str = "tts") -> float:
        """Sample, log temperature / power / throttle, and pause between blocks when the rule says so.

        Returns how many seconds were spent paused.  One pause stops at 10 minutes so a stuck sensor cannot
        hold the job."""
        temp, power, mask = sample()
        self.update(temp, power, mask)
        if self.temp is not None or self.power is not None or self.mask:
            log(stage_telemetry_line(stage, self.temp, self.power, _mask_text(self.mask)))
        if not self.should_pause():
            return 0.0
        log(f"{stage}: cooling pause (GPU {self.temp if self.temp is not None else '-'} C)")
        paused = 0.0
        while paused < MAX_PAUSE_S:
            sleep(PAUSE_STEP_S)
            paused += PAUSE_STEP_S
            temp, power, mask = sample()
            self.update(temp, power, mask)
            if self.cool_enough():
                log(f"{stage}: cooling pause ended after {paused:.0f} s")
                return paused
        log(f"{stage}: cooling pause stopped after {paused:.0f} s")
        return paused


def _mask_text(mask: int) -> str:
    from dubber.diag.procs import decode_throttle

    if not mask:
        return "none"
    return decode_throttle(mask)
