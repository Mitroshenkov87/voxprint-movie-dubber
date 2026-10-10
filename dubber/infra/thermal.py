"""Cooling pauses between blocks, and the GPU line written for each stage.

Suite thermal rule (project-notes suite/COLLABORATION.md, section 3): full speed for the first ~2.5 h of a job;
after that, pause between batches while the GPU 5-min median is >= 83 C or it keeps throttling for > 60 s;
resume at <= 75 C.  Hardware longevity beats speed.

The 2.5 hours count from the start of the whole dub (``job_started`` in the stage config), not from the start of
the speech stage.  The median is taken over the temperature samples of the last five minutes (one per block and
one every few seconds during a pause).  There is no user-facing mode: the rule runs on its own.  A missing sensor
never pauses the job.
"""
from __future__ import annotations

import os
import statistics
import subprocess
import time
from collections import deque
from typing import Callable, Deque, Optional, Tuple

FULL_SPEED_S = 2.5 * 3600
#: pause while the median GPU temperature over MEDIAN_WINDOW_S is at or above HOT_C
HOT_C = 83.0
MEDIAN_WINDOW_S = 300.0
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

    def __init__(self, now: Callable[[], float] = time.monotonic, started_ago_s: float = 0.0) -> None:
        """``started_ago_s``: how long the dub has already been running when this stage starts (its full-speed window
        is shared with the stages before)."""
        self._now = now
        self.t0 = now() - max(0.0, float(started_ago_s or 0.0))
        self.temp: Optional[float] = None
        self._temps: Deque[Tuple[float, float]] = deque()
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
        if self.temp is not None:
            self._temps.append((t, self.temp))
        while self._temps and t - self._temps[0][0] > MEDIAN_WINDOW_S:
            self._temps.popleft()
        if self.mask & THROTTLE_BITS:
            if self._throttle_since is None:
                self._throttle_since = t
        else:
            self._throttle_since = None

    def median_temp(self) -> Optional[float]:
        """Median of the temperature samples of the last five minutes; None when there is none."""
        if not self._temps:
            return None
        return float(statistics.median(v for _t, v in self._temps))

    def throttle_seconds(self, now: Optional[float] = None) -> float:
        if self._throttle_since is None:
            return 0.0
        t = self._now() if now is None else now
        return max(0.0, t - self._throttle_since)

    def should_pause(self, now: Optional[float] = None) -> bool:
        """True only after the full-speed window, and only for a hot 5-min median or a throttle that has lasted."""
        t = self._now() if now is None else now
        if self.elapsed(t) < FULL_SPEED_S:
            return False
        median = self.median_temp()
        if median is not None and median >= HOT_C:
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
        median = self.median_temp()
        log(f"{stage}: cooling pause (GPU {self.temp if self.temp is not None else '-'} C, "
            f"5-min median {'-' if median is None else f'{median:.0f}'} C)")
        paused = 0.0
        while paused < MAX_PAUSE_S:
            sleep(PAUSE_STEP_S)
            paused += PAUSE_STEP_S
            temp, power, mask = sample()
            self.update(temp, power, mask)
            if self.cool_enough():
                self._temps.clear()                    # a fresh median after the pause: the hot samples are history
                log(f"{stage}: cooling pause ended after {paused:.0f} s")
                return paused
        log(f"{stage}: cooling pause stopped after {paused:.0f} s")
        return paused


def _mask_text(mask: int) -> str:
    from dubber.diag.procs import decode_throttle

    if not mask:
        return "none"
    return decode_throttle(mask)
