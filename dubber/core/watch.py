"""Watch mode logic (no Qt): when the "Watch" button becomes available, and when the player must wait for the dubbing.

Decision (2026-10-08): not strict real time.  The first ``READY_AHEAD_S`` (5 min) are dubbed before Watch becomes available; the
user starts it by hand; the dubbing keeps running ahead; if playback catches up with the dubbed part the player pauses ("buffering")
and resumes once ``RESUME_AHEAD_S`` are ready again (or the film is finished).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

READY_AHEAD_S = 300.0
LOW_WATER_S = 2.0
RESUME_AHEAD_S = 45.0


@dataclass
class WatchState:
    """Playback position against how far the dub has been mixed."""
    total: float
    dubbed_until: float = 0.0
    finished: bool = False
    playing: bool = False
    buffering: bool = False

    @property
    def ready(self) -> bool:
        """The Watch button is enabled."""
        return self.finished or self.dubbed_until >= min(READY_AHEAD_S, max(0.0, self.total - 0.5))

    def update(self, position: float, dubbed_until: float, finished: bool = False) -> str:
        """Feed the player position and the dubbing progress; returns the action for the player: play | pause | none."""
        self.dubbed_until, self.finished = max(self.dubbed_until, dubbed_until), finished or self.finished
        if not self.playing:
            return "none"
        if self.buffering:
            if self.finished or self.dubbed_until >= min(self.total, position + RESUME_AHEAD_S):
                self.buffering = False
                return "play"
            return "none"
        if not self.finished and position >= self.dubbed_until - LOW_WATER_S:
            self.buffering = True
            return "pause"
        return "none"

    def start(self) -> bool:
        """Begin playback when enough of the film is dubbed. Returns False while Watch stays disabled."""
        if not self.ready:
            return False
        self.playing, self.buffering = True, False
        return True

    def stop(self) -> None:
        """Stop playback and clear buffering."""
        self.playing = self.buffering = False


class Eta:
    """Remaining time from the dubbed seconds per wall-clock second (exponential moving average)."""

    def __init__(self, alpha: float = 0.3) -> None:
        self.alpha = alpha
        self.rate = 0.0
        self._last: Optional[Tuple[float, float]] = None

    def update(self, now: float, done_s: float, total_s: float) -> float:
        """Remaining wall-clock seconds after this sample, or -1 until a dubbing rate is known."""
        if self._last is not None:
            dt, dd = now - self._last[0], done_s - self._last[1]
            if dt > 0 and dd >= 0:
                r = dd / dt
                self.rate = r if self.rate == 0 else self.alpha * r + (1 - self.alpha) * self.rate
        self._last = (now, done_s)
        if self.rate <= 0:
            return -1.0
        return max(0.0, (total_s - done_s) / self.rate)
