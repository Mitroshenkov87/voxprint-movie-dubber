"""Click-burst trigger for a short hidden picture.

Six clicks inside two seconds fire it once.  The sibling Voxprint app can copy this module: it depends only on
the standard library, draws nothing, and does not know about the splash screen.
"""
from __future__ import annotations

import time
from typing import Callable, List

CLICKS = 6
WINDOW_S = 2.0
SHOW_S = 3.0


class ClickBurst:
    """True once, the first time ``CLICKS`` arrive within ``WINDOW_S``.  Later clicks stay False."""

    def __init__(self, clicks: int = CLICKS, window_s: float = WINDOW_S, clock: Callable[[], float] = time.monotonic) -> None:
        self.clicks = clicks
        self.window_s = window_s
        self._clock = clock
        self._times: List[float] = []
        self.fired = False

    def click(self, now: float | None = None) -> bool:
        if self.fired:
            return False
        t = self._clock() if now is None else now
        self._times = [x for x in self._times if t - x <= self.window_s]
        self._times.append(t)
        if len(self._times) >= self.clicks:
            self.fired = True
            return True
        return False
