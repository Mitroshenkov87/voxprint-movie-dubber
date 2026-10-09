"""Time fitting of synthesised lines (research note 04): shift into pauses -> start a little earlier -> Signalsmith Stretch up to
1.15x -> (runner) shorter translation and best-of-N takes -> otherwise the line may run up to 1.25x and is flagged."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from dubber.core.project import Line

MAX_STRETCH = 1.15          # inaudible
HARD_STRETCH = 1.25         # last resort before the line is flagged "too long"
GAP = 0.06                  # minimum silence between two dub lines
EARLY = 0.25                # a line may start this much before the original line


@dataclass
class Placement:
    id: int
    start: float
    stretch: float
    verdict: str            # fits | shifted | stretched | too_long

    @property
    def needs_retry(self) -> bool:
        return self.verdict == "too_long"


def place(lines: Sequence[Line], seconds: Dict[int, float], total: Optional[float] = None) -> List[Placement]:
    """Placement of every dubbed line (``seconds``: synthesised length by line id).  Lines are processed in time order and never
    overlap each other."""
    order = sorted([ln for ln in lines if ln.id in seconds], key=lambda ln: ln.start)
    out: List[Placement] = []
    prev_end = 0.0
    for i, ln in enumerate(order):
        d = float(seconds[ln.id])
        nxt = order[i + 1].start if i + 1 < len(order) else (total if total is not None else ln.end + 3.0)
        limit = max(ln.end, nxt - GAP)
        earliest = max(ln.start - EARLY, prev_end + GAP if out else 0.0)
        start = max(ln.start, earliest)
        if d <= ln.duration and start + d <= limit:
            p = Placement(ln.id, start, 1.0, "fits")
        elif start + d <= limit:
            p = Placement(ln.id, start, 1.0, "shifted")
        elif earliest + d <= limit:
            p = Placement(ln.id, limit - d, 1.0, "shifted")
        else:
            avail = max(1e-3, limit - earliest)
            f = d / avail
            if f <= MAX_STRETCH:
                p = Placement(ln.id, earliest, round(f, 3), "stretched")
            else:
                p = Placement(ln.id, earliest, round(min(f, HARD_STRETCH), 3), "too_long")
        prev_end = p.start + d / p.stretch
        out.append(p)
    return out


def slot_for(line: Line, lines: Sequence[Line], total: Optional[float] = None) -> float:
    """Room for a line: up to the next line (minus the gap) plus the early start allowance."""
    later = [ln.start for ln in lines if ln.start > line.start]
    nxt = min(later) if later else (total if total is not None else line.end + 3.0)
    return max(line.duration, nxt - GAP - line.start) + EARLY


def best_take(durations: Sequence[float], slot: float) -> int:
    """Best-of-N: among takes that fit without stretching, the longest (most natural pace); else the one needing the least stretch."""
    fitting = [(d, i) for i, d in enumerate(durations) if d <= slot]
    if fitting:
        return max(fitting)[1]
    return min(range(len(durations)), key=lambda i: durations[i])
