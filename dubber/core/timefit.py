"""Time fitting of synthesised lines (research note 04): shift into pauses -> start a little earlier -> Signalsmith Stretch up to
1.15x -> (runner) shorter translation and best-of-N takes -> otherwise the line may run up to 1.25x and is flagged."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from dubber.core.project import Line

MAX_STRETCH = 1.15          # inaudible
HARD_STRETCH = 1.25         # last resort before the line is flagged "too long"
GAP = 0.06                  # minimum silence between two dub lines
EARLY = 0.25                # a line may start this much before the original line


@dataclass
class Placement:
    """Where one dubbed line sits: start in seconds, stretch factor, and a fit verdict."""
    id: int
    start: float
    stretch: float
    verdict: str            # fits | shifted | stretched | too_long

    @property
    def needs_retry(self) -> bool:
        """Whether the verdict is ``too_long`` and a shorter take should be tried."""
        return self.verdict == "too_long"


def is_short_interjection(line: Line) -> bool:
    """Whoa!, Yes!, No., Wait., Boy.: a source slot of at most 1 s, or at most two words."""
    src = (line.text or "").strip()
    if not src:
        return False
    return line.duration <= 1.0 + 1e-6 or len(re.findall(r"\w+", src)) <= 2


def interjection_end(line: Line, next_start: Optional[float]) -> float:
    """How far a short interjection may run before any stretch: min(next start - GAP, end + 0.8 s)."""
    cap = line.end + 0.8
    if next_start is not None:
        cap = min(next_start - GAP, cap)
    return cap


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
        if is_short_interjection(ln):
            p = _place_interjection(ln, d, earliest, start, nxt)
            prev_end = p.start + d / p.stretch
            out.append(p)
            continue
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


def _place_interjection(ln: Line, d: float, earliest: float, start: float, nxt: float) -> Placement:
    """Use the following pause (up to 0.8 s) before stretching. Flag only when the line still overlaps the next one."""
    soft = max(ln.end, interjection_end(ln, nxt))
    if d <= ln.duration and start + d <= soft:
        return Placement(ln.id, start, 1.0, "fits")
    if earliest + d <= soft:
        return Placement(ln.id, max(earliest, min(start, soft - d)), 1.0, "shifted")
    avail = max(1e-3, (nxt - GAP) - earliest)
    if earliest + d / HARD_STRETCH <= nxt - GAP + 1e-6:
        stretch = 1.0 if d <= avail else round(min(d / avail, HARD_STRETCH), 3)
        return Placement(ln.id, earliest, stretch, "stretched" if stretch > 1.001 else "shifted")
    return Placement(ln.id, earliest, HARD_STRETCH, "too_long")


def slot_for(line: Line, lines: Sequence[Line], total: Optional[float] = None) -> float:
    """Room for a line: up to the next line (minus the gap) plus the early start allowance."""
    later = [ln.start for ln in lines if ln.start > line.start]
    nxt = min(later) if later else (total if total is not None else line.end + 3.0)
    return max(line.duration, nxt - GAP - line.start) + EARLY


def best_take(durations: Sequence[float], slot: float, completeness: Optional[Sequence[float]] = None) -> int:
    """Best-of-N, meaning first: among the takes that fit ``slot`` the most complete wording (``completeness``: share of the
    full translation kept, 1.0 = all; default all 1.0), then the longest (most natural pace).  When none fits: the most complete
    wording again, its shortest take - a full line read a little fast is better than a line that lost its meaning (the placement
    may still squeeze it up to HARD_STRETCH and flags it)."""
    n = len(durations)
    comp = list(completeness) if completeness is not None else [1.0] * n
    fitting = [i for i in range(n) if durations[i] <= slot and comp[i] >= 0.7 - 1e-6]
    if fitting:
        return max(fitting, key=lambda i: (round(comp[i], 3), durations[i]))
    top = max(comp)
    near = [i for i in range(n) if comp[i] >= top - 1e-6]
    # a variant that keeps >= 85 % and fits within the hard stretch beats an over-long full take
    hard = [i for i in range(n) if durations[i] <= slot * HARD_STRETCH / MAX_STRETCH and comp[i] >= 0.85]
    if not any(durations[i] <= slot * HARD_STRETCH / MAX_STRETCH for i in near) and hard:
        return max(hard, key=lambda i: (round(comp[i], 3), -durations[i]))
    return min(near, key=lambda i: durations[i])
