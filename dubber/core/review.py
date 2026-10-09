"""What deserves a look after the automatic run - shown as small, non-blocking badges on the Characters and Lines tabs.

Qt-free: returns ``{"characters": [(i18n key, params)], "lines": [...]}``; an empty list means the tab needs no attention.  Nothing here
ever stops a dub: the program always finishes with its best choice; the badges only point at what a person may want to check.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from dubber.core import script
from dubber.core.project import Project

Hint = Tuple[str, Dict[str, Any]]
TINY_SPEAKER_S = 4.0          # a "speaker" with less speech than this (and < 3 % of the dialogue) is probably a split of someone else
TINY_SPEAKER_SHARE = 0.03


def too_long_count(p: Project) -> int:
    """Lines that could not be fitted (after the dub: the time-fitting verdict; before it: the length estimate)."""
    if (p.stages.get("tts") or {}).get("done"):
        return sum(1 for ln in p.lines if ln.fit == "too_long" and not ln.keep_original)
    lang = p.settings.get("target_lang", "ru")
    lines = p.lines
    return sum(1 for i, ln in enumerate(lines)
               if script.too_long(ln, lang, next_start=lines[i + 1].start if i + 1 < len(lines) else None))


def diarized(p: Project) -> bool:
    st = p.stages.get("diarization") or {}
    return bool(st.get("done")) and "skipped" not in str(st.get("summary", ""))


def attention(p: Project) -> Dict[str, List[Hint]]:
    out: Dict[str, List[Hint]] = {"characters": [], "lines": []}
    if not p.lines:
        return out
    n_long = too_long_count(p)
    if n_long:
        out["lines"].append(("review.lines_long", {"n": n_long}))
    n_soft = sum(1 for ln in p.lines if ln.softened)
    if n_soft:
        out["lines"].append(("review.lines_softened", {"n": n_soft}))
    if p.settings.get("multi_voice"):
        if (p.stages.get("diarization") or {}).get("done") and not diarized(p):
            out["characters"].append(("review.chars_not_found", {}))
        elif len(p.speakers) > 1:
            total = sum(s.seconds for s in p.speakers) or 1.0
            tiny = [s for s in p.speakers if s.seconds < TINY_SPEAKER_S and s.seconds / total < TINY_SPEAKER_SHARE]
            if tiny:
                out["characters"].append(("review.chars_tiny", {"n": len(tiny)}))
    return out
