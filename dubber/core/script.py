"""Building the script: dialogue windows, lines from recognition or subtitles, translations, speakers, length estimates."""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from dubber.core.project import Line
from dubber.core.subtitles import Cue, clean_text, is_music, sdh_speaker

Window = Tuple[float, float]


# ---------------------------------------------------------------------------------------------- dialogue windows
def dialogue_windows(segments: Iterable[Window], merge_gap: float = 0.6, pad: float = 0.15, min_len: float = 0.3,
                     total: Optional[float] = None) -> List[Window]:
    """Merge VAD segments closer than ``merge_gap`` into windows, pad them; only these windows are separated and dubbed
    (everything outside stays the original soundtrack - research decision 2, "fail-open")."""
    out: List[List[float]] = []
    for s, e in sorted((float(a), float(b)) for a, b in segments):
        if e - s <= 0:
            continue
        s, e = max(0.0, s - pad), e + pad
        if total is not None:
            e = min(e, total)
        if out and s - out[-1][1] < merge_gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(round(a, 3), round(b, 3)) for a, b in out if b - a >= min_len]


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


# ---------------------------------------------------------------------------------------------- lines
def lines_from_asr(segments: Sequence[Dict[str, Any]], max_len: float = 12.0) -> List[Line]:
    """ASR segments -> lines; long segments are split at word gaps (word timestamps) so each line fits one breath."""
    out: List[Line] = []
    for seg in segments:
        words = seg.get("words") or []
        if seg["end"] - seg["start"] <= max_len or len(words) < 4:
            out.append(Line(0, float(seg["start"]), float(seg["end"]), seg.get("text", "").strip(), source="asr"))
            continue
        cur: List[Dict[str, Any]] = []
        for w in words:
            if cur and (w["start"] - cur[0]["start"] > max_len or (w["start"] - cur[-1]["end"] > 0.5 and w["start"] - cur[0]["start"] > max_len / 2)):
                out.append(Line(0, cur[0]["start"], cur[-1]["end"], "".join(x["w"] for x in cur).strip(), source="asr"))
                cur = []
            cur.append(w)
        if cur:
            out.append(Line(0, cur[0]["start"], cur[-1]["end"], "".join(x["w"] for x in cur).strip(), source="asr"))
    return _number(out)


def lines_from_cues(cues: Sequence[Cue], target: bool) -> List[Line]:
    """Subtitle cues -> lines.  ``target``: the cue text is already the translation (subtitles in the dub language)."""
    out = []
    for c in cues:
        music = is_music(c.text)
        text = clean_text(c.text)
        if not text and not music:
            continue
        ln = Line(0, c.start, c.end, source="subs", keep_original=music, kind="music" if music else "speech", tag=sdh_speaker(c.text))
        if target:
            ln.translation = text
        else:
            ln.text = text
        out.append(ln)
    return _number(out)


def _number(lines: List[Line]) -> List[Line]:
    lines.sort(key=lambda ln: ln.start)
    for i, ln in enumerate(lines, 1):
        ln.id = i
    return lines


def attach_text(lines: List[Line], cues: Sequence[Cue], attr: str) -> int:
    """Fill ``attr`` (``text`` / ``translation``) of each line from the cues it overlaps most; returns how many got text."""
    n = 0
    for ln in lines:
        best = [c for c in cues if overlap(ln.start, ln.end, c.start, c.end) > 0.3 * min(ln.duration, c.end - c.start)]
        if best:
            txt = " ".join(clean_text(c.text) for c in best).strip()
            if txt:
                setattr(ln, attr, txt)
                n += 1
            if any(is_music(c.text) for c in best):
                ln.keep_original, ln.kind = True, "music"
            if not ln.tag:
                ln.tag = next((t for t in (sdh_speaker(c.text) for c in best) if t), "")
    return n


def speakers_from_tags(lines: Sequence[Line], min_cover: float = 0.6) -> Dict[int, str]:
    """Speaker per line from subtitle speaker tags ("JOHN: ...") when most speech lines carry one (>= ``min_cover`` and at
    least two names); untagged lines take the nearest tagged line before them (else after).  {} = not enough tags."""
    speech = [ln for ln in lines if not ln.keep_original]
    tagged = [ln for ln in speech if ln.tag]
    if not speech or len(tagged) < min_cover * len(speech) or len({ln.tag.upper() for ln in tagged}) < 2:
        return {}
    out: Dict[int, str] = {}
    last = ""
    for ln in sorted(speech, key=lambda x: x.start):
        if ln.tag:
            last = ln.tag.strip().title()
        out[ln.id] = last
    first = next(ln.tag.strip().title() for ln in sorted(tagged, key=lambda x: x.start))
    return {k: (v or first) for k, v in out.items()}


def snap_to_windows(lines: List[Line], windows: Sequence[Window], max_move: float = 0.6) -> None:
    """Subtitle timing is approximate: snap line edges to the nearest speech window edge when it is close."""
    if not windows:
        return
    starts = np.array([w[0] for w in windows])
    ends = np.array([w[1] for w in windows])
    for ln in lines:
        i = int(np.argmin(np.abs(starts - ln.start)))
        if abs(starts[i] - ln.start) <= max_move:
            ln.start = float(starts[i])
        j = int(np.argmin(np.abs(ends - ln.end)))
        if abs(ends[j] - ln.end) <= max_move and ends[j] > ln.start:
            ln.end = float(ends[j])


# ---------------------------------------------------------------------------------------------- speakers
def assign_speakers(lines: List[Line], turns: Sequence[Dict[str, Any]]) -> None:
    """Each line gets the diarization speaker that covers most of it (turns: ``{"start","end","speaker"}``)."""
    for ln in lines:
        score: Dict[str, float] = {}
        for t in turns:
            o = overlap(ln.start, ln.end, t["start"], t["end"])
            if o > 0:
                score[t["speaker"]] = score.get(t["speaker"], 0.0) + o
        if score:
            ln.speaker = max(score, key=score.get)


def cluster_speakers(features: np.ndarray, threshold: float = 0.35, max_speakers: int = 8) -> List[int]:
    """Agglomerative clustering (cosine, average linkage) of per-line voice fingerprints - the fallback when pyannote is not
    available.  Returns a cluster index per row (0 = the most frequent)."""
    n = len(features)
    if n == 0:
        return []
    if n == 1:
        return [0]
    from scipy.cluster.hierarchy import fcluster, linkage

    f = features - features.mean(axis=0)
    f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)
    z = linkage(f, method="average", metric="cosine")
    labels = fcluster(z, t=threshold, criterion="distance")
    if len(set(labels)) > max_speakers:
        labels = fcluster(z, t=max_speakers, criterion="maxclust")
    order = {lab: i for i, (lab, _) in enumerate(sorted(((lab, (labels == lab).sum()) for lab in set(labels)), key=lambda x: -x[1]))}
    return [order[lab] for lab in labels]


# ---------------------------------------------------------------------------------------------- length estimates
_VOWELS = {"ru": "аеёиоуыэюя", "en": "aeiouy", "de": "aeiouyäöü"}
SYLLABLES_PER_S = {"ru": 5.6, "en": 5.0, "de": 5.2}


def syllables(text: str, lang: str) -> int:
    v = _VOWELS.get(lang, "aeiouy")
    t = text.lower()
    if lang in ("en", "de"):
        return max(1, len(re.findall(f"[{v}]+", t)))
    return max(1, sum(t.count(c) for c in v))


def estimate_seconds(text: str, lang: str) -> float:
    """Spoken length of ``text`` at a normal dubbing pace (+ pauses at punctuation)."""
    pauses = len(re.findall(r"[,;:]", text)) * 0.12 + len(re.findall(r"[.!?…]", text.rstrip(".!?… "))) * 0.25
    return syllables(text, lang) / SYLLABLES_PER_S.get(lang, 5.2) + pauses


def too_long(line: Line, lang: str, slack: float = 1.15, next_start: Optional[float] = None) -> bool:
    """Flag for the Script table: the translation will not fit its slot even after squeezing (shown before any synthesis)."""
    if line.keep_original or not line.translation.strip():
        return False
    room = line.duration
    if next_start is not None:
        room = max(room, min(next_start - 0.1, line.end + 0.8) - line.start)
    return estimate_seconds(line.translation, lang) > room * slack


_FILLERS = {
    "ru": ["ну", "вот", "знаешь", "знаете", "в общем", "так сказать", "как бы", "просто", "вообще-то", "же", "ведь", "ладно", "слушай", "послушай"],
    "en": ["well", "you know", "i mean", "just", "actually", "really", "so", "like", "okay", "oh"],
    "de": ["also", "na ja", "eigentlich", "halt", "eben", "doch", "ja", "mal"],
}


def shorten(text: str, lang: str) -> List[str]:
    """Shorter variants of a translation, most conservative first: drop parentheticals, filler words, repeated words,
    then the last clause.  The time fitter tries them in order (research note 04, step "shorten the translation")."""
    out: List[str] = []

    def add(t: str) -> None:
        t = re.sub(r"\s+([,.!?…])", r"\1", re.sub(r"\s+", " ", t)).strip(" ,;")
        t = re.sub(r"^[,;]\s*", "", t)
        if t and t != text and t not in out:
            out.append(t[0].upper() + t[1:] if text[:1].isupper() else t)
    t = re.sub(r"\s*[(\[][^)\]]*[)\]]", "", text)
    add(t)
    for f in sorted(_FILLERS.get(lang, []), key=len, reverse=True):
        t = re.sub(rf"(?i)(^|[\s,])({re.escape(f)})(?=[\s,.!?]|$),?", r"\1", t)
    add(t)
    t = re.sub(r"(?i)\b(\w+)(\s+\1\b)+", r"\1", t)
    add(t)
    parts = re.split(r"(?<=[,;:])\s+", t)
    if len(parts) > 1:
        add(" ".join(parts[:-1]).rstrip(",;:") + ".")
    return out
