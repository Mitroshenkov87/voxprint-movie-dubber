"""Lines from recognised words: sentence boundaries + pauses + speaker turns, length capped at natural break points.

Speech recognition gives a stream of words with timestamps; its own segments are cut by its 30 s decoding windows, not by
meaning, so they often end mid-sentence and run across two speakers (real case: a two-person dialogue came out as 4-8 s chunks
like "...well i got a hot day" / "coming over later so good for you man and i know...").  Lines are rebuilt here from words:

1. hard breaks: a pause longer than ``hard_pause`` (0.6 s) or a change of speaker (diarization turns, when known) - a line never
   spans either;
2. sentence breaks: a word ending in ``. ? ! …`` (not an abbreviation, "A.I." inside a sentence is not an end);
3. a piece still longer than ``max_s`` / ``max_words`` is split at its best inner point: the longest pause, a comma, a
   conjunction, near the middle (recursively).  Without punctuation (some recognisers drop it) the cap is lower and the split
   points are the pauses alone.

Junk tokens of the recogniser (letters of another script inside an English text, ``=#`` debris) are removed first.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

Word = Dict[str, Any]                  # {"w": " text", "start": s, "end": s}

HARD_PAUSE = 0.6
MAX_S = 8.0
MAX_WORDS = 24
MAX_S_UNPUNCTUATED = 6.0
_END = re.compile(r"[.!?…]+[\"'»”)\]]*$")
_ABBREV = {"mr.", "mrs.", "ms.", "dr.", "st.", "vs.", "etc.", "jr.", "sr.", "prof.", "e.g.", "i.e.", "u.s.", "a.m.", "p.m.", "т.е.", "т.д.", "т.п."}
_JUNK_CHARS = re.compile(r"[=#<>\[\]{}|\\^~_*@\ufffd]+")
_MIXED = re.compile(r"[^\W\d_]{2,}\d+[^\W\d_]+|\d+[^\W\d_]{3,}\d")      # "Choi443buz": letters and digits interleaved
_CONJ = {"and", "but", "so", "because", "or", "then", "which", "that", "when", "if", "while", "until", "although",
         "и", "но", "а", "потому", "или", "что", "когда", "если", "und", "aber", "weil", "oder", "dass", "wenn"}
_LATIN_LANGS = {"en", "de", "fr", "es", "it", "pt", "nl", "pl", "cs", "sv", "da", "no", "fi", "tr", "ro", "hu"}
_CYR_LANGS = {"ru", "uk", "be", "bg", "sr", "kk"}


def _allowed(c: str, lang: Optional[str]) -> bool:
    if not c.isalpha() or not lang:
        return True
    if lang in _LATIN_LANGS:
        return ord(c) < 0x250 or 0x1E00 <= ord(c) <= 0x1EFF
    if lang in _CYR_LANGS:
        return ord(c) < 0x250 or 0x400 <= ord(c) <= 0x52F
    return True


def _strip_foreign(word: str, lang: Optional[str]) -> str:
    """Letters of a script the language does not use are recogniser debris: "Wellсем," -> "Well,"; "очему." -> ""."""
    kept = "".join(c for c in word if _allowed(c, lang))
    return kept if any(c.isalpha() for c in kept) or not any(c.isalpha() for c in word) else ""


def clean_words(words: Sequence[Word], lang: Optional[str] = None) -> List[Word]:
    out: List[Word] = []
    for w in words:
        t = _strip_foreign(_JUNK_CHARS.sub("", str(w.get("w", ""))), lang)
        if not t.strip() or _MIXED.search(t):
            continue
        if not any(c.isalnum() for c in t):          # stray punctuation: attach it to the previous word
            if out:
                out[-1] = {**out[-1], "w": out[-1]["w"] + t.strip()}
            continue
        if not t.startswith(" ") and out:              # a word piece (".I." of "A.I.", "-cleaned"): glue it to the word
            out[-1] = {**out[-1], "w": out[-1]["w"] + t, "end": max(out[-1]["end"], float(w["end"]))}
            continue
        out.append({"w": t, "start": float(w["start"]), "end": max(float(w["end"]), float(w["start"]))})
    return out


def words_of(segments: Sequence[Dict[str, Any]], lang: Optional[str] = None) -> List[Word]:
    """All words of the recognised segments, in time order (a segment without word timings becomes one pseudo word)."""
    raw: List[Word] = []
    for s in segments:
        ws = s.get("words") or []
        if ws:
            raw += list(ws)
        elif str(s.get("text", "")).strip():
            raw.append({"w": " " + str(s["text"]).strip(), "start": s["start"], "end": s["end"]})
    raw.sort(key=lambda w: float(w["start"]))
    return collapse_loops(clean_words(raw, lang))


def _norm(w: str) -> str:
    return re.sub(r"[^\w]", "", w.lower())


def collapse_loops(words: List[Word], max_repeat: int = 3) -> List[Word]:
    """A word repeated more than ``max_repeat`` times in a row is a decoding loop of the recogniser (real case: "B.I." x6 over a
    laugh at the end of clip 1): the run is reduced to one word.  "Yeah, yeah, yeah" (3) stays."""
    out: List[Word] = []
    i = 0
    while i < len(words):
        j = i
        while j + 1 < len(words) and _norm(words[j + 1]["w"]) == _norm(words[i]["w"]) and _norm(words[i]["w"]):
            j += 1
        run = j - i + 1
        out += words[i:j + 1] if run <= max_repeat else [words[i]]
        i = j + 1
    return out


def is_sentence_end(word: str, next_word: Optional[str]) -> bool:
    t = word.strip()
    if not _END.search(t):
        return False
    if t.endswith((".", ".\"", ".'")) and not t.endswith((". ", "...")):
        low = t.lower().rstrip("\"'»”)")
        if low in _ABBREV:
            return False
        # an acronym with dots ("A.I.") followed by a lower-case word is inside the sentence
        if re.fullmatch(r"(?:[A-Za-zА-Яа-я]\.){2,}", low.rstrip(",")) and next_word and next_word.strip()[:1].islower():
            return False
    return True


def punctuated(words: Sequence[Word]) -> bool:
    """True when the recogniser wrote sentence punctuation (at least one end mark per ~30 words)."""
    n = len(words)
    ends = sum(1 for w in words if _END.search(w["w"].strip()))
    return n < 12 or ends * 30 >= n


def _speaker_at(t0: float, t1: float, turns: Sequence[Dict[str, Any]]) -> Optional[str]:
    best, best_o = None, 0.0
    for tr in turns:
        o = min(t1, float(tr["end"])) - max(t0, float(tr["start"]))
        if o > best_o:
            best, best_o = str(tr["speaker"]), o
    if best is None:                                   # no overlap: the nearest turn
        mid = (t0 + t1) / 2
        near = min(turns, key=lambda tr: min(abs(mid - float(tr["start"])), abs(mid - float(tr["end"]))), default=None)
        best = str(near["speaker"]) if near else None
    return best


def _split_long(piece: List[Word], max_s: float, max_words: int, min_side: float = 1.0) -> List[List[Word]]:
    dur = piece[-1]["end"] - piece[0]["start"]
    if (dur <= max_s and len(piece) <= max_words) or len(piece) < 4:
        return [piece]
    best_i, best_score = None, -1e9
    for i in range(1, len(piece)):
        left, right = piece[i - 1]["end"] - piece[0]["start"], piece[-1]["end"] - piece[i]["start"]
        if (left < min_side or right < min_side) and len(piece) > 6:
            continue
        gap = max(0.0, piece[i]["start"] - piece[i - 1]["end"])
        prev = piece[i - 1]["w"].strip()
        nxt = piece[i]["w"].strip().lower().strip(",.")
        score = 4.0 * min(gap, 0.6)
        if prev.endswith((",", ";", ":", "—", "–")):
            score += 1.0
        if nxt in _CONJ:
            score += 0.4
        score -= 0.8 * abs(left - right) / max(dur, 1e-6)
        if score > best_score:
            best_i, best_score = i, score
    if best_i is None:
        best_i = len(piece) // 2
    return _split_long(piece[:best_i], max_s, max_words, min_side) + _split_long(piece[best_i:], max_s, max_words, min_side)


def segment(words: Sequence[Word], turns: Optional[Sequence[Dict[str, Any]]] = None, hard_pause: float = HARD_PAUSE,
            max_s: float = MAX_S, max_words: int = MAX_WORDS) -> List[Tuple[float, float, str]]:
    """``(start, end, text)`` per line."""
    ws = [w for w in words if str(w.get("w", "")).strip()]
    if not ws:
        return []
    has_punct = punctuated(ws)
    if not has_punct:
        max_s = min(max_s, MAX_S_UNPUNCTUATED)
    spk = [_speaker_at(w["start"], w["end"], turns) for w in ws] if turns else [None] * len(ws)
    pieces: List[List[Word]] = [[ws[0]]]
    for i in range(1, len(ws)):
        prev, cur = ws[i - 1], ws[i]
        gap = cur["start"] - prev["end"]
        brk = gap > hard_pause or (turns and spk[i] != spk[i - 1]) or \
            (has_punct and is_sentence_end(prev["w"], cur["w"]))
        if brk:
            pieces.append([cur])
        else:
            pieces[-1].append(cur)
    out: List[Tuple[float, float, str]] = []
    lower = not any(c.isupper() for w in ws for c in w["w"])
    for piece in pieces:
        for part in _split_long(piece, max_s, max_words):
            text = re.sub(r"\s+", " ", "".join(w["w"] for w in part)).strip()
            if lower:
                text = re.sub(r"\bi\b", "I", text)
                text = re.sub(r"\bi'(m|ve|ll|d)\b", lambda m: "I'" + m.group(1), text)
            if text and text[0].islower():
                text = text[0].upper() + text[1:]
            out.append((round(part[0]["start"], 3), round(part[-1]["end"], 3), text))
    return out
