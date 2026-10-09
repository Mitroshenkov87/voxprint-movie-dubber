"""Subtitles: parsing (SRT / WebVTT / ASS), clean-up, song detection, sidecar search and the online services.

Search order (research note 05): subtitles inside the file -> files next to the film -> SubDL (the user's API key) ->
OpenSubtitles (the user's API key, optional login).  The HTTP layer is injectable (``http``) so everything is testable offline.
Only text and timing are used; downloaded files stay inside the project folder.
"""
from __future__ import annotations

import io
import json
import re
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

USER_AGENT = "VoxprintMovieDubber v0.1"
SUB_EXTS = (".srt", ".vtt", ".ass", ".ssa")
LANG_ALIASES = {"ru": ("ru", "rus", "russian", "рус"), "en": ("en", "eng", "english"), "de": ("de", "ger", "deu", "german")}


@dataclass
class Cue:
    start: float
    end: float
    text: str


# ---------------------------------------------------------------------------------------------- parsing
_TS = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})")


def _ts(s: str) -> float:
    m = _TS.search(s)
    if not m:
        raise ValueError(f"bad timestamp: {s!r}")
    h, mi, se, ms = m.groups()
    return int(h or 0) * 3600 + int(mi) * 60 + int(se) + int(ms.ljust(3, "0")) / 1000.0


def parse_srt(text: str) -> List[Cue]:
    cues: List[Cue] = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n").strip("\ufeff \n")):
        lines = [ln for ln in block.split("\n") if ln.strip()]
        idx = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if idx is None:
            continue
        a, b = lines[idx].split("-->", 1)
        try:
            cues.append(Cue(_ts(a), _ts(b), "\n".join(lines[idx + 1:]).strip()))
        except ValueError:
            continue
    return cues


def parse_vtt(text: str) -> List[Cue]:
    return parse_srt(re.sub(r"^WEBVTT.*?\n", "", text.lstrip("\ufeff"), count=1, flags=re.S))


def parse_ass(text: str) -> List[Cue]:
    cues: List[Cue] = []
    fmt: List[str] = []
    for ln in text.replace("\r", "").split("\n"):
        if ln.startswith("Format:") and not fmt:
            cand = [x.strip().lower() for x in ln[7:].split(",")]
            if "start" in cand and "text" in cand:
                fmt = cand
        elif ln.startswith("Dialogue:") and fmt:
            parts = ln[9:].split(",", len(fmt) - 1)
            if len(parts) < len(fmt):
                continue
            d = dict(zip(fmt, parts))
            body = re.sub(r"\{[^}]*\}", "", d["text"]).replace("\\N", "\n").replace("\\n", "\n").strip()
            try:
                cues.append(Cue(_ts(d["start"].strip() + "0"), _ts(d["end"].strip() + "0"), body))
            except ValueError:
                continue
    return sorted(cues, key=lambda c: c.start)


def parse_any(text: str, name: str = "") -> List[Cue]:
    low = name.lower()
    if low.endswith((".ass", ".ssa")) or "[Script Info]" in text[:2000]:
        return parse_ass(text)
    if low.endswith(".vtt") or text.lstrip("\ufeff").startswith("WEBVTT"):
        return parse_vtt(text)
    return parse_srt(text)


def decode_bytes(data: bytes) -> str:
    for enc in ("utf-8-sig", "cp1251", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def load_file(path: Path) -> List[Cue]:
    return parse_any(decode_bytes(Path(path).read_bytes()), str(path))


def _fmt(t: float) -> str:
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def write_srt(cues: Sequence[Cue], path: Path) -> None:
    out = [f"{i}\n{_fmt(c.start)} --> {_fmt(c.end)}\n{c.text}\n" for i, c in enumerate(cues, 1)]
    Path(path).write_text("\n".join(out), encoding="utf-8", newline="\n")


# ---------------------------------------------------------------------------------------------- clean-up
MUSIC_MARKS = ("♪", "♫", "♬")
_MUSIC_WORDS = re.compile(r"^\W*[\[(](music|song|singing|sings|music playing|музыка|песня|поёт|поет|Musik|Gesang)[^\])]*[\])]\W*$", re.I)
_SDH = re.compile(r"[\[(][^\])]{1,60}[\])]")
_SPEAKER = re.compile(r"^\s*[A-ZА-ЯЁ][A-ZА-ЯЁ .'-]{1,24}:\s*")
_TAGS = re.compile(r"</?[a-zA-Z][^>]*>|\{\\[^}]*\}")


def is_music(text: str) -> bool:
    """Song lyrics / music cue: the line keeps the original audio (decision: songs are never dubbed)."""
    t = text.strip()
    return any(m in t for m in MUSIC_MARKS) or bool(_MUSIC_WORDS.match(t))


def clean_text(text: str) -> str:
    """Remove formatting tags, SDH descriptions ``[door slams]``, speaker labels ``JOHN:`` and dialogue dashes."""
    t = _TAGS.sub("", text)
    t = _SDH.sub("", t)
    parts = []
    for ln in t.split("\n"):
        ln = _SPEAKER.sub("", ln).strip()
        ln = re.sub(r"^[-–—]\s*", "", ln)
        if ln:
            parts.append(ln)
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def sdh_speaker(text: str) -> str:
    """``JOHN: ...`` -> ``JOHN`` (a hint for the speaker detection), else ''."""
    m = _SPEAKER.match(_TAGS.sub("", text))
    return m.group(0).strip().rstrip(":").strip() if m else ""


# ---------------------------------------------------------------------------------------------- timing helpers
def estimate_offset(cues: Sequence[Cue], windows: Sequence[Tuple[float, float]], search_s: float = 8.0, step: float = 0.1) -> float:
    """Shift (seconds) that best aligns the cues with the detected speech windows (subtitles from another release are often
    offset by a constant).  0 when nothing is better."""
    if not cues or not windows:
        return 0.0
    import numpy as np

    end = max(max(c.end for c in cues), max(w[1] for w in windows)) + search_s + 1
    res = 0.05
    n = int(end / res) + 1
    a = np.zeros(n, dtype=np.float32)
    b = np.zeros(n, dtype=np.float32)
    for c in cues:
        a[int(max(0, c.start) / res):int(max(0, c.end) / res)] = 1
    for s, e in windows:
        b[int(s / res):int(e / res)] = 1
    best, best_score = 0.0, float((a * b).sum())
    k = int(search_s / step)
    for i in range(-k, k + 1):
        sh = int(round(i * step / res))
        score = float((np.roll(a, sh) * b).sum()) if sh else best_score
        if score > best_score * 1.02:
            best, best_score = i * step, score
    return round(best, 2)


def shift(cues: Sequence[Cue], offset: float) -> List[Cue]:
    return [Cue(max(0.0, c.start + offset), max(0.0, c.end + offset), c.text) for c in cues]


# ---------------------------------------------------------------------------------------------- sidecar files
def lang_matches(name: str, lang: str) -> bool:
    toks = re.split(r"[ ._\-\[\]()]+", name.lower())
    return any(a in toks for a in LANG_ALIASES.get(lang, (lang,)))


def sidecar_candidates(video: Path, lang: str) -> List[Path]:
    """Subtitle files next to the film (same stem first), the ones naming ``lang`` first; also a ``Subs``/``Subtitles`` folder."""
    video = Path(video)
    folders = [video.parent] + [video.parent / d for d in ("Subs", "subs", "Subtitles", "subtitles")]
    found: List[Path] = []
    for d in folders:
        try:
            for p in d.iterdir():
                if p.suffix.lower() in SUB_EXTS and p.is_file():
                    found.append(p)
        except OSError:
            continue
    stem = video.stem.lower()

    def rank(p: Path) -> Tuple[int, int, str]:
        return (0 if lang_matches(p.stem, lang) else 1, 0 if p.stem.lower().startswith(stem) else 1, p.name)
    return sorted([p for p in found if lang_matches(p.stem, lang) or p.stem.lower() == stem], key=rank)


# ---------------------------------------------------------------------------------------------- online services
Http = Callable[[str, str, Dict[str, str], Optional[bytes]], bytes]      # (method, url, headers, body) -> response body


def default_http(method: str, url: str, headers: Dict[str, str], body: Optional[bytes] = None, timeout: float = 30.0) -> bytes:
    req = urllib.request.Request(url, data=body, method=method, headers={"User-Agent": USER_AGENT, **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - fixed https hosts
        return r.read()


def guess_title(video: Path) -> Tuple[str, Optional[int], Optional[Tuple[int, int]]]:
    """``The.Movie.2019.1080p.mkv`` -> ("The Movie", 2019, None); ``Show.S01E02`` -> ("Show", None, (1, 2))."""
    stem = Path(video).stem
    ep = re.search(r"[Ss](\d{1,2})[Ee](\d{1,3})", stem)
    year = re.search(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", stem)
    cut = min([m.start() for m in (ep, year) if m] or [len(stem)])
    title = re.sub(r"[._]+", " ", stem[:cut]).strip(" -([")
    return title, int(year.group(1)) if year else None, (int(ep.group(1)), int(ep.group(2))) if ep else None


def _first_sub_in_zip(data: bytes) -> Tuple[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = [n for n in z.namelist() if n.lower().endswith(SUB_EXTS) and not n.startswith("__MACOSX")]
        if not names:
            raise ValueError("no subtitle file in the archive")
        info = z.getinfo(names[0])
        if info.file_size > 20 * 1024 * 1024:
            raise ValueError("subtitle file too large")
        return Path(names[0]).name, z.read(names[0])


def subdl_search(api_key: str, video: Path, lang: str, http: Http = default_http) -> Optional[Tuple[str, bytes]]:
    """SubDL API v1 (``api.subdl.com``; its terms allow a user's own key inside an application)."""
    title, year, ep = guess_title(video)
    q: Dict[str, Any] = {"api_key": api_key, "film_name": title, "languages": lang.upper(), "subs_per_page": 10}
    if year:
        q["year"] = year
    if ep:
        q.update(type="tv", season_number=ep[0], episode_number=ep[1])
    data = json.loads(http("GET", "https://api.subdl.com/api/v1/subtitles?" + urllib.parse.urlencode(q), {}, None) or b"{}")
    subs = data.get("subtitles") or []
    if not data.get("status") or not subs:
        return None
    url = subs[0].get("url") or ""
    if not url:
        return None
    blob = http("GET", "https://dl.subdl.com" + url if url.startswith("/") else url, {}, None)
    return _first_sub_in_zip(blob)


def opensubtitles_search(api_key: str, video: Path, lang: str, http: Http = default_http, username: str = "",
                         password: str = "") -> Optional[Tuple[str, bytes]]:
    """OpenSubtitles REST API (``api.opensubtitles.com``): search, then ``/download`` for a link (login optional, raises quota)."""
    base = "https://api.opensubtitles.com/api/v1"
    headers = {"Api-Key": api_key, "Content-Type": "application/json", "Accept": "application/json"}
    if username and password:
        tok = json.loads(http("POST", base + "/login", headers, json.dumps({"username": username, "password": password}).encode()) or b"{}")
        if tok.get("token"):
            headers["Authorization"] = "Bearer " + tok["token"]
    title, year, ep = guess_title(video)
    q: Dict[str, Any] = {"query": title, "languages": lang}
    if year:
        q["year"] = year
    if ep:
        q.update(season_number=ep[0], episode_number=ep[1])
    data = json.loads(http("GET", base + "/subtitles?" + urllib.parse.urlencode(q), headers, None) or b"{}")
    for item in data.get("data") or []:
        files = (item.get("attributes") or {}).get("files") or []
        if not files:
            continue
        link = json.loads(http("POST", base + "/download", headers, json.dumps({"file_id": files[0]["file_id"]}).encode()) or b"{}")
        if link.get("link"):
            return (link.get("file_name") or "opensubtitles.srt"), http("GET", link["link"], {}, None)
    return None


@dataclass
class Found:
    source: str            # embedded | sidecar | subdl | opensubtitles
    cues: List[Cue]
    label: str = ""


def find_online(video: Path, lang: str, settings: Dict[str, Any], http: Http = default_http,
                log: Callable[[str], None] = lambda m: None) -> Optional[Found]:
    """SubDL, then OpenSubtitles, with the user's keys from Settings; network errors only log."""
    attempts = []
    if settings.get("subdl_key"):
        attempts.append(("subdl", lambda: subdl_search(settings["subdl_key"], video, lang, http)))
    if settings.get("opensubtitles_key"):
        attempts.append(("opensubtitles", lambda: opensubtitles_search(settings["opensubtitles_key"], video, lang, http,
                                                                       settings.get("opensubtitles_user", ""),
                                                                       settings.get("opensubtitles_password", ""))))
    for name, fn in attempts:
        try:
            got = fn()
        except Exception as exc:  # noqa: BLE001 - offline, quota, bad key: try the next source
            log(f"{name}: {type(exc).__name__}: {str(exc)[:160]}")
            continue
        if got:
            fname, data = got
            cues = parse_any(decode_bytes(data), fname)
            if cues:
                return Found(name, cues, fname)
    return None
