"""A dubbing project: one folder per film, resumable, with a cache per stage.

Layout::

    <project>/project.json      settings (source, languages, multi-voice, voices, volumes) + stage states (input hash, timing)
    <project>/lines.json        the script: one entry per line (time, original, translation, speaker, flags)
    <project>/speakers.json     characters and their voices
    <project>/audio/            extracted soundtrack (16 kHz mono for analysis, 48 kHz stereo for the mix)
    <project>/stems/            separated speech / background
    <project>/tts/              synthesised lines (line_<id>.wav, cached by text+voice hash)
    <project>/watch/            mixed chunks for Watch mode (chunk_<n>.wav, CHUNK_S seconds each)
    <project>/out/              results (preview.wav, dub_track.wav)

Every write goes through a temp file with a unique name and ``os.replace`` (an interrupted run never leaves a half-written JSON).
A stage is skipped when it is marked done with the same input hash (:meth:`Project.stage_fresh`).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

SCHEMA = 1
CHUNK_S = 30.0


@dataclass
class Line:
    id: int
    start: float
    end: float
    text: str = ""                    # original language
    translation: str = ""             # target language (from subtitles or MT; editable)
    spoken: str = ""                  # what the dub actually says (a shortened translation when the line did not fit)
    speaker: str = "S1"
    keep_original: bool = False       # songs / music / shouting: the original stays
    kind: str = "speech"              # speech | music
    source: str = ""                  # asr | subs | manual
    audio: str = ""                   # synthesised file (relative to the project)
    audio_s: float = 0.0
    place_start: float = -1.0         # where the dub line is placed (after time fitting)
    stretch: float = 1.0              # time factor applied (>1 = faster)
    fit: str = ""                     # fits | shifted | stretched | too_long | kept
    edited: bool = False
    tag: str = ""                     # speaker name from the subtitles ("JOHN: ...", SDH), if any
    softened: str = ""                # the translation before the profanity filter changed it ("" = unchanged)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Line":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class Voice:
    kind: str = "clone"               # clone (own lines) | library (shared library) | actor (blend of both) | auto (by key role)
    id: str = ""                      # library voice id

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "Voice":
        d = d or {}
        return cls(kind=str(d.get("kind") or "clone"), id=str(d.get("id") or ""))


@dataclass
class Speaker:
    id: str
    name: str = ""
    voice: Voice = field(default_factory=lambda: Voice("auto"))   # auto: key character -> actor-like, others -> library match
    ref_audio: str = ""               # reference clip built from the speaker's lines (relative)
    ref_text: str = ""
    seconds: float = 0.0              # total speech of the speaker
    key: Optional[bool] = None        # key character: None = automatic (share of the dialogue), True/False = set by the user

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Speaker":
        return cls(id=str(d["id"]), name=str(d.get("name") or d["id"]), voice=Voice.from_dict(d.get("voice")),
                   ref_audio=str(d.get("ref_audio") or ""), ref_text=str(d.get("ref_text") or ""), seconds=float(d.get("seconds") or 0.0),
                   key=d.get("key") if isinstance(d.get("key"), bool) else None)


DEFAULT_SETTINGS: Dict[str, Any] = {
    "source": "",
    "source_lang": "en",
    "target_lang": "ru",
    "multi_voice": False,             # OFF: one voice for every line, diarization skipped
    "single_voice": {"kind": "clone", "id": ""},
    "original_volume": 0.15,          # original speech under the dub (voice-over style); 0 = removed
    "audio_track": 0,
    "subtitle_choice": "auto",        # auto | none | <path>
    "output_format": "same",          # same | mkv | mp4
    "actor_weight": 0.7,
    "key_share": 0.2,                 # a speaker with >= this share of the dialogue time is a key character              # actor-like voices: share of the actor's own timbre (dubber.core.actor_voice)
    "profanity": "keep",              # keep (as in the original) | soften (no mat; see dubber.core.profanity)
}


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    for i in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:                  # Windows: a reader has the file open for a moment
            time.sleep(0.05 * (i + 1))
    tmp.unlink(missing_ok=True)
    raise PermissionError(f"cannot replace {path}")


def write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=1))


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return default


def write_json_gz(path: Path, data: Any) -> None:
    """Reproducible gzip (mtime=0, no file name in the header): the same data gives the same bytes."""
    raw = json.dumps(data, ensure_ascii=False, sort_keys=True).encode("utf-8")
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    with open(tmp, "wb") as fh, gzip.GzipFile(filename="", mode="wb", fileobj=fh, mtime=0) as gz:
        gz.write(raw)
    os.replace(tmp, path)


def read_json_gz(path: Path, default: Any = None) -> Any:
    try:
        with gzip.open(path, "rb") as gz:
            return json.loads(gz.read().decode("utf-8"))
    except (OSError, ValueError, EOFError):
        return default


def hash_of(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def file_fingerprint(path: Path) -> str:
    try:
        st = Path(path).stat()
        return f"{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        return "missing"


class Project:
    def __init__(self, folder: Path) -> None:
        self.folder = Path(folder)
        data = read_json(self.folder / "project.json", {}) or {}
        self.settings: Dict[str, Any] = {**DEFAULT_SETTINGS, **(data.get("settings") or {})}
        self.stages: Dict[str, Dict[str, Any]] = dict(data.get("stages") or {})
        self.lines: List[Line] = [Line.from_dict(d) for d in read_json(self.folder / "lines.json", []) or []]
        self.speakers: List[Speaker] = [Speaker.from_dict(d) for d in read_json(self.folder / "speakers.json", []) or []]

    # ------------------------------------------------------------------ creation
    @classmethod
    def create(cls, folder: Path, source: Path, **settings: Any) -> "Project":
        p = cls(folder)
        p.settings.update(settings)
        p.settings["source"] = str(Path(source).resolve())
        p.save()
        return p

    @staticmethod
    def folder_for(source: Path, root: Path) -> Path:
        """``<root>/<film name>-<short hash of the full path>`` - the same film always maps to the same (resumable) project."""
        src = Path(source).resolve()
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in src.stem)[:60] or "film"
        return Path(root) / f"{safe}-{hashlib.sha1(str(src).encode('utf-8')).hexdigest()[:8]}"

    # ------------------------------------------------------------------ files
    @property
    def source(self) -> Path:
        return Path(self.settings["source"])

    def path(self, *parts: str) -> Path:
        p = self.folder.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def rel(self, p: Path) -> str:
        return Path(p).resolve().relative_to(self.folder.resolve()).as_posix()

    def abs(self, rel: str) -> Path:
        return self.folder / rel

    def save(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        write_json(self.folder / "project.json", {"schema": SCHEMA, "settings": self.settings, "stages": self.stages})
        write_json(self.folder / "lines.json", [asdict(ln) for ln in self.lines])
        write_json(self.folder / "speakers.json", [{**asdict(s), "voice": asdict(s.voice)} for s in self.speakers])

    # ------------------------------------------------------------------ stage cache
    def stage_fresh(self, key: str, inputs: str) -> bool:
        st = self.stages.get(key) or {}
        return bool(st.get("done")) and st.get("inputs") == inputs

    def mark_done(self, key: str, inputs: str, seconds: float, **extra: Any) -> None:
        self.stages[key] = {"done": True, "inputs": inputs, "seconds": round(seconds, 2), "at": time.time(), **extra}
        self.save()

    def invalidate(self, *keys: str) -> None:
        for k in keys:
            self.stages.pop(k, None)
        self.save()

    # ------------------------------------------------------------------ script helpers
    def line(self, line_id: int) -> Optional[Line]:
        return next((ln for ln in self.lines if ln.id == line_id), None)

    def speaker(self, sid: str) -> Optional[Speaker]:
        return next((s for s in self.speakers if s.id == sid), None)

    def dub_lines(self) -> List[Line]:
        return [ln for ln in self.lines if not ln.keep_original and ln.translation.strip()]

    def merge_speakers(self, keep: str, absorb: Iterable[str]) -> None:
        absorb = [a for a in absorb if a != keep]
        for ln in self.lines:
            if ln.speaker in absorb:
                ln.speaker = keep
        self.speakers = [s for s in self.speakers if s.id not in absorb]
        sp = self.speaker(keep)
        if sp:
            sp.seconds = sum(ln.duration for ln in self.lines if ln.speaker == keep)
            sp.ref_audio, sp.ref_text = "", ""             # rebuilt from the merged lines
        self.invalidate("voices", "tts", "fit", "mix", "mux")

    def reassign(self, line_id: int, speaker: str) -> None:
        ln = self.line(line_id)
        if ln and ln.speaker != speaker:
            ln.speaker = speaker
            ln.audio, ln.audio_s = "", 0.0
            if not self.speaker(speaker):
                self.speakers.append(Speaker(speaker, speaker))
            self.invalidate("tts", "fit", "mix", "mux")

    def is_key(self, sp: "Speaker") -> bool:
        """Key character (gets an actor-like voice in "auto"): set by the user, else >= ``key_share`` of all dialogue time."""
        if sp.key is not None:
            return sp.key
        total = sum(s.seconds for s in self.speakers)
        return total > 0 and sp.seconds / total >= float(self.settings.get("key_share", 0.2))

    def voice_for(self, line: Line) -> Voice:
        if not self.settings.get("multi_voice"):
            return Voice.from_dict(self.settings.get("single_voice"))
        sp = self.speaker(line.speaker)
        return sp.voice if sp else Voice()
