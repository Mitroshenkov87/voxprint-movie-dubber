"""What is inside the film: duration, audio tracks, embedded subtitles (ffprobe); extraction of text subtitles and audio."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dubber import ffmpeg
from dubber.core import subtitles as subs

TEXT_SUB_CODECS = {"subrip", "srt", "ass", "ssa", "mov_text", "webvtt", "text"}
LANG3 = {"rus": "ru", "eng": "en", "deu": "de", "ger": "de", "ru": "ru", "en": "en", "de": "de"}


@dataclass
class Track:
    index: int              # index among streams of its type (ffmpeg "0:a:N" / "0:s:N")
    stream: int             # absolute stream index
    codec: str
    lang: str = ""
    title: str = ""
    channels: Optional[int] = None

    @property
    def lang2(self) -> str:
        return LANG3.get(self.lang.lower(), self.lang.lower()[:2])

    def label(self) -> str:
        bits = [self.lang or "und", self.codec]
        if self.channels:
            bits.append(f"{self.channels} ch")
        if self.title:
            bits.append(self.title)
        return " · ".join(bits)


@dataclass
class MediaInfo:
    path: Path
    duration: float = 0.0
    video_codec: str = ""
    audio: List[Track] = field(default_factory=list)
    subtitles: List[Track] = field(default_factory=list)

    @property
    def text_subtitles(self) -> List[Track]:
        return [t for t in self.subtitles if t.codec in TEXT_SUB_CODECS]

    @property
    def container(self) -> str:
        return self.path.suffix.lower().lstrip(".")


def probe(path: Path) -> MediaInfo:
    data = ffmpeg.probe(Path(path))
    info = MediaInfo(Path(path), float(data.get("duration") or 0.0))
    a = s = 0
    for st in data["streams"]:                      # type: ignore[index]
        kind = st.get("type")
        if kind == "video" and not info.video_codec:
            info.video_codec = str(st.get("codec") or "")
        elif kind == "audio":
            info.audio.append(Track(a, int(st.get("index") or 0), str(st.get("codec") or ""), str(st.get("lang") or ""),
                                    str(st.get("title") or ""), st.get("channels")))
            a += 1
        elif kind == "subtitle":
            info.subtitles.append(Track(s, int(st.get("index") or 0), str(st.get("codec") or ""), str(st.get("lang") or ""),
                                        str(st.get("title") or "")))
            s += 1
    return info


def extract_subtitle(path: Path, track: Track, out_srt: Path) -> List[subs.Cue]:
    ffmpeg.run(["-i", path, "-map", f"0:s:{track.index}", "-c:s", "srt", out_srt])
    return subs.load_file(out_srt)


def embedded_for(info: MediaInfo, lang: str) -> Optional[Track]:
    """Best text subtitle track in ``lang`` (non-"forced"/"SDH" titles first)."""
    cands = [t for t in info.text_subtitles if t.lang2 == lang]
    cands.sort(key=lambda t: (("forced" in t.title.lower()), ("sdh" in t.title.lower())))
    return cands[0] if cands else None


def extract_audio(path: Path, out: Path, sr: int, channels: int, track: int = 0, start: Optional[float] = None,
                  length: Optional[float] = None) -> None:
    args: List = []
    if start is not None:
        args += ["-ss", f"{start:.3f}"]
    args += ["-i", path]
    if length is not None:
        args += ["-t", f"{length:.3f}"]
    args += ["-map", f"0:a:{track}", "-vn", "-sn", "-ac", channels, "-ar", sr, "-c:a", "pcm_s16le", out]
    ffmpeg.run(args)


def mux_dub(src: Path, dub_wav: Path, out: Path, lang: str, title: str) -> int:
    """Same file + the dub as a new audio track (video and the original tracks are stream-copied).  mp4 gets AAC, mkv too
    (plays on TVs); the dub becomes the default audio track, the original stays selectable."""
    lang3 = {"ru": "rus", "en": "eng", "de": "deu"}.get(lang, lang)
    n_audio = len(probe(src).audio)
    args = ["-i", src, "-i", dub_wav, "-map", "0", "-map", "1:a:0", "-c", "copy", f"-c:a:{n_audio}", "aac", f"-b:a:{n_audio}", "224k",
            f"-metadata:s:a:{n_audio}", f"language={lang3}", f"-metadata:s:a:{n_audio}", f"title={title}"]
    for i in range(n_audio):
        args += [f"-disposition:a:{i}", "0"]
    args += [f"-disposition:a:{n_audio}", "default"]
    if Path(out).suffix.lower() == ".mp4":
        args += ["-c:s", "mov_text", "-movflags", "+faststart"]
    args.append(out)
    ffmpeg.run(args, timeout=4 * 3600)
    return n_audio


def preview_clip(src: Path, start: float, length: float, dub_wav: Path, out: Path) -> Path:
    """A short file with the dubbed fragment for an external player (only used when the built-in player cannot play the film).
    The video is re-encoded here (native mpeg4, no GPL encoder needed) so it starts exactly at ``start`` and stays in sync."""
    ffmpeg.run(["-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", src, "-i", dub_wav, "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "mpeg4", "-q:v", "4", "-c:a", "aac", "-b:a", "192k", "-shortest", out], timeout=900)
    return Path(out)
