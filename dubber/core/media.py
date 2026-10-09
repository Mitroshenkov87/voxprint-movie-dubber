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
    info = MediaInfo(Path(path), float(data["duration"] or 0.0))
    a = s = 0
    for st in data["streams"]:
        kind = st["type"]
        if kind == "video" and not info.video_codec:
            info.video_codec = st["codec"]
        elif kind == "audio":
            info.audio.append(Track(a, st["index"], st["codec"], st["lang"], st["title"], st["channels"]))
            a += 1
        elif kind == "subtitle":
            info.subtitles.append(Track(s, st["index"], st["codec"], st["lang"], st["title"]))
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


def pick_original_track(info: Optional[MediaInfo], target_lang: str = "") -> int:
    """Audio track to dub from, chosen automatically: the original-language track.  A track already in the dub language, a
    commentary or an audio-description track is skipped; a title saying "original" wins; otherwise the first track left."""
    if info is None or not info.audio:
        return 0

    def bad(t: Track) -> bool:
        title = t.title.lower()
        return any(w in title for w in ("comment", "коммент", "description", "описани", "director"))

    def rank(t: Track):
        title = t.title.lower()
        return (bad(t), bool(target_lang) and t.lang2 == target_lang, not ("original" in title or "оригинал" in title), t.index)

    return min(info.audio, key=rank).index


def extract_audio(path: Path, out: Path, sr: int, channels: int, track: int = 0, start: Optional[float] = None,
                  length: Optional[float] = None) -> None:
    args: List[object] = []
    if start is not None:
        args += ["-ss", f"{start:.3f}"]
    args += ["-i", path]
    if length is not None:
        args += ["-t", f"{length:.3f}"]
    args += ["-map", f"0:a:{track}", "-vn", "-sn", "-ac", channels, "-ar", sr, "-c:a", "pcm_s16le", out]
    ffmpeg.run(args)


#: audio codecs an MP4 file can hold as they are (others are converted to AAC on an MP4 export; the video is never re-encoded)
MP4_AUDIO = {"aac", "ac3", "eac3", "mp3", "alac"}
MP4_TEXT_SUBS = {"subrip", "srt", "ass", "ssa", "mov_text", "webvtt", "text"}


def mux_dub(src: Path, dub_wav: Path, out: Path, lang: str, title: str) -> int:
    """Same file + the dub as a new audio track (video stream-copied, never re-encoded).  The dub is AAC (plays on TVs) and becomes
    the default audio track, the original stays selectable.

    MKV (default): every stream of the film is copied.  MP4 (remux only): video + audio + text subtitles (as mov_text); an original
    audio track whose codec MP4 cannot hold (TrueHD, DTS, FLAC, PCM, Vorbis, ...) is converted to AAC - only that audio; picture-based
    subtitles and attachments (fonts) are left out because MP4 cannot carry them."""
    lang3 = {"ru": "rus", "en": "eng", "de": "deu"}.get(lang, lang)
    info = probe(src)
    n_audio = len(info.audio)
    mp4 = Path(out).suffix.lower() == ".mp4"
    if mp4:
        args = ["-i", src, "-i", dub_wav, "-map", "0:v?", "-map", "0:a?"]
        args += [x for t in info.subtitles if t.codec.lower() in MP4_TEXT_SUBS for x in ("-map", f"0:s:{t.index}")]
        args += ["-map", "1:a:0", "-c", "copy"]
        for t in info.audio:
            if t.codec.lower() not in MP4_AUDIO:
                ch = int(t.channels or 2)
                args += [f"-c:a:{t.index}", "aac", f"-b:a:{t.index}", f"{min(640, 96 * max(2, ch))}k"]
    else:
        args = ["-i", src, "-i", dub_wav, "-map", "0", "-map", "1:a:0", "-c", "copy"]
    args += [f"-c:a:{n_audio}", "aac", f"-b:a:{n_audio}", "224k",
             f"-metadata:s:a:{n_audio}", f"language={lang3}", f"-metadata:s:a:{n_audio}", f"title={title}"]
    for i in range(n_audio):
        args += [f"-disposition:a:{i}", "0"]
    args += [f"-disposition:a:{n_audio}", "default"]
    if mp4:
        args += ["-c:s", "mov_text", "-movflags", "+faststart"]
        if info.video_codec.lower() in ("hevc", "h265"):
            args += ["-tag:v", "hvc1"]                       # HEVC in MP4 plays on Apple devices / TVs only with this tag
    args.append(out)
    ffmpeg.run(args, timeout=4 * 3600)
    return n_audio


def preview_clip(src: Path, start: float, length: float, dub_wav: Path, out: Path) -> Path:
    """A short file with the dubbed fragment for an external player (only used when the built-in player cannot play the film).
    The video is re-encoded here (native mpeg4, no GPL encoder needed) so it starts exactly at ``start`` and stays in sync."""
    ffmpeg.run(["-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", src, "-i", dub_wav, "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "mpeg4", "-q:v", "4", "-c:a", "aac", "-b:a", "192k", "-shortest", out], timeout=900)
    return Path(out)
