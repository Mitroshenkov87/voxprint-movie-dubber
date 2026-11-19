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
    """One audio or subtitle stream from ffprobe."""
    index: int              # index among streams of its type (ffmpeg "0:a:N" / "0:s:N")
    stream: int             # absolute stream index
    codec: str
    lang: str = ""
    title: str = ""
    channels: Optional[int] = None

    @property
    def lang2(self) -> str:
        """Two-letter language code. Known three-letter tags are mapped; others are cut to two letters."""
        return LANG3.get(self.lang.lower(), self.lang.lower()[:2])

    def label(self) -> str:
        """Short label: language, codec, channel count when known, and title."""
        bits = [self.lang or "und", self.codec]
        if self.channels:
            bits.append(f"{self.channels} ch")
        if self.title:
            bits.append(self.title)
        return " · ".join(bits)


@dataclass
class MediaInfo:
    """Duration, video codec, and the audio and subtitle tracks of one film file."""
    path: Path
    duration: float = 0.0
    video_codec: str = ""
    audio: List[Track] = field(default_factory=list)
    subtitles: List[Track] = field(default_factory=list)

    @property
    def text_subtitles(self) -> List[Track]:
        """Subtitle tracks whose codec is text rather than a picture."""
        return [t for t in self.subtitles if t.codec in TEXT_SUB_CODECS]

    @property
    def container(self) -> str:
        """File extension without the dot, in lower case."""
        return self.path.suffix.lower().lstrip(".")


def probe(path: Path) -> MediaInfo:
    """Duration, video codec, and audio and subtitle tracks from ffprobe."""
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
    """Convert one subtitle stream to SRT at ``out_srt`` and return its cues."""
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
    """Extract one audio track as 16-bit PCM. ``start`` and ``length``, when set, are seconds."""
    args: List[object] = []
    if start is not None:
        args += ["-ss", f"{start:.3f}"]
    args += ["-i", path]
    if length is not None:
        args += ["-t", f"{length:.3f}"]
    args += ["-map", f"0:a:{track}", "-vn", "-sn", "-ac", channels, "-ar", sr, "-c:a", "pcm_s16le", out]
    ffmpeg.run(args)


def poster_frame(src: Path, at_s: float, dest: Path) -> bool:
    """Write one JPEG of the frame at ``at_s`` seconds.

    The player shows this still until Play. A decode failure returns False and leaves the
    player on its previous picture.

    Args:
        src: Film path.
        at_s: Time in seconds.
        dest: JPEG path to write. Parent directories are created.

    Returns:
        True when ``dest`` exists and is not empty.
    """
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg.run(["-ss", f"{max(0.0, float(at_s)):.3f}", "-i", src, "-frames:v", "1", dest], timeout=60)
    except (OSError, ffmpeg.FfmpegError):
        return False
    return dest.is_file() and dest.stat().st_size > 0


#: audio codecs an MP4 file can hold as they are (others are converted to AAC on an MP4 export)
MP4_AUDIO = {"aac", "ac3", "eac3", "mp3", "alac"}
MP4_TEXT_SUBS = {"subrip", "srt", "ass", "ssa", "mov_text", "webvtt", "text"}


def _subtitle_filter(path: Path) -> str:
    """Return an ffmpeg ``subtitles`` filter for ``path``.

    The burned-in option is the one place a dub re-encodes the picture. The
    filter reads the file itself, so the path is not also mapped as a track.

    Args:
        path: A subtitle file on disk.
    """
    text = Path(path).resolve().as_posix()
    for char in ("\\", ":", "'", ",", "[", "]", ";"):
        text = text.replace(char, "\\" + char)
    return f"subtitles='{text}'"


def _mux_tail(
    args: List[object],
    info: MediaInfo,
    *,
    kept: int,
    lang3: str,
    title: str,
    mp4: bool,
    burn: bool,
    map_subs: bool,
) -> None:
    """Append the dub-track codec, dispositions, and the MP4 finishing flags."""
    args += [
        f"-c:a:{kept}",
        "aac",
        f"-b:a:{kept}",
        "224k",
        f"-metadata:s:a:{kept}",
        f"language={lang3}",
        f"-metadata:s:a:{kept}",
        f"title={title}",
    ]
    for index in range(kept):
        args += [f"-disposition:a:{index}", "0"]
    args += [f"-disposition:a:{kept}", "default"]
    if not mp4:
        return
    if map_subs:
        args += ["-c:s", "mov_text"]
    args += ["-movflags", "+faststart"]
    if not burn and info.video_codec.lower() in ("hevc", "h265"):
        args += ["-tag:v", "hvc1"]  # HEVC in MP4 plays on Apple devices / TVs only with this tag


def _mux_copy(src: Path, dub_wav: Path, out: Path, lang: str, title: str) -> int:
    """Copy the film and add the dub. The picture is not re-encoded."""
    lang3 = {"ru": "rus", "en": "eng", "de": "deu"}.get(lang, lang)
    info = probe(src)
    n_audio = len(info.audio)
    mp4 = Path(out).suffix.lower() == ".mp4"
    if mp4:
        args: List[object] = ["-i", src, "-i", dub_wav, "-map", "0:v?", "-map", "0:a?"]
        args += [x for t in info.subtitles if t.codec.lower() in MP4_TEXT_SUBS for x in ("-map", f"0:s:{t.index}")]
        args += ["-map", "1:a:0", "-c", "copy"]
        for track in info.audio:
            if track.codec.lower() not in MP4_AUDIO:
                channels = int(track.channels or 2)
                args += [f"-c:a:{track.index}", "aac", f"-b:a:{track.index}", f"{min(640, 96 * max(2, channels))}k"]
    else:
        args = ["-i", src, "-i", dub_wav, "-map", "0", "-map", "1:a:0", "-c", "copy"]
    _mux_tail(args, info, kept=n_audio, lang3=lang3, title=title, mp4=mp4, burn=False, map_subs=mp4)
    args.append(out)
    ffmpeg.run(args, timeout=4 * 3600)
    return n_audio


def _mux_arranged(
    src: Path,
    dub_wav: Path,
    out: Path,
    lang: str,
    title: str,
    *,
    keep_original: bool,
    delivery: str,
    subtitle_file: Optional[Path],
) -> int:
    """Mux one of the plain choices: drop the original, drop subs, add a track, or burn.

    Burning uses the ``subtitles`` filter and re-encodes the picture with mpeg4,
    the same encoder as a preview clip. Every other choice still copies the picture.
    """
    lang3 = {"ru": "rus", "en": "eng", "de": "deu"}.get(lang, lang)
    info = probe(src)
    mp4 = Path(out).suffix.lower() == ".mp4"
    burn = delivery == "burn" and subtitle_file is not None
    soft = delivery == "track"
    kept = len(info.audio) if keep_original else 0
    args: List[object] = ["-i", src, "-i", dub_wav]
    extra = soft and subtitle_file is not None
    if extra:
        args += ["-i", subtitle_file]
    if keep_original and not mp4 and soft and not burn:
        args += ["-map", "0", "-map", "1:a:0"]
        if extra:
            args += ["-map", "2:s:0"]
        args += ["-c", "copy"]
        _mux_tail(args, info, kept=kept, lang3=lang3, title=title, mp4=False, burn=False, map_subs=False)
        args.append(out)
        ffmpeg.run(args, timeout=4 * 3600)
        return kept
    if burn:
        args += ["-map", "0:v:0"]
    else:
        args += ["-map", "0:v?"]
    if keep_original:
        args += ["-map", "0:a?"]
    mapped_subs = False
    if soft:
        if mp4:
            for track in info.subtitles:
                if track.codec.lower() in MP4_TEXT_SUBS:
                    args += ["-map", f"0:s:{track.index}"]
                    mapped_subs = True
        else:
            args += ["-map", "0:s?", "-map", "0:d?", "-map", "0:t?"]
            mapped_subs = True
        if extra:
            args += ["-map", "2:s:0"]
            mapped_subs = True
    elif not mp4:
        args += ["-map", "0:d?", "-map", "0:t?"]
    args += ["-map", "1:a:0"]
    if burn and subtitle_file is not None:
        # mpeg4 needs no GPL build. The preview clip uses the same encoder.
        args += ["-vf", _subtitle_filter(subtitle_file), "-c:v", "mpeg4", "-q:v", "4", "-c:a", "copy"]
    else:
        args += ["-c", "copy"]
    if keep_original and mp4:
        for track in info.audio:
            if track.codec.lower() not in MP4_AUDIO:
                channels = int(track.channels or 2)
                args += [f"-c:a:{track.index}", "aac", f"-b:a:{track.index}", f"{min(640, 96 * max(2, channels))}k"]
    _mux_tail(
        args,
        info,
        kept=kept,
        lang3=lang3,
        title=title,
        mp4=mp4,
        burn=burn,
        map_subs=mapped_subs and mp4,
    )
    args.append(out)
    ffmpeg.run(args, timeout=4 * 3600)
    return kept


def mux_dub(
    src: Path,
    dub_wav: Path,
    out: Path,
    lang: str,
    title: str,
    *,
    keep_original: bool = True,
    subtitle_delivery: str = "track",
    subtitle_path: Optional[Path] = None,
) -> int:
    """Add the dub as the default audio track.

    The picture is copied, except when ``subtitle_delivery`` is ``burn`` and
    ``subtitle_path`` exists: that option draws the cues into the picture with
    the ffmpeg ``subtitles`` filter and re-encodes the video. ``keep_original``
    leaves the film's audio selectable. ``track`` copies text subtitles and,
    when ``subtitle_path`` is set, adds that file as one more subtitle track.
    ``none`` leaves subtitle streams out.

    The default arguments match the older call: original audio stays, text
    subtitles are copied, and the video is not re-encoded.

    Args:
        src: The film.
        dub_wav: The finished dub track.
        out: The file to write.
        lang: Dub language (``ru``, ``en``, or ``de``).
        title: Title of the new audio track.
        keep_original: When False, the film's audio is not mapped.
        subtitle_delivery: ``none``, ``track``, or ``burn``.
        subtitle_path: Dub subtitle file for ``track`` or ``burn``.

    Returns:
        The output index of the dub audio track.
    """
    delivery = subtitle_delivery if subtitle_delivery in ("none", "track", "burn") else "track"
    ready = Path(subtitle_path) if subtitle_path is not None and Path(subtitle_path).is_file() else None
    if keep_original and delivery == "track" and ready is None:
        return _mux_copy(src, dub_wav, out, lang, title)
    return _mux_arranged(
        src,
        dub_wav,
        out,
        lang,
        title,
        keep_original=keep_original,
        delivery=delivery,
        subtitle_file=ready,
    )


def preview_clip(src: Path, start: float, length: float, dub_wav: Path, out: Path) -> Path:
    """A short file with the dubbed fragment for an external player (only used when the built-in player cannot play the film).
    The video is re-encoded here (native mpeg4, no GPL encoder needed) so it starts exactly at ``start`` and stays in sync."""
    ffmpeg.run(["-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", src, "-i", dub_wav, "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "mpeg4", "-q:v", "4", "-c:a", "aac", "-b:a", "192k", "-shortest", out], timeout=900)
    return Path(out)
