"""ffmpeg / ffprobe discovery and the few operations the pipeline needs (extract audio, probe, add a dubbed track).

Search order: ``VOXPRINT_FFMPEG`` (file or folder) -> ``bin/`` next to the program -> the tools folder of the app data
(where a future installer step may put an LGPL build) -> ``PATH`` (only if ``-version`` really runs) -> the ``imageio-ffmpeg``
wheel (a GPL build; offline fallback, no ffprobe).  The video stream is NEVER re-encoded (``-c:v copy``).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, TypedDict

from dubber import paths
from dubber.appinfo import resource_dir

EXE = ".exe" if os.name == "nt" else ""
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


class FfmpegError(RuntimeError):
    """ffmpeg is missing or a command failed; the message carries the tail of ffmpeg's stderr."""


@dataclass
class FfmpegInfo:
    """ffmpeg and ffprobe that were found, the version line, and candidates that did not run."""
    ffmpeg: Optional[str] = None
    ffprobe: Optional[str] = None
    version: str = ""
    source: str = ""                       # where it was found
    problems: List[str] = field(default_factory=list)


def _works(exe: str) -> str:
    """Return the first line of ``exe -version`` or '' if it does not run."""
    try:
        p = subprocess.run([exe, "-version"], capture_output=True, text=True, timeout=20, creationflags=_NO_WINDOW,
                           encoding="utf-8", errors="replace")
        if p.returncode == 0 and p.stdout:
            return p.stdout.splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return ""


def _candidates() -> List[tuple]:
    out: List[tuple] = []
    env = os.environ.get("VOXPRINT_FFMPEG")
    if env:
        p = Path(env)
        out.append(("VOXPRINT_FFMPEG", p / f"ffmpeg{EXE}" if p.is_dir() else p))
    for d, label in ((resource_dir() / "bin", "program folder"), (Path(sys.executable).parent / "bin", "python/exe folder"),
                     (paths.app_home() / "tools", "app tools folder")):
        out.append((label, d / f"ffmpeg{EXE}"))
    return out


def find_ffmpeg() -> FfmpegInfo:
    """Locate ffmpeg (and ffprobe next to it or on PATH).  Never raises."""
    info = FfmpegInfo()
    cands = [(lbl, str(p)) for lbl, p in _candidates() if Path(p).is_file()]
    on_path = shutil.which("ffmpeg")
    if on_path:
        cands.append(("PATH", on_path))
    for label, exe in cands:
        ver = _works(exe)
        if ver:
            info.ffmpeg, info.version, info.source = exe, ver, label
            break
        info.problems.append(f"{label}: {exe} does not run")
    if info.ffmpeg is None:
        try:
            import imageio_ffmpeg

            exe = imageio_ffmpeg.get_ffmpeg_exe()
            ver = _works(exe)
            if ver:
                info.ffmpeg, info.version, info.source = exe, ver, "imageio-ffmpeg wheel (GPL build, fallback)"
        except Exception as exc:  # noqa: BLE001
            info.problems.append(f"imageio-ffmpeg: {type(exc).__name__}: {exc}")
    if info.ffmpeg:
        sibling = Path(info.ffmpeg).with_name(f"ffprobe{EXE}")
        probe = str(sibling) if sibling.is_file() else shutil.which("ffprobe")
        if probe and _works(probe):
            info.ffprobe = probe
    return info


_cached: Optional[FfmpegInfo] = None


def ffmpeg_info(refresh: bool = False) -> FfmpegInfo:
    """Return the cached :func:`find_ffmpeg` result, searching on the first call and when ``refresh`` is true."""
    global _cached
    if _cached is None or refresh:
        _cached = find_ffmpeg()
    return _cached


def require_ffmpeg() -> str:
    """Return the path of a working ffmpeg executable.

    Raises:
        FfmpegError: No ffmpeg binary was found.
    """
    exe = ffmpeg_info().ffmpeg
    if not exe:
        raise FfmpegError("ffmpeg was not found (install ffmpeg, or set VOXPRINT_FFMPEG to its path)")
    return exe


def run(args: Sequence[object], timeout: float = 3600) -> subprocess.CompletedProcess:
    """Run ffmpeg with ``args`` (without the program name); raises :class:`FfmpegError` on failure."""
    cmd = [require_ffmpeg(), "-hide_banner", "-nostdin", "-y", "-loglevel", "error", *map(str, args)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, creationflags=_NO_WINDOW,
                           encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired as exc:
        raise FfmpegError(f"ffmpeg timed out after {timeout:.0f} s") from exc
    except OSError as exc:
        raise FfmpegError(f"cannot start ffmpeg: {exc}") from exc
    if p.returncode != 0:
        raise FfmpegError(f"ffmpeg exited with {p.returncode}: {(p.stderr or '').strip()[-600:]}")
    return p


class StreamInfo(TypedDict):
    """One media stream: index, type, codec, language, channel count, sample rate in hertz, and title."""
    index: int
    type: str
    codec: str
    lang: str
    channels: Optional[int]
    rate: Optional[int]
    title: str


class ProbeResult(TypedDict):
    """Media duration in seconds and the streams :func:`probe` reported."""
    duration: Optional[float]
    streams: List[StreamInfo]


def _as_int(value: object) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _stream_from_ffprobe(raw: Dict[str, object]) -> StreamInfo:
    tags = raw.get("tags")
    tag_map = tags if isinstance(tags, dict) else {}
    return {
        "index": _as_int(raw.get("index")) or 0,
        "type": str(raw.get("codec_type") or ""),
        "codec": str(raw.get("codec_name") or ""),
        "lang": str(tag_map.get("language") or ""),
        "channels": _as_int(raw.get("channels")),
        "rate": _as_int(raw.get("sample_rate")),
        "title": str(tag_map.get("title") or ""),
    }


def probe(path: Path) -> ProbeResult:
    """Streams and duration of a media file: ``{"duration": s, "streams": [{"index","type","codec","lang","channels","rate"}]}``."""
    info = ffmpeg_info()
    if info.ffprobe:
        p = subprocess.run([info.ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
                           capture_output=True, text=True, timeout=60, creationflags=_NO_WINDOW, encoding="utf-8", errors="replace")
        if p.returncode != 0:
            raise FfmpegError(f"ffprobe failed: {(p.stderr or '').strip()[-400:]}")
        data = json.loads(p.stdout or "{}")
        raw_streams = data.get("streams", [])
        streams = [_stream_from_ffprobe(s) for s in raw_streams if isinstance(s, dict)]
        dur = data.get("format", {}).get("duration")
        return {"duration": float(dur) if dur else None, "streams": streams}
    # no ffprobe: parse the banner that "ffmpeg -i" prints to stderr
    exe = require_ffmpeg()
    p = subprocess.run([exe, "-hide_banner", "-i", str(path)], capture_output=True, text=True, timeout=60,
                       creationflags=_NO_WINDOW, encoding="utf-8", errors="replace")
    text = p.stderr or ""
    dur = None
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if match:
        dur = int(match.group(1)) * 3600 + int(match.group(2)) * 60 + float(match.group(3))
    streams = []
    for found in re.finditer(r"Stream #0:(\d+)(?:\((\w+)\))?[^:]*:\s*(Audio|Video|Subtitle|Data|Attachment):\s*([^\s,(]+)(.*)", text):
        rate = re.search(r"(\d+) Hz", found.group(5))
        streams.append({"index": int(found.group(1)), "type": found.group(3).lower(), "codec": found.group(4),
                        "lang": found.group(2) or "", "channels": None, "rate": int(rate.group(1)) if rate else None, "title": ""})
    return {"duration": dur, "streams": streams}


def extract_audio(src: Path, dst: Path, sample_rate: int = 16000, channels: int = 1, audio_index: int = 0) -> None:
    """Decode the ``audio_index``-th audio stream of ``src`` to a PCM16 WAV (resampled / downmixed)."""
    args: List[object] = ["-i", src, "-map", f"0:a:{audio_index}", "-vn", "-sn", "-ac", channels, "-ar", sample_rate,
                          "-c:a", "pcm_s16le", dst]
    run(args)


def add_dub_track(src: Path, dub_audio: Path, out: Path, language: str = "rus", title: str = "AI dub",
                  codec: str = "aac", bitrate: str = "192k") -> int:
    """Copy everything from ``src`` and append ``dub_audio`` as a new audio track.  Returns the index of the new audio track
    (among the output audio streams).  Video and the original audio are stream-copied (never re-encoded)."""
    n_audio = sum(1 for stream in probe(src)["streams"] if stream["type"] == "audio")
    ca = f"-c:a:{n_audio}"
    args: List[object] = ["-i", src, "-i", dub_audio, "-map", "0", "-map", "1:a:0", "-c", "copy", ca, codec,
                          f"-b:a:{n_audio}", bitrate, f"-metadata:s:a:{n_audio}", f"language={language}",
                          f"-metadata:s:a:{n_audio}", f"title={title}", out]
    run(args)
    return n_audio
