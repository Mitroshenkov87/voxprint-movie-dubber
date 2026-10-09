"""Audio helpers (numpy / soundfile only): read, write, resample, energy speech detection, MFCC features, Signalsmith stretch."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def read(path: Path | str, sr: Optional[int] = None, mono: bool = True) -> Tuple[np.ndarray, int]:
    import soundfile as sf

    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    x = data.mean(axis=1) if mono else data
    if sr and rate != sr:
        x = resample(x, rate, sr)
        rate = sr
    return x.astype(np.float32), int(rate)


def read_range(path: Path, start: float, end: float, mono: bool = False) -> Tuple[np.ndarray, int]:
    import soundfile as sf

    info = sf.info(str(path))
    a, b = max(0, int(start * info.samplerate)), max(0, int(end * info.samplerate))
    data, rate = sf.read(str(path), start=a, stop=min(b, info.frames), dtype="float32", always_2d=True)
    if b - a > len(data):
        data = np.vstack([data, np.zeros((b - a - len(data), data.shape[1]), dtype=np.float32)])
    return (data.mean(axis=1) if mono else data), int(rate)


def write(path: Path, x: np.ndarray, sr: int) -> None:
    import soundfile as sf

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(x, -1.0, 1.0), sr, subtype="PCM_16")


def duration(path: Path) -> float:
    import soundfile as sf

    info = sf.info(str(path))
    return info.frames / float(info.samplerate)


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return x
    try:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(sr_from, sr_to)
        return resample_poly(x, sr_to // g, sr_from // g, axis=0).astype(np.float32)
    except Exception:  # noqa: BLE001
        n = int(len(x) * sr_to / sr_from)
        return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def energy_vad(x: np.ndarray, sr: int, frame_s: float = 0.03, thresh_db: float = -38.0, min_speech_s: float = 0.25,
               min_silence_s: float = 0.3) -> List[Tuple[float, float]]:
    """Speech-like regions by frame energy relative to the loudest frames (fallback when Silero VAD is not installed)."""
    n = max(1, int(frame_s * sr))
    frames = len(x) // n
    if frames == 0:
        return []
    e = np.sqrt(np.mean(x[: frames * n].reshape(frames, n) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(e)
    ref = np.percentile(db, 95)
    active = (db > ref + thresh_db / 2) if ref > -60 else np.zeros(frames, bool)
    segs, start = [], None
    for i, a in enumerate(active):
        if a and start is None:
            start = i
        elif not a and start is not None:
            segs.append([start * frame_s, i * frame_s])
            start = None
    if start is not None:
        segs.append([start * frame_s, frames * frame_s])
    merged: List[List[float]] = []
    for s in segs:
        if merged and s[0] - merged[-1][1] < min_silence_s:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return [(round(a, 3), round(b, 3)) for a, b in merged if b - a >= min_speech_s]


def mfcc_stats(x: np.ndarray, sr: int, n_mels: int = 32, n_mfcc: int = 13) -> np.ndarray:
    """Mean + std of MFCCs (a small speaker fingerprint for the fallback speaker clustering)."""
    if sr != 16000:
        x = resample(x, sr, 16000)
        sr = 16000
    n_fft, hop = 512, 160
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    frames = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(frames)[:, None]
    spec = np.abs(np.fft.rfft(x[idx] * np.hanning(n_fft), axis=1)) ** 2
    mel = _mel_bank(sr, n_fft, n_mels)
    logmel = np.log(spec @ mel.T + 1e-10)
    from scipy.fft import dct

    c = dct(logmel, type=2, axis=1, norm="ortho")[:, 1:n_mfcc + 1]
    voiced = spec.sum(axis=1) > np.percentile(spec.sum(axis=1), 40)
    c = c[voiced] if voiced.sum() > 5 else c
    return np.concatenate([c.mean(axis=0), c.std(axis=0)]).astype(np.float32)


def _mel_bank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    def hz2mel(h):
        return 2595 * np.log10(1 + h / 700.0)

    def mel2hz(m):
        return 700 * (10 ** (m / 2595.0) - 1)
    pts = mel2hz(np.linspace(hz2mel(60), hz2mel(sr / 2 - 200), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, n_mels + 1):
        a, b, c = bins[i - 1], bins[i], bins[i + 1]
        if b > a:
            fb[i - 1, a:b] = (np.arange(a, b) - a) / (b - a)
        if c > b:
            fb[i - 1, b:c] = (c - np.arange(b, c)) / (c - b)
    return fb


def stretch(x: np.ndarray, sr: int, factor: float) -> np.ndarray:
    """Make ``x`` ``factor`` times faster without changing the pitch (Signalsmith Stretch, MIT; fallback: ffmpeg ``atempo``)."""
    if abs(factor - 1.0) < 1e-3 or len(x) == 0:
        return x
    try:
        import python_stretch as ps

        st = ps.Signalsmith.Stretch()
        st.preset(1, sr)
        st.setTimeFactor(float(factor))
        y = st.process(x.reshape(1, -1).astype(np.float32))
        return np.asarray(y, dtype=np.float32).reshape(-1)
    except ImportError:
        return _atempo(x, sr, factor)


def _atempo(x: np.ndarray, sr: int, factor: float) -> np.ndarray:
    import os
    import tempfile

    from dubber import ffmpeg

    d = tempfile.mkdtemp(prefix="vmd-stretch-")
    a, b = os.path.join(d, "in.wav"), os.path.join(d, "out.wav")
    try:
        write(Path(a), x, sr)
        ffmpeg.run(["-i", a, "-filter:a", f"atempo={factor:.4f}", b])
        y, _ = read(Path(b))
        return y
    finally:
        import shutil

        shutil.rmtree(d, ignore_errors=True)


def trim_silence(x: np.ndarray, sr: int, thresh_db: float = -45.0, pad_s: float = 0.04) -> np.ndarray:
    """Cut leading / trailing silence of a synthesised line (TTS output usually has some)."""
    if len(x) == 0:
        return x
    n = max(1, int(0.01 * sr))
    frames = len(x) // n
    if frames == 0:
        return x
    db = 20 * np.log10(np.sqrt(np.mean(x[: frames * n].reshape(frames, n) ** 2, axis=1)) + 1e-9)
    on = np.where(db > thresh_db)[0]
    if len(on) == 0:
        return x[:0]
    a = max(0, on[0] * n - int(pad_s * sr))
    b = min(len(x), (on[-1] + 1) * n + int(pad_s * sr))
    return x[a:b]
