"""Mixing the dub track (research notes 01/06): outside dialogue windows the original soundtrack is kept untouched; inside the
windows the separated background + the original speech at ``original_volume`` (voice-over style) + the dub lines.  Lines marked
"keep original" (songs) keep the full original.  Crossfades at window edges; the result is peak-limited.

Works on any time range, so the same code makes the whole track, the 1-minute preview and the Watch-mode chunks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from dubber.core import audio
from dubber.core.project import Line

FADE_S = 0.08


def _env(n: int, sr: int, windows: Sequence[Tuple[float, float]], t0: float) -> np.ndarray:
    """1 inside the dialogue windows, 0 outside, with short linear ramps (relative to range start ``t0``)."""
    env = np.zeros(n, dtype=np.float32)
    r = max(1, int(FADE_S * sr))
    for s, e in windows:
        a, b = int((s - t0) * sr), int((e - t0) * sr)
        if b <= 0 or a >= n:
            continue
        lo, hi = max(0, a), min(n, b)
        env[lo:hi] = 1.0
        for k in range(r):                              # ramps
            if 0 <= a - r + k < n:
                env[a - r + k] = max(env[a - r + k], k / r)
            if 0 <= b + k < n:
                env[b + k] = max(env[b + k], 1 - k / r)
    return env


def mix_range(t0: float, t1: float, original: Path, windows: Sequence[Tuple[float, float]], lines: Sequence[Line],
              line_audio: Dict[int, Tuple[np.ndarray, int]], background: Optional[Path] = None, speech: Optional[Path] = None,
              original_volume: float = 0.15, dub_gain_db: float = 0.0) -> Tuple[np.ndarray, int]:
    """Stereo float32 mix of ``[t0, t1)`` at the sample rate of ``original``."""
    orig, sr = audio.read_range(original, t0, t1, mono=False)
    if orig.shape[1] == 1:
        orig = np.repeat(orig, 2, axis=1)
    orig = orig[:, :2]
    n = len(orig)
    keep = [(ln.start, ln.end) for ln in lines if ln.keep_original]
    win = [w for w in windows if not any(k[0] < w[1] and k[1] > w[0] and (min(k[1], w[1]) - max(k[0], w[0])) > 0.5 * (w[1] - w[0]) for k in keep)]
    env = _env(n, sr, win, t0)[:, None]
    if background is not None and Path(background).exists():
        bg, bsr = audio.read_range(background, t0, t1, mono=False)
        bg = _match(bg, bsr, sr, n)
        sp = None
        if speech is not None and Path(speech).exists():
            s_, ssr = audio.read_range(speech, t0, t1, mono=False)
            sp = _match(s_, ssr, sr, n)
        inside = bg + (sp * original_volume if sp is not None else 0.0)
    else:                                              # no separation: duck the whole original under the dub
        inside = orig * max(original_volume, 0.12)
    mix = orig * (1.0 - env) + inside * env
    g = 10 ** (dub_gain_db / 20)
    for ln in lines:
        if ln.keep_original or ln.id not in line_audio or ln.place_start < 0:
            continue
        x, lsr = line_audio[ln.id]
        if lsr != sr:
            x = audio.resample(x, lsr, sr)
        a = int((ln.place_start - t0) * sr)
        b = a + len(x)
        if b <= 0 or a >= n:
            continue
        xa, xb = max(0, -a), len(x) - max(0, b - n)
        mix[max(0, a):min(n, b)] += (x[xa:xb] * g)[:, None]
    peak = float(np.abs(mix).max()) if n else 0.0
    if peak > 0.98:
        mix *= 0.98 / peak
    return mix.astype(np.float32), sr


def _match(x: np.ndarray, sr_from: int, sr_to: int, n: int) -> np.ndarray:
    if sr_from != sr_to:
        x = audio.resample(x, sr_from, sr_to)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    x = x[:, :2]
    if len(x) < n:
        x = np.vstack([x, np.zeros((n - len(x), 2), dtype=np.float32)])
    return x[:n]


def loudness_match_gain(dub: np.ndarray, reference: np.ndarray) -> float:
    """dB gain that brings the dub lines to the RMS of the original speech (a simple stand-in for LUFS matching)."""
    def rms(v):
        return float(np.sqrt(np.mean(np.square(v)))) if len(v) else 0.0
    a, b = rms(dub), rms(reference)
    if a < 1e-6 or b < 1e-6:
        return 0.0
    return float(np.clip(20 * np.log10(b / a), -12, 12))


def chunk_bounds(total: float, chunk_s: float) -> List[Tuple[float, float]]:
    out, t = [], 0.0
    while t < total - 1e-6:
        out.append((t, min(total, t + chunk_s)))
        t += chunk_s
    return out
