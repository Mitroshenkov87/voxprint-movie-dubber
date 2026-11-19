"""Where the player takes the dubbed sound from (no Qt): the Watch-mode chunks written while dubbing, or one finished WAV.

Both sources answer ``read(t, n)``: ``n`` stereo float32 frames at 48 kHz starting at film time ``t``.  Parts that are not dubbed
yet (or outside the file) come back as silence - the Watch logic pauses the film before that happens.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

SR = 48000


def _load(path: Path) -> Optional[np.ndarray]:
    import soundfile as sf

    try:
        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except (OSError, RuntimeError):
        return None
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    if sr != SR:
        from dubber.core.audio import resample

        x = np.stack([resample(x[:, c], sr, SR) for c in range(2)], axis=1).astype(np.float32)
    return x[:, :2]


class ChunkSource:
    """``<project>/watch/chunk_NNNNN.wav`` (``index.json`` gives the chunk length); a few chunks stay in memory."""

    def __init__(self, folder: Path, keep: int = 4) -> None:
        self.folder = Path(folder)
        self.keep = keep
        self._cache: "OrderedDict[int, np.ndarray]" = OrderedDict()

    @property
    def chunk_s(self) -> float:
        """Return the chunk length in seconds, or 30 when index.json is missing."""
        try:
            return float(json.loads((self.folder / "index.json").read_text(encoding="utf-8")).get("chunk_s") or 30.0)
        except (OSError, ValueError):
            return 30.0

    def _chunk(self, i: int) -> Optional[np.ndarray]:
        if i in self._cache:
            self._cache.move_to_end(i)
            return self._cache[i]
        path = self.folder / f"chunk_{i:05d}.wav"
        if not path.is_file():
            return None
        x = _load(path)
        if x is not None:
            self._cache[i] = x
            while len(self._cache) > self.keep:
                self._cache.popitem(last=False)
        return x

    def read(self, t: float, n: int) -> np.ndarray:
        """Return n stereo frames at film time t, with silence for chunks that are not ready."""
        out = np.zeros((n, 2), np.float32)
        cs = self.chunk_s
        pos = max(0, int(round(t * SR)))
        done = 0
        while done < n:
            i = int((pos + done) // int(cs * SR))
            off = (pos + done) - i * int(cs * SR)
            x = self._chunk(i)
            take = min(n - done, int(cs * SR) - off)
            if x is not None and off < len(x):
                part = x[off: off + take]
                out[done: done + len(part)] = part
            done += take
        return out


class WavSource:
    """One finished WAV (the preview fragment) that starts at film time ``offset``."""

    def __init__(self, path: Path, offset: float = 0.0) -> None:
        self.offset = offset
        loaded = _load(Path(path))
        self.x: np.ndarray = loaded if loaded is not None else np.zeros((0, 2), np.float32)

    @property
    def span(self) -> Tuple[float, float]:
        """Return the film-time start and end of this file."""
        return self.offset, self.offset + len(self.x) / SR

    def read(self, t: float, n: int) -> np.ndarray:
        """Return n stereo frames at film time t, with silence outside this file."""
        out = np.zeros((n, 2), np.float32)
        a = int(round((t - self.offset) * SR))
        lo, hi = max(a, 0), min(a + n, len(self.x))
        if hi > lo:
            out[lo - a: hi - a] = self.x[lo:hi]
        return out
