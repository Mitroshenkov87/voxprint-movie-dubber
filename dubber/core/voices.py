"""Voices: the shared Voxprint voice library (read-only) and reference clips cloned from a character's own lines.

The library is the one of Voxprint AI Audiobook Builder: ``<Voxprint home>/voices/<id>/`` with a LoRA adapter
(``adapter_model.safetensors`` + ``adapter_config.json``), ``ref_sample.wav`` (+ its text in ``training_meta.json``
``ref_sample_text``) and ``voice.json`` (name, language, gender ...).  The dubber never writes there.
A library voice is used with its adapter on the Qwen3-TTS base model it was trained on (``voice.json`` ``base_model``).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from dubber.core import audio
from dubber.core.project import Line
from dubber.engines.diarization import try_speaker_embeddings
from dubber.infra import shared_paths
from dubber.workers.common import wer

REQUIRED_FILES = ("adapter_model.safetensors", "adapter_config.json")
REF_MIN_S, REF_MAX_S = 5.0, 15.0          # research note 03: a reference of <= 15 s clones best
REF_PREF_S = 3.0                          # prefer lines at least this long
REF_MIN_LINE_S = 1.5                      # shorter lines are never a cloning reference
REF_SIM_MIN = 0.75                        # cosine similarity to the anchor line (same voice)
REF_PAD_S = 0.12                          # context kept on each side of a chunk, clamped to the neighbours
REF_FADE_S = 0.010                        # fade in/out so a hard cut does not click
REF_GAP_S = 0.25                          # silence between reference chunks
REF_WER_MAX = 0.35                        # re-transcription may replace the joined text only under this
REF_SHOUT_DB = 6.0                        # a 1.5-3 s line louder than this above the anchor is a shout


@dataclass
class LibraryVoice:
    id: str
    name: str
    path: Path
    language: str = ""
    gender: str = ""
    base_model: str = ""
    license: str = ""
    ref_text: str = ""
    adapter_scale: float = 1.0
    repo_id: str = ""

    @property
    def ref_audio(self) -> Optional[Path]:
        p = self.path / "ref_sample.wav"
        return p if p.is_file() else None

    @property
    def preview(self) -> Optional[Path]:
        for n in ("preview.wav", "ref_sample.wav"):
            if (self.path / n).is_file():
                return self.path / n
        return None


def _read(p: Path) -> Dict:
    try:
        d = json.loads(p.read_text(encoding="utf-8-sig"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _scale(value) -> float:
    """``adapter_scale`` of voice.json clamped to 0.1..1.0; voices without it keep full strength (as in the Audiobook Builder)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 1.0
    return 1.0 if v != v else round(min(1.0, max(0.1, v)), 2)


def list_library(root: Optional[Path] = None) -> List[LibraryVoice]:
    """Usable voices of the shared library (complete adapter + reference clip), sorted by name.  Missing folder -> []."""
    root = Path(root) if root is not None else shared_paths.voices_dir()
    out: List[LibraryVoice] = []
    try:
        folders = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))
    except OSError:
        return []
    for d in folders:
        if not all((d / n).is_file() for n in REQUIRED_FILES):
            continue
        info, meta = _read(d / "voice.json"), _read(d / "training_meta.json")
        out.append(LibraryVoice(id=d.name, name=str(info.get("name") or info.get("voice_name") or d.name), path=d,
                                language=str(info.get("language") or ""), gender=str(info.get("gender") or info.get("voice_type") or ""),
                                base_model=str(info.get("base_model") or ""), license=str(info.get("license") or ""),
                                ref_text=str(meta.get("ref_sample_text") or ""), adapter_scale=_scale(info.get("adapter_scale")),
                                repo_id=str(info.get("repo_id") or "")))
    out.sort(key=lambda v: v.name.lower())
    return out


def default_single_voice(target_lang: str, root: Optional[Path] = None) -> Optional[LibraryVoice]:
    """The library voice used when one voice reads the whole film and the user chose none: a voice in the dub language with a
    reference clip (first by name; one recorded for this language beats a multilingual one).  None -> clone the film's voice."""
    usable = [v for v in list_library(root) if v.ref_audio is not None]
    same = [v for v in usable if (v.language or "").lower()[:2] == (target_lang or "").lower()[:2]]
    return same[0] if same else None


def get_library_voice(voice_id: str, root: Optional[Path] = None) -> Optional[LibraryVoice]:
    return next((v for v in list_library(root) if v.id == voice_id), None)


def pick_reference_lines(lines: Sequence[Line], speaker: Optional[str], max_s: float = REF_MAX_S,
                         wav: Optional[Path] = None, sr: int = 24000) -> List[Line]:
    """Lines for the cloning reference: one voice, longest first, up to ``max_s``.

    The anchor is the longest candidate. A line stays only when its speaker embedding is at least ``REF_SIM_MIN``
    (cosine) from that anchor. Lines of ``REF_PREF_S`` (3 s) and more are preferred. A line of 1.5-3 s is used only
    when it matches the anchor and is not a shout, and only to reach ``REF_MIN_S`` — the 15 s budget is left
    unfilled rather than filled with short or shouted lines. Mixing two voices is never worth those 5 s.
    ``speaker`` None = any speaker (single-voice mode clones the main character). ``wav`` is the speech stem;
    without it every candidate is treated as the anchor's voice (tests that only check duration)."""
    cands = [ln for ln in lines if not ln.keep_original and (speaker is None or ln.speaker == speaker)
             and ln.duration >= REF_MIN_LINE_S - 1e-6]
    if not cands:
        return []
    x: Optional[np.ndarray] = None
    rate = sr
    embs: Dict[int, np.ndarray] = {}
    if wav is not None and Path(wav).is_file():
        x, rate = audio.read(wav, sr)
        embs = _embeddings(cands, x, rate)
    anchor = max(cands, key=lambda ln: (ln.duration, -ln.start))
    anchor_emb = embs.get(anchor.id)
    anchor_rms = _median_rms(_segment(x, rate, anchor), rate) if x is not None else 0.0

    def accept(ln: Line) -> bool:
        if anchor_emb is not None and ln.id in embs and ln.id != anchor.id:
            if _cosine(embs[ln.id], anchor_emb) < REF_SIM_MIN:
                return False
        if ln.duration < REF_PREF_S and x is not None and _is_shouted(_segment(x, rate, ln), rate, anchor_rms):
            return False
        return True

    pool = [ln for ln in cands if accept(ln)]
    long = sorted((ln for ln in pool if ln.duration >= REF_PREF_S - 1e-6), key=lambda ln: -ln.duration)
    short = sorted((ln for ln in pool if ln.duration < REF_PREF_S - 1e-6), key=lambda ln: -ln.duration)
    chosen: List[Line] = []
    total = 0.0
    for ln in long:
        if total + ln.duration > max_s + 1e-6:
            continue
        chosen.append(ln)
        total += ln.duration
    if total < REF_MIN_S:
        for ln in short:
            if total + ln.duration > max_s + 1e-6:
                continue
            chosen.append(ln)
            total += ln.duration
            if total >= REF_MIN_S:
                break
    return sorted(chosen, key=lambda ln: ln.start)


def build_reference(lines: Sequence[Line], speech_wav: Path, out_wav: Path, sr: int = 24000,
                    bounds: Optional[Sequence[Line]] = None) -> Tuple[float, str]:
    """Concatenate the chosen lines from the separated speech stem -> ``out_wav``; returns (seconds, joined text).

    Each chunk is padded by ``REF_PAD_S`` on both sides (clamped to the neighbouring line and to the file) and faded
    in and out over ``REF_FADE_S`` so the join does not click. ``REF_GAP_S`` of silence stays between chunks.
    ``bounds`` are the lines whose edges limit the pad (the whole script); omitted, only ``lines`` themselves."""
    x, rate = audio.read(speech_wav, sr)
    neighbours = list(bounds) if bounds is not None else list(lines)
    parts: List[np.ndarray] = []
    gap = np.zeros(int(round(REF_GAP_S * rate)), dtype=np.float32)
    for ln in lines:
        a, b = _chunk_span(ln, neighbours, len(x), rate)
        seg = x[a:b]
        if len(seg):
            parts += [_fade(seg, rate), gap]
    if not parts:
        return 0.0, ""
    ref = np.concatenate(parts[:-1])
    peak = float(np.abs(ref).max()) or 1.0
    audio.write(out_wav, ref * min(1.0, 0.9 / peak), rate)
    return len(ref) / rate, " ".join(ln.text for ln in lines if ln.text).strip()


def reference_log(n_lines: int, seconds: float, status: str) -> str:
    """One log line for a built reference, e.g. ``reference: 2 lines, 12.3 s, check OK``."""
    return f"reference: {n_lines} lines, {seconds:.1f} s, check {status}"


def assess_reference(lines: Sequence[Line], joined: str, hypothesis: str) -> Tuple[bool, Optional[int], str]:
    """Compare a re-transcription of the reference with the joined line text.

    Returns ``(ok, drop_index, ref_text)``. ``ok`` is false when the normalised word error rate is above
    ``REF_WER_MAX`` or the hypothesis starts with words the reference text does not. ``drop_index`` is then the
    worst chunk (the first chunk when those extra leading words are there). ``ref_text`` is the re-transcription
    when it is the closer match, otherwise the joined line text."""
    hyp = (hypothesis or "").strip()
    err = wer(joined, hyp)
    leading = _extra_leading(joined, hyp)
    ok = bool(hyp) and err <= REF_WER_MAX and not leading
    if ok:
        return True, None, hyp if _hypothesis_closer(joined, hyp) else joined
    return False, _worst_chunk(lines, hyp, leading), joined


# ---------------------------------------------------------------------------------------------- voice match
def _segment(x: Optional[np.ndarray], sr: int, ln: Line) -> np.ndarray:
    if x is None or sr <= 0:
        return np.zeros(0, dtype=np.float32)
    n = len(x)
    a = int(np.clip(round(ln.start * sr), 0, n))
    b = int(np.clip(round(ln.end * sr), 0, n))
    return x[a:b]


def _embeddings(lines: Sequence[Line], x: np.ndarray, sr: int) -> Dict[int, np.ndarray]:
    """One speaker vector per line: pyannote when that model is available, otherwise MFCC mean + pitch."""
    segs = [_segment(x, sr, ln) for ln in lines]
    got = try_speaker_embeddings(segs, sr)
    if got is not None and len(got) == len(lines):
        return {ln.id: np.asarray(e, dtype=np.float32).reshape(-1) for ln, e in zip(lines, got)}
    return {ln.id: _cheap_embedding(seg, sr) for ln, seg in zip(lines, segs)}


def _cheap_embedding(x: np.ndarray, sr: int) -> np.ndarray:
    """MFCC mean (unit length) plus a pitch histogram, so a different voice cannot hide in the timbre."""
    if len(x) < 8:
        return np.zeros(13 + 16, dtype=np.float32)
    mf = _unit(audio.mfcc_stats(x, sr)[:13])
    bins = np.zeros(16, dtype=np.float32)
    f0 = _pitch_hz(x, sr)
    if f0 > 0:
        t = (np.log(f0) - np.log(70.0)) / (np.log(500.0) - np.log(70.0))
        i = int(np.clip(t, 0.0, 0.999) * 16)
        bins[i] = 1.0
        if i + 1 < 16:
            bins[i + 1] = 0.35
        if i > 0:
            bins[i - 1] = 0.35
    return np.concatenate([mf, _unit(bins)])


def _pitch_hz(x: np.ndarray, sr: int) -> float:
    """Rough fundamental (Hz) from the spectrum peak between 70 and 500 Hz; 0 when there is none."""
    y = np.asarray(x, dtype=np.float64)
    if len(y) < 16:
        return 0.0
    y = y - float(y.mean())
    step = max(1, sr // 8000)
    y = y[::step]
    y = y * np.hanning(len(y))
    spec = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), step / sr)
    band = (freqs >= 70.0) & (freqs <= 500.0)
    if not np.any(band) or float(spec[band].max()) <= 0:
        return 0.0
    return float(freqs[band][int(np.argmax(spec[band]))])


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.asarray(v, dtype=np.float32)
    return (np.asarray(v, dtype=np.float32) / n)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _median_rms(x: np.ndarray, sr: int) -> float:
    if len(x) == 0:
        return 0.0
    frame = max(1, int(0.03 * sr))
    vals = [float(np.sqrt(np.mean(np.square(x[i:i + frame].astype(np.float64))) + 1e-12))
            for i in range(0, len(x), frame)]
    return float(np.median(vals)) if vals else 0.0


def _is_shouted(x: np.ndarray, sr: int, anchor_rms: float) -> bool:
    """True when this line's median RMS is more than ``REF_SHOUT_DB`` above the anchor's."""
    if anchor_rms <= 1e-8 or len(x) == 0:
        return False
    return 20.0 * np.log10((_median_rms(x, sr) + 1e-12) / anchor_rms) > REF_SHOUT_DB


def _chunk_span(ln: Line, bounds: Sequence[Line], n: int, sr: int) -> Tuple[int, int]:
    prev_end, next_start = 0.0, (n / sr if sr else ln.end)
    for other in bounds:
        if other.id == ln.id:
            continue
        if other.end <= ln.start + 1e-3:
            prev_end = max(prev_end, float(other.end))
        if other.start >= ln.end - 1e-3:
            next_start = min(next_start, float(other.start))
    a = max(prev_end, ln.start - REF_PAD_S)
    b = min(next_start, ln.end + REF_PAD_S)
    ia = int(np.clip(round(a * sr), 0, n))
    ib = int(np.clip(round(b * sr), 0, n))
    if ib <= ia:
        ia = int(np.clip(round(ln.start * sr), 0, n))
        ib = int(np.clip(round(ln.end * sr), 0, n))
    return ia, ib


def _fade(seg: np.ndarray, sr: int) -> np.ndarray:
    y = np.array(seg, dtype=np.float32, copy=True)
    n = min(len(y) // 2, int(round(REF_FADE_S * sr)))
    if n <= 0:
        return y
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    y[:n] *= ramp
    y[-n:] *= ramp[::-1]
    return y


def _words(text: str) -> List[str]:
    return re.sub(r"[^\w\s']", " ", (text or "").lower()).split()


def _extra_leading(joined: str, hypothesis: str) -> bool:
    """True when the hypothesis has words in front of the reference (a spurious syllable glued on)."""
    ref, hyp = _words(joined), _words(hypothesis)
    if not ref or not hyp or hyp[0] == ref[0]:
        return False
    return ref[0] in hyp[1:]


def _hypothesis_closer(joined: str, hypothesis: str) -> bool:
    """The re-transcription is the closer match when it agrees with the joined text and is not empty."""
    if not hypothesis.strip():
        return False
    return wer(joined, hypothesis) <= REF_WER_MAX


def _worst_chunk(lines: Sequence[Line], hypothesis: str, leading: bool) -> Optional[int]:
    if not lines:
        return None
    if leading:
        return 0
    heard = set(_words(hypothesis))
    scores = []
    for ln in lines:
        words = _words(ln.text)
        scores.append(0.0 if not words else sum(w in heard for w in words) / len(words))
    return int(np.argmin(scores))
