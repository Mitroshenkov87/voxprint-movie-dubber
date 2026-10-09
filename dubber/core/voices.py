"""Voices: the shared Voxprint voice library (read-only) and reference clips cloned from a character's own lines.

The library is the one of Voxprint AI Audiobook Builder: ``<Voxprint home>/voices/<id>/`` with a LoRA adapter
(``adapter_model.safetensors`` + ``adapter_config.json``), ``ref_sample.wav`` (+ its text in ``training_meta.json``
``ref_sample_text``) and ``voice.json`` (name, language, gender ...).  The dubber never writes there.
A library voice is used with its adapter on the Qwen3-TTS base model it was trained on (``voice.json`` ``base_model``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from dubber.core import audio
from dubber.core.project import Line
from dubber.infra import shared_paths

REQUIRED_FILES = ("adapter_model.safetensors", "adapter_config.json")
REF_MIN_S, REF_MAX_S = 5.0, 15.0          # research note 03: a reference of <= 15 s clones best


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
                                ref_text=str(meta.get("ref_sample_text") or "")))
    out.sort(key=lambda v: v.name.lower())
    return out


def get_library_voice(voice_id: str, root: Optional[Path] = None) -> Optional[LibraryVoice]:
    return next((v for v in list_library(root) if v.id == voice_id), None)


def pick_reference_lines(lines: Sequence[Line], speaker: Optional[str], max_s: float = REF_MAX_S) -> List[Line]:
    """Lines for the cloning reference: clean speech (not songs), 1.5-8 s each, longest first, up to ``max_s`` in total.
    ``speaker`` None = any speaker (single-voice mode clones the main character)."""
    pool = [ln for ln in lines if not ln.keep_original and (speaker is None or ln.speaker == speaker) and 1.5 <= ln.duration <= 8.0]
    pool.sort(key=lambda ln: -ln.duration)
    chosen, total = [], 0.0
    for ln in pool:
        if total + ln.duration > max_s:
            continue
        chosen.append(ln)
        total += ln.duration
    return sorted(chosen, key=lambda ln: ln.start)


def build_reference(lines: Sequence[Line], speech_wav: Path, out_wav: Path, sr: int = 24000) -> Tuple[float, str]:
    """Concatenate the chosen lines from the separated speech stem (0.25 s gaps) -> ``out_wav``; returns (seconds, joined text)."""
    x, rate = audio.read(speech_wav, sr)
    parts: List[np.ndarray] = []
    gap = np.zeros(int(0.25 * sr), dtype=np.float32)
    for ln in lines:
        seg = x[int(ln.start * sr):int(ln.end * sr)]
        if len(seg):
            parts += [seg, gap]
    if not parts:
        return 0.0, ""
    ref = np.concatenate(parts[:-1])
    peak = float(np.abs(ref).max()) or 1.0
    audio.write(out_wav, ref * min(1.0, 0.9 / peak), sr)
    return len(ref) / sr, " ".join(ln.text for ln in lines if ln.text).strip()
