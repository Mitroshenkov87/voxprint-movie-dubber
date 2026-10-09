"""'Actor-like voice' (multi-voice mode option): a temporary per-project voice that sounds like the actor speaking the dub language.

For one speaker it blends
* the actor's own clean reference clip (built from his lines; Qwen3-TTS clones cross-lingually from it), and
* the closest voice of the shared library (cosine similarity of speaker embeddings: the actor's x-vector against each library
  voice's ``speaker_centroid.safetensors``; voices of the other gender are skipped when the gender is known),
by mixing the two speaker embeddings (``weight`` = share of the actor, default 0.7) and running the library LoRA adapter at a
reduced strength (``library scale x (1 - weight)``).  When the actor clip is too short or too noisy the library voice is used
as is (fallback).

The temporary voice lives in ``<project>/voices/actor_<speaker>/`` (record + embeddings) and disappears with the project;
:func:`save_to_library` turns it into a normal library voice (Audiobook Builder format) only when the user asks for it and
confirms that he may use this person's voice (EULA: the user is responsible for consent / rights).

The model-dependent part (the actor's x-vector, the prompt) runs in the TTS engine (``dubber.engines.tts``); this module is numpy
only.  NEEDS VALIDATION ON A GPU: blend weight, adapter strength and the quality thresholds are first guesses.
"""
from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_WEIGHT = 0.7          # share of the actor's timbre
MIN_REF_S = 4.0               # shorter actor clips -> library fallback
MIN_SNR_DB = 12.0             # noisier actor clips -> library fallback
CENTROID_FILE = "speaker_centroid.safetensors"
RECORD = "actor_voice.json"


@dataclass
class Candidate:
    id: str
    path: Path
    gender: str = ""
    adapter_scale: float = 1.0


def folder(project_folder: Path, speaker_id: str) -> Path:
    return Path(project_folder) / "voices" / f"actor_{speaker_id}"


def ref_quality(x: np.ndarray, sr: int) -> Tuple[float, float, bool]:
    """(seconds, rough SNR in dB, good enough).  SNR = loud frames vs quiet frames (speech vs the floor between words)."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    secs = len(x) / float(sr) if sr else 0.0
    hop = max(1, int(0.02 * sr))
    n = len(x) // hop
    if n < 10:
        return secs, 0.0, False
    e = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    snr = 20 * np.log10(np.percentile(e, 90) / max(np.percentile(e, 10), 1e-6))
    return secs, float(snr), bool(secs >= MIN_REF_S and snr >= MIN_SNR_DB)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, np.float32).reshape(-1), np.asarray(b, np.float32).reshape(-1)
    if a.size != b.size or not a.any() or not b.any():
        return -1.0
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def load_centroid(path: Path) -> Optional[np.ndarray]:
    f = Path(path) / CENTROID_FILE
    if not f.is_file():
        return None
    try:
        from safetensors.numpy import load_file

        return np.asarray(load_file(str(f))["speaker_embedding"], dtype=np.float32).reshape(-1)
    except Exception:  # noqa: BLE001
        return None


def pick_closest(actor: Optional[np.ndarray], candidates: Sequence[Candidate], gender: str = "") -> Tuple[Optional[Candidate], float]:
    """Library voice with the most similar embedding (same gender first when known); without an embedding: the first of that gender."""
    pool = [c for c in candidates if not gender or not c.gender or c.gender == gender] or list(candidates)
    if not pool:
        return None, -1.0
    if actor is None:
        return pool[0], -1.0
    scored = [(cosine(actor, emb), c) for c in pool for emb in [load_centroid(c.path)] if emb is not None]
    if not scored:
        return pool[0], -1.0
    score, best = max(scored, key=lambda t: t[0])
    return best, score


def blend(actor: np.ndarray, library: Optional[np.ndarray], weight: float) -> np.ndarray:
    """Weighted mix of two speaker embeddings, rescaled to the actor's norm (the model expects that magnitude)."""
    a = np.asarray(actor, np.float32).reshape(-1)
    if library is None or library.size != a.size:
        return a
    w = float(min(1.0, max(0.0, weight)))
    norm_a = float(np.linalg.norm(a))
    norm_l = float(np.linalg.norm(library))
    mix = w * a / max(norm_a, 1e-9) + (1.0 - w) * library / max(norm_l, 1e-9)
    norm_m = float(np.linalg.norm(mix))
    return (mix / max(norm_m, 1e-9) * norm_a).astype(np.float32)


def adapter_scale(library_scale: float, weight: float) -> float:
    return round(max(0.0, float(library_scale) * (1.0 - float(weight))), 3)


def write_record(dirpath: Path, data: Dict[str, Any], actor_emb: Optional[np.ndarray] = None,
                 blended: Optional[np.ndarray] = None) -> None:
    d = Path(dirpath)
    d.mkdir(parents=True, exist_ok=True)
    if actor_emb is not None:
        np.save(d / "actor_embedding.npy", np.asarray(actor_emb, np.float32))
    if blended is not None:
        np.save(d / "blended_embedding.npy", np.asarray(blended, np.float32))
    tmp = d / f".{RECORD}.{uuid.uuid4().hex[:8]}.tmp"
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(d / RECORD)


def read_record(dirpath: Path) -> Dict[str, Any]:
    try:
        return json.loads((Path(dirpath) / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_to_library(dirpath: Path, name: str, consent_note: str, library_root: Path, language: str = "") -> Path:
    """Turn a made actor voice into a library voice (Audiobook Builder format).  ``consent_note`` is required (EULA)."""
    if not consent_note.strip():
        raise ValueError("a consent / rights note is required to keep a voice of a real person")
    d = Path(dirpath)
    rec = read_record(d)
    lib = Path(rec.get("library_path") or "")
    if not rec or not (lib / "adapter_model.safetensors").is_file():
        raise ValueError("this actor voice has not been made yet (dub at least one line first)")
    slug = "".join(c if c.isalnum() else "-" for c in name.lower()).strip("-") or "actor"
    target = Path(library_root) / slug
    n = 2
    while target.exists():
        target, n = Path(library_root) / f"{slug}-{n}", n + 1
    stage = Path(library_root) / f".{slug}.{uuid.uuid4().hex[:8]}.tmp"
    stage.mkdir(parents=True)
    try:
        for f in ("adapter_model.safetensors", "adapter_config.json"):
            shutil.copy2(lib / f, stage / f)
        if rec.get("ref_audio") and Path(rec["ref_audio"]).is_file():
            shutil.copy2(rec["ref_audio"], stage / "ref_sample.wav")
        emb_file = d / "blended_embedding.npy"
        if emb_file.is_file():
            from safetensors.numpy import save_file

            save_file({"speaker_embedding": np.load(emb_file).astype(np.float32)}, str(stage / CENTROID_FILE))
        (stage / "training_meta.json").write_text(json.dumps({"ref_sample_audio": "ref_sample.wav",
                                                              "ref_sample_text": rec.get("ref_text", "")}, ensure_ascii=False),
                                                  encoding="utf-8")
        info = {"schema": 3, "id": target.name, "name": name, "language": language, "base_model": rec.get("base_model", ""),
                "adapter_scale": rec.get("adapter_scale", 0.5), "license": "personal use",
                "description": f"Actor-like voice made by Voxprint AI Movie Dubber from a film speaker blended with '{rec.get('library_id', '')}'.",
                "prepared_by": "Voxprint AI Movie Dubber", "commercial_use": False,
                "consent": {"scope": "personal", "method": "user statement", "confirmed": True, "statement": consent_note.strip()}}
        (stage / "voice.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        stage.replace(target)
        return target
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def candidates_from_library(lib_voices: List[Any]) -> List[Candidate]:
    """Library voices that can serve as the blend partner (they need a centroid to be compared; others still count as fallback)."""
    return [Candidate(v.id, Path(v.path), (v.gender or "").lower(), float(getattr(v, "adapter_scale", 1.0))) for v in lib_voices]
