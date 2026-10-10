"""Who speaks when: pyannote community-1 (gated, needs the user's HF token) or the fallback MFCC clustering per line."""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from dubber import models
from dubber.core import audio
from dubber.core.project import Line
from dubber.core.script import cluster_speakers


def pyannote_turns(wav16: str, device: str, allow_download: bool, log: Callable[[str], None],
                   num_speakers: Optional[int] = None) -> List[Dict[str, Any]]:
    """Diarize a 16 kHz WAV with pyannote and return each turn's start, end, and speaker, in seconds. ``num_speakers`` asks for exactly that many when set."""
    import torch
    from pyannote.audio import Pipeline

    repo = models.SPECS["diar"].repo
    folder, _ = models.ensure(repo, allow_download, token=os.environ.get("HF_TOKEN") or None, log=log)
    try:
        pipe = Pipeline.from_pretrained(str(folder))
    except Exception:  # noqa: BLE001
        pipe = Pipeline.from_pretrained(repo, token=os.environ.get("HF_TOKEN") or None)
    if hasattr(pipe, "embedding_batch_size"):
        pipe.embedding_batch_size = 4
    pipe.to(torch.device(device))
    x, _ = audio.read(wav16, 16000)
    kw = {"num_speakers": num_speakers} if num_speakers else {}
    out = pipe({"waveform": torch.from_numpy(x)[None, :], "sample_rate": 16000}, **kw)
    ann = getattr(out, "exclusive_speaker_diarization", None) or getattr(out, "speaker_diarization", None) or out
    return [{"start": round(t.start, 3), "end": round(t.end, 3), "speaker": str(lab)} for t, _, lab in ann.itertracks(yield_label=True)]


def try_speaker_embeddings(segments: Sequence[np.ndarray], sr: int) -> Optional[List[np.ndarray]]:
    """One pyannote speaker embedding per waveform, or None when that model is not available.

    The reference picker uses this and falls back to an MFCC + pitch vector, so tests run without the model."""
    try:
        import torch
        from pyannote.audio import Inference, Model
    except ImportError:
        return None
    if not segments:
        return []
    try:
        spec = models.SPECS["embed"]
        folder, _ = models.ensure(spec.repo, token=os.environ.get("HF_TOKEN") or None)
        model = Model.from_pretrained(str(folder))
        infer = Inference(model, window="whole")
        out: List[np.ndarray] = []
        for seg in segments:
            wav = np.asarray(seg, dtype=np.float32).reshape(-1)
            if len(wav) < 16:
                out.append(np.zeros(1, dtype=np.float32))
                continue
            emb = infer({"waveform": torch.from_numpy(wav)[None, :], "sample_rate": int(sr)})
            out.append(np.asarray(emb, dtype=np.float32).reshape(-1))
        return out
    except Exception:  # noqa: BLE001 - no weights, no token, CPU-only trial: the caller has a fallback
        return None


def cluster_lines(lines: Sequence[Line], speech_wav: str, threshold: float = 0.35) -> Dict[int, str]:
    """Fallback: one MFCC fingerprint per line, clustered; returns line id -> ``S1``, ``S2``... (S1 = most lines)."""
    x, sr = audio.read(speech_wav, 16000)
    feats, ids = [], []
    for ln in lines:
        if ln.keep_original:
            continue
        seg = x[int(ln.start * sr):int(ln.end * sr)]
        if len(seg) < sr // 4:
            continue
        feats.append(audio.mfcc_stats(seg, sr))
        ids.append(ln.id)
    if not feats:
        return {}
    labels = cluster_speakers(np.stack(feats), threshold)
    return {i: f"S{lab + 1}" for i, lab in zip(ids, labels)}
