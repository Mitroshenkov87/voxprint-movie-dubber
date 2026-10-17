"""Who speaks when: the ungated pyannote community-1 mirror, or MFCC clustering per line."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from dubber import model_resolver
from dubber.core import audio
from dubber.core.project import Line
from dubber.core.script import cluster_speakers

#: Gated upstream ids a pipeline config may name. The value is the subfolder in the diarization mirror.
_GATED_SUBMODELS: Tuple[Tuple[str, str], ...] = (
    ("pyannote/wespeaker-voxceleb-resnet34-LM", "embedding"),
    ("pyannote/segmentation-3.0", "segmentation"),
    ("pyannote/speaker-diarization-community-1", ""),
    ("pyannote/embedding", "embedding"),
)


def rewrite_pipeline_config(text: str, folder: Path) -> str:
    """Point gated Hugging Face ids in a pipeline config at local files under ``folder``.

    ``$model/...`` is left unchanged. pyannote binds that prefix to the pipeline directory,
    which is the diarization mirror, so embedding, segmentation and PLDA load from that folder.
    The checksummed ``config.yaml`` is not modified by this function.
    """
    out = text
    root = folder.resolve()
    for repo_id, sub in _GATED_SUBMODELS:
        local = root if not sub else (root / sub)
        quoted = '"' + local.as_posix() + '"'
        out = re.sub(rf"(?<![\w./-]){re.escape(repo_id)}(?![\w.-])", quoted, out)
    return out


def pipeline_checkpoint(folder: Path) -> Path:
    """Directory or rewritten config to pass to ``Pipeline.from_pretrained``.

    When the config already uses ``$model/...``, the directory is returned and pyannote
    resolves those subfolders locally. A gated repo id is rewritten into ``config.local.yaml``
    in the same folder; ``config.yaml`` itself stays byte-identical for the checksum.
    """
    config = folder / "config.yaml"
    text = config.read_text(encoding="utf-8")
    rewritten = rewrite_pipeline_config(text, folder)
    if rewritten == text:
        return folder
    dest = folder / "config.local.yaml"
    dest.write_text(rewritten, encoding="utf-8")
    return dest


def load_diar_pipeline(folder: Path) -> Any:
    """Load the community-1 pipeline from the local mirror. No Hugging Face token is sent.

    pyannote.audio is imported here because a partial install does not have it.
    """
    from pyannote.audio import Pipeline

    pipe = Pipeline.from_pretrained(str(pipeline_checkpoint(folder)), token=False)
    if pipe is None:
        raise RuntimeError("Pipeline.from_pretrained returned None")
    return pipe


def pyannote_turns(wav16: str, device: str, allow_download: bool, log: Callable[[str], None],
                   num_speakers: Optional[int] = None) -> List[Dict[str, Any]]:
    """Diarize a 16 kHz WAV with pyannote and return each turn's start, end, and speaker, in seconds. ``num_speakers`` asks for exactly that many when set."""
    import torch

    del allow_download, log
    folder = model_resolver.resolve("diarization")
    pipe = load_diar_pipeline(Path(folder))
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
        folder = model_resolver.resolve("embedding")
        model = Model.from_pretrained(str(folder), token=False)
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
    except Exception:  # noqa: BLE001 - no weights, or a CPU-only trial: the caller has a fallback
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
