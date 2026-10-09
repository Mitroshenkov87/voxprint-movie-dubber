"""Model registry of the dubber on top of the shared Voxprint model store (:mod:`dubber.infra.model_store`).

The models live in the folder shared with Voxprint AI Audiobook Builder (``%LOCALAPPDATA%\\Voxprint\\models`` by default), one plain
folder per model, so the 4.5 GB TTS base model is downloaded once for both programs.  The HF token (gated pyannote) comes from
the ``HF_TOKEN`` environment variable of the worker process - never from the command line and never written to the report.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from dubber import paths
from dubber.infra import model_store
from dubber.infra.model_store import ModelUnavailable  # noqa: F401  (re-exported)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo: str
    title: str
    size_gb: float
    license: str
    gated: bool = False
    patterns: Tuple[str, ...] = ()          # allow_patterns for the download ('' = everything)
    weights: Tuple[str, ...] = ("*.safetensors", "*.bin")     # at least one of these must exist for the folder to count as complete


OPUS_FILES = ("config.json", "generation_config.json", "tokenizer_config.json", "vocab.json", "source.spm", "target.spm",
              "pytorch_model.bin", "model.safetensors")

SPECS: Dict[str, ModelSpec] = {s.key: s for s in (
    ModelSpec("tts_1_7b", "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "Qwen3-TTS 1.7B Base (voice cloning)", 4.5, "Apache-2.0"),
    ModelSpec("tts_0_6b", "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "Qwen3-TTS 0.6B Base (voice cloning)", 2.5, "Apache-2.0"),
    ModelSpec("asr", "deepdml/faster-whisper-large-v3-turbo-ct2", "faster-whisper large-v3-turbo (CTranslate2)", 1.6, "MIT",
              weights=("model.bin",)),
    ModelSpec("sep", "JusperLee/TIGER-DnR", "TIGER-DnR (dialogue / effects / music separation)", 0.02, "Apache-2.0 (weights), MIT (code)",
              weights=("model.safetensors",)),
    ModelSpec("diar", "pyannote/speaker-diarization-community-1", "pyannote speaker-diarization community-1", 0.05, "CC-BY-4.0, gated",
              gated=True, weights=("*.bin", "*.ckpt", "*.safetensors", "config.yaml")),
    ModelSpec("mt_en_ru", "Helsinki-NLP/opus-mt-en-ru", "Opus-MT en->ru", 0.3, "CC-BY-4.0", patterns=OPUS_FILES),
    ModelSpec("mt_ru_en", "Helsinki-NLP/opus-mt-ru-en", "Opus-MT ru->en", 0.3, "CC-BY-4.0", patterns=OPUS_FILES),
    ModelSpec("mt_en_de", "Helsinki-NLP/opus-mt-en-de", "Opus-MT en->de", 0.3, "CC-BY-4.0", patterns=OPUS_FILES),
    ModelSpec("mt_de_en", "Helsinki-NLP/opus-mt-de-en", "Opus-MT de->en", 0.3, "CC-BY-4.0", patterns=OPUS_FILES),
)}


#: Models fetched by the installer (the gated diarization model needs a Hugging Face token and is left out).
INSTALL_MODELS = ("tts_1_7b", "asr", "sep", "mt_en_ru", "mt_ru_en", "mt_en_de", "mt_de_en")


def mt_spec(source: str, target: str) -> Optional[ModelSpec]:
    """Opus-MT model for a direct pair among en/ru/de (None when there is no direct model)."""
    return SPECS.get(f"mt_{source}_{target}")


def total_download_gb(keys: List[str], missing_only: bool = True) -> float:
    """Approximate download size for ``keys`` (missing models only by default)."""
    return round(sum(SPECS[k].size_gb for k in keys if not (missing_only and locate(SPECS[k].repo))), 1)


def local_dir(repo: str, root: Optional[Path] = None) -> Path:
    """``owner/name`` -> ``<models>/owner--name``."""
    return model_store.local_dir_for(repo, root)


def verify_dir(path: Path, spec: Optional[ModelSpec] = None) -> bool:
    """A folder counts as a complete model when it has a config and at least one weights file (+ manifest sizes, if pinned)."""
    if spec is None:
        return model_store.verify_structure(path)
    return model_store.is_ready(path, spec.repo, spec.weights, spec.patterns or None)


def _spec_for_repo(repo: str) -> Optional[ModelSpec]:
    for s in SPECS.values():
        if s.repo == repo:
            return s
    return None


def locate(repo: str) -> Optional[Path]:
    """Complete copy in the shared models folder (or, read-only, in the dubber's own folder of builds before the shared store)."""
    spec = _spec_for_repo(repo)
    p = local_dir(repo)
    if verify_dir(p, spec):
        return p
    legacy = paths.legacy_models_dir() / model_store.folder_name(repo)
    if verify_dir(legacy, spec):
        return legacy
    return None


def ensure(repo: str, allow_download: bool = True, token: Optional[str] = None,
           log: Callable[[str], None] = lambda m: None) -> Tuple[Path, Dict[str, object]]:
    """Return ``(folder, info)``; ``info`` = ``{"source": "present|downloaded|legacy", "download_s": float}``.  Raises ModelUnavailable."""
    found = locate(repo)
    if found:
        return found, {"source": "legacy folder" if found.parent == paths.legacy_models_dir() else "present", "download_s": 0.0}
    spec = _spec_for_repo(repo)
    if spec and spec.gated and not token:
        token = os.environ.get("HF_TOKEN") or None
        if not token:
            try:
                from huggingface_hub import get_token

                token = get_token()
            except Exception:  # noqa: BLE001
                token = None
    info = model_store.ensure_model(repo, allow_download=allow_download, token=token,
                                    patterns=(spec.patterns or None) if spec else None,
                                    weights=spec.weights if spec else model_store.DEFAULT_WEIGHTS, gated=bool(spec and spec.gated),
                                    size_gb=spec.size_gb if spec else 0.0, log_fn=log)
    return info.path, {"source": info.source, "download_s": info.download_s}
