"""Model registry of the dubber on top of the shared Voxprint model store (:mod:`dubber.infra.model_store`).

The models live in the folder shared with Voxprint AI Audiobook Builder (``%LOCALAPPDATA%\\Voxprint\\models`` by default), one plain
folder per model, so the 4.5 GB TTS base model is downloaded once for both programs.  The HF token (gated pyannote) comes from
the ``HF_TOKEN`` environment variable of the worker process - never from the command line and never written to the report.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TypedDict

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
    target_token: str = ""                 # multi-target Opus-MT: ``>>rus<<`` in front of every sentence
    require_config: bool = True            # False for a checkpoint that has no config.json (Mel-Band RoFormer)


#: Inference files of an Opus-MT tc-big model. The duplicate pytorch_model.bin / tf_model.h5 are not downloaded.
TC_BIG_FILES = ("config.json", "generation_config.json", "tokenizer_config.json", "vocab.json",
                "special_tokens_map.json", "source.spm", "target.spm", "model.safetensors")
WHISPER_FILES = ("config.json", "preprocessor_config.json", "tokenizer.json", "vocabulary.json", "model.bin")
TIGER_FILES = ("config.json", "model.safetensors")
EMBED_FILES = ("config.yaml", "pytorch_model.bin")
ROFORMER_FILES = ("MelBandRoformer.ckpt",)

SPECS: Dict[str, ModelSpec] = {s.key: s for s in (
    ModelSpec("tts_1_7b", "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "Qwen3-TTS 1.7B Base (voice cloning)", 4.5, "Apache-2.0"),
    ModelSpec("tts_0_6b", "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "Qwen3-TTS 0.6B Base (voice cloning)", 2.5, "Apache-2.0"),
    ModelSpec("asr", "deepdml/faster-whisper-large-v3-turbo-ct2", "faster-whisper large-v3-turbo (CTranslate2)", 1.6, "MIT",
              patterns=WHISPER_FILES, weights=("model.bin",)),
    ModelSpec("sep", "JusperLee/TIGER-DnR", "TIGER-DnR (dialogue / effects / music separation)", 0.02,
              "Apache-2.0 (weights), MIT (code)", patterns=TIGER_FILES, weights=("model.safetensors",)),
    ModelSpec("diar", "pyannote/speaker-diarization-community-1", "pyannote speaker-diarization community-1", 0.05, "CC-BY-4.0, gated",
              gated=True, weights=("*.bin", "*.ckpt", "*.safetensors", "config.yaml")),
    ModelSpec("embed", "pyannote/embedding", "pyannote speaker embedding", 0.09, "MIT, gated",
              gated=True, patterns=EMBED_FILES, weights=("pytorch_model.bin",)),
    ModelSpec("roformer", "KimberleyJSN/melbandroformer", "Mel-Band RoFormer (vocals)", 0.85, "MIT",
              patterns=ROFORMER_FILES, weights=("*.ckpt",), require_config=False),
    ModelSpec("mt_en_ru", "Helsinki-NLP/opus-mt-tc-big-en-zle", "Opus-MT tc-big en->ru", 0.5, "CC-BY-4.0",
              patterns=TC_BIG_FILES, target_token=">>rus<<"),
    ModelSpec("mt_ru_en", "Helsinki-NLP/opus-mt-tc-big-zle-en", "Opus-MT tc-big ru->en", 0.5, "CC-BY-4.0", patterns=TC_BIG_FILES),
    ModelSpec("mt_ru_de", "Helsinki-NLP/opus-mt-tc-big-zle-de", "Opus-MT tc-big ru->de", 0.5, "CC-BY-4.0", patterns=TC_BIG_FILES),
    ModelSpec("mt_de_ru", "Helsinki-NLP/opus-mt-tc-big-de-zle", "Opus-MT tc-big de->ru", 0.5, "CC-BY-4.0",
              patterns=TC_BIG_FILES, target_token=">>rus<<"),
    ModelSpec("mt_en_de", "Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-gmw", "Opus-MT tc-bible-big en->de", 0.9,
              "Apache-2.0", patterns=TC_BIG_FILES, target_token=">>deu<<"),
    ModelSpec("mt_de_en", "Helsinki-NLP/opus-mt-tc-bible-big-gmw-deu_eng_fra_por_spa", "Opus-MT tc-bible-big de->en", 0.9,
              "Apache-2.0", patterns=TC_BIG_FILES, target_token=">>eng<<"),
)}


#: Models fetched by the installer. Gated models (diarization, speaker embedding) need a Hugging Face token and are left out.
#: Mel-Band RoFormer is optional and is fetched when separation is set to it.
INSTALL_MODELS = ("tts_1_7b", "asr", "sep", "mt_en_ru", "mt_ru_en", "mt_en_de", "mt_de_en", "mt_ru_de", "mt_de_ru")


def mt_spec(source: str, target: str) -> Optional[ModelSpec]:
    """Opus-MT tc-big model for a direct pair (None when that pair has no tc-big model)."""
    return SPECS.get(f"mt_{source}_{target}")


def mt_inputs(token: str, texts: Sequence[str]) -> List[str]:
    """Sentences as a multi-target Opus-MT model expects them: ``>>rus<< text`` when ``token`` is set."""
    if not token:
        return list(texts)
    return [f"{token} {text}" for text in texts]


def total_download_gb(keys: List[str], missing_only: bool = True) -> float:
    """Approximate download size for ``keys`` (missing models only by default)."""
    return round(sum(SPECS[k].size_gb for k in keys if not (missing_only and locate(SPECS[k].repo))), 1)


def local_dir(repo: str, root: Optional[Path] = None) -> Path:
    """``owner/name`` -> ``<models>/owner--name``."""
    spec = _spec_for_repo(repo)
    if spec is None:
        return model_store.local_dir_for(repo, root)
    return model_store.local_dir_for(repo, root, spec.weights, spec.require_config)


def verify_dir(path: Path, spec: Optional[ModelSpec] = None) -> bool:
    """A folder counts as a complete model when it has the expected weights (+ manifest checksums, if pinned)."""
    if spec is None:
        return model_store.verify_structure(path)
    return model_store.is_ready(path, spec.repo, spec.weights, spec.patterns or None, spec.require_config)


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


class EnsureInfo(TypedDict):
    source: str
    download_s: float


def ensure(repo: str, allow_download: bool = True, token: Optional[str] = None,
           log: Callable[[str], None] = lambda m: None) -> Tuple[Path, EnsureInfo]:
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
                                    size_gb=spec.size_gb if spec else 0.0, log_fn=log,
                                    require_config=spec.require_config if spec else True)
    return info.path, {"source": info.source, "download_s": info.download_s}
