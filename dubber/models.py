"""Model registry, locator and downloader.

Layout: ``<models>/<owner>--<name>/`` (same as Voxprint Audiobook Builder, so its copies are re-used read-only: the 4.5 GB TTS base
model is not downloaded twice).  Order: own folder -> other Voxprint programs' folders -> download from Hugging Face
(only when allowed).  A download goes to ``<name>.partial`` and is renamed after the check, so a half-downloaded model is never
used.  The HF token (gated pyannote) comes from the ``HF_TOKEN`` environment variable of the worker process - never from
the command line and never written to the report.
"""
from __future__ import annotations

import os
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from dubber import paths


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


def mt_spec(source: str, target: str) -> Optional[ModelSpec]:
    """Opus-MT model for a direct pair among en/ru/de (None when there is no direct model)."""
    return SPECS.get(f"mt_{source}_{target}")


def total_download_gb(keys: List[str], missing_only: bool = True) -> float:
    """Approximate download size for ``keys`` (missing models only by default)."""
    return round(sum(SPECS[k].size_gb for k in keys if not (missing_only and locate(SPECS[k].repo))), 1)


def local_dir(repo: str, root: Optional[Path] = None) -> Path:
    """``owner/name`` -> ``<models>/owner--name``."""
    return (root or paths.models_dir()) / repo.replace("/", "--")


def verify_dir(path: Path, spec: Optional[ModelSpec] = None) -> bool:
    """A folder counts as a complete model when it has a config and at least one weights file."""
    if not path.is_dir():
        return False
    weights = spec.weights if spec else ("*.safetensors", "*.bin")
    has_weights = any(any(path.glob(w)) for w in weights)
    has_config = any((path / n).exists() for n in ("config.json", "config.yaml", "params.json"))
    return has_weights and has_config


def _spec_for_repo(repo: str) -> Optional[ModelSpec]:
    for s in SPECS.values():
        if s.repo == repo:
            return s
    return None


def locate(repo: str) -> Optional[Path]:
    """Existing complete copy: own folder first, then sibling programs' folders; None when absent."""
    spec = _spec_for_repo(repo)
    own = local_dir(repo)
    if verify_dir(own, spec):
        return own
    for root in paths.foreign_model_roots():
        p = local_dir(repo, root)
        if verify_dir(p, spec):
            return p
    return None


class ModelUnavailable(RuntimeError):
    """The model is missing and cannot be fetched (downloads disabled, no token for a gated model, network error)."""


def _dir_size(p: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(p):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def ensure(repo: str, allow_download: bool = True, token: Optional[str] = None,
           log: Callable[[str], None] = lambda m: None) -> Tuple[Path, Dict[str, object]]:
    """Return ``(folder, info)`` for ``repo``; ``info`` = ``{"source": "own|foreign|downloaded", "download_s": float, "path": str}``.

    Raises :class:`ModelUnavailable` with an explanatory message otherwise.
    """
    found = locate(repo)
    if found:
        src = "own" if found == local_dir(repo) else f"reused from {found.parent}"
        return found, {"source": src, "download_s": 0.0}
    spec = _spec_for_repo(repo)
    if not allow_download:
        raise ModelUnavailable(f"{repo} is not on this computer and downloading is switched off")
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or None
    if spec and spec.gated and not token:
        try:
            from huggingface_hub import get_token

            token = get_token()
        except Exception:  # noqa: BLE001
            token = None
        if not token:
            raise ModelUnavailable(f"{repo} is a gated model: accept its terms on huggingface.co and provide a Hugging Face token "
                                   f"(UI field 'HF token' or the HF_TOKEN environment variable)")
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # noqa: BLE001
        raise ModelUnavailable(f"huggingface_hub is not installed: {exc}") from exc
    final = local_dir(repo)
    part = final.with_name(final.name + ".partial")
    part.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    size_gb = spec.size_gb if spec else 0.0

    def watch() -> None:
        while not stop.wait(5.0):
            log(f"downloading {repo}: {_dir_size(part) / 1024 ** 3:.2f}" + (f" / ~{size_gb:.1f}" if size_gb else "") + " GB")

    w = threading.Thread(target=watch, daemon=True)
    w.start()
    t0 = time.time()
    try:
        kw = dict(repo_id=repo, local_dir=str(part), token=token)
        if spec and spec.patterns:
            kw["allow_patterns"] = list(spec.patterns)
        snapshot_download(**kw)               # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001
        raise ModelUnavailable(f"download of {repo} failed: {type(exc).__name__}: {' '.join(str(exc).split())[:160]}") from exc
    finally:
        stop.set()
    if not verify_dir(part, spec):
        raise ModelUnavailable(f"{repo}: the download finished but the folder looks incomplete ({part})")
    if final.exists():
        shutil.rmtree(final, ignore_errors=True)
    os.replace(part, final)
    return final, {"source": "downloaded", "download_s": round(time.time() - t0, 1)}
