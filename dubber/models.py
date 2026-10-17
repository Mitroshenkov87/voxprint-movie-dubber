"""Model registry. VOXPrint MDL downloads the weights; this module only finds them on disk.

The files live in the suite model store (``VOXPRINT_MODELS_DIR``, otherwise the shared
models folder). ``mdl.json`` at the install root is the app manifest. Qwen3-TTS is
declared there with the Audiobook Builder repository and revision because
voxprint-common 0.2.4 cannot resolve a shared-component id into files.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TypedDict

from dubber.appinfo import resource_dir
from dubber.infra import model_store
from dubber.infra.model_store import ModelUnavailable  # noqa: F401  (re-exported)
from dubber.infra.resources import snapshot
from dubber.infra.vram_tier import detect

APP_ID = "movie-dubber"


@dataclass(frozen=True)
class ModelSpec:
    """One Hugging Face model in the dubber registry: repo, size in gigabytes, license, and completeness rules."""
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
#: Inference files of the diarization mirror. Checksums are the SHA256SUMS of that revision.
DIAR_FILES = ("config.yaml", "embedding/pytorch_model.bin", "segmentation/pytorch_model.bin",
              "plda/plda.npz", "plda/xvec_transform.npz")
ROFORMER_FILES = ("MelBandRoformer.ckpt",)

SPECS: Dict[str, ModelSpec] = {s.key: s for s in (
    ModelSpec("tts_1_7b", "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "Qwen3-TTS 1.7B Base (voice cloning)", 4.5, "Apache-2.0"),
    ModelSpec("tts_0_6b", "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "Qwen3-TTS 0.6B Base (voice cloning)", 2.5, "Apache-2.0"),
    ModelSpec("asr", "deepdml/faster-whisper-large-v3-turbo-ct2", "faster-whisper large-v3-turbo (CTranslate2)", 1.6, "MIT",
              patterns=WHISPER_FILES, weights=("model.bin",)),
    ModelSpec("sep", "JusperLee/TIGER-DnR", "TIGER-DnR (dialogue / effects / music separation)", 0.02,
              "Apache-2.0 (weights), MIT (code)", patterns=TIGER_FILES, weights=("model.safetensors",)),
    ModelSpec("diar", "Mitroshenkov87/voxprint-pyannote-speaker-diarization-community-1",
              "pyannote speaker-diarization community-1", 0.04, "CC-BY-4.0",
              patterns=DIAR_FILES, weights=("segmentation/*.bin", "embedding/*.bin", "config.yaml")),
    ModelSpec("embed", "Mitroshenkov87/voxprint-pyannote-embedding", "pyannote speaker embedding", 0.09, "MIT",
              patterns=EMBED_FILES, weights=("pytorch_model.bin",)),
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


#: Full registry list used by the VRAM-tier helper. ``fetch-models`` with no key list uses :func:`plan_keys`.
#: Both TTS sizes are listed; :func:`dubber.infra.vram_tier.install_models` keeps the one the card can run.
INSTALL_MODELS = ("tts_1_7b", "tts_0_6b", "asr", "sep", "roformer", "diar", "embed",
                  "mt_en_ru", "mt_ru_en", "mt_en_de", "mt_de_en", "mt_ru_de", "mt_de_ru")
#: Speaker models. One-voice dubbing runs without them, so a missing download does not stop startup.
OPTIONAL_MODELS = ("diar", "embed")
REQUIRED_MODELS = tuple(key for key in INSTALL_MODELS if key not in OPTIONAL_MODELS)


def mt_spec(source: str, target: str) -> Optional[ModelSpec]:
    """Opus-MT tc-big model for a direct pair (None when that pair has no tc-big model)."""
    return SPECS.get(f"mt_{source}_{target}")


def mt_inputs(token: str, texts: Sequence[str]) -> List[str]:
    """Sentences as a multi-target Opus-MT model expects them: ``>>rus<< text`` when ``token`` is set."""
    if not token:
        return list(texts)
    return [f"{token} {text}" for text in texts]


def total_download_gb(keys: List[str], missing_only: bool = True) -> float:
    """Approximate size for ``keys`` (missing models only by default). MDL does the download."""
    return round(sum(SPECS[k].size_gb for k in keys if not (missing_only and locate(SPECS[k].repo))), 1)


def manifest_path() -> Path:
    """Return ``mdl.json`` next to the program (the MDL app manifest)."""
    return resource_dir() / "mdl.json"


def mdl_id(key: str) -> str:
    """Map a registry key to the MDL model id (``tts_1_7b`` and ``tts_0_6b`` share ``qwen3-tts``)."""
    if key.startswith("tts_"):
        return "qwen3-tts"
    return key.replace("_", "-")


def plan_keys(total_gb: Optional[float] = None) -> List[str]:
    """Required models for this card. Diarization and embedding stay optional.

    A 24 GB card gets Qwen3-TTS 1.7B. Every other reading, including unknown and 8 GB,
    gets the 0.6B model. This is the set ``voxprint-mdl enqueue`` fetches by default.
    """
    if total_gb is None:
        total_gb = snapshot(use_torch=False).vram_total_gb
    tts = "tts_1_7b" if detect(float(total_gb or 0.0)) == "24gb" else "tts_0_6b"
    keys: List[str] = []
    for key in INSTALL_MODELS:
        if key in OPTIONAL_MODELS:
            continue
        if key in ("tts_1_7b", "tts_0_6b"):
            if key == tts:
                keys.append(key)
            continue
        keys.append(key)
    return keys


def missing_required(total_gb: Optional[float] = None) -> List[str]:
    """Registry keys from :func:`plan_keys` that are not on disk. This does not download."""
    return [key for key in plan_keys(total_gb) if locate(SPECS[key].repo) is None]


def mdl_argv(command: str, *extra: str, model_ids: Optional[Sequence[str]] = None) -> List[str]:
    """Build a ``voxprint-mdl`` command for this app, including ``--manifest``.

    Discovery only scans the suite install folders. ``--manifest`` points at this
    program's ``mdl.json`` so a source checkout and a non-default prefix still work.
    """
    exe = shutil.which("voxprint-mdl")
    argv = [exe] if exe else [sys.executable, "-m", "voxprint_common.mdl"]
    argv.extend([command, "--app", APP_ID, "--manifest", str(manifest_path())])
    for model_id in model_ids or ():
        argv.extend(["--model", model_id])
    argv.extend(extra)
    return argv


def mdl_env(base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Return a copy of the environment with Hugging Face tokens removed."""
    env = dict(os.environ if base is None else base)
    env.pop("HF_TOKEN", None)
    env.pop("HUGGINGFACE_HUB_TOKEN", None)
    return env


def gui_argv() -> List[str]:
    """Build ``voxprint-mdl --gui --app movie-dubber``.

    ``--gui`` is a top-level flag in voxprint-common 0.2.4. It opens the
    localized Manage models window and is not a download subcommand.
    """
    exe = shutil.which("voxprint-mdl")
    argv = [exe] if exe else [sys.executable, "-m", "voxprint_common.mdl"]
    argv.extend(["--gui", "--app", APP_ID])
    return argv


def launch_mdl() -> None:
    """Open ``voxprint-mdl --gui --app movie-dubber``.

    ``voxprint_common.mdl.ensure_ready(app='movie-dubber')`` is not called. That
    function downloads inside the process, and this app has no downloader.

    Raises:
        OSError: The MDL window could not be started.
    """
    subprocess.Popen(gui_argv(), env=mdl_env())


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
    """Complete copy in the shared models folder, including a folder linked from MDL blobs."""
    spec = _spec_for_repo(repo)
    p = local_dir(repo)
    if verify_dir(p, spec):
        return p
    weights = spec.weights if spec else model_store.DEFAULT_WEIGHTS
    patterns = (spec.patterns or None) if spec else None
    require_config = spec.require_config if spec else True
    linked = model_store.link_blobs(repo, weights, patterns, require_config)
    if linked is not None and verify_dir(linked, spec):
        return linked
    return None


class EnsureInfo(TypedDict):
    """How :func:`ensure` resolved a model: a source label and the download time in seconds."""
    source: str
    download_s: float


def ensure(repo: str, allow_download: bool = True, token: Optional[str] = None,
           log: Callable[[str], None] = lambda m: None) -> Tuple[Path, EnsureInfo]:
    """Return ``(folder, info)`` when the model is already on disk. Raises ModelUnavailable otherwise.

    ``allow_download`` and ``token`` are ignored. This function does not download and
    does not call ``ensure_ready(app='movie-dubber')``. Ask VOXPrint MDL:
    ``voxprint-mdl enqueue --app movie-dubber``.
    """
    del allow_download, token, log
    found = locate(repo)
    if found:
        return found, {"source": "present", "download_s": 0.0}
    raise ModelUnavailable(
        f"{repo} is not on this computer. VOXPrint MDL owns downloads. "
        "Run: voxprint-mdl enqueue --app movie-dubber"
    )
