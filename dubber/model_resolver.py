"""Model roles: role id -> MDL entry in ``mdl.json`` -> local folder.

Every pipeline stage asks this module for its weights. Stage code names a role such
as ``asr`` or ``tts.1_7b``; it never names a repository, a revision or a path. The
repository, the pinned revision, the licence and the files live in ``mdl.json``, which
VOXPrint MDL reads to download them. This module never downloads and never calls
``voxprint_common.mdl.ensure_ready`` (that helper downloads inside the process).

Placeholders: Model Forge will publish the models the suite ships for most roles. Until
those outputs land, ``mdl.json`` points at the current upstream weights and marks those
entries with ``"placeholder": true`` (voxprint-common 0.2.4 ignores unknown fields).
Swapping a placeholder is a ``mdl.json`` change only; no stage code changes.

A missing model raises :class:`ModelUnavailable`. The CLI maps it to exit code 4
(``models``) and the window shows its Manage models notice
(:func:`is_model_missing` recognises the message after it crossed the worker process).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dubber.infra.model_store import ModelUnavailable

#: Hint shown with every missing-model error. ``Manage models`` is the window button.
FETCH_HINT = "Open Manage models, or run: voxprint-mdl enqueue --app movie-dubber"


@dataclass(frozen=True)
class Role:
    """Which ``mdl.json`` variant a role uses: the model id and, for sized models, the VRAM tier."""
    mdl_id: str
    tier: Optional[str] = None


#: Stage roles. Translation roles (``translation.<src>-<tgt>``) are derived from the ``mt-*`` ids.
ROLES: Dict[str, Role] = {
    "tts.1_7b": Role("qwen3-tts", "24gb"),
    "tts.0_6b": Role("qwen3-tts", "16gb"),
    "asr": Role("asr"),
    "separation": Role("sep"),
    "separation.best": Role("roformer"),
    "diarization": Role("diar"),
    "embedding": Role("embed"),
}

#: Registry keys (``dubber.models.SPECS``, settings, CLI) -> role ids.
_KEY_ROLES: Dict[str, str] = {
    "tts_1_7b": "tts.1_7b", "tts_0_6b": "tts.0_6b", "asr": "asr", "sep": "separation",
    "roformer": "separation.best", "diar": "diarization", "embed": "embedding",
}


@dataclass(frozen=True)
class ModelEntry:
    """One resolved ``mdl.json`` variant for a role."""
    role: str
    mdl_id: str
    version: str
    tier: str
    repo: str
    license: str
    display_name: str
    placeholder: bool


def role_for_key(key: str) -> str:
    """Map a registry key such as ``tts_1_7b`` or ``mt_en_ru`` to its role id."""
    if key in _KEY_ROLES:
        return _KEY_ROLES[key]
    if key.startswith("mt_"):
        parts = key.split("_")
        if len(parts) == 3:
            return translation_role(parts[1], parts[2])
    return key


def translation_role(src: str, tgt: str) -> str:
    """Role id of the direct Opus-MT model for one language pair."""
    return f"translation.{src}-{tgt}"


def _binding(role: str) -> Role:
    if role in ROLES:
        return ROLES[role]
    if role.startswith("translation.") and "-" in role:
        src, tgt = role.split(".", 1)[1].split("-", 1)
        return Role(f"mt-{src}-{tgt}")
    raise ModelUnavailable(f"unknown model role {role!r}")


def manifest_path() -> Path:
    """``mdl.json`` next to the program, the same file ``voxprint-mdl --manifest`` gets."""
    from dubber import models

    return models.manifest_path()


_CACHE: Dict[Tuple[str, float], List[ModelEntry]] = {}


def _rows(path: Path) -> List[Tuple[Any, Dict[str, Any]]]:
    """Validated MDL models (voxprint-common parser) paired with their raw ``mdl.json`` rows."""
    from voxprint_common.mdl import load_manifest

    parsed = load_manifest(path)
    raw = json.loads(path.read_text(encoding="utf-8-sig")).get("models") or []
    return list(zip(parsed.models, raw))


def entries(path: Optional[Path] = None) -> List[ModelEntry]:
    """Every ``mdl.json`` variant, cached until the file changes.

    Raises:
        ModelUnavailable: ``mdl.json`` is missing or does not parse.
    """
    target = Path(path) if path is not None else manifest_path()
    try:
        key = (str(target), target.stat().st_mtime)
    except OSError as exc:
        raise ModelUnavailable(f"the model manifest {target} is missing. Reinstall the app.") from exc
    if key in _CACHE:
        return _CACHE[key]
    try:
        rows = _rows(target)
    except Exception as exc:  # noqa: BLE001 - CatalogError, bad JSON: one clear message
        raise ModelUnavailable(f"the model manifest {target} cannot be read: {exc}") from exc
    out = [ModelEntry(role="", mdl_id=m.id, version=m.version, tier=m.tier.value, repo=str(row.get("repo") or ""),
                      license=str(row.get("license") or ""), display_name=m.display_name or m.id,
                      placeholder=bool(row.get("placeholder", False)))
           for m, row in rows]
    _CACHE.clear()
    _CACHE[key] = out
    return out


def entry(role: str, path: Optional[Path] = None) -> ModelEntry:
    """The ``mdl.json`` variant bound to ``role``.

    Raises:
        ModelUnavailable: The role is unknown or ``mdl.json`` has no matching entry.
    """
    want = _binding(role)
    for item in entries(path):
        if item.mdl_id == want.mdl_id and (want.tier is None or item.tier == want.tier):
            if not item.repo:
                raise ModelUnavailable(f"mdl.json entry {item.mdl_id} for role {role} has no repo")
            return ModelEntry(role, item.mdl_id, item.version, item.tier, item.repo, item.license, item.display_name,
                              item.placeholder)
    tier = f" (tier {want.tier})" if want.tier else ""
    raise ModelUnavailable(f"mdl.json has no entry {want.mdl_id}{tier} for model role {role}")


def locate(role: str) -> Optional[Path]:
    """The complete local folder for ``role``, or None when it is not on disk (or not declared)."""
    from dubber import models

    try:
        item = entry(role)
    except ModelUnavailable:
        return None
    return models.locate(item.repo)


def resolve(role: str) -> Path:
    """The local folder of the model bound to ``role``. Does not download.

    Raises:
        ModelUnavailable: The model is not declared or not on disk yet.
    """
    item = entry(role)
    from dubber import models

    found = models.locate(item.repo)
    if found is None:
        raise ModelUnavailable(f"{item.display_name} (model role {role}) is not on this computer. {FETCH_HINT}")
    return found


def is_model_missing(message: str) -> bool:
    """True when a stage error is a missing model (the text crosses the worker process as ``ModelUnavailable: ...``)."""
    return "ModelUnavailable" in (message or "")
