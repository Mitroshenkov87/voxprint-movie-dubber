"""Settings shared by every Voxprint program (Audiobook Builder, Movie Dubber): ``<Voxprint home>/state/suite.json``.

Source: voxprint-audiobook-builder@578646515234 ``infra/suite_settings.py`` (agreed spec v1; adapted to this program's modules).
``%LOCALAPPDATA%\\Voxprint\\state\\suite.json`` on Windows, ``~/.local/share/voxprint/state/suite.json`` on Linux::

    {"schema": 1,
     "ui_language": "ru",            # ISO code; default: the OS language if supported, else "en"
     "theme": "glass-dark",          # unknown -> the program's default theme ("glass-dark", our only theme)
     "models_dir": null,             # absolute path, or null = the default folder; VOXPRINT_MODELS_DIR always wins
     "gpu": "auto"}                  # "auto" | "cuda:N" | "cpu"

* UTF-8, written atomically (temp file + ``os.replace``); keys this program does not know are kept as they are.
* A missing key means its default; a missing or damaged file means all defaults (and is rewritten on the next change).
* ``models_dir`` stays null unless the user picks a folder; the older ``state/models_dir.txt`` still counts when the file has no
  ``models_dir`` (:func:`dubber.infra.shared_paths.configured_models_dir`).
* App-specific settings never go here.

Stdlib only; nothing here may fail the start of the program.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger("dubber.suite")

SCHEMA = 1
FILENAME = "suite.json"
KEYS = ("ui_language", "theme", "models_dir", "gpu")
DEFAULT_THEME = "glass-dark"
THEMES = (DEFAULT_THEME,)
DEFAULT_GPU = "auto"
_GPU_RE = re.compile(r"^cuda:(\d{1,2})$")


def path() -> Path:
    """Return the path of ``<Voxprint home>/state/suite.json``."""
    from dubber.infra import shared_paths

    return shared_paths.voxprint_home() / "state" / FILENAME


def read_raw() -> Dict[str, Any]:
    """The file as a dict (``{}`` when missing, unreadable or not an object; a BOM is accepted)."""
    try:
        data = json.loads(path().read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def has(key: str) -> bool:
    """True when ``key`` is written in the file (``models_dir: null`` counts: it is an explicit "default folder")."""
    return key in read_raw()


# ------------------------------------------------------------------------------------------------- normalizers
def normalize_language(value: object) -> Optional[str]:
    """Return a supported UI language code, or None when ``value`` is not one."""
    from dubber.i18n import normalize_code

    return normalize_code(value) if isinstance(value, str) else None


def default_language() -> str:
    """Return the OS language when this program has that catalog, otherwise English."""
    from dubber.i18n import DEFAULT_LANG, system_language

    return system_language() or DEFAULT_LANG


def normalize_theme(value: object) -> str:
    """Return ``value`` when it is a known theme, otherwise ``glass-dark``."""
    return value if isinstance(value, str) and value in THEMES else DEFAULT_THEME


def normalize_gpu(value: object) -> Optional[str]:
    """``auto`` / ``cpu`` / ``cuda:N`` (``cuda`` alone = ``cuda:0``), or ``None`` for anything else."""
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    if v in ("auto", "cpu"):
        return v
    if v in ("cuda", "gpu"):
        return "cuda:0"
    m = _GPU_RE.match(v)
    return f"cuda:{int(m.group(1))}" if m else None


def normalize_models_dir(value: object) -> Optional[Path]:
    """An absolute path, or ``None`` (null, empty or relative = the default folder)."""
    if not isinstance(value, str) or not value.strip():
        return None
    p = Path(os.path.expandvars(os.path.expanduser(value.strip().strip('"'))))
    return p if p.is_absolute() else None


# ------------------------------------------------------------------------------------------------- reading
def ui_language() -> Optional[str]:
    """The shared UI language if the file names one this program supports, else None (the caller falls back)."""
    return normalize_language(read_raw().get("ui_language"))


def theme() -> str:
    """Return the shared theme from suite.json, or ``glass-dark``."""
    return normalize_theme(read_raw().get("theme"))


def models_dir() -> Optional[Path]:
    """Return the absolute models folder from suite.json, or None for the default folder."""
    return normalize_models_dir(read_raw().get("models_dir"))


def gpu() -> str:
    """``auto`` / ``cpu`` / ``cuda:N``. A hand-edited object ``{"device": "cuda:0", "vram_fraction": 0.5}`` still yields the device;
    :func:`dubber.infra.resources.vram_fraction` reads the cap. ``set_value`` keeps storing a string."""
    value = read_raw().get("gpu")
    if isinstance(value, dict):
        value = value.get("device", value.get("gpu"))
    return normalize_gpu(value) or DEFAULT_GPU


def device_setting() -> str:
    """The shared ``gpu`` value in this program's device terms: auto | cuda | cpu (cuda:N -> cuda; the index is in :func:`gpu`)."""
    g = gpu()
    return "cuda" if g.startswith("cuda") else g


def values() -> Dict[str, Any]:
    """Return the four shared settings with defaults filled in."""
    md = models_dir()
    return {"ui_language": ui_language() or default_language(), "theme": theme(), "models_dir": str(md) if md else None, "gpu": gpu()}


# ------------------------------------------------------------------------------------------------- writing
def _write(data: Dict[str, Any]) -> None:
    f = path()
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f"{f.name}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=True), encoding="utf-8")
    try:
        for i in range(10):
            try:
                os.replace(tmp, f)
                return
            except PermissionError:                  # the other program has the file open for a moment (Windows)
                time.sleep(0.1 * (i + 1))
        raise PermissionError(f"cannot replace {f}")
    finally:
        if tmp.exists():
            tmp.unlink()


def set_value(key: str, value: Any) -> Any:
    """Validate and store one shared value (other keys, known or not, stay).  ``ValueError`` for an unknown key / bad value."""
    if key == "ui_language":
        stored: Any = normalize_language(value)
        if stored is None:
            raise ValueError(f"unsupported UI language {value!r}")
    elif key == "theme":
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"invalid theme {value!r}")
        stored = value.strip()
    elif key == "models_dir":
        if value is None or (isinstance(value, str) and not value.strip()):
            stored = None
        else:
            p = normalize_models_dir(str(value))
            if p is None:
                raise ValueError(f"models_dir must be an absolute path or null, got {value!r}")
            stored = str(p)
    elif key == "gpu":
        stored = normalize_gpu(value)
        if stored is None:
            raise ValueError(f"gpu must be auto, cpu or cuda:N, got {value!r}")
    else:
        raise ValueError(f"not a shared setting: {key!r}")
    data = read_raw()
    if data.get(key, object()) == stored and data.get("schema") == SCHEMA:
        return stored
    data["schema"] = SCHEMA
    data[key] = stored
    _write(data)
    return stored


def set_quietly(key: str, value: Any) -> None:
    """Store one shared value, logging and ignoring any error."""
    try:
        set_value(key, value)
    except Exception as exc:  # noqa: BLE001
        log.warning("shared setting %s not saved: %s", key, exc)


def sync_cli() -> int:
    """``--sync-suite-settings`` (installer): write the models folder chosen before (``models_dir.txt``) and our UI language
    into the file when it has none yet.  Exit code 0 when done, 1 on any error (the installer ignores it)."""
    try:
        from dubber.infra import shared_paths

        if not has("models_dir"):
            md = shared_paths.legacy_models_dir_choice()
            set_value("models_dir", str(md) if md else None)
        if not has("ui_language"):
            from dubber import i18n

            set_value("ui_language", i18n.saved_language() or default_language())
        print(json.dumps(values()), flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"suite settings: {exc}", flush=True)
        return 1
