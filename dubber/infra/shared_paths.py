"""Folders shared by all Voxprint programs (spec agreed with Voxprint AI Audiobook Builder; logic copied from its ``infra/paths.py``).

* Voxprint home: ``$VOXPRINT_HOME``, else ``%LOCALAPPDATA%\\Voxprint`` (Windows) / ``$XDG_DATA_HOME/voxprint`` (default
  ``~/.local/share/voxprint``).
* Models: ``$VOXPRINT_MODELS_DIR``, else ``models_dir`` of the shared ``<home>/state/suite.json`` (:mod:`dubber.infra.suite`;
  null = the default), else - when suite.json has no ``models_dir`` - the older ``<home>/state/models_dir.txt`` (one absolute
  path, UTF-8 with or without BOM), else ``<home>/models``.  A configured folder that cannot be created (unplugged drive) falls back to the default.
  A folder that holds a Voxprint backup is never used as the models folder.
* Voices: ``<home>/voices/<id>/`` - the Audiobook Builder voice library; this program only reads it.

The dubber's own data (logs, projects, reports) stays in ``dubber.paths.app_home`` (``%LOCALAPPDATA%\\VoxprintMovieDubber``).
Everything that decides a shared location lives in this one module.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

SHARED_NAME = "Voxprint"
MODELS_DIR_ENV = "VOXPRINT_MODELS_DIR"
MODELS_DIR_FILE = "models_dir.txt"
BACKUP_MANIFEST = "voxprint-backup.json"
BACKUP_DIRNAME = "Voxprint-backup"
BACKUP_SUBDIRS = ("models", "model", "voices", "tools")
BACKUP_CONTENT = ("models", "model", "voices")


def voxprint_home() -> Path:
    """Root data folder shared by the Voxprint programs (created on demand)."""
    env = os.environ.get("VOXPRINT_HOME")
    if env:
        p = Path(env)
    elif sys.platform == "win32":
        p = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))) / SHARED_NAME
    else:
        xdg = os.environ.get("XDG_DATA_HOME", "").strip()
        p = (Path(xdg) if xdg and os.path.isabs(xdg) else Path.home() / ".local" / "share") / SHARED_NAME.lower()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sub(name: str) -> Path:
    p = voxprint_home() / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def shared_state_dir() -> Path:
    """Return ``<Voxprint home>/state``, creating it on demand."""
    return _sub("state")


def default_models_dir() -> Path:
    """``<home>/models`` (``%LOCALAPPDATA%\\Voxprint\\models`` on Windows)."""
    return _sub("models")


def legacy_models_dir_choice() -> Optional[Path]:
    """The folder in ``state/models_dir.txt`` (absolute paths only), or None."""
    try:
        lines = (voxprint_home() / "state" / MODELS_DIR_FILE).read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeDecodeError):
        lines = []
    text = next((ln.strip().strip('"') for ln in lines if ln.strip()), "")
    if not text:
        return None
    p = Path(os.path.expandvars(os.path.expanduser(text)))
    return p if p.is_absolute() else None


def configured_models_dir() -> Optional[Path]:
    """The models folder chosen by the user or None: ``$VOXPRINT_MODELS_DIR``, else suite.json ``models_dir`` (an explicit null
    = the default folder), else ``state/models_dir.txt``; absolute paths only."""
    text = os.environ.get(MODELS_DIR_ENV, "").strip().strip('"')
    if text:
        p = Path(os.path.expandvars(os.path.expanduser(text)))
        return p if p.is_absolute() else None
    from dubber.infra import suite

    if suite.has("models_dir"):
        return suite.models_dir()
    return legacy_models_dir_choice()


def set_models_dir(folder: Optional[Path]) -> None:
    """Remember ``folder`` as the models folder; None or the default forgets the choice.  Written to suite.json (null for the
    default) and to the older ``models_dir.txt`` that earlier builds of both programs read."""
    from dubber.infra import suite

    f = shared_state_dir() / MODELS_DIR_FILE
    if folder is None or _same(Path(folder), voxprint_home() / "models"):
        f.unlink(missing_ok=True)
        suite.set_value("models_dir", None)
        return
    f.write_text(str(folder) + "\n", encoding="utf-8")
    suite.set_value("models_dir", str(folder))


def _same(a: Path, b: Path) -> bool:
    try:
        return os.path.normcase(str(a.resolve())) == os.path.normcase(str(b.resolve()))
    except OSError:
        return False


def _looks_like_backup(d: Path) -> bool:
    if (d / BACKUP_MANIFEST).is_file():
        return True
    return d.name.lower() == BACKUP_DIRNAME.lower() and any((d / s).is_dir() for s in BACKUP_CONTENT)


def backup_root_of(folder: Optional[Path]) -> Optional[Path]:
    """The Voxprint backup behind ``folder`` (the folder itself, a ``Voxprint-backup`` inside it, or its parent), else None."""
    if folder is None:
        return None
    p = Path(folder)
    try:
        if _looks_like_backup(p):
            return p
        if _looks_like_backup(p / BACKUP_DIRNAME):
            return p / BACKUP_DIRNAME
        if p.name.lower() in BACKUP_SUBDIRS and _looks_like_backup(p.parent):
            return p.parent
    except OSError:
        return None
    return None


def models_dir() -> Path:
    """Where models are stored and downloaded: the configured folder if it can be created, else :func:`default_models_dir`."""
    p = configured_models_dir()
    if p is not None and backup_root_of(p) is not None:
        p = None
    if p is not None:
        try:
            p.mkdir(parents=True, exist_ok=True)
            return p
        except OSError:
            pass
    return default_models_dir()


def voices_dir() -> Path:
    """The shared voice library ``<home>/voices`` (written by the Audiobook Builder; read-only here, not created)."""
    return voxprint_home() / "voices"
