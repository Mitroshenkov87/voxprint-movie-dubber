"""Application folders and the user's Desktop.

* data root: ``%LOCALAPPDATA%\\VoxprintMovieDubber`` (Windows) / ``$XDG_DATA_HOME/voxprint-movie-dubber`` (Linux);
  override with ``VOXPRINT_DUBBER_HOME``.  Sub-folders: ``logs``, ``state``, ``reports``, ``projects``.
* Desktop: the *real* Desktop folder (Windows ``SHGetKnownFolderPath``: follows OneDrive "Known Folder Move"), with
  fallbacks; override with ``VOXPRINT_DESKTOP`` (tests).  If nothing is writable the report goes to ``reports``.
* Models and voices live in the suite home from ``voxprint_common.paths`` (``<home>/models`` and ``<home>/voices``).
  A folder that holds a Voxprint backup is never used as the models folder.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

from dubber.appinfo import APP_NAME
from voxprint_common.paths import home as suite_home
from voxprint_common.paths import models_dir as suite_models_dir
from voxprint_common.paths import state_dir as suite_state_dir

IS_WINDOWS = sys.platform == "win32"


def app_home() -> Path:
    """Root data directory (created on demand)."""
    env = os.environ.get("VOXPRINT_DUBBER_HOME")
    if env:
        p = Path(env)
    elif IS_WINDOWS:
        p = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")) / APP_NAME
    else:
        xdg = os.environ.get("XDG_DATA_HOME", "").strip()
        p = (Path(xdg) if xdg and os.path.isabs(xdg) else Path.home() / ".local" / "share") / "voxprint-movie-dubber"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sub(name: str) -> Path:
    p = app_home() / name
    p.mkdir(parents=True, exist_ok=True)
    return p


BACKUP_MANIFEST = "voxprint-backup.json"
BACKUP_DIRNAME = "Voxprint-backup"
BACKUP_SUBDIRS = ("models", "model", "voices", "tools")
BACKUP_CONTENT = ("models", "model", "voices")
_MODELS_DIR_FILE = "models_dir.txt"


def default_models_dir() -> Path:
    """``<suite home>/models``. ``VOXPRINT_MODELS_DIR`` still wins in :func:`models_dir`."""
    return suite_home() / "models"


def voices_dir() -> Path:
    """The shared voice library ``<suite home>/voices`` (written by the Audiobook Builder; not created here)."""
    return suite_home() / "voices"


def legacy_models_dir() -> Path:
    """``<app home>/models`` used by builds before the shared store (read-only fallback; not created)."""
    return app_home() / "models"


def _same(left: Path, right: Path) -> bool:
    try:
        return os.path.normcase(str(left.resolve())) == os.path.normcase(str(right.resolve()))
    except OSError:
        return False


def _looks_like_backup(folder: Path) -> bool:
    if (folder / BACKUP_MANIFEST).is_file():
        return True
    return folder.name.lower() == BACKUP_DIRNAME.lower() and any((folder / name).is_dir() for name in BACKUP_CONTENT)


def backup_root_of(folder: Optional[Path]) -> Optional[Path]:
    """The Voxprint backup behind ``folder`` (the folder, a ``Voxprint-backup`` inside it, or its parent), else None."""
    if folder is None:
        return None
    candidate = Path(folder)
    try:
        if _looks_like_backup(candidate):
            return candidate
        if _looks_like_backup(candidate / BACKUP_DIRNAME):
            return candidate / BACKUP_DIRNAME
        if candidate.name.lower() in BACKUP_SUBDIRS and _looks_like_backup(candidate.parent):
            return candidate.parent
    except OSError:
        return None
    return None


def set_models_dir(folder: Optional[Path]) -> None:
    """Remember ``folder`` as the models folder. None, or the default, forgets the choice.

    Written to ``suite.json`` and to the older ``state/models_dir.txt`` that earlier builds still read.
    """
    from voxprint_common.suite_settings import set_models_dir as store

    note = suite_state_dir() / _MODELS_DIR_FILE
    if folder is None or _same(Path(folder), default_models_dir()):
        note.unlink(missing_ok=True)
        store(None)
        return
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(str(folder) + "\n", encoding="utf-8")
    store(folder)


def models_dir() -> Path:
    """Where models are stored: the suite choice when it can be created, else :func:`default_models_dir`.

    A backup folder is refused. Creating the directory is this program's job; ``voxprint_common.paths`` does not.
    """
    chosen = suite_models_dir()
    if backup_root_of(chosen) is not None:
        chosen = default_models_dir()
    try:
        chosen.mkdir(parents=True, exist_ok=True)
        return chosen
    except OSError:
        fallback = default_models_dir()
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def projects_dir() -> Path:
    """Default parent of dubbing projects (cache of every stage, resumable)."""
    return _sub("projects")


def logs_dir() -> Path:
    """Log folder under the app home, created on demand."""
    return _sub("logs")


def state_dir() -> Path:
    """State folder under the app home, created on demand."""
    return _sub("state")


def reports_dir() -> Path:
    """Copy of every diagnostic report and the fallback place when the Desktop is not writable."""
    return _sub("reports")


def new_work_dir(prefix: str = "vmd-") -> Path:
    """A fresh scratch folder in the OS temp dir (callers delete it when done)."""
    return Path(tempfile.mkdtemp(prefix=prefix))


# ---------------------------------------------------------------------------------------------- Desktop
def _windows_known_folder_desktop() -> str:
    """Desktop path from the shell (follows OneDrive folder redirection); '' when unavailable."""
    try:
        import ctypes
        from ctypes import wintypes

        class GUID(ctypes.Structure):
            _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD), ("Data3", wintypes.WORD), ("Data4", ctypes.c_ubyte * 8)]

        def _guid(s: str) -> "GUID":
            import uuid

            u = uuid.UUID(s)
            g = GUID()
            g.Data1, g.Data2, g.Data3 = u.time_low, u.time_mid, u.time_hi_version
            for i, b in enumerate(u.bytes[8:]):
                g.Data4[i] = b
            return g

        folderid_desktop = _guid("B4BFCC3A-DB2C-424C-B029-7FE99A87C641")
        buf = ctypes.c_wchar_p()
        hr = ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(folderid_desktop), 0, None, ctypes.byref(buf))  # type: ignore[attr-defined]
        if hr == 0 and buf.value:
            val = buf.value
            ctypes.windll.ole32.CoTaskMemFree(buf)  # type: ignore[attr-defined]
            return val
    except Exception:  # noqa: BLE001 - fall back to the environment based guesses
        pass
    return ""


def desktop_candidates() -> List[Path]:
    """Possible Desktop folders, best first (not filtered by existence)."""
    out: List[Path] = []
    env = os.environ.get("VOXPRINT_DESKTOP")
    if env:
        out.append(Path(env))
    if IS_WINDOWS:
        kf = _windows_known_folder_desktop()
        if kf:
            out.append(Path(kf))
        for var in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial"):
            if os.environ.get(var):
                out.append(Path(os.environ[var]) / "Desktop")
        if os.environ.get("USERPROFILE"):
            out.append(Path(os.environ["USERPROFILE"]) / "Desktop")
    else:
        xdg = os.environ.get("XDG_DESKTOP_DIR")
        if xdg:
            out.append(Path(xdg))
    out.append(Path.home() / "Desktop")
    seen, uniq = set(), []
    for p in out:
        if str(p) not in seen:
            seen.add(str(p))
            uniq.append(p)
    return uniq


def desktop_dir() -> Path:
    """The first existing, writable Desktop candidate; otherwise the home folder; finally the app's ``reports`` folder."""
    for p in desktop_candidates():
        try:
            if p.is_dir() and os.access(p, os.W_OK):
                return p
        except OSError:
            continue
    home = Path.home()
    if home.is_dir() and os.access(home, os.W_OK):
        return home
    return reports_dir()
