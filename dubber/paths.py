"""Application folders and the user's Desktop.

* data root: ``%LOCALAPPDATA%\\VoxprintMovieDubber`` (Windows) / ``$XDG_DATA_HOME/voxprint-movie-dubber`` (Linux);
  override with ``VOXPRINT_DUBBER_HOME``.  Sub-folders: ``logs``, ``state``, ``reports``, ``projects``.
* Desktop: the *real* Desktop folder (Windows ``SHGetKnownFolderPath``: follows OneDrive "Known Folder Move"), with
  fallbacks; override with ``VOXPRINT_DESKTOP`` (tests).  If nothing is writable the report goes to ``reports``.
* Models are NOT here: they live in the folder shared with Voxprint AI Audiobook Builder (:mod:`dubber.infra.shared_paths`).
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import List

from dubber.appinfo import APP_NAME

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


def models_dir() -> Path:
    """The shared Voxprint models folder (see :mod:`dubber.infra.shared_paths`)."""
    from dubber.infra import shared_paths

    return shared_paths.models_dir()


def legacy_models_dir() -> Path:
    """``<app home>/models`` used by builds before the shared store (read-only fallback; not created)."""
    return app_home() / "models"


def projects_dir() -> Path:
    """Default parent of dubbing projects (cache of every stage, resumable)."""
    return _sub("projects")


def logs_dir() -> Path:
    return _sub("logs")


def state_dir() -> Path:
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
