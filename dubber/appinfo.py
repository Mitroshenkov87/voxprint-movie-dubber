"""Application metadata (single place).  ``BUILD`` is filled by the build scripts (``build_info.json`` next to the program).

``APP_VERSION``, ``APP_BUILD`` and ``CODENAME`` are the only copy of the release identity.  The installer
(``installer/VoxprintMovieDubber.iss``), the About line, diagnostics and the GitHub release title all read them from here.

A codename is one Biblical Hebrew word in ASCII transliteration.  It describes the state of that build (readiness, a
milestone), not a theme of the product.  It is shown only when it is not empty.  Build 999 is ``Hineni`` (Genesis 22:1,
"here I am": readiness).
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

APP_NAME = "VoxprintMovieDubber"            # technical name: folders, exe, logs - never localised
APP_DISPLAY_NAME = "Voxprint AI Movie Dubber"
APP_VERSION = "1.0.0-rc"
APP_BUILD = 999
CODENAME = "Hineni"


def format_version(version: str = APP_VERSION, build: int = APP_BUILD, codename: str = CODENAME) -> str:
    """Human version, e.g. ``1.0.0 RC · build 999 "Hineni"``.  The codename is omitted when it is empty."""
    pretty = version[:-3] + " RC" if version.endswith("-rc") else version
    label = f"{pretty} · build {int(build)}"
    name = (codename or "").strip()
    if name:
        label += f' "{name}"'
    return label


def version_label() -> str:
    """The version line shown in the window, the splash and the installer."""
    return format_version()


def release_title() -> str:
    """GitHub release title, e.g. ``Voxprint AI Movie Dubber 1.0.0 RC · build 999 "Hineni"``."""
    return f"{APP_DISPLAY_NAME} {version_label()}"


def resource_dir() -> Path:
    """Folder with bundled resources: the repository or installation folder."""
    base = getattr(sys, "_MEIPASS", None)
    return Path(base) if base else Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def build_info() -> Dict[str, Any]:
    """``build_info.json`` written by the build scripts (git commit, build date, flavour); ``{}`` when running from source."""
    try:
        return json.loads((resource_dir() / "build_info.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def version_line() -> str:
    """One line for About and the diagnostics header, e.g. ``1.0.0 RC · build 999 "Hineni" (commit abc, run from source)``."""
    bi = build_info()
    parts = []
    if bi.get("commit"):
        parts.append(f"commit {bi['commit']}")
    if bi.get("built"):
        parts.append(f"built {bi['built']}")
    if bi.get("flavour"):
        parts.append(str(bi["flavour"]))
    parts.append("frozen exe" if getattr(sys, "frozen", False) else "run from source")
    return f"{version_label()} ({', '.join(parts)})"
