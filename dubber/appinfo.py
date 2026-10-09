"""Application metadata (single place).  ``BUILD`` is filled by the build scripts (``build_info.json`` next to the program)."""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

APP_NAME = "VoxprintMovieDubber"            # technical name: folders, exe, logs - never localised
APP_DISPLAY_NAME = "Voxprint AI Movie Dubber"
APP_VERSION = "0.1.0-pre.4"
AUTHOR = "Aleksandr Mitroshenkov"


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
    """One line for the report header, e.g. ``0.1.0-pre (commit abc1234, built 2026-10-04, frozen)``."""
    bi = build_info()
    parts = []
    if bi.get("commit"):
        parts.append(f"commit {bi['commit']}")
    if bi.get("built"):
        parts.append(f"built {bi['built']}")
    if bi.get("flavour"):
        parts.append(str(bi["flavour"]))
    parts.append("frozen exe" if getattr(sys, "frozen", False) else "run from source")
    return f"{APP_VERSION} ({', '.join(parts)})"
