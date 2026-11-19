"""Application metadata.

``BUILD.json`` at the repository root is the offset and the codename.
``voxprint_common.buildinfo`` reads that file and a sibling ``build_info.json``.
The release build number is ``GITHUB_RUN_NUMBER`` plus the offset. The installer
workflow writes the result into ``build_info.json``. An installed copy reads that
stamp and does not need the git checkout.

A recorded build of 0 stays 0. A missing recorded build is ``dev``, never 0.
``VOXPRINT_BUILD`` set to digits forces that number.

The version line keeps a middle dot: ``1.0.0 RC · build N "Codename"``.
The splash uses :func:`voxprint_common.buildinfo.label`, which has no middle dot.

A codename is one Biblical Hebrew word in ASCII transliteration. It describes the state of
that build (readiness, a milestone), not a theme of the product. It is shown only when it
is not empty.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from voxprint_common.buildinfo import BuildInfo, load

APP_NAME = "VoxprintMovieDubber"            # technical name: folders, exe, logs - never localised
APP_DISPLAY_NAME = "VOXPrint AI Movie Dubber"
APP_VERSION = "1.0.0-rc"
BUILD_FILE = "BUILD.json"
DEV_BUILD = "dev"


def resource_dir() -> Path:
    """Folder with bundled resources: the repository or installation folder."""
    base = getattr(sys, "_MEIPASS", None)
    return Path(base) if base else Path(__file__).resolve().parent.parent


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


@lru_cache(maxsize=1)
def build_info() -> dict[str, Any]:
    """``build_info.json`` written by the installer workflow; ``{}`` when the stamp is absent.

    The build number and codename come from :func:`voxprint_common.buildinfo.load`.
    This dict keeps the extra stamp fields (commit, built, flavour) for :func:`version_line`.
    """
    return _read_json(resource_dir() / "build_info.json")


def clear_build_cache() -> None:
    """Drop the cached read of ``build_info.json`` (tests)."""
    build_info.cache_clear()


def _identity() -> BuildInfo:
    """Build identity from ``BUILD.json`` and ``build_info.json``.

    Movie Dubber stores ``1.0.0-rc`` in :data:`APP_VERSION`, not in ``BUILD.json``.
    The library label therefore shows ``1.0.0 RC`` even when the file has no version.
    """
    info = load(resource_dir(), app="movie-dubber")
    if info.version:
        return info
    return BuildInfo(
        app=info.app or "movie-dubber",
        version="1.0.0",
        rc=None,
        build=info.build,
        codename=info.codename,
        show_rc=APP_VERSION.endswith("-rc"),
    )


def app_build() -> int | str:
    """Build number for this process, or the marker ``dev`` when none was recorded."""
    number = _identity().build
    return DEV_BUILD if number is None else number


def app_codename() -> str:
    """Codename for this process."""
    return _identity().codename


#: Import-time snapshot. Call :func:`app_build` / :func:`app_codename` to read again.
APP_BUILD = app_build()
CODENAME = app_codename()


def format_version(version: str | None = None, build: int | str | None = None, codename: str | None = None) -> str:
    """Human version, e.g. ``1.0.0 RC · build 1001 "Bochan"``. The codename is omitted when it is empty."""
    version = APP_VERSION if version is None else version
    if build is None:
        build = app_build()
    if codename is None:
        codename = app_codename()
    text = str(version)
    pretty = text[:-3] + " RC" if text.endswith("-rc") else text
    shown = build if isinstance(build, str) else int(build)
    label = f"{pretty} · build {shown}"
    name = str(codename or "").strip()
    if name:
        label += f' "{name}"'
    return label


def version_label() -> str:
    """The version line shown in the window, About, ``--version`` and the installer."""
    return format_version()


def release_title() -> str:
    """GitHub release title, e.g. ``VOXPrint AI Movie Dubber 1.0.0 RC · build 1001 "Bochan"``."""
    return f"{APP_DISPLAY_NAME} {version_label()}"


def version_line() -> str:
    """One line for About, ``--version`` and the diagnostics header."""
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
