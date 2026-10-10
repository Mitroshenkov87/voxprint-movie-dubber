"""Application metadata.

``BUILD.json`` at the repository root is the only copy of the build offset and the codename.
The release build number is ``GITHUB_RUN_NUMBER`` plus that offset. The installer workflow
writes the result into ``build_info.json`` (``build`` and ``codename``). An installed copy
reads that stamp and does not need the git checkout.

A local run, with no stamp and no ``GITHUB_RUN_NUMBER``, uses the offset alone.
``VOXPRINT_BUILD=dev`` (or a missing ``BUILD.json`` and no stamp) is the ``dev`` marker.
``VOXPRINT_BUILD=<digits>`` forces that number.

A codename is one Biblical Hebrew word in ASCII transliteration. It describes the state of
that build (readiness, a milestone), not a theme of the product. It is shown only when it
is not empty.
"""
from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

APP_NAME = "VoxprintMovieDubber"            # technical name: folders, exe, logs - never localised
APP_DISPLAY_NAME = "Voxprint AI Movie Dubber"
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
def build_json() -> dict[str, Any]:
    """``BUILD.json``: ``offset`` and ``codename``. Empty when the file is missing or broken."""
    return _read_json(resource_dir() / BUILD_FILE)


@lru_cache(maxsize=1)
def build_info() -> dict[str, Any]:
    """``build_info.json`` written by the installer workflow; ``{}`` when running from a checkout without a stamp."""
    return _read_json(resource_dir() / "build_info.json")


def clear_build_cache() -> None:
    """Drop cached reads of ``BUILD.json`` and ``build_info.json`` (tests)."""
    build_json.cache_clear()
    build_info.cache_clear()


def _as_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _offset_of(source: Mapping[str, Any]) -> int | None:
    return _as_int(source.get("offset"))


def resolve_build(environ: Mapping[str, str] | None = None, baked: Mapping[str, Any] | None = None,
                  source: Mapping[str, Any] | None = None) -> int | str:
    """The build number, or the marker ``dev``.

    Order: ``VOXPRINT_BUILD`` (``dev`` or digits), then the baked ``build`` field, then
    ``GITHUB_RUN_NUMBER`` + offset, then the offset, then ``dev``.
    """
    env = os.environ if environ is None else environ
    marker = str(env.get("VOXPRINT_BUILD", "")).strip()
    if marker.lower() == DEV_BUILD:
        return DEV_BUILD
    forced = _as_int(marker) if marker else None
    if forced is not None:
        return forced
    info = build_info() if baked is None else baked
    stamped = info.get("build") if isinstance(info, Mapping) else None
    if isinstance(stamped, str) and stamped.strip().lower() == DEV_BUILD:
        return DEV_BUILD
    stamped_n = _as_int(stamped)
    if stamped_n is not None:
        return stamped_n
    src = build_json() if source is None else source
    offset = _offset_of(src) if isinstance(src, Mapping) else None
    if offset is None:
        return DEV_BUILD
    run = _as_int(env.get("GITHUB_RUN_NUMBER", ""))
    if run is not None:
        return run + offset
    return offset


def resolve_codename(baked: Mapping[str, Any] | None = None, source: Mapping[str, Any] | None = None) -> str:
    """Codename from the baked stamp when it has one, otherwise from ``BUILD.json``."""
    info = build_info() if baked is None else baked
    stamped = info.get("codename") if isinstance(info, Mapping) else None
    if isinstance(stamped, str) and stamped.strip():
        return stamped.strip()
    src = build_json() if source is None else source
    raw = src.get("codename") if isinstance(src, Mapping) else None
    if isinstance(raw, str):
        return raw.strip()
    return ""


def app_build() -> int | str:
    """Build number for this process (see :func:`resolve_build`)."""
    return resolve_build()


def app_codename() -> str:
    """Codename for this process (see :func:`resolve_codename`)."""
    return resolve_codename()


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
    """The version line shown in the window, the splash and the installer."""
    return format_version()


def release_title() -> str:
    """GitHub release title, e.g. ``Voxprint AI Movie Dubber 1.0.0 RC · build 1001 "Bochan"``."""
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
