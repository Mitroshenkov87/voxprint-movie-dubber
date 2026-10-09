"""Localisation: flat JSON catalogs ``dubber/locales/<code>.json`` and ``tr(key, **params)`` (never raises).

Languages: ``ru`` and ``en`` (the audiobook program also has ``de``; add it the same way - copy en.json, translate, add the code to
``LANGS``).  Order: ``VOXPRINT_LANG``, the shared Voxprint choice (``suite.json`` ``ui_language``, when it is a language we
have), our saved choice (``state/language``), the OS language, English.
The diagnostic REPORT is always English (so it can be pasted anywhere and read by anyone); only the UI is localised.
"""
from __future__ import annotations

import json
import locale
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

from dubber.appinfo import resource_dir

log = logging.getLogger("dubber.i18n")
LANGS = ("en", "ru")
DEFAULT_LANG = "en"
LANG_NAMES = {"en": "English", "ru": "Русский"}

_catalogs: Dict[str, Dict[str, str]] = {}
_current: Optional[str] = None


def locales_dir() -> Path:
    return Path(__file__).resolve().parent / "locales" if not getattr(sys, "_MEIPASS", None) else resource_dir() / "dubber" / "locales"


def load_catalog(lang: str) -> Dict[str, str]:
    """Load and cache a catalog; a missing or broken file gives an empty dict."""
    if lang not in _catalogs:
        data: Dict[str, str] = {}
        try:
            data = json.loads((locales_dir() / f"{lang}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("locale %s not loaded: %s", lang, exc)
        _catalogs[lang] = data
    return _catalogs[lang]


def normalize_code(value: Optional[str]) -> Optional[str]:
    """'ru-RU' / 'en_US.UTF-8' -> 'ru' / 'en'; unsupported -> None."""
    if not value:
        return None
    code = value.strip().lower().replace("_", "-").split(".")[0].split("@")[0].split("-")[0]
    return code if code in LANGS else None


def system_language() -> Optional[str]:
    """UI language of the OS if supported."""
    if sys.platform == "win32":
        try:
            import ctypes

            buf = ctypes.create_unicode_buffer(85)
            if ctypes.windll.kernel32.GetUserDefaultLocaleName(buf, len(buf)):  # type: ignore[attr-defined]
                code = normalize_code(buf.value)
                if code:
                    return code
        except Exception:  # noqa: BLE001
            pass
    for var in ("LC_ALL", "LC_MESSAGES", "LANG"):
        code = normalize_code(os.environ.get(var))
        if code:
            return code
    try:
        return normalize_code(locale.getlocale()[0])
    except Exception:  # noqa: BLE001
        return None


def _saved_file() -> Path:
    from dubber import paths

    return paths.state_dir() / "language"


def saved_language() -> Optional[str]:
    try:
        return normalize_code(_saved_file().read_text(encoding="utf-8"))
    except OSError:
        return None


def detect_language() -> str:
    code = normalize_code(os.environ.get("VOXPRINT_LANG"))
    if code:
        return code
    try:
        from dubber.infra import suite

        code = suite.ui_language()
        if code:
            return code
    except Exception:  # noqa: BLE001 - the shared file must never block the start
        pass
    return saved_language() or system_language() or DEFAULT_LANG


def current() -> str:
    global _current
    if _current is None:
        _current = detect_language()
    return _current


def set_language(lang: str, save: bool = True) -> None:
    """Switch the UI language (and remember it)."""
    global _current
    if lang in LANGS:
        _current = lang
        if save:
            try:
                _saved_file().write_text(lang, encoding="utf-8")
            except OSError:
                pass
            from dubber.infra import suite

            suite.set_quietly("ui_language", lang)


def reset() -> None:
    """Forget the cached language and catalogs (tests)."""
    global _current
    _current = None
    _catalogs.clear()


def tr(key: str, **params: object) -> str:
    """Translate ``key``; missing -> English -> the key itself.  ``{name}`` placeholders are filled from ``params``."""
    text = load_catalog(current()).get(key) or load_catalog(DEFAULT_LANG).get(key) or key
    if params:
        try:
            return text.format(**params)
        except (KeyError, IndexError, ValueError):
            return text
    return text
