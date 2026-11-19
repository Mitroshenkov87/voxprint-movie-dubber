"""How a library voice is named on screen.

The library may store a name per language. A Russian screen uses that Russian
name. When the library only has a Latin name, the Russian screen writes it in
Cyrillic, so Boris is shown as Борис. The same text is used everywhere a
library voice is shown.
"""
from __future__ import annotations

import json

from dubber.core.voices import LibraryVoice
from dubber.i18n import current, tr

_DIGRAPHS = (
    ("shch", "щ"),
    ("sch", "щ"),
    ("yo", "ё"),
    ("zh", "ж"),
    ("kh", "х"),
    ("ts", "ц"),
    ("ch", "ч"),
    ("sh", "ш"),
    ("yu", "ю"),
    ("ya", "я"),
    ("ye", "е"),
)
_LETTERS = {
    "a": "а",
    "b": "б",
    "c": "к",
    "d": "д",
    "e": "е",
    "f": "ф",
    "g": "г",
    "h": "х",
    "i": "и",
    "j": "й",
    "k": "к",
    "l": "л",
    "m": "м",
    "n": "н",
    "o": "о",
    "p": "п",
    "q": "к",
    "r": "р",
    "s": "с",
    "t": "т",
    "u": "у",
    "v": "в",
    "w": "в",
    "x": "кс",
    "y": "й",
    "z": "з",
}


def library_voice_label(voice: LibraryVoice) -> str:
    """Return the name shown for ``voice`` in the current interface language.

    A name stored for this language wins. Otherwise a Russian screen
    transliterates a Latin name. The voice's own language, when it has one,
    is the short catalog label, for example ``рус.``.

    Args:
        voice: One installed library voice.
    """
    info = _voice_info(voice)
    name = _named_for(info, current())
    if not name:
        name = _plain_name(info, voice.name)
        if current() == "ru":
            name = transliterate_ru(name)
    tongue = _language_label(voice.language)
    if tongue:
        return f"{name} ({tongue})"
    return name


def transliterate_ru(text: str) -> str:
    """Return ``text`` with Latin letters written in Cyrillic.

    Letters that are already Cyrillic stay as they are. The first letter of a
    Latin word keeps its capital.

    Args:
        text: A voice name.
    """
    if not any("A" <= char <= "Z" or "a" <= char <= "z" for char in text):
        return text
    out: list[str] = []
    index = 0
    while index < len(text):
        piece, size = _take(text, index)
        out.append(piece)
        index += size
    return "".join(out)


def _take(text: str, index: int) -> tuple[str, int]:
    for latin, cyrillic in _DIGRAPHS:
        chunk = text[index : index + len(latin)]
        if chunk.lower() != latin:
            continue
        return _match_case(chunk, cyrillic), len(latin)
    char = text[index]
    mapped = _LETTERS.get(char.lower())
    if mapped is None:
        return char, 1
    return _match_case(char, mapped), 1


def _match_case(source: str, mapped: str) -> str:
    if source[:1].isupper():
        return mapped[:1].upper() + mapped[1:]
    return mapped


def _voice_info(voice: LibraryVoice) -> dict[str, object]:
    try:
        loaded = json.loads((voice.path / "voice.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _named_for(info: dict[str, object], lang: str) -> str:
    for key in ("display_names", "names"):
        found = _pick_lang(info.get(key), lang)
        if found:
            return found
    for key in ("name", "display_name"):
        found = _pick_lang(info.get(key), lang)
        if found:
            return found
    return ""


def _plain_name(info: dict[str, object], fallback: str) -> str:
    for key in ("display_name", "name"):
        stored = info.get(key)
        if isinstance(stored, str) and stored.strip():
            return stored.strip()
    return fallback


def _pick_lang(value: object, lang: str) -> str:
    if not isinstance(value, dict):
        return ""
    for candidate in (lang, lang[:2]):
        text = value.get(candidate)
        if isinstance(text, str) and text.strip():
            return text.strip()
    return ""


def _language_label(code: str) -> str:
    token = (code or "").lower().replace("_", "-").split("-")[0]
    if not token:
        return ""
    key = f"shell.voice.lang.{token}"
    label = tr(key)
    if label == key:
        return token
    return label
