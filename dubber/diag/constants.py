"""Fixed inputs of the benchmark (so that reports from different runs/machines are comparable)."""
from __future__ import annotations

from pathlib import Path

from dubber.appinfo import resource_dir

LANG_NAMES = {"ru": "Russian", "en": "English", "de": "German"}      # as the Qwen3-TTS API spells them
LANG_UI = {"ru": "Русский", "en": "English", "de": "Deutsch"}

#: Test phrases per target language: [0] very short (a typical one-word/one-clause line), [1] a normal film line, [2] a long sentence.
TTS_PHRASES = {
    "ru": ["Привет! Это короткая проверка дубляжа.",
           "Добрый вечер. Я надеялся, что вы придёте сегодня, и очень рад вас видеть.",
           "Мы почти не успели поговорить, но дождь закончился, и теперь у нас есть время рассказать друг другу всё, что накопилось за эти годы."],
    "en": ["Hello! This is a short dubbing test.",
           "Good evening. I was hoping you would come tonight, and I am very glad to see you.",
           "We hardly had time to talk, but the rain has stopped, and now we have time to tell each other everything that has piled up over the years."],
    "de": ["Hallo! Das ist ein kurzer Synchrontest.",
           "Guten Abend. Ich hatte gehofft, dass Sie heute kommen, und freue mich sehr, Sie zu sehen.",
           "Wir hatten kaum Zeit zu reden, aber der Regen hat aufgehört, und jetzt haben wir Zeit, uns alles zu erzählen, was sich über die Jahre angesammelt hat."],
}
WARMUP_PHRASE = {"ru": "Привет.", "en": "Hello.", "de": "Hallo."}


def test_clip_dir() -> Path:
    """Folder of the bundled synthetic clip (``assets/test_clip``)."""
    return resource_dir() / "assets" / "test_clip"
