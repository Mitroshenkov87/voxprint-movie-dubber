"""Library voice names follow the interface language."""
from __future__ import annotations

from dubber import i18n
from dubber.core.voices import LibraryVoice
from dubber.ui.voice_label import library_voice_label, transliterate_ru


def _voice(folder, name: str, language: str, extra: str = "") -> LibraryVoice:
    folder.mkdir(parents=True)
    payload = extra or f'{{"name": "{name}", "language": "{language}"}}'
    (folder / "voice.json").write_text(payload, encoding="utf-8")
    return LibraryVoice(id=folder.name, name=name, path=folder, language=language)


def test_a_russian_screen_writes_a_latin_library_name_in_cyrillic(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    boris = _voice(tmp_path / "boris", "Boris", "en")
    maria = _voice(tmp_path / "maria", "Maria", "ru")
    ivan = _voice(
        tmp_path / "ivan",
        "Ivan",
        "ru",
        '{"name": "Ivan", "names": {"en": "Ivan", "ru": "Иван"}, "language": "ru"}',
    )
    i18n.reset()
    i18n.set_language("ru")
    assert library_voice_label(boris) == "Борис (англ.)"
    assert library_voice_label(maria) == "Мариа (рус.)"
    assert library_voice_label(ivan) == "Иван (рус.)"
    i18n.set_language("en")
    assert library_voice_label(boris) == "Boris (English)"
    assert library_voice_label(maria) == "Maria (Russian)"
    assert library_voice_label(ivan) == "Ivan (Russian)"
    i18n.reset()


def test_transliteration_keeps_a_name_that_is_already_cyrillic():
    assert transliterate_ru("Борис") == "Борис"
    assert transliterate_ru("Elena") == "Елена"
