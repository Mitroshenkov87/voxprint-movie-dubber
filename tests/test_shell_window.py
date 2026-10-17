"""The shared audio screen: Dub, Expert mode, and no model ids in the window."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QApplication, QPushButton, QWidget

from dubber import i18n
from dubber.ui.shell_window import ShellWindow
from voxprint_common.i18n import reset as reset_shell
from voxprint_common.ui.variants import surface_for


def _window(tmp_path, monkeypatch, lang: str):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("VOXPRINT_LANG", lang)
    i18n.reset()
    reset_shell()
    if QApplication.instance() is None:
        QApplication([])
    return ShellWindow()


def test_simple_screen_dubs_in_plain_words_and_expert_keeps_values(tmp_path, monkeypatch):
    window = _window(tmp_path, monkeypatch, "en")
    window.show()
    QApplication.processEvents()
    button = window.view.root.findChild(QPushButton, "generate")
    panel = window.view.root.findChild(QWidget, "expertPanel")
    assert button is not None and button.text() == "Dub"
    assert panel is not None and not panel.isVisible()
    assert window.shell.get("dub.actor_weight") == 0.5
    window.shell.choose("quality", "fast")
    window.expert.set_enabled(True)
    QApplication.processEvents()
    assert panel.isVisible()
    assert window.shell.get("dub.tts_size") == "small"
    window.expert.set_enabled(False)
    assert window.shell.get("dub.tts_size") == "small"
    assert window.shell.get("dub.actor_weight") == 0.5
    window.close()


def test_audio_surface_keeps_the_film_path(tmp_path, monkeypatch):
    window = _window(tmp_path, monkeypatch, "en")
    window.show()
    QApplication.processEvents()
    assert window.view._surface == surface_for("movie-dubber")
    assert window.view.root.findChild(QWidget, "voicePicker") is not None
    assert window.view.root.findChild(QWidget, "player") is not None
    assert window.view.root.findChild(QPushButton, "playResult") is None
    film = tmp_path / "clip.mkv"
    film.write_bytes(b"x")
    window.set_video(film)
    window.shell.set_prompt(f"{film.resolve()} · Warm")
    assert window._restore_source() == film.resolve()
    assert window.shell.prompt == str(film.resolve())
    window.close()


def test_russian_catalog_names_the_dub_button(tmp_path, monkeypatch):
    window = _window(tmp_path, monkeypatch, "ru")
    button = window.view.root.findChild(QPushButton, "generate")
    assert button is not None and button.text() == "Дублировать"
    drop = window.view.root.findChild(QWidget, "fileDrop")
    assert drop is not None and "видео" in drop.text().lower()
    window.close()


def test_window_source_does_not_name_model_ids():
    text = (Path(__file__).resolve().parents[1] / "dubber" / "ui" / "shell_window.py").read_text(encoding="utf-8")
    for token in ("qwen3-tts", "roformer", "mdl.json", "tts_0_6b", "tts_1_7b"):
        assert token not in text
