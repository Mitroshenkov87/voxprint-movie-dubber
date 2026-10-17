"""The shared audio screen: Dub, Expert mode, and no model ids in the window."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QApplication, QComboBox, QPushButton, QSpinBox, QWidget

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
    assert window.shell.get("dub.actor_weight") == 50
    window.shell.choose("quality", "fast")
    window.expert.set_enabled(True)
    QApplication.processEvents()
    assert panel.isVisible()
    likeness = window.view.root.findChild(QSpinBox, "dub.actor_weight")
    assert likeness is not None and likeness.value() == 50 and likeness.text() == "50%"
    assert window.shell.get("dub.tts_size") == "small"
    window.expert.set_enabled(False)
    assert window.shell.get("dub.tts_size") == "small"
    assert window.shell.get("dub.actor_weight") == 50
    window.close()


def test_audio_surface_keeps_the_film_path(tmp_path, monkeypatch):
    window = _window(tmp_path, monkeypatch, "en")
    window.show()
    QApplication.processEvents()
    assert window.view._surface == surface_for("movie-dubber")
    voice = window.view.root.findChild(QWidget, "voicePicker")
    label = window.view.root.findChild(QWidget, "voiceLabel")
    assert voice is not None and label is not None
    assert not voice.isVisible() and not label.isVisible()
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
    balanced = window.view.root.findChild(QPushButton, "chip-quality-balanced")
    keep = window.view.root.findChild(QPushButton, "chip-voices-keep")
    neutral = window.view.root.findChild(QPushButton, "chip-voices-neutral")
    best = window.view.root.findChild(QPushButton, "chip-quality-best")
    expert = window.view.root.findChild(QPushButton, "expertToggle")
    assert balanced is not None and balanced.text() == "Сбалансированно"
    assert best is not None and best.text() == "Лучшее качество"
    assert keep is not None and keep.text() == "Оставить голоса актёров"
    assert neutral is not None and neutral.text() == "Нейтральные голоса"
    assert expert is not None and expert.text() == "Режим эксперта"
    custom = window.view.root.findChild(QPushButton, "chip-voices-pick")
    assert custom is not None and custom.text() == "Свой выбор"
    drop = window.view.root.findChild(QWidget, "fileDrop")
    assert drop is not None and "видео" in drop.text().lower()
    window.close()


def test_review_wizard_keeps_characters_lines_and_the_player(tmp_path, monkeypatch):
    window = _window(tmp_path, monkeypatch, "en")
    window.show()
    QApplication.processEvents()
    overflow = window.view.root.findChild(QPushButton, "overflow")
    assert overflow is not None and overflow.menu() is not None
    labels = [action.text() for action in overflow.menu().actions()]
    assert "Characters" in labels
    assert "Lines" in labels
    assert "Player" in labels
    assert "Check and repair" in labels
    window._open_review(1)
    QApplication.processEvents()
    wizard = window._wizard
    studio = window._studio
    assert wizard is not None and wizard.isVisible()
    assert studio is not None
    assert studio.chars.sld_one.maximum() == 100
    assert studio.lines.table.columnCount() == 6
    assert studio.player.btn_play.text()
    assert studio.stack.currentIndex() == 0
    note = window.view.root.findChild(QWidget, "simpleNote")
    assert note is not None and "Prepare the film first" in note.text()
    wizard.close()
    window.close()


def test_empty_history_strip_is_hidden_and_video_region_is_only_a_hook(tmp_path, monkeypatch):
    window = _window(tmp_path, monkeypatch, "en")
    window.show()
    QApplication.processEvents()
    strip = window.view.root.findChild(QWidget, "historyStrip")
    assert strip is not None and not strip.isVisible() and strip.height() == 0
    assert window._video_result is None
    window.close()


def test_custom_voices_wait_until_prepare_then_save_a_library_voice(tmp_path, monkeypatch):
    from dubber.core.project import Line, Project, Speaker, Voice

    window = _window(tmp_path, monkeypatch, "ru")
    window.show()
    QApplication.processEvents()
    assert window.shell.get("dub.multi_voice") is False
    dialog = window._open_voice_cast()
    QApplication.processEvents()
    assert window.shell.get("dub.multi_voice") is True
    title = dialog.findChild(QWidget, "voiceCastTitle")
    hint = dialog.findChild(QWidget, "voiceCastHint")
    prepare = dialog.findChild(QPushButton, "voiceCastPrepare")
    done = dialog.findChild(QPushButton, "voiceCastDone")
    cancel = dialog.findChild(QPushButton, "voiceCastCancel")
    assert title is not None and title.text() == "Свой выбор голосов"
    assert hint is not None and hint.text() == "Голоса можно назначить после подготовки"
    assert prepare is not None and prepare.isVisible()
    assert done is not None and done.text() == "Готово"
    assert cancel is not None and cancel.text() == "Отмена"
    assert dialog.findChild(QPushButton, "voiceListen-S1") is None
    folder = tmp_path / "proj"
    source = tmp_path / "film.mkv"
    source.write_bytes(b"x")
    voices = {
        "maria": '{"name": "Maria", "language": "ru"}',
        "ivan": '{"name": "Ivan", "names": {"en": "Ivan", "ru": "Иван"}, "language": "ru"}',
    }
    for folder_name, payload in voices.items():
        voice_dir = tmp_path / "home" / "voices" / folder_name
        voice_dir.mkdir(parents=True)
        (voice_dir / "adapter_model.safetensors").write_bytes(b"x")
        (voice_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
        (voice_dir / "voice.json").write_text(payload, encoding="utf-8")
    project = Project.create(folder, source)
    project.speakers = [
        Speaker("S1", "Ann", voice=Voice("library", "maria"), seconds=12),
        Speaker("S2", "Speaker 2", seconds=4),
    ]
    project.lines = [
        Line(1, 0.0, 1.0, text="Hello", speaker="S1"),
        Line(2, 1.2, 2.0, text="Again", speaker="S1"),
        Line(3, 2.2, 3.0, text="Yes", speaker="S2"),
    ]
    project.stages["translation"] = {"done": True}
    project.settings["multi_voice"] = True
    project.save()
    studio = window._ensure_studio(sync_file=False, push=False)
    studio.project = project
    dialog.reload()
    QApplication.processEvents()
    assert not prepare.isVisible()
    name = dialog.findChild(QWidget, "voiceName-S1")
    picker = dialog.findChild(QComboBox, "voicePick-S1")
    assert name is not None and name.text() == "Голос 1"
    meta = dialog.findChild(QWidget, "voiceMeta-S1")
    excerpt = dialog.findChild(QPushButton, "voiceExcerpt-S1")
    library = dialog.findChild(QPushButton, "voiceLibrary-S1")
    other = dialog.findChild(QWidget, "voiceName-S2")
    assert other is not None and other.text() == "Голос 2"
    assert meta is not None and "12" in meta.text()
    assert excerpt is not None and excerpt.text() == "Прослушать отрывок"
    assert library is not None and library.text() == "Прослушать голос"
    assert dialog.findChild(QPushButton, "voiceListen-S1") is None
    assert picker is not None and picker.currentText() == "Maria (рус.)"
    assert "Библиотека" not in picker.currentText()
    assert "Ann" not in name.text()
    assert hint is not None and hint.text() == "Послушайте отрывок каждого голоса из фильма и выберите для него голос озвучки."
    studio.chars.load(project)
    assert studio.chars.cards["S2"].edt_name.text() == "Голос 2"
    dialog.place_on_parent()
    QApplication.processEvents()
    for button in dialog.findChildren(QPushButton):
        if not button.isVisible():
            continue
        corner = button.mapTo(dialog, button.rect().topLeft())
        assert corner.x() >= 0
        assert corner.x() + button.width() <= dialog.width()
        assert corner.y() >= 0
        assert corner.y() + button.height() <= dialog.height()
    ivan = next(index for index in range(picker.count()) if picker.itemText(index) == "Иван (рус.)")
    picker.setCurrentIndex(ivan)
    assert project.speaker("S1").voice.kind == "library"
    assert project.speaker("S1").voice.id == "ivan"
    dialog.close()
    window.close()


def test_window_source_does_not_name_model_ids():
    text = (Path(__file__).resolve().parents[1] / "dubber" / "ui" / "shell_window.py").read_text(encoding="utf-8")
    for token in ("qwen3-tts", "roformer", "mdl.json", "tts_0_6b", "tts_1_7b"):
        assert token not in text
