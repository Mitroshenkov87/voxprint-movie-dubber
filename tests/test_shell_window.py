"""The shared audio screen: Dub, Expert mode, and no model ids in the window."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QFontMetrics, QImage
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QPushButton,
    QSpinBox,
    QWidget,
)

from dubber import i18n
from dubber.ui.shell_window import ShellWindow
from voxprint_common.i18n import reset as reset_shell
from voxprint_common.ui.presets import PresetMatch
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
    assert window.view.root.findChild(QWidget, "voicePicker") is None
    assert window.view.root.findChild(QWidget, "voiceLabel") is None
    assert window.view._surface.voice_picker is False
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
    assert window.view.root.findChild(QWidget, "videoResult") is not None
    assert window.view.root.findChild(QLabel, "videoPoster") is not None
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
    assert picker is not None and picker.currentText() == "Мариа (рус.)"
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


def test_more_menu_switches_the_interface_language(tmp_path, monkeypatch):
    window = _window(tmp_path, monkeypatch, "ru")
    window.show()
    QApplication.processEvents()
    overflow = window.view.root.findChild(QPushButton, "overflow")
    assert overflow is not None and overflow.menu() is not None
    labels = [action.text() for action in overflow.menu().actions()]
    assert "Язык интерфейса" in labels
    assert "Очередь" in labels
    button = window.view.root.findChild(QPushButton, "generate")
    assert button is not None and button.text() == "Дублировать"
    window.shell.set_value("dub.source_lang", "de")
    assert window.set_ui_language("en")
    QApplication.processEvents()
    button = window.view.root.findChild(QPushButton, "generate")
    overflow = window.view.root.findChild(QPushButton, "overflow")
    assert button is not None and button.text() == "Dub"
    assert overflow is not None and overflow.text().startswith("More")
    assert window.shell.get("dub.source_lang") == "de"
    labels = [action.text() for action in overflow.menu().actions()]
    assert "Interface language" in labels
    window.close()


def test_queue_runs_one_film_and_cancel_leaves_the_rest(tmp_path, monkeypatch):
    window = _window(tmp_path, monkeypatch, "en")
    window.show()
    QApplication.processEvents()
    launched: list[Path] = []
    monkeypatch.setattr(window, "_launch_pipeline", lambda: launched.append(Path(window._source)) or True)
    first = tmp_path / "one.mkv"
    second = tmp_path / "two.mkv"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    window.add_films([first, second])
    assert launched == [first.resolve()]
    assert window._queue.running() == str(first.resolve())
    assert window._queue.waiting() == 1
    note = window.view.root.findChild(QWidget, "simpleNote")
    assert note is not None and "1" in note.text()
    dialog = window.open_queue()
    QApplication.processEvents()
    assert dialog.findChild(QWidget, "queueTitle") is not None
    removes = [button for button in dialog.findChildren(QPushButton, "queueRemove") if button.isEnabled()]
    assert len(removes) == 1
    window.queue_cancel()
    assert window._queue.running() is None
    assert launched == [first.resolve()]
    assert window._queue.waiting() == 1
    window.queue_remove(str(second.resolve()))
    assert window._queue.waiting() == 0
    assert note is not None
    window.close()
    again = ShellWindow()
    assert again._queue.running() is None
    assert again._queue.waiting() == 0
    again.close()


def test_queue_is_still_there_after_the_window_closes(tmp_path, monkeypatch):
    window = _window(tmp_path, monkeypatch, "en")
    monkeypatch.setattr(window, "_launch_pipeline", lambda: True)
    film = tmp_path / "queued.mkv"
    film.write_bytes(b"a")
    window.add_films([film])
    assert window._queue.running() == str(film.resolve())
    window.close()
    again = ShellWindow()
    assert again._queue.running() == str(film.resolve())
    again.close()


def test_source_language_and_stop_stage_are_expert_fields(tmp_path, monkeypatch):
    window = _window(tmp_path, monkeypatch, "en")
    window.show()
    QApplication.processEvents()
    assert window.view.root.findChild(QWidget, "dub.source_lang") is None
    assert window.view.root.findChild(QWidget, "dub.until") is None
    window.expert.set_enabled(True)
    QApplication.processEvents()
    source = window.view.root.findChild(QComboBox, "dub.source_lang")
    until = window.view.root.findChild(QComboBox, "dub.until")
    assert source is not None and source.currentData() == "auto"
    assert until is not None and until.currentData() == "mux"
    source.setCurrentIndex(source.findData("en"))
    until.setCurrentIndex(until.findData("translation"))
    QApplication.processEvents()
    assert window.shell.get("dub.source_lang") == "en"
    assert window.shell.get("dub.until") == "translation"
    window.close()


def test_poster_uses_the_shell_region_or_a_still(tmp_path, monkeypatch):
    def _still(_src, _at_s, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        image = QImage(8, 8, QImage.Format.Format_RGB32)
        image.fill(0xFF225588)
        return image.save(str(dest), "JPG")

    monkeypatch.setattr("dubber.ui.shell_window.poster_frame", _still)
    window = _window(tmp_path, monkeypatch, "en")
    window.show()
    QApplication.processEvents()
    film = tmp_path / "clip.mkv"
    film.write_bytes(b"x")
    window.set_video(film)
    QApplication.processEvents()
    region = window.view.root.findChild(QWidget, "videoResult")
    poster = window.view.root.findChild(QLabel, "videoPoster")
    assert region is not None
    assert poster is not None and poster.isVisible() and poster.pixmap() is not None and not poster.pixmap().isNull()
    window.close()


def test_plain_rows_line_up_in_russian_and_burning_notes_the_reencode(tmp_path, monkeypatch):
    window = _window(tmp_path, monkeypatch, "ru")
    window.resize(1200, 900)
    window.show()
    QApplication.processEvents()
    chips = {
        "quality": ("fast", "balanced", "best"),
        "voices": ("keep", "neutral", "pick"),
        "language": ("ru", "en", "de"),
        "original": ("keep", "replace"),
        "background": ("quiet", "original", "loud"),
        "subtitles": ("none", "track", "burn"),
        "format": ("mkv", "mp4"),
    }
    words = {
        "quality": "Качество",
        "voices": "Голоса",
        "language": "Язык озвучки",
        "original": "Оригинальная дорожка",
        "background": "Фон из оригинала",
        "subtitles": "Субтитры",
        "format": "Формат",
    }
    chip_words = {
        ("original", "keep"): "Оставить второй дорожкой",
        ("background", "original"): "Как в оригинале",
        ("subtitles", "track"): "Отдельной дорожкой",
        ("subtitles", "burn"): "Вшить в видео",
    }
    first_x = None
    widths = []
    for row_id, values in chips.items():
        label = window.view.root.findChild(QLabel, f"preset-{row_id}")
        assert label is not None and label.text() == words[row_id]
        assert label.width() >= QFontMetrics(label.font()).horizontalAdvance(label.text())
        widths.append(label.minimumWidth())
        for value in values:
            chip = window.view.root.findChild(QPushButton, f"chip-{row_id}-{value}")
            assert chip is not None and chip.isVisible()
            expected = chip_words.get((row_id, value))
            if expected is not None:
                assert chip.text() == expected
            assert chip.width() >= QFontMetrics(chip.font()).horizontalAdvance(chip.text())
            corner = chip.mapTo(window, chip.rect().bottomRight())
            assert corner.x() <= 1200 and corner.y() <= 900
        first = window.view.root.findChild(QPushButton, f"chip-{row_id}-{values[0]}")
        assert first is not None
        left = first.mapTo(window, first.rect().topLeft()).x()
        if first_x is None:
            first_x = left
        else:
            assert abs(left - first_x) <= 2
    assert len(set(widths)) == 1
    note = window.view.root.findChild(QLabel, "subtitleBurnNote")
    assert note is not None and not note.isVisible()
    burn = window.view.root.findChild(QPushButton, "chip-subtitles-burn")
    assert burn is not None
    burn.click()
    QApplication.processEvents()
    assert window.shell.get("dub.subtitles") == "burn"
    assert note.isVisible() and note.text() == "видео будет перекодировано"
    window.expert.set_enabled(True)
    QApplication.processEvents()
    mix = window.view.root.findChild(QDoubleSpinBox, "dub.mix_original")
    assert mix is not None and mix.decimals() == 2 and mix.value() == 0.15
    assert window.shell.get("dub.mix_original") == 0.15
    background = window.shell.preset_status("background")
    assert background is not None and background.kind is PresetMatch.EXACT
    delivery = window.view.root.findChild(QComboBox, "dub.subtitles")
    assert delivery is not None and delivery.currentData() == "burn"
    window.close()


def test_window_source_does_not_name_model_ids():
    text = (Path(__file__).resolve().parents[1] / "dubber" / "ui" / "shell_window.py").read_text(encoding="utf-8")
    for token in ("qwen3-tts", "roformer", "mdl.json", "tts_0_6b", "tts_1_7b"):
        assert token not in text
