import pytest

pytest.importorskip("PySide6")

from dubber.ui import theme


def _lum(hex_color):
    h = hex_color.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def ratio(a, b):
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


@pytest.mark.parametrize("fg,bg", theme.CONTRAST_PAIRS)
def test_wcag_aa_contrast(fg, bg):
    assert ratio(fg, bg) >= 4.5, (fg, bg, ratio(fg, bg))


def test_style_sheet_formats_cleanly():
    css = theme.build_style(True)
    assert "{{" not in css and "QPushButton" in css
    assert theme.PRIMARY in css and theme.PRIMARY_HOVER in css and theme.TEXT_ON_PRIMARY in css
    assert theme.MENU_BG in css and theme.MENU_SELECTION in css
    from dubber.ui import splash
    assert splash.BAR == "#b98cff"


def _app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _wait(cond, timeout=60.0):
    import time
    app = _app()
    t0 = time.time()
    while not cond():
        app.processEvents()
        if time.time() - t0 > timeout:
            raise AssertionError("timed out")
        time.sleep(0.02)
    app.processEvents()


def test_window_builds_with_four_steps_and_both_languages():
    _app()
    from dubber.ui.window import MainWindow
    w = MainWindow()
    for name in ("btn_settings", "btn_diag", "btn_gear", "film", "chars", "lines", "dubp", "player"):
        assert hasattr(w, name), name
    assert len(w.step_buttons) == 4 and w.stack.count() == 4
    assert not w.step_buttons[1].isEnabled()                 # nothing prepared yet
    assert w.target_lang() in ("ru", "en", "de")
    w.set_language("ru")
    assert w.btn_diag.text() == "Диагностика" and w.step_buttons[3].text().endswith("Озвучка")
    assert w.step_buttons[2].text() == "3. Реплики" and w.film.btn_dub.text() == "Дублировать"
    w.set_language("en")
    assert w.step_buttons[2].text() == "3. Lines" and w.film.btn_dub.text() == "Dub"
    assert not w.film.options_body.isVisibleTo(w.film)            # Options start collapsed
    w.close()


def test_locales_have_the_same_keys_and_cover_the_ui():
    import json
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    en = json.loads((root / "dubber" / "locales" / "en.json").read_text(encoding="utf-8"))
    ru = json.loads((root / "dubber" / "locales" / "ru.json").read_text(encoding="utf-8"))
    assert set(en) == set(ru)
    used = set()
    for f in list((root / "dubber").rglob("*.py")):
        used |= set(re.findall(r'tr\("([a-z_]+\.[a-z_0-9]+)"', f.read_text(encoding="utf-8")))
    assert not used - set(en), used - set(en)
    from dubber.pipeline import stages
    assert all(f"stage.{k}" in en for k in stages.ORDER)


def _library_voice(name="narrator-anna", lang="ru", ref=False):
    from dubber.infra import shared_paths
    lib = shared_paths.voices_dir() / name                     # a voice made by the Audiobook Builder
    lib.mkdir(parents=True)
    for n in ("adapter_model.safetensors", "adapter_config.json"):
        (lib / n).write_bytes(b"{}")
    (lib / "voice.json").write_text('{"name": "Anna", "language": "%s"}' % lang, encoding="utf-8")
    if ref:
        import numpy as np
        from dubber.core import audio
        audio.write(lib / "ref_sample.wav", np.zeros(24000, np.float32), 24000)
    return lib


def _film(tmp_path, clip):
    import shutil
    from test_pipeline import SRT
    films = tmp_path / "films"
    films.mkdir()
    shutil.copy(clip, films / "clip.mkv")
    (films / "clip.en.srt").write_text(SRT, encoding="utf-8")
    return films / "clip.mkv"


def test_one_button_dub_with_preset_defaults(tmp_path, clip):
    """Drop a film, press Dub: original track, Russian, subtitles automatic, profanity as in the original, one voice,
    voice-over volume, <name>.dub-ru.mkv next to the film - and the progress stays on the Film screen."""
    from pathlib import Path
    from dubber import settings
    from dubber.pipeline import stages as S
    _app()
    from dubber.ui.window import MainWindow
    w = MainWindow(cfg=S.MOCK_CFG)
    w.set_source(_film(tmp_path, clip))
    p = w.project.settings
    assert (p["target_lang"], p["subtitle_choice"], p["profanity"], p["multi_voice"], p["output_format"]) == ("ru", "auto", "keep", False, "mkv")
    assert p["audio_track_auto"] and w.film.cmb_audio.currentIndex() == 0 and w.film.cmb_audio.itemText(0).startswith("Automatic")
    assert "clip.dub-ru.mkv" in w.film.lbl_plan.text()
    assert not w.step_buttons[1].isEnabled() and not w.step_buttons[2].isEnabled()
    assert w.dub_all()
    assert w.film.progress.isVisibleTo(w.film) and not w.film.btn_dub.isEnabled()
    _wait(lambda: w.job_kind == "", timeout=120)
    out = Path(w.project.settings["output_file"])
    assert out.exists() and out.name == "clip.dub-ru.mkv", w.film.lbl_prepare.text()
    assert w.stack.currentIndex() == 0 and "clip.dub-ru.mkv" in w.film.lbl_prepare.text()
    assert w.film.btn_show.isVisibleTo(w.film) and w.film.btn_watch.isVisibleTo(w.film)
    # Characters stay disabled with one voice; Lines is an optional review tab
    assert not w.step_buttons[1].isEnabled() and "different voices" in w.step_buttons[1].toolTip().lower()
    assert w.step_buttons[2].isEnabled()
    # options are remembered for the next film
    w.film.cmb_profanity.setCurrentIndex(1)
    w.film.cmb_format.setCurrentIndex(1)
    s = settings.load()
    assert s["dub_profanity"] == "soften" and s["output_format"] == "mp4"
    assert w.new_project_settings(w.info)["profanity"] == "soften"
    assert "clip.dub-ru.mp4" in w.film.lbl_plan.text()
    w.close()


def test_single_voice_defaults_to_a_library_voice_in_the_dub_language(tmp_path, clip):
    from dubber.pipeline import stages as S
    _app()
    _library_voice("narrator-anna", "ru", ref=True)
    _library_voice("narrator-bob", "en", ref=True)
    from dubber.ui.window import MainWindow
    w = MainWindow(cfg=S.MOCK_CFG)
    w.set_source(_film(tmp_path, clip))
    assert w.project.settings["single_voice"]["kind"] == "library"
    assert w.project.settings["single_voice"]["id"] == "narrator-anna"
    assert w.project.settings["single_voice"]["actor_weight"] == 0.5
    assert "Anna" in w.film.lbl_plan.text()
    w.film.cmb_voice.setCurrentIndex(w.film.cmb_voice.findData("clone:"))       # the user's own choice wins from now on
    assert w.project.settings["single_voice"]["kind"] == "clone" and w.project.settings["single_voice_user"]
    w.close()


def test_review_path_and_attention_badges(tmp_path, clip):
    """Prepare and review -> Lines; multi-voice from the Options enables Characters; badges only when something needs a look."""
    from dubber.core.project import Project
    from dubber.pipeline import stages as S
    _app()
    _library_voice()
    from dubber.ui.window import MainWindow
    w = MainWindow(cfg=S.MOCK_CFG)
    w.set_source(_film(tmp_path, clip))
    assert "next to the film" in w.film.lbl_subs_status.text()
    w.prepare()
    _wait(lambda: w.job_kind == "")
    assert w.stack.currentIndex() == 2 and w.lines.table.rowCount() == 4      # one voice: straight to Lines
    assert w.step_buttons[1].property("attention") == "false"
    # multi-voice is switched on in the Film screen's Options
    w.film.chk_multi.setChecked(True)
    assert Project(w.project.folder).settings["multi_voice"] is True and w.step_buttons[1].isEnabled()
    assert not w.film.cmb_voice.isVisibleTo(w.film)
    w.chars.find_speakers.emit()
    _wait(lambda: w.job_kind == "")
    assert len(w.chars.cards) == 2
    first = next(iter(w.chars.cards.values()))
    first.cmb_voice.setCurrentIndex(first.cmb_voice.findData("library:narrator-anna"))
    assert Project(w.project.folder).speaker(first.sid).voice.id == "narrator-anna"
    # softening marks lines -> a badge on Lines and a one-line hint on the Film screen, nothing blocks
    w.project.lines[0].translation = "Чёрт, это бред."
    w.project.save()
    w.film.cmb_profanity.setCurrentIndex(1)
    from dubber.ui.pages import COL_TRANS
    w.lines.table.item(1, COL_TRANS).setText("Очень-очень длинная реплика, которая никак не поместится в этот короткий промежуток времени никогда.")
    w._update_steps()
    assert w.step_buttons[2].property("attention") == "true" and w.step_buttons[2].toolTip()
    assert w.film.lbl_attention.isVisibleTo(w.film) and "Lines" in w.film.lbl_attention.text()
    assert w.film.btn_dub.isEnabled()
    w.close()


def test_review_attention_is_qt_free(tmp_path):
    from dubber.core import review
    from dubber.core.project import Line, Project, Speaker
    p = Project(tmp_path / "p")
    p.lines = [Line(1, 0.0, 1.0, "Hi", "Привет", softened="Бл..."), Line(2, 2.0, 2.5, "x", "Очень длинная фраза, которая не влезет никак вообще")]
    assert [k for k, _ in review.attention(p)["lines"]] == ["review.lines_long", "review.lines_softened"]
    assert review.attention(p)["characters"] == []
    p.settings["multi_voice"] = True
    p.speakers = [Speaker("S1", seconds=120.0), Speaker("S2", seconds=1.5)]
    assert review.attention(p)["characters"] == [("review.chars_tiny", {"n": 1})]


def test_watch_chunk_source_reads_across_chunk_borders(tmp_path):
    import json
    import numpy as np
    from dubber.core import audio
    from dubber.ui.dub_audio import SR, ChunkSource, WavSource
    d = tmp_path / "watch"
    for i in range(2):
        audio.write(d / f"chunk_{i:05d}.wav", np.full((SR * 2, 2), 0.1 * (i + 1), np.float32), SR)
    (d / "index.json").write_text(json.dumps({"chunk_s": 2.0}), encoding="utf-8")
    src = ChunkSource(d)
    x = src.read(1.5, SR)                       # 1.5 .. 2.5 s: half from chunk 0, half from chunk 1
    assert x[0, 0] == pytest.approx(0.1, abs=1e-3) and x[-1, 0] == pytest.approx(0.2, abs=1e-3)
    assert not src.read(10.0, 100).any()        # not dubbed yet -> silence
    w = WavSource(d / "chunk_00000.wav", offset=5.0)
    y = w.read(4.5, SR)
    assert not y[: SR // 2 - 10].any() and y[SR // 2 + 10, 0] == pytest.approx(0.1, abs=1e-3)


def test_settings_and_diagnostics_dialogs(tmp_path):
    _app()
    from dubber import settings
    from dubber.infra import shared_paths
    from dubber.ui import dialogs
    d = dialogs.SettingsDialog()
    assert not d.chk_download.icon().isNull()
    d.edt_subdl.setText("k1")
    d.edt_hf.setText("hf_x")
    d.cmb_device.setCurrentIndex(d.cmb_device.findData("cuda"))
    d._models_choice = tmp_path / "models-elsewhere"
    d.accept()
    s = settings.load()
    assert s["subdl_key"] == "k1" and s["hf_token"] == "hf_x" and s["device"] == "cuda"
    assert shared_paths.configured_models_dir() == tmp_path / "models-elsewhere"
    assert settings.redacted(s)["hf_token"] == "set"
    g = dialogs.DiagnosticsDialog(target_lang=lambda: "de")
    assert not g.chk_download.icon().isNull()
    opt = g.build_options()
    assert opt.target_lang == "de" and opt.hf_token == "hf_x"
    g.close()
