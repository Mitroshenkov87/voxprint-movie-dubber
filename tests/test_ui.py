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
    for name in ("btn_settings", "btn_diag", "btn_gear", "film", "chars", "script", "dubp", "player"):
        assert hasattr(w, name), name
    assert len(w.step_buttons) == 4 and w.stack.count() == 4
    assert not w.step_buttons[1].isEnabled()                 # nothing prepared yet
    assert w.target_lang() in ("ru", "en", "de")
    w.set_language("ru")
    assert w.btn_diag.text() == "Диагностика" and w.step_buttons[3].text().endswith("Озвучка")
    w.set_language("en")
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


def test_full_flow_offscreen_with_cpu_engines(tmp_path, clip):
    """Film -> prepare -> characters (multi-voice + library voice) -> script edit -> dub -> Watch, all with CPU stand-ins."""
    import shutil
    from pathlib import Path
    from dubber.core.project import Project
    from dubber.pipeline import stages as S
    from dubber.infra import shared_paths
    from test_pipeline import SRT
    _app()
    films = tmp_path / "films"
    films.mkdir()
    shutil.copy(clip, films / "clip.mkv")
    (films / "clip.en.srt").write_text(SRT, encoding="utf-8")
    lib = shared_paths.voices_dir() / "narrator-anna"                # a voice made by the Audiobook Builder
    lib.mkdir(parents=True)
    for n in ("adapter_model.safetensors", "adapter_config.json"):
        (lib / n).write_bytes(b"{}")
    (lib / "voice.json").write_text('{"name": "Anna", "language": "ru"}', encoding="utf-8")

    from dubber.ui.window import MainWindow
    w = MainWindow(cfg=S.MOCK_CFG)
    w.set_source(films / "clip.mkv")
    assert w.project is not None and "Duration" in w.film.lbl_info.text()
    assert "next to the film" in w.film.lbl_subs_status.text()
    w.prepare()
    _wait(lambda: not w.job.isRunning())
    _wait(lambda: w.job_kind == "")
    assert w.stack.currentIndex() == 1 and w.step_buttons[2].isEnabled()
    assert w.script.table.rowCount() == 4
    # single voice by default; the library voice is offered without copying it
    assert w.chars.cmb_single.findData("library:narrator-anna") >= 0
    assert not (w.project.folder / "voices" / "narrator-anna").exists()
    # multi-voice: needs the speaker search, then shows cards
    w.chars.chk_multi.setChecked(True)
    assert Project(w.project.folder).settings["multi_voice"] is True
    assert w.chars.btn_find.isVisibleTo(w.chars)
    w.chars.find_speakers.emit()
    _wait(lambda: w.job_kind == "")
    assert len(w.chars.cards) == 2
    first = next(iter(w.chars.cards.values()))
    first.cmb_voice.setCurrentIndex(first.cmb_voice.findData("library:narrator-anna"))
    assert Project(w.project.folder).speaker(first.sid).voice.id == "narrator-anna"
    # script edit is saved at once
    from dubber.ui.pages import COL_TRANS
    w.script.table.item(0, COL_TRANS).setText("Добрый вечер.")
    assert Project(w.project.folder).lines[0].translation == "Добрый вечер."
    # dub
    w.go(3)
    w.run_pipeline("dub", "mux")
    _wait(lambda: w.job_kind == "", timeout=120)
    out = Path(w.project.settings["output_file"])
    assert out.exists() and out.name == "clip.dub-ru.mkv", w.dubp.lbl_stage.text()
    assert w.watch_state.ready and w.dubp.btn_watch.isEnabled() and w.dubp.btn_open.isEnabled()
    src = w.watch_source()
    x = src.read(1.0, 4800)
    assert x.shape == (4800, 2) and abs(x).max() > 0
    w.close()


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
    d.edt_subdl.setText("k1")
    d.edt_hf.setText("hf_x")
    d.cmb_device.setCurrentIndex(d.cmb_device.findData("cpu"))
    d._models_choice = tmp_path / "models-elsewhere"
    d.accept()
    s = settings.load()
    assert s["subdl_key"] == "k1" and s["hf_token"] == "hf_x" and s["device"] == "cpu"
    assert shared_paths.configured_models_dir() == tmp_path / "models-elsewhere"
    assert settings.redacted(s)["hf_token"] == "set"
    g = dialogs.DiagnosticsDialog(target_lang=lambda: "de")
    opt = g.build_options()
    assert opt.target_lang == "de" and opt.hf_token == "hf_x"
    g.close()
