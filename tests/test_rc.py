"""Release-candidate behaviour: version line, likeness, length-aware speech, GPU policy, residency, splash."""
from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_version_label_quotes_the_codename_and_omits_an_empty_one(monkeypatch):
    from dubber import appinfo

    monkeypatch.delenv("VOXPRINT_BUILD", raising=False)
    monkeypatch.delenv("GITHUB_RUN_NUMBER", raising=False)
    monkeypatch.setattr(appinfo, "build_info", lambda: {})
    monkeypatch.setattr(appinfo, "build_json", lambda: {"offset": 983, "codename": "Bochan"})
    assert appinfo.APP_VERSION == "1.0.0-rc"
    assert appinfo.app_build() == 983 and appinfo.app_codename() == "Bochan"
    assert appinfo.version_label() == '1.0.0 RC · build 983 "Bochan"'
    assert appinfo.format_version(build=983, codename="") == "1.0.0 RC · build 983"
    assert appinfo.format_version(build=983, codename="  ") == "1.0.0 RC · build 983"
    assert appinfo.release_title() == 'Voxprint AI Movie Dubber 1.0.0 RC · build 983 "Bochan"'


def test_installer_reads_appinfo_and_docs_state_the_codename_rule():
    iss = (ROOT / "installer" / "VoxprintMovieDubber.iss").read_text(encoding="utf-8-sig")
    assert "FileOpen" in iss and "appinfo.py" in iss
    assert '#define AppVersion "0.1.0-pre' not in iss
    assert "VersionLabel" in iss and "VersionInfoProductTextVersion" in iss
    building = (ROOT / "docs" / "BUILDING.md").read_text(encoding="utf-8")
    assert "BUILD.json" in building and "Bochan" in building and "Biblical Hebrew" in building and "not a theme" in building
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Windows 11 24H2" in readme and "2025" in readme and "Bochan" in readme and "BUILD.json" in readme
    assert "BUILD.json" in iss
    yml = (ROOT / ".github" / "workflows" / "build-installer.yml").read_text(encoding="utf-8")
    assert "release_title" in yml
    note = (ROOT / "dubber" / "assets" / "README.md").read_text(encoding="utf-8")
    assert "not depict a real person" in note or "does not depict a real person" in note


def test_locale_keys_for_the_splash_and_the_cuda_warning():
    en = json.loads((ROOT / "dubber" / "locales" / "en.json").read_text(encoding="utf-8"))
    ru = json.loads((ROOT / "dubber" / "locales" / "ru.json").read_text(encoding="utf-8"))
    assert en["splash.easter_egg"] == "When the dub is still loading..."
    assert ru["splash.easter_egg"] == "Когда дубляж ещё грузится..."
    for key in ("asr.cuda_fallback", "asr.cuda_fallback_title", "splash.easter_egg"):
        assert en[key] and ru[key]


def test_default_likeness_is_half_and_persists_per_speaker_and_single_voice(tmp_path):
    from dubber.core import actor_voice
    from dubber.core.project import DEFAULT_SETTINGS, Line, Project, Speaker

    assert actor_voice.DEFAULT_WEIGHT == 0.5
    assert DEFAULT_SETTINGS["actor_weight"] == 0.5
    assert DEFAULT_SETTINGS["single_voice"]["actor_weight"] == 0.5
    src = tmp_path / "film.mkv"
    src.write_bytes(b"x")
    p = Project.create(tmp_path / "proj", src)
    assert p.settings["actor_weight"] == 0.5
    p.speakers = [Speaker("S1", "Ann", actor_weight=0.35)]
    p.lines = [Line(1, 0.0, 1.0, "hi", "привет", speaker="S1")]
    p.settings["single_voice"] = {"kind": "clone", "id": "", "actor_weight": 0.2}
    p.save()
    again = Project(p.folder)
    assert again.speaker("S1").actor_weight == 0.35
    assert again.settings["single_voice"]["actor_weight"] == 0.2
    # a project that already stored 0.7 keeps 0.7
    p.settings["actor_weight"] = 0.7
    p.save()
    assert Project(p.folder).settings["actor_weight"] == 0.7


def test_likeness_sliders_and_auto_voice_keep_the_weight(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from dubber.core.project import Line, Project, Speaker
    from dubber.pipeline import stages as S
    from dubber.ui.pages import CharactersPage, FilmPage
    from dubber.ui.window import MainWindow

    QApplication.instance() or QApplication([])
    src = tmp_path / "film.mkv"
    src.write_bytes(b"x")
    p = Project.create(tmp_path / "proj", src, multi_voice=True)
    p.speakers = [Speaker("S1", "Ann", seconds=4.0)]
    p.lines = [Line(1, 0.0, 2.0, "hello there", "привет", speaker="S1")]
    p.save()
    chars = CharactersPage()
    chars.load(p)
    card = chars.cards["S1"]
    assert card.sld_like.value() == 50
    card.sld_like.setValue(55)
    chars.set_actor_weight("S1", card.sld_like.value())
    assert Project(p.folder).speaker("S1").actor_weight == 0.55

    film = FilmPage()
    p.settings["multi_voice"] = False
    p.settings["single_voice"] = {"kind": "clone", "id": "", "actor_weight": 0.3}
    film.load(p)
    assert film.sld_likeness.value() == 30
    assert not film.sld_likeness.isHidden() or not p.settings["multi_voice"]
    film.sld_likeness.setValue(60)
    film.store(p)
    assert p.settings["single_voice"]["kind"] == "clone"
    assert p.settings["single_voice"]["actor_weight"] == 0.6

    window = MainWindow(cfg=S.MOCK_CFG)
    p.settings["single_voice_user"] = False
    window.apply_auto(p)
    assert p.settings["single_voice"]["actor_weight"] == 0.6
    window.close()


def test_prepare_for_voice_shortens_and_merges_once():
    from dubber.core.project import Line
    from dubber.core.script import estimate_seconds, prepare_for_voice

    wordy = "We should leave this crowded place at once, yes."
    short_slot = Line(1, 0.0, 0.5, "hi", wordy, speaker="S1")
    nxt = Line(2, 1.2, 2.0, "bye", "Goodbye.", speaker="S2")
    lines = [short_slot, nxt]
    before = estimate_seconds(wordy, "en")
    dropped = prepare_for_voice(lines, "en", total=4.0)
    assert dropped == []
    assert estimate_seconds(lines[0].translation, "en") < before
    again = prepare_for_voice(lines, "en", total=12.0)
    assert again == [] and lines[0].translation

    long = "Extraordinary circumstances require immediate attention from every responsible person tonight."
    a = Line(1, 0.0, 0.6, "a", long, speaker="S1")
    b = Line(2, 0.8, 1.6, "b", "Yes.", speaker="S1")
    group = [a, b]
    absorbed = prepare_for_voice(group, "en", total=4.0)
    assert absorbed == [2] and len(group) == 1 and "Yes" in group[0].translation
    assert prepare_for_voice(group, "en", total=4.0) == []


def test_tts_retries_a_too_long_line_once(tmp_path, monkeypatch):
    from dubber.core.project import Line, Project
    from dubber.engines import tts as tts_mod
    from dubber.pipeline import stages as S

    calls = []
    real = tts_mod.MockTTS.synthesize_batch

    def wrapped(self, texts, voice, seed=None):
        calls.append(seed)
        return real(self, texts, voice, seed)

    monkeypatch.setattr(tts_mod.MockTTS, "synthesize_batch", wrapped)
    src = tmp_path / "film.mkv"
    src.write_bytes(b"x")
    p = Project.create(tmp_path / "proj", src, target_lang="en")
    text = "Extraordinary circumstances require immediate attention from every responsible person in the building tonight."
    p.lines = [Line(1, 0.0, 0.5, "go", text, speaker="S1")]
    p.settings["duration"] = 1.0
    p.settings["single_ref"] = {}
    S.st_tts(p, S.MOCK_CFG, lambda *a, **k: None)
    assert 1 <= len(calls) <= 2


def test_cuda_asr_refuses_a_silent_cpu_fallback(monkeypatch):
    from dubber.engines import asr
    from dubber.pipeline import stages as S

    monkeypatch.setattr(asr, "cuda_gpu_present", lambda: True)
    with pytest.raises(asr.AsrCudaFallback):
        asr.ensure_cuda_asr("auto", "cpu")
    asr.ensure_cuda_asr("cpu", "cpu")
    asr.ensure_cuda_asr("cuda", "cuda")
    monkeypatch.setattr(asr, "cuda_gpu_present", lambda: False)
    asr.ensure_cuda_asr("auto", "cpu")
    assert asr.is_cuda_fallback_failure("asr: AsrCudaFallback: stopped")
    assert asr.is_cuda_fallback_failure("would run on the CPU even though an NVIDIA GPU is available")
    assert not asr.is_cuda_fallback_failure("out of memory")

    def boom(*_a, **_k):
        raise asr.AsrCudaFallback("would run on the CPU even though")

    monkeypatch.setattr(S, "transcribe_faster_whisper", boom)
    with pytest.raises(asr.AsrCudaFallback):
        S._reference_hypothesis(Path("clip.wav"), {"asr": "whisper", "device": "auto", "allow_download": False}, lambda *_a, **_k: None)


def test_separation_batch_size_and_overlapped_copy(tmp_path):
    from dubber.core import audio
    from dubber.engines import separation as sep

    assert sep.chunk_batch_size(16) == 4
    assert sep.chunk_batch_size(24) == 8
    assert sep.chunk_batch_size(16.0, 2.5) == 2
    assert sep.chunk_batch_size(10, 1.0) == 1
    assert sep.safe_compute_dtype(True, True) == "bf16"
    assert sep.safe_compute_dtype(True, False) == "fp32"
    assert sep.safe_compute_dtype(False, True) == "fp32"

    sr = 8000
    audio.write(tmp_path / "mix.wav", np.zeros(sr * 20, np.float32), sr)
    entered = threading.Event()
    release = threading.Event()
    state = {"batches": 0}

    class Tracking(sep._BackgroundCopies):
        def submit(self, fn):
            def wrapped() -> None:
                if state["batches"] >= 1:
                    entered.set()
                    release.wait(2)
                fn()
            super().submit(wrapped)

    def model(seg, _sr):
        return seg, seg

    def separate_batch(segs, _sr):
        state["batches"] += 1
        if state["batches"] == 2:
            assert entered.wait(2)
            release.set()
        return [(s, s) for s in segs]

    model.separate_batch = separate_batch
    monkey = Tracking
    original = sep._BackgroundCopies
    sep._BackgroundCopies = monkey
    try:
        windows = [(0.0, 0.4), (3.0, 3.4), (6.0, 6.4), (9.0, 9.4)]
        sep.separate_windows(tmp_path / "mix.wav", windows, tmp_path / "sp.wav", tmp_path / "bg.wav", model, batch_size=2)
    finally:
        sep._BackgroundCopies = original
    assert state["batches"] == 2
    assert (tmp_path / "sp.wav").is_file()


def test_resident_keeps_what_fits_and_prefetch_does_not_evict(monkeypatch):
    from dubber.infra import resident

    resident.reset()
    resident.disable()
    try:
        assert resident.unload_plan([("a", 5.0), ("b", 1.0)], "c", 4.0, free_gb=3.0, total_gb=16.0) == ["a"]
        assert resident.unload_plan([("a", 9.0)], "b", 9.0, 0.0, 0.0) == []
        resident.enable()
        built = {"n": 0}

        def build():
            built["n"] += 1
            return {"n": built["n"]}

        first = resident.slot("asr", 1.0, build, free_gb=10.0, total_gb=16.0)
        second = resident.slot("asr", 1.0, build, free_gb=10.0, total_gb=16.0)
        assert first is second and built["n"] == 1
        resident.slot("big", 8.0, lambda: {"big": True}, free_gb=2.0, total_gb=16.0)
        resident.register_loader("next", lambda: {"poison": True})
        assert resident.prefetch("next", 5.0, free_gb=2.0, total_gb=16.0) is None
        assert "big" in resident.loaded() and "next" not in resident.loaded()
        assert "next" in resident.noted()
    finally:
        resident.reset()
        resident.disable()


def test_prefetch_overlap_runs_beside_the_gpu_work():
    import time

    from dubber.pipeline.prefetch import overlap

    marks = []

    def gpu() -> str:
        marks.append(("gpu", time.monotonic()))
        time.sleep(0.05)
        marks.append(("gpu-end", time.monotonic()))
        return "ok"

    def cpu() -> None:
        marks.append(("cpu", time.monotonic()))
        time.sleep(0.05)

    assert overlap(gpu, cpu) == "ok"
    assert [m[0] for m in marks][0] in ("gpu", "cpu")
    cpu_t = next(t for name, t in marks if name == "cpu")
    gpu_end = next(t for name, t in marks if name == "gpu-end")
    assert cpu_t < gpu_end


def test_thermal_full_speed_then_pause_and_resume():
    from dubber.infra.thermal import (COOL_C, FULL_SPEED_S, HOT_C, MAX_PAUSE_S, MEDIAN_WINDOW_S, THROTTLE_BITS, THROTTLE_HOLD_S,
                                      Guard, stage_telemetry_line, telemetry_from_summary)

    # the suite rule (project-notes suite/COLLABORATION.md, section 3)
    assert (FULL_SPEED_S, HOT_C, MEDIAN_WINDOW_S, THROTTLE_HOLD_S, COOL_C) == (2.5 * 3600, 83.0, 300.0, 60.0, 75.0)
    assert stage_telemetry_line("asr", 70, 120, "none") == "asr: GPU temperature 70 C, power 120 W, throttle none"
    assert telemetry_from_summary("asr", {}) == ""
    line = telemetry_from_summary("asr", {"n": 2, "temp_max_c": 81, "power_max_w": 140, "throttle": "HW thermal slowdown"})
    assert line.startswith("asr: GPU temperature 81 C") and "140 W" in line

    clock = {"t": 0.0}
    guard = Guard(now=lambda: clock["t"])
    guard.update(HOT_C + 5, 200, 0)
    assert not guard.should_pause()                       # full speed for the first 2.5 hours, however hot
    clock["t"] = FULL_SPEED_S + 400                       # the early sample has left the 5-minute window
    guard.update(HOT_C, 200, 0)
    assert guard.median_temp() == HOT_C and guard.should_pause()
    guard.update(COOL_C, 100, 0)
    assert guard.cool_enough()

    # one hot spike among cool samples does not pause: it is the 5-min median that counts
    spiky = Guard(now=lambda: clock["t"], started_ago_s=FULL_SPEED_S + 1)
    for temp in (78.0, 79.0, 95.0):
        spiky.update(temp, 150, 0)
    assert spiky.median_temp() == 79.0 and not spiky.should_pause()
    spiky.update(90.0, 150, 0)
    assert spiky.median_temp() == pytest.approx(84.5) and spiky.should_pause()

    guard.update(70, 100, THROTTLE_BITS)
    guard._temps.clear()
    guard.update(70, 100, THROTTLE_BITS)
    assert not guard.should_pause()
    clock["t"] += THROTTLE_HOLD_S + 1
    assert guard.should_pause()

    quiet = Guard(now=lambda: FULL_SPEED_S + 10)
    quiet.update(None, None, 0)
    assert not quiet.should_pause()

    # the full-speed window counts from the start of the dub, not of the speech stage
    late = Guard(now=lambda: 0.0, started_ago_s=FULL_SPEED_S + 1)
    late.update(HOT_C + 1, 200, 0)
    assert late.should_pause()

    paused = {"n": 0.0}

    def sleep(step: float) -> None:
        paused["n"] += step
        clock["t"] += step

    hot = Guard(now=lambda: clock["t"])
    hot.t0 = 0.0
    clock["t"] = FULL_SPEED_S + 5
    spent = hot.pause_after_block(sample=lambda: (90.0, 220.0, 0), sleep=sleep, log=lambda _m: None)
    assert spent == pytest.approx(MAX_PAUSE_S)

    temps = iter([88.0] + [80.0] * 3 + [74.0] * 10)
    cooled = Guard(now=lambda: clock["t"])
    cooled.t0 = clock["t"] - FULL_SPEED_S - 1
    spent = cooled.pause_after_block(sample=lambda: (next(temps), 200.0, 0), sleep=sleep, log=lambda _m: None)
    assert 0 < spent < MAX_PAUSE_S and cooled.median_temp() is None     # resumed at <= 75 C with a fresh median


def test_dialogue_groups_batch_only_long_lines():
    from dubber.engines.tts import LONG_LINE_CHARS, is_long_line, next_dialogue_group

    assert is_long_line("x" * LONG_LINE_CHARS)
    assert is_long_line("hi", 8.0)
    assert not is_long_line("hi", 1.0)
    queue = [1, 2, 3, 4, 5]
    texts = {i: "x" * 130 for i in queue}
    assert next_dialogue_group(queue, texts, 12, graphs=True) == [1]
    queue = [1, 2, 3, 4]
    texts = {i: "short" for i in queue}
    assert next_dialogue_group(queue, texts, 12, {i: 1.0 for i in queue}, False) == [1]
    queue = [4, 1, 3, 2, 0]
    texts = {i: "y" * (130 + i) for i in queue}
    group = next_dialogue_group(list(queue), texts, 3, graphs=False)
    assert group == sorted(group, key=lambda i: len(texts[i])) and len(group) == 3


def test_click_burst_fires_once_inside_two_seconds():
    from dubber.ui.easter_egg import CLICKS, WINDOW_S, ClickBurst

    assert CLICKS == 6 and WINDOW_S == 2.0
    burst = ClickBurst()
    for i in range(5):
        assert burst.click(i * 0.3) is False
    assert burst.click(1.5) is True
    assert burst.click(1.6) is False
    late = ClickBurst()
    for t in (0.0, 0.5, 1.0, 1.5, 2.0):
        assert late.click(t) is False
    assert late.click(2.6) is False


def test_splash_overlay_shows_once_and_does_not_close_the_splash():
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication

    from dubber.ui import splash

    QApplication.instance() or QApplication([])

    def press(widget) -> None:
        local = QPointF(8, 8)
        event = QMouseEvent(QEvent.Type.MouseButtonPress, local, local, Qt.MouseButton.LeftButton,
                            Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        widget.mousePressEvent(event)

    screen = splash.show()
    try:
        for _ in range(5):
            press(screen)
        assert screen._egg is None or not screen._egg.isVisible()
        press(screen)
        assert screen.isVisible()
        assert screen._egg is not None and screen._egg.isVisible()
        assert screen._egg.caption.text()
        press(screen._egg)
        assert not screen._egg.isVisible()
        assert screen.isVisible()
        for _ in range(6):
            press(screen)
        assert not screen._egg.isVisible()
    finally:
        screen.close()


def test_session_probe_on_the_test_clip(tmp_path, clip):
    from dubber.core.project import Project
    from dubber.pipeline.session import WorkerSession, close_session

    if not clip.is_file():
        pytest.skip("test clip is missing")
    film = tmp_path / "clip.mkv"
    shutil.copy(clip, film)
    project = Project.create(tmp_path / "proj", film)
    session = WorkerSession({"allow_download": False, "device": "cpu", "inprocess": True})
    try:
        summary = session.run_stage(project.folder, "probe", {"allow_download": False, "device": "cpu"}, lambda _line: None, timeout=90)
    finally:
        close_session()
    assert "min" in summary
