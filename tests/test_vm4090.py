"""Regressions from the first RTX 4090 run of v1.0.0-rc.1002 (Windows Server 2025, driver 610.88)."""
from __future__ import annotations

import inspect
import logging
import shutil
import time

import numpy as np
import pytest

from dubber.core import audio
from dubber.core.project import Line, Project
from dubber.engines import separation as sep
from dubber.engines import tts as tts_mod


def test_tiger_shape_contract_rejects_track_major_output():
    """A fake TIGER that returns [ntrack, T] must not swap stems or raise for N>3."""
    def tiger(batch):
        arr = np.asarray(batch)
        n = int(arr.shape[0])
        if n == 1:
            seg = arr.reshape(-1)
            return seg, np.zeros_like(seg), np.zeros_like(seg)
        if n > sep.N_TRACKS:
            raise IndexError("index 3 is out of bounds for dimension 0 with size 3")
        width = int(arr.shape[-1])
        tracks = np.stack([np.full(width, 1.0), np.full(width, 2.0), np.full(width, 3.0)])
        return tracks, np.zeros_like(tracks), np.zeros_like(tracks)

    segs = [np.full(40, 0.25, np.float32), np.full(30, 0.5, np.float32), np.full(20, 0.75, np.float32), np.full(16, 0.1, np.float32)]
    outs = sep.separate_with_forward(tiger, segs)
    assert len(outs) == 4
    for seg, (speech, _back) in zip(segs, outs):
        assert np.allclose(speech, seg)

    pair = segs[:2]
    outs = sep.separate_with_forward(tiger, pair)
    for seg, (speech, _back) in zip(pair, outs):
        assert np.allclose(speech, seg)
        assert not np.allclose(speech[:8], np.full(8, 1.0))


def test_true_batch_is_one_call_unless_the_leading_axis_is_the_track_count():
    calls = {"n": 0}

    def honest(batch):
        calls["n"] += 1
        arr = np.asarray(batch, np.float32)
        if arr.shape[0] == 1:
            seg = arr.reshape(-1)
            return seg, np.zeros_like(seg), np.zeros_like(seg)
        rows = arr[:, 0, :]
        return rows, np.zeros_like(rows), np.zeros_like(rows)

    four = [np.full(12, float(i + 1), np.float32) for i in range(4)]
    calls["n"] = 0
    outs = sep.separate_with_forward(honest, four)
    assert calls["n"] == 1
    for seg, (speech, _back) in zip(four, outs):
        assert np.allclose(speech[: len(seg)], seg)

    three = four[:3]
    calls["n"] = 0
    outs = sep.separate_with_forward(honest, three)
    assert calls["n"] == 4  # the stacked call is ambiguous (3 tracks), then one each
    for seg, (speech, _back) in zip(three, outs):
        assert np.allclose(speech, seg)


def test_tiger_builder_has_no_separate_batch():
    src = inspect.getsource(sep.build_tiger)
    assert "run.separate_batch" not in src
    assert sep.chunk_batch_size(16) == 4 and sep.chunk_batch_size(24) == 8


def test_tiger_stage_uses_batch_one(tmp_path, monkeypatch):
    from dubber.pipeline import stages as S

    seen = {}

    def grab(mix, windows, speech, bg, fn, log, batch_size=1):
        seen["batch"] = batch_size
        audio.write(speech, np.zeros(1600, np.float32), 16000)
        audio.write(bg, np.zeros(1600, np.float32), 16000)

    monkeypatch.setattr(S, "separate_windows", grab)
    monkeypatch.setattr(S, "tiger_model", lambda *_a, **_k: "tiger")
    monkeypatch.setattr(S, "snapshot", lambda use_torch=False: type("Snap", (), {"vram_total_gb": 24.0, "vram_budget_gb": 8.0})())
    src = tmp_path / "film.mkv"
    src.write_bytes(b"x")
    p = Project.create(tmp_path / "proj", src)
    p.path("analysis", "windows.json").write_text("[[0, 1]]", encoding="utf-8")
    S.st_separation(p, {**S.MOCK_CFG, "separation": "tiger"}, lambda *_a, **_k: None)
    assert seen["batch"] == 1

    monkeypatch.setattr(S, "roformer_model", lambda *_a, **_k: "roformer")
    S.st_separation(p, {**S.MOCK_CFG, "separation": "roformer"}, lambda *_a, **_k: None)
    assert seen["batch"] == 8


def test_adapter_name_is_a_torch_module_name_and_collision_safe():
    key = "actor:single:0.50>miriam"
    name = tts_mod.adapter_name(key)
    assert "." not in name and ":" not in name and ">" not in name
    assert name == tts_mod.adapter_name(key)
    assert tts_mod.adapter_name("a.b") != tts_mod.adapter_name("a:b")
    assert name[0].isalpha() or name[0] == "_"
    dotted = tts_mod.adapter_name("1.voice")
    assert not dotted[0].isdigit()


def test_activate_uses_the_sanitized_adapter_name(monkeypatch):
    import sys
    import types

    seen = {}

    class Layers:
        def enable_adapter_layers(self):
            return None

        def disable_adapter_layers(self):
            return None

    class Peft:
        def __init__(self):
            self.base_model = Layers()

        def load_adapter(self, folder, adapter_name):
            seen["load"] = adapter_name

        def set_adapter(self, name):
            seen["set"] = name

    class PeftModel:
        @staticmethod
        def from_pretrained(_model, _folder, adapter_name):
            seen["from"] = adapter_name
            return Peft()

    class Talker:
        pass

    class Inner:
        def __init__(self):
            self.model = type("M", (), {"talker": Talker()})()

    monkeypatch.setattr(tts_mod, "set_lora_scale", lambda *_a, **_k: seen.setdefault("scale", _a[1]))
    mod = sys.modules.get("peft") or types.ModuleType("peft")
    monkeypatch.setitem(sys.modules, "peft", mod)
    monkeypatch.setattr(mod, "PeftModel", PeftModel, raising=False)
    engine = tts_mod.QwenTTS.__new__(tts_mod.QwenTTS)
    engine._peft = None
    engine._adapters = {}
    engine.budget = None
    engine.model = object()
    engine.backend = "standard/sdpa"
    engine._inner = lambda: Inner()  # type: ignore[method-assign]
    voice = tts_mod.VoiceSpec(key="actor:single:0.50>miriam", kind="library", adapter_dir="/voices/miriam")
    engine._activate(voice)
    expect = tts_mod.adapter_name(voice.key)
    assert seen["from"] == expect and seen["set"] == expect and seen["scale"] == expect
    assert expect in engine._adapters


def test_missing_graphs_package_is_logged_and_triton_is_not_the_cause(monkeypatch):
    monkeypatch.setattr(tts_mod.importlib.util, "find_spec", lambda _name: None)
    reason = tts_mod.graphs_skip_reason(True, True)
    assert "faster-qwen3-tts" in reason and "triton" in reason
    assert ("graphs", "sdpa") not in tts_mod.QwenTTS.candidates(True, need_adapters=True)
    assert ("graphs", "sdpa") not in tts_mod.QwenTTS.candidates(True, need_adapters=False)


def test_batch_timeout_falls_back_to_one_line_and_a_single_line_stops():
    class Slow(tts_mod.MockTTS):
        def __init__(self):
            super().__init__()
            self.calls = []
            self.notes = []
            self.log = self.notes.append

        def synthesis_timeout(self, texts):
            return 0.05

        def synthesize_batch(self, texts, voice, seed=None):
            self.calls.append(len(texts))
            if len(texts) > 1:
                time.sleep(0.2)
            return super().synthesize_batch(texts, voice, seed)

    engine = Slow()
    long = "y" * 130
    items = [(i, f"{long} {i}") for i in range(4)]
    engine.line_seconds = {i: 9.0 for i in range(4)}
    got = {}
    engine.run_queue(items, tts_mod.VoiceSpec("v", "clone"), lambda i, w: got.__setitem__(i, w))
    assert len(got) == 4
    assert any(n > 1 for n in engine.calls)
    assert engine.calls[-4:] == [1, 1, 1, 1]
    assert any("TTS: line" in note for note in engine.notes)

    class Stuck(tts_mod.MockTTS):
        def synthesis_timeout(self, texts):
            return 0.05

        def synthesize_batch(self, texts, voice, seed=None):
            time.sleep(0.2)
            return super().synthesize_batch(texts, voice, seed)

    with pytest.raises(RuntimeError, match="no progress"):
        Stuck().run_queue([(1, "Hi")], tts_mod.VoiceSpec("v", "clone"), lambda *_a: None)


def test_reference_check_warn_is_capped(tmp_path, monkeypatch):
    from dubber.pipeline import stages as S

    calls = {"n": 0}

    def hyp(_wav, _cfg, _emit):
        calls["n"] += 1
        return "not the reference"

    monkeypatch.setattr(S, "_reference_hypothesis", hyp)
    monkeypatch.setattr(S.voices, "pick_reference_lines", lambda *_a, **_k: [Line(1, 0, 1, "a"), Line(2, 1, 2, "b")])
    monkeypatch.setattr(S.voices, "build_reference", lambda *_a, **_k: (4.0, "a b"))
    monkeypatch.setattr(S.voices, "assess_reference", lambda *_a, **_k: (False, 0, "a b"))
    src = tmp_path / "clip.wav"
    src.write_bytes(b"x")
    film = tmp_path / "film.mkv"
    film.write_bytes(b"x")
    project = Project.create(tmp_path / "proj", film)
    logs = []
    S._make_reference(project, {"asr": "whisper"}, lambda *_a, **k: logs.append(k.get("text", "")),
                      None, src, tmp_path / "out.wav")
    assert calls["n"] == S.MAX_REF_REBUILDS + 1
    assert any("rebuilt" in (line or "") for line in logs)

    calls["n"] = 0
    monkeypatch.setattr(S.voices, "pick_reference_lines", lambda *_a, **_k: [Line(1, 0, 1, "only")])
    monkeypatch.setattr(S.voices, "assess_reference", lambda *_a, **_k: (False, None, "only"))
    logs.clear()
    S._make_reference(project, {"asr": "whisper"}, lambda *_a, **k: logs.append(k.get("text", "")),
                      None, src, tmp_path / "out.wav")
    assert calls["n"] == 1
    assert any("warn" in (line or "") for line in logs)


def test_download_retries_getaddrinfo_then_succeeds(tmp_path, monkeypatch):
    from dubber.infra import model_store

    sleeps = []
    monkeypatch.setattr(model_store.time, "sleep", lambda seconds: sleeps.append(seconds))
    calls = {"n": 0}
    good = {"config.json": b"{}", "model.safetensors": b"w"}

    def dl(_repo, dest, _revision, _patterns, _token):
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError("[Errno 11001] getaddrinfo failed")
        for name, data in good.items():
            path = dest / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

    info = model_store.ensure_model("org/retry", root=tmp_path, downloader=dl, log_fn=lambda _m: None)
    assert info.source == "downloaded" and calls["n"] == 3
    assert sleeps == [1.0, 2.0]


def test_download_does_not_retry_a_local_error(tmp_path, monkeypatch):
    from dubber.infra import model_store

    monkeypatch.setattr(model_store.time, "sleep", lambda _s: (_ for _ in ()).throw(AssertionError("slept")))
    calls = {"n": 0}

    def dl(*_a):
        calls["n"] += 1
        raise RuntimeError("permission denied")

    with pytest.raises(model_store.ModelUnavailable, match="permission denied"):
        model_store.ensure_model("org/local", root=tmp_path, downloader=dl)
    assert calls["n"] == 1


def test_fetch_fails_loud_on_missing_and_skips_gated_without_a_token(tmp_path, monkeypatch):
    from dubber import cli, models

    monkeypatch.setattr(models, "hf_token", lambda explicit=None: None)

    def ensure(repo, *_a, **_k):
        if "pyannote" in repo:
            raise AssertionError("gated model downloaded without a token")
        raise models.ModelUnavailable("download of org/sep failed: OSError: getaddrinfo failed")

    monkeypatch.setattr(models, "ensure", ensure)
    monkeypatch.setattr(models, "locate", lambda _repo: None)
    report = tmp_path / "models.txt"
    code = cli.main(["fetch-models", "--models", "sep,diar,embed", "--report", str(report)])
    text = report.read_text(encoding="utf-8")
    assert code == cli.EXIT_MODELS
    assert "FAILED: sep" in text
    assert "GATED: diar, embed" in text
    assert "HF_TOKEN" in text
    only = tmp_path / "gated.txt"
    code = cli.main(["fetch-models", "--models", "diar,embed", "--report", str(only)])
    assert code == 0
    body = only.read_text(encoding="utf-8")
    assert "GATED: diar, embed" in body and "one-voice" in body.lower()


def test_installer_model_list_matches_fetch_models():
    from dubber import models
    from dubber.infra import vram_tier as vt

    assert "tts_0_6b" in models.INSTALL_MODELS
    assert "roformer" in models.INSTALL_MODELS
    assert "diar" in models.INSTALL_MODELS and "embed" in models.INSTALL_MODELS
    small = vt.install_models(8.0, models.INSTALL_MODELS)
    assert "tts_0_6b" in small and "tts_1_7b" not in small
    assert vt.install_models(24.0, models.INSTALL_MODELS) == list(models.INSTALL_MODELS)


def test_stage_failure_logs_the_traceback_and_keeps_the_user_message_short(tmp_path, clip, caplog, monkeypatch):
    from dubber.pipeline import runner as R
    from dubber.pipeline import stages as S

    films = tmp_path / "films"
    films.mkdir()
    shutil.copy(clip, films / "clip.mkv")
    (films / "clip.en.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nHello.\n", encoding="utf-8")
    project = Project.create(tmp_path / "proj", films / "clip.mkv", target_lang="ru", source_lang="en")

    def boom(_p, _cfg, _emit):
        raise RuntimeError("stem exploded")

    caplog.set_level(logging.ERROR, logger="dubber.pipeline")
    monkeypatch.setitem(S.FUNCS, "separation", boom)
    res = R.Runner(project, S.MOCK_CFG).run()
    assert not res.ok
    assert res.message.startswith("separation: RuntimeError: stem exploded")
    assert "Traceback" not in res.message
    assert "Traceback" in res.debug and "stem exploded" in res.debug
    assert "stem exploded" in caplog.text and "Traceback" in caplog.text


def test_json_error_carries_debug(monkeypatch, capsys):
    import json

    from dubber import cli

    def explode(_req):
        raise RuntimeError("stage blew up")

    monkeypatch.setattr(cli, "_dispatch", explode)
    code = cli.main(["version", "--json"])
    assert code == cli.EXIT_JOB
    data = json.loads(capsys.readouterr().out)
    assert data["error"] == "RuntimeError: stage blew up"
    assert "Traceback" in data["debug"]
    assert "Traceback" not in data["error"]


def test_preview_reuses_cached_separation(tmp_path, clip, monkeypatch):
    from dubber.pipeline import runner as R
    from dubber.pipeline import stages as S

    films = tmp_path / "films"
    films.mkdir()
    shutil.copy(clip, films / "clip.mkv")
    (films / "clip.en.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nHello.\n\n2\n00:00:01,200 --> 00:00:02,000\nAgain.\n",
        encoding="utf-8")
    project = Project.create(tmp_path / "proj", films / "clip.mkv", target_lang="ru", source_lang="en")
    res = R.Runner(project, S.MOCK_CFG).run(until_stage="voices")
    assert res.ok, res.message
    calls = {"n": 0}

    def boom(*_a, **_k):
        calls["n"] += 1
        raise RuntimeError("separation should be reused")

    monkeypatch.setitem(S.FUNCS, "separation", boom)
    parent = Project(project.folder)
    start = R.best_preview_start(parent, length=10)
    sub = R.preview_project(parent, start, 10)
    r2 = R.Runner(sub, S.MOCK_CFG).run(until_stage="mix")
    assert calls["n"] == 0
    assert r2.ok, r2.message
    assert "reused from the full film" in Project(sub.folder).stages["separation"]["summary"]
    assert (sub.folder / "out" / "dub_track.wav").exists()
    assert audio.duration(sub.folder / "out" / "dub_track.wav") == pytest.approx(10, abs=0.5)


def test_poster_frame_or_a_clear_failure(tmp_path, clip):
    from dubber.core import media
    from dubber.ffmpeg import ffmpeg_info

    dest = tmp_path / "poster.jpg"
    ok = media.poster_frame(clip, 0.0, dest)
    if ffmpeg_info().ffmpeg:
        assert ok and dest.stat().st_size > 0
    else:
        assert ok is False


def test_session_failure_keeps_the_worker_traceback():
    from dubber.pipeline.runner import stage_error

    err = stage_error("RuntimeError: stem exploded", "Traceback (most recent call last):\n  stem exploded")
    assert str(err) == "RuntimeError: stem exploded"
    assert "Traceback" in err.debug
