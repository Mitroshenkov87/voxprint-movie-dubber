import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from dubber import ffmpeg, i18n, models, paths
from dubber.pipeline import stages
from dubber.workers import common

ROOT = Path(__file__).resolve().parent.parent
needs_ffmpeg = pytest.mark.skipif(not ffmpeg.ffmpeg_info(True).ffmpeg, reason="no ffmpeg on this machine")


# ------------------------------------------------------------------ paths
def test_desktop_env_override_and_fallbacks(tmp_path, monkeypatch):
    assert paths.desktop_dir() == Path(os.environ["VOXPRINT_DESKTOP"])
    monkeypatch.setenv("VOXPRINT_DESKTOP", str(tmp_path / "nope"))          # does not exist -> next candidate or the home folder
    monkeypatch.setenv("HOME", str(tmp_path / "nohome"))
    d = paths.desktop_dir()
    assert d.is_dir() and os.access(d, os.W_OK)


def test_data_folders_follow_override(tmp_path):
    assert paths.models_dir().parent == Path(os.environ["VOXPRINT_DUBBER_HOME"])
    assert paths.reports_dir().is_dir()


# ------------------------------------------------------------------ i18n
def test_catalogs_have_identical_keys():
    en, ru = i18n.load_catalog("en"), i18n.load_catalog("ru")
    assert set(en) == set(ru)
    assert all(v.strip() for v in ru.values())


def test_tr_params_and_missing_key():
    i18n.set_language("ru", save=False)
    assert i18n.tr("ui.btn_diag") != "ui.btn_diag"
    assert i18n.tr("no.such.key") == "no.such.key"


# ------------------------------------------------------------------ models
def test_model_layout_and_verify(tmp_path):
    d = models.local_dir("owner/name", tmp_path)
    assert d.name == "owner--name"                                    # same layout as the Audiobook Builder
    assert not models.verify_dir(d)
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    assert not models.verify_dir(d)                                   # no weights yet
    (d / "model.safetensors").write_bytes(b"x")
    assert models.verify_dir(d)


def test_ensure_without_download_raises(monkeypatch):
    with pytest.raises(models.ModelUnavailable):
        models.ensure("nobody/nothing-here", allow_download=False)


# ------------------------------------------------------------------ pure helpers
def test_wer():
    assert common.wer("Hello, world!", "hello world") == 0.0
    assert common.wer("a b c d", "a x c d") == pytest.approx(0.25)
    assert common.wer("", "") == 0.0


def test_fit_plan_verdicts():
    slots = [{"start": 0, "end": 2}, {"start": 3, "end": 5}, {"start": 6, "end": 8}]
    lines = [{"id": 0, "seconds": 1.5}, {"id": 1, "seconds": 2.2}, {"id": 2, "seconds": 3.0}]
    plan = stages.fit_plan(slots, lines)
    assert [i["verdict"] for i in plan["items"]] == ["fits", "squeeze", "too_long"]
    assert (plan["ok"], plan["squeeze"], plan["too_long"]) == (1, 1, 1)


def test_mix_track_places_and_ducks(tmp_path):
    sr = 16000
    base = np.full(sr * 4, 0.3, dtype="float32")
    sf.write(tmp_path / "orig.wav", base, sr)
    t = np.arange(sr) / sr
    sf.write(tmp_path / "l0.wav", (0.4 * np.sin(2 * np.pi * 440 * t)).astype("float32"), sr)
    info = stages.mix_track(str(tmp_path / "orig.wav"), [{"start": 1.0, "end": 2.0}], [{"path": str(tmp_path / "l0.wav")}], tmp_path / "mix.wav")
    assert info["lines"] == 1
    mix, _ = sf.read(tmp_path / "mix.wav")
    assert abs(mix[int(0.2 * sr)] - 0.3) < 0.01                       # untouched before the line
    assert np.abs(mix[sr:2 * sr]).max() > 0.3                         # the dub is audible
    assert abs(np.mean(mix[int(3.2 * sr):])) == pytest.approx(0.3, abs=0.01)


def test_mix_track_survives_line_outside_audio(tmp_path):
    sr = 8000
    sf.write(tmp_path / "o.wav", np.zeros(sr, dtype="float32"), sr)
    sf.write(tmp_path / "l.wav", np.ones(sr, dtype="float32") * 0.1, sr)
    info = stages.mix_track(str(tmp_path / "o.wav"), [{"start": 99.0, "end": 100.0}], [{"path": str(tmp_path / "l.wav")}], tmp_path / "m.wav")
    assert info["lines"] == 0


def test_unavailable_stages_are_marked(tmp_path):
    ctx = stages.StageContext(tmp_path / "x.mkv", "ru", tmp_path)
    res = stages.Stage().run(ctx)
    assert res.implemented is False and res.ok


# ------------------------------------------------------------------ ffmpeg on the bundled clip
@needs_ffmpeg
def test_probe_extract_and_mux(clip, tmp_path):
    info = ffmpeg.probe(clip)
    assert 18 < float(info["duration"]) < 19 and any(s["type"] == "audio" for s in info["streams"])
    ctx = stages.StageContext(clip, "ru", tmp_path)
    res = stages.ExtractStage().run(ctx)
    assert res.ok and Path(ctx.data["wav16"]).is_file()
    data, sr = sf.read(ctx.data["wav16"])
    assert sr == 16000 and data.ndim == 1
    dub = tmp_path / "dub.wav"
    sf.write(dub, np.zeros(sr * 2, dtype="float32"), sr)
    out = tmp_path / "out.mkv"
    idx = ffmpeg.add_dub_track(clip, dub, out)
    streams = ffmpeg.probe(out)["streams"]
    assert idx == 1 and sum(1 for s in streams if s["type"] == "audio") == 2
    assert any(s["type"] == "video" for s in streams)


def test_bundled_clip_assets_present():
    d = ROOT / "assets" / "test_clip"
    for name in ("voxprint-test-clip.mkv", "ref_voice.wav", "expected.json", "README.txt"):
        assert (d / name).is_file(), name
    assert len(json.loads((d / "expected.json").read_text(encoding="utf-8"))["lines"]) == 4


# ------------------------------------------------------------------ workers run in-process on CPU
def test_gpu_worker_without_gpu_does_not_crash():
    from dubber.workers import w_gpu
    out = w_gpu.run({}, common.WorkerContext())
    assert out["status"] in ("OK", "WARN") and out["details"]


def test_diarization_without_pyannote_is_skip_not_fail():
    from dubber.workers import w_diar
    import importlib.util
    try:
        has = importlib.util.find_spec("pyannote.audio") is not None
    except ImportError:
        has = False
    if has:
        pytest.skip("pyannote is installed here")
    assert w_diar.run({"wav16": "x"}, common.WorkerContext())["status"] == "SKIP"


def test_vad_worker_finds_speech_in_bundled_clip(clip, tmp_path):
    pytest.importorskip("silero_vad")
    ctx = stages.StageContext(clip, "ru", tmp_path)
    stages.ExtractStage().run(ctx)
    from dubber.workers import w_vad
    out = w_vad.run({"wav16": ctx.data["wav16"], "out_json": str(tmp_path / "vad.json")}, common.WorkerContext())
    assert out["status"] == "OK"
    segs = json.loads((tmp_path / "vad.json").read_text())["segments"]
    assert 3 <= len(segs) <= 8


def test_rope_shim_is_idempotent():
    pytest.importorskip("transformers")
    a = common.apply_qwen_tts_compat()
    b = common.apply_qwen_tts_compat()
    assert b == ""
    assert a in ("", "applied RoPE compat shim for qwen-tts-hf on this transformers version")


def test_tiger_vendored_model_constructs():
    pytest.importorskip("torch")
    from dubber.third_party.look2hear.models import TIGERDNR
    m = TIGERDNR()
    n = sum(p.numel() for p in m.parameters())
    assert 1_000_000 < n < 100_000_000
    assert not hasattr(TIGERDNR, "from_pretrained")           # the HF-hub mixin was removed on purpose


# ------------------------------------------------------------------ packaging
def test_installer_files_are_consistent():
    iss = (ROOT / "installer" / "VoxprintMovieDubber.iss").read_text(encoding="utf-8")
    assert "install-runtime.ps1" in iss and "UninstallDisplayName" in iss
    assert (ROOT / "installer" / "install-runtime.ps1").is_file()
    assert (ROOT / "assets" / "voxprint-dubber.ico").is_file()
    reqs = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "faster-qwen3-tts" in reqs and "torch" not in [ln.split(">")[0].strip() for ln in reqs.splitlines()]


def test_script_files_use_crlf_in_checkout_policy():
    attrs = (ROOT / ".gitattributes").read_text()
    assert "*.iss text eol=crlf" in attrs and "*.ps1 text eol=crlf" in attrs


def test_fetch_models_cli_rejects_unknown_key():
    r = subprocess.run([sys.executable, str(ROOT / "main.py"), "--fetch-models", "--models", "nope"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "nope" in r.stdout + r.stderr


def test_cli_version_prints_a_version():
    r = subprocess.run([sys.executable, str(ROOT / "main.py"), "--version"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "0.1.0" in r.stdout
