"""Shared Voxprint infrastructure: folders, model store (lock, manifest, users), GPU lock, stdio guard."""
import json
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from dubber.infra import gpu_lock, model_store, shared_paths, stdio_guard


def test_models_dir_env_file_and_fallback(tmp_path, monkeypatch):
    home = Path(os.environ["VOXPRINT_HOME"])
    assert shared_paths.models_dir() == home / "models"
    chosen = tmp_path / "big-disk" / "models"
    shared_paths.set_models_dir(chosen)
    assert (home / "state" / "models_dir.txt").read_text(encoding="utf-8").strip() == str(chosen)
    assert shared_paths.models_dir() == chosen
    monkeypatch.setenv("VOXPRINT_MODELS_DIR", str(tmp_path / "env-models"))
    assert shared_paths.models_dir() == tmp_path / "env-models"
    monkeypatch.setenv("VOXPRINT_MODELS_DIR", "relative/path")                    # only absolute paths count (same as the Audiobook Builder)
    assert shared_paths.models_dir() == home / "models"
    monkeypatch.delenv("VOXPRINT_MODELS_DIR")
    shared_paths.set_models_dir(None)
    assert shared_paths.models_dir() == home / "models"


def test_unusable_configured_folder_falls_back(tmp_path, monkeypatch):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    monkeypatch.setenv("VOXPRINT_MODELS_DIR", str(blocker / "models"))           # cannot be created (parent is a file)
    assert shared_paths.models_dir() == shared_paths.default_models_dir()


def test_backup_folder_is_never_the_models_folder(tmp_path, monkeypatch):
    b = tmp_path / "Voxprint-backup"
    (b / "models").mkdir(parents=True)
    monkeypatch.setenv("VOXPRINT_MODELS_DIR", str(b / "models"))
    assert shared_paths.models_dir() == shared_paths.default_models_dir()


def test_model_lock_is_exclusive_and_removed(tmp_path):
    p = tmp_path / ".x--y.lock"
    a, b = model_store.ModelLock(p), model_store.ModelLock(p)
    assert a.try_acquire()
    held = {}

    def other():                       # another thread behaves like another process for flock on the same path
        held["b"] = b.try_acquire()

    t = threading.Thread(target=other)
    t.start()
    t.join()
    if os.name != "nt":
        assert held["b"] is False
    a.release(remove=True)
    assert not p.exists()


def test_sweep_stale_locks(tmp_path):
    (tmp_path / ".a--b.lock").write_bytes(b"")
    (tmp_path / ".not-ours.lock").write_bytes(b"data")
    assert model_store.sweep_stale_locks(tmp_path) == 1
    assert (tmp_path / ".not-ours.lock").exists()


def _fake_downloader(files):
    def dl(repo, dest, revision, patterns, token):
        for name, data in files.items():
            p = dest / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
    return dl


def test_ensure_downloads_into_partial_then_renames(tmp_path):
    seen = []
    info = model_store.ensure_model("org/model", root=tmp_path, downloader=_fake_downloader({"config.json": b"{}", "model.safetensors": b"w"}),
                                    log_fn=seen.append)
    assert info.source == "downloaded" and info.path == tmp_path / "org--model"
    assert (info.path / "model.safetensors").exists()
    assert not (tmp_path / "org--model.partial").exists() and not (tmp_path / ".org--model.lock").exists()
    again = model_store.ensure_model("org/model", root=tmp_path, downloader=None)
    assert again.source == "present"


def test_manifest_mismatch_deletes_bad_file_and_keeps_partial(tmp_path, monkeypatch):
    import hashlib

    good = b"weights"
    manifest = {"org/m": {"revision": "a" * 40, "files": {"config.json": {"size": 2, "sha256": hashlib.sha256(b"{}").hexdigest()},
                                                         "model.safetensors": {"size": len(good), "sha256": hashlib.sha256(good).hexdigest()}}}}
    monkeypatch.setattr(model_store, "load_manifest", lambda path=None: manifest)
    with pytest.raises(model_store.ModelUnavailable, match="checksum"):
        model_store.ensure_model("org/m", root=tmp_path, downloader=_fake_downloader({"config.json": b"{}", "model.safetensors": b"WEIGHTS"}))
    part = tmp_path / "org--m.partial"
    assert (part / "config.json").exists() and not (part / "model.safetensors").exists()   # resume: only the bad file goes
    info = model_store.ensure_model("org/m", root=tmp_path, downloader=_fake_downloader({"model.safetensors": good}))
    assert info.source == "downloaded" and (info.path / ".verified").exists()


def test_download_off_and_gated(tmp_path):
    with pytest.raises(model_store.ModelUnavailable, match="switched off"):
        model_store.ensure_model("org/none", root=tmp_path, allow_download=False)
    with pytest.raises(model_store.ModelUnavailable, match="gated"):
        model_store.ensure_model("org/gated", root=tmp_path, gated=True)


def test_stalled_download_gives_up(tmp_path, monkeypatch):
    monkeypatch.setattr(model_store, "WATCH_INTERVAL", 0.05)
    stop = threading.Event()

    def hang(*a):
        stop.wait(5)
    with pytest.raises(model_store.ModelUnavailable, match="stalled"):
        model_store.ensure_model("org/slow", root=tmp_path, downloader=hang, stall_s=0.2)
    stop.set()


def test_users_file(tmp_path):
    model_store.register_user("audiobook-builder", tmp_path)
    model_store.register_user(root=tmp_path)
    assert model_store.read_users(tmp_path) == {"audiobook-builder": True, "movie-dubber": True}
    assert model_store.unregister_user(root=tmp_path) == ["audiobook-builder"]
    assert model_store.unregister_user("audiobook-builder", tmp_path) == []
    assert json.loads((tmp_path / ".users.json").read_text()) == {}


def test_bundled_manifest_is_well_formed():
    m = model_store.load_manifest()
    assert "Qwen/Qwen3-TTS-12Hz-1.7B-Base" in m
    for repo, e in m.items():
        assert len(e["revision"]) == 40, repo
        assert all(len(f["sha256"]) == 64 and f["size"] > 0 for f in e["files"].values())


# ------------------------------------------------------------------ GPU lock
def test_gpu_lock_acquire_release_and_wait():
    path = gpu_lock.lock_file()
    with gpu_lock.gpu_job("tts", eta_s=100):
        data = json.loads(path.read_text())
        assert data["owner"] == "movie-dubber" and data["job"] == "tts"
        datetime.fromisoformat(data["started"]) and datetime.fromisoformat(data["eta"])
    assert not path.exists()
    now = datetime.now(timezone.utc)
    path.write_text(json.dumps({"owner": "audiobook-builder", "job": "train", "started": now.isoformat(), "eta": (now + timedelta(hours=1)).isoformat()}))
    with pytest.raises(gpu_lock.GpuLockTimeout):
        with gpu_lock.gpu_job("tts", timeout=0.2, poll=0.05):
            pass
    assert json.loads(path.read_text())["owner"] == "audiobook-builder"           # never removed by us


def test_gpu_lock_staleness_rules(tmp_path):
    p = tmp_path / "l"
    p.write_text("x")
    now = datetime.now(timezone.utc)
    old = {"owner": "audiobook-builder", "eta": (now - timedelta(hours=3)).isoformat()}
    assert gpu_lock.is_stale(old, p, busy=lambda: False)
    assert not gpu_lock.is_stale(old, p, busy=lambda: True)                       # a python process still uses the GPU
    recent = {"owner": "audiobook-builder", "eta": (now - timedelta(hours=1)).isoformat()}
    assert not gpu_lock.is_stale(recent, p, busy=lambda: False)


def test_gpu_lock_left_by_our_crashed_run_is_taken_over():
    path = gpu_lock.lock_file()
    path.write_text(json.dumps({"owner": "movie-dubber", "job": "old", "started": "2026-01-01T00:00:00+00:00",
                                "eta": "2099-01-01T00:00:00+00:00", "pid": 999999999}))
    assert gpu_lock.try_acquire("new", 60) is None
    gpu_lock.release()


def test_stdio_guard_swallows_dead_handles():
    class Dead:
        def write(self, t):
            raise OSError(22, "Invalid argument")

        def flush(self):
            raise OSError(22, "Invalid argument")
    g = stdio_guard.GuardedStream(Dead())
    assert g.write("x") == 0
    g.flush()
    assert stdio_guard.GuardedStream(None).write("x") == 0


def test_guarded_stream_survives_unencodable_text():
    import io

    from dubber.infra.stdio_guard import GuardedStream

    raw = io.BytesIO()
    inner = io.TextIOWrapper(raw, encoding="cp1251", errors="strict")
    g = GuardedStream(inner)
    g.write("power plan: \ufffd\u2713 balanced\n")       # not encodable in cp1251: must not kill the stream
    g.write("next line\n")
    g.flush()
    out = raw.getvalue().decode("cp1251")
    assert "balanced" in out and "next line" in out


def test_fetch_worker_reports_sizes(tmp_path, monkeypatch):
    from dubber import models
    from dubber.workers import w_fetch

    folder = tmp_path / "m"
    folder.mkdir()
    (folder / "model.bin").write_bytes(b"x" * 1024)
    monkeypatch.setattr(models, "ensure", lambda repo, allow, log=None: (folder, {"source": "present", "download_s": 0.0}))

    class Ctx:
        def log(self, m):
            pass

    res = w_fetch.run({"keys": ["asr"]}, Ctx())
    assert res["status"] == "OK" and res["metrics"]["models"]["asr"]["source"] == "present"
