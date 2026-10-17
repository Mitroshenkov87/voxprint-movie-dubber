"""Shared Voxprint infrastructure: folders, model store (locate, manifest, users), GPU lock, stdio guard."""
import hashlib
import json
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from dubber import models
from dubber.engines.separation import _expose_roformer
from dubber import paths
from dubber.infra import model_store, stdio_guard
from voxprint_common.gpu_lock import Acquired, GpuLockTimeout, gpu_job, is_stale, release, try_acquire
from dubber.ui.pages import TARGET_LANGS


def test_models_dir_env_file_and_fallback(tmp_path, monkeypatch):
    home = Path(os.environ["VOXPRINT_HOME"])
    assert paths.models_dir() == home / "models"
    chosen = tmp_path / "big-disk" / "models"
    paths.set_models_dir(chosen)
    assert (home / "state" / "models_dir.txt").read_text(encoding="utf-8").strip() == str(chosen)
    assert paths.models_dir() == chosen
    monkeypatch.setenv("VOXPRINT_MODELS_DIR", str(tmp_path / "env-models"))
    assert paths.models_dir() == tmp_path / "env-models"
    monkeypatch.setenv("VOXPRINT_MODELS_DIR", "relative/path")                    # only absolute paths count
    assert paths.models_dir() == chosen
    monkeypatch.delenv("VOXPRINT_MODELS_DIR")
    paths.set_models_dir(None)
    assert paths.models_dir() == home / "models"


def test_unusable_configured_folder_falls_back(tmp_path, monkeypatch):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    monkeypatch.setenv("VOXPRINT_MODELS_DIR", str(blocker / "models"))           # cannot be created (parent is a file)
    assert paths.models_dir() == paths.default_models_dir()


def test_backup_folder_is_never_the_models_folder(tmp_path, monkeypatch):
    b = tmp_path / "Voxprint-backup"
    (b / "models").mkdir(parents=True)
    monkeypatch.setenv("VOXPRINT_MODELS_DIR", str(b / "models"))
    assert paths.models_dir() == paths.default_models_dir()


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


def test_the_app_does_not_download_models():
    source = Path(model_store.__file__).read_text(encoding="utf-8")
    models_source = Path(models.__file__).read_text(encoding="utf-8")
    for text in (source, models_source):
        assert "snapshot_download" not in text
        assert "huggingface_hub" not in text
        assert "ensure_model" not in text
    assert "ensure_ready(" not in source
    assert models_source.count("ensure_ready(app='movie-dubber')") == 2
    assert "is not called" in models_source


def test_size_mismatch_is_not_ready_and_the_file_stays(tmp_path, monkeypatch):
    good = b"weights"
    manifest = {"org/m": {"revision": "a" * 40, "license": "MIT",
                          "files": {"config.json": {"size": 2, "sha256": hashlib.sha256(b"{}").hexdigest()},
                                    "model.safetensors": {"size": len(good), "sha256": hashlib.sha256(good).hexdigest()}}}}
    monkeypatch.setattr(model_store, "load_manifest", lambda path=None: manifest)
    folder = tmp_path / "org--m"
    folder.mkdir()
    (folder / "config.json").write_bytes(b"{}")
    (folder / "model.safetensors").write_bytes(b"WEIGHTS")
    assert model_store.is_ready(folder, "org/m") is False
    assert (folder / "model.safetensors").read_bytes() == b"WEIGHTS"


def test_link_blobs_hardlinks_a_complete_model(tmp_path, monkeypatch):
    payload = b"weights"
    digest = hashlib.sha256(payload).hexdigest()
    config = b"{}"
    config_digest = hashlib.sha256(config).hexdigest()
    manifest = {"org/m": {"revision": "c" * 40, "license": "MIT", "files": {
        "config.json": {"size": len(config), "sha256": config_digest},
        "model.safetensors": {"size": len(payload), "sha256": digest},
    }}}
    monkeypatch.setattr(model_store, "load_manifest", lambda path=None: manifest)
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    (blobs / digest).write_bytes(payload)
    (blobs / config_digest).write_bytes(config)
    linked = model_store.link_blobs("org/m", root=tmp_path)
    assert linked == tmp_path / "org--m"
    assert (linked / "model.safetensors").read_bytes() == payload
    assert (linked / ".verified").is_file()
    assert model_store.is_ready(linked, "org/m")


def test_ensure_names_mdl_and_does_not_fetch(monkeypatch):
    monkeypatch.setattr(models, "locate", lambda _repo: None)
    with pytest.raises(models.ModelUnavailable, match="voxprint-mdl enqueue --app movie-dubber"):
        models.ensure("org/missing", allow_download=True)


def test_registry_has_no_gated_model_and_pins_the_mirrors(tmp_path):
    from dubber.engines.diarization import pipeline_checkpoint, rewrite_pipeline_config

    assert all(spec.gated is False for spec in models.SPECS.values())
    diar = models.SPECS["diar"]
    embed = models.SPECS["embed"]
    assert diar.repo == "Mitroshenkov87/voxprint-pyannote-speaker-diarization-community-1"
    assert embed.repo == "Mitroshenkov87/voxprint-pyannote-embedding"
    assert not diar.repo.startswith("pyannote/") and not embed.repo.startswith("pyannote/")
    manifest = model_store.load_manifest()
    assert "pyannote/embedding" not in manifest
    assert "pyannote/speaker-diarization-community-1" not in manifest
    assert manifest[diar.repo]["revision"] == "9c851af727f1085a361777d13e32734f7f5ebb30"
    assert manifest[embed.repo]["revision"] == "4655ca656e1d7b1d8fe8100b39b4fdbf3015c7b8"
    assert manifest[diar.repo]["license"] == "CC-BY-4.0"
    assert manifest[embed.repo]["license"] == "MIT"
    assert manifest[diar.repo]["files"]["config.yaml"]["sha256"] == "5ce2bfa9a938dc132cec1172592d65173cbb8f444ea1e4133f10f9391de155be"
    assert manifest[diar.repo]["files"]["segmentation/pytorch_model.bin"]["sha256"] == "7ad24338d844fb95985486eb1a464e32d229f6d7a03c9abe60f978bacf3f816e"
    assert manifest[embed.repo]["files"]["pytorch_model.bin"]["sha256"] == "4bcec986de13da7af7ac88736572692359950df63669989c4f78b294934c9089"
    root = Path(__file__).resolve().parents[1]
    notice = (root / "NOTICE").read_text(encoding="utf-8")
    models_doc = (root / "docs" / "MODELS.md").read_text(encoding="utf-8")
    third = (root / "docs" / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    for text in (notice, models_doc, third):
        assert "9c851af727f1085a361777d13e32734f7f5ebb30" in text
        assert "4655ca656e1d7b1d8fe8100b39b4fdbf3015c7b8" in text
        assert "no changes besides removing the access gate" in text.lower()
        assert "https://huggingface.co/pyannote/speaker-diarization-community-1" in text
        assert "https://huggingface.co/Mitroshenkov87/voxprint-pyannote-speaker-diarization-community-1" in text
        assert "https://huggingface.co/pyannote/embedding" in text
        assert "https://huggingface.co/Mitroshenkov87/voxprint-pyannote-embedding" in text
    sample = "segmentation: pyannote/segmentation-3.0\nembedding: pyannote/wespeaker-voxceleb-resnet34-LM\nplda: $model/plda\n"
    folder = Path("/models/diar")
    rewritten = rewrite_pipeline_config(sample, folder)
    assert "pyannote/" not in rewritten
    assert "$model/plda" in rewritten
    assert (folder / "segmentation").resolve().as_posix() in rewritten
    assert (folder / "embedding").resolve().as_posix() in rewritten
    kept = "embedding: $model/embedding\nsegmentation: $model/segmentation\n"
    assert rewrite_pipeline_config(kept, folder) == kept
    local = tmp_path / "diar"
    local.mkdir()
    original = "embedding: $model/embedding\nsegmentation: pyannote/segmentation-3.0\n"
    (local / "config.yaml").write_text(original, encoding="utf-8")
    checkpoint = pipeline_checkpoint(local)
    assert checkpoint.name == "config.local.yaml"
    assert (local / "config.yaml").read_text(encoding="utf-8") == original
    body = checkpoint.read_text(encoding="utf-8")
    assert "pyannote/" not in body and "$model/embedding" in body


def test_users_file(tmp_path):
    model_store.register_user("audiobook-builder", tmp_path)
    model_store.register_user(root=tmp_path)
    assert model_store.read_users(tmp_path) == {"audiobook-builder": True, "movie-dubber": True}
    assert model_store.unregister_user(root=tmp_path) == ["audiobook-builder"]
    assert model_store.unregister_user("audiobook-builder", tmp_path) == []
    assert json.loads((tmp_path / ".users.json").read_text()) == {}


def test_plan_keys_leave_diarization_optional_and_pick_one_tts():
    big = models.plan_keys(24)
    small = models.plan_keys(8)
    unknown = models.plan_keys(0)
    assert "tts_1_7b" in big and "tts_0_6b" not in big
    assert "tts_0_6b" in small and "tts_1_7b" not in small
    assert "tts_0_6b" in unknown and "tts_1_7b" not in unknown
    for keys in (big, small, unknown):
        assert "diar" not in keys and "embed" not in keys
        assert "asr" in keys and "sep" in keys
    raw = json.loads(models.manifest_path().read_text(encoding="utf-8"))
    assert raw["schema"] == 1 and raw["app"] == "movie-dubber"
    assert "TODO" in raw["notes"] and "Qwen3-TTS" in raw["notes"]
    tiers = {row["vram_tier"] for row in raw["models"] if row["id"] == "qwen3-tts"}
    assert tiers == {"24gb", "16gb", "any"}
    diar = next(row for row in raw["models"] if row["id"] == "diar")
    assert diar["on_demand"] is True
    assert diar["repo"].startswith("Mitroshenkov87/voxprint-pyannote-")
    assert all(not str(row.get("repo", "")).startswith("pyannote/") for row in raw["models"])


def test_bundled_manifest_is_well_formed():
    m = model_store.load_manifest()
    assert "Qwen/Qwen3-TTS-12Hz-1.7B-Base" in m
    for repo, e in m.items():
        assert len(e["revision"]) == 40, repo
        assert e.get("license"), repo
        assert all(len(f["sha256"]) == 64 and f["size"] > 0 for f in e["files"].values())


def test_registry_licences_match_manifest_and_tc_big_pairs():
    manifest = model_store.load_manifest()
    assert "Qwen/Qwen3-ForcedAligner-0.6B" not in manifest
    for spec in models.SPECS.values():
        assert spec.license
        if spec.repo not in manifest:
            continue
        assert manifest[spec.repo]["license"] == spec.license
        pinned = set(manifest[spec.repo]["files"])
        if spec.patterns:
            assert set(spec.patterns) == pinned
    for repo in manifest:
        assert any(s.repo == repo for s in models.SPECS.values()), repo
    assert models.mt_spec("en", "ru").repo == "Helsinki-NLP/opus-mt-tc-big-en-zle"
    assert models.mt_spec("en", "ru").target_token == ">>rus<<"
    assert models.mt_spec("ru", "en").repo == "Helsinki-NLP/opus-mt-tc-big-zle-en"
    assert models.mt_spec("ru", "en").target_token == ""
    assert models.mt_spec("ru", "de").repo == "Helsinki-NLP/opus-mt-tc-big-zle-de"
    assert models.mt_spec("de", "ru").repo == "Helsinki-NLP/opus-mt-tc-big-de-zle"
    assert models.mt_spec("de", "ru").target_token == ">>rus<<"
    assert models.mt_spec("en", "de").repo == "Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-gmw"
    assert models.mt_spec("en", "de").target_token == ">>deu<<"
    assert models.mt_spec("en", "de").license == "Apache-2.0"
    assert models.mt_spec("de", "en").repo == "Helsinki-NLP/opus-mt-tc-bible-big-gmw-deu_eng_fra_por_spa"
    assert models.mt_spec("de", "en").target_token == ">>eng<<"
    assert models.mt_spec("de", "en").license == "Apache-2.0"
    for src in TARGET_LANGS:
        for tgt in TARGET_LANGS:
            if src == tgt:
                assert models.mt_spec(src, tgt) is None
            else:
                spec = models.mt_spec(src, tgt)
                assert spec is not None and spec.repo in manifest, f"{src}->{tgt}"
    assert models.mt_inputs(">>rus<<", ["Hello"]) == [">>rus<< Hello"]
    assert models.mt_inputs("", ["Hello"]) == ["Hello"]
    joined = " ".join(str(models.SPECS[k].repo) for k in models.SPECS)
    assert "opus-mt-en-ru" not in joined and "opus-mt-en-de" not in joined
    assert "opus-mt-de-en" not in joined and "opus-mt-ru-en" not in joined
    root = Path(__file__).resolve().parents[1]
    text = (root / "docs" / "MODELS.md").read_text(encoding="utf-8") + (root / "NOTICE").read_text(encoding="utf-8")
    assert "https://creativecommons.org/licenses/by/4.0/" in text
    assert "https://www.apache.org/licenses/LICENSE-2.0" in text
    assert "https://huggingface.co/Helsinki-NLP" in text
    assert "ac9b0614ab3cd7f77219e18ba494dfd93956c348" in text
    assert "en → de" in text or "en->de" in text


def test_checkpoint_model_is_checksummed_without_a_config(tmp_path, monkeypatch):
    payload = b"ckpt-bytes"
    repo = "org/ckpt"
    digest = hashlib.sha256(payload).hexdigest()
    manifest = {repo: {"revision": "b" * 40, "license": "MIT",
                       "files": {"weights.ckpt": {"size": len(payload), "sha256": digest}}}}
    monkeypatch.setattr(model_store, "load_manifest", lambda path=None: manifest)
    folder = tmp_path / "done"
    folder.mkdir()
    (folder / "weights.ckpt").write_bytes(payload)
    assert not model_store.verify_structure(folder, ("*.ckpt",), require_config=True)
    assert model_store.is_ready(folder, repo, ("*.ckpt",), ("weights.ckpt",), require_config=False)
    (folder / "weights.ckpt").write_bytes(b"nope")
    assert model_store.is_ready(folder, repo, ("*.ckpt",), ("weights.ckpt",), require_config=False) is False
    (folder / "weights.ckpt").write_bytes(payload)
    model_store.write_verified_marker(folder, repo)
    assert (folder / ".verified").is_file()
    assert model_store.is_ready(folder, repo, ("*.ckpt",), ("weights.ckpt",), require_config=False)
    linked = tmp_path / "link"
    linked.mkdir()
    (linked / "MelBandRoformer.ckpt").write_bytes(payload)
    assert _expose_roformer(linked, "vocals_mel_band_roformer.ckpt") == "vocals_mel_band_roformer.ckpt"
    assert (linked / "vocals_mel_band_roformer.ckpt").resolve() == (linked / "MelBandRoformer.ckpt").resolve()


# ------------------------------------------------------------------ GPU lock
def test_gpu_lock_acquire_release_and_wait(tmp_path):
    path = tmp_path / "gpu.lock"
    with gpu_job("movie-dubber", "tts", eta=100, path=path):
        data = json.loads(path.read_text())
        assert data["app"] == "movie-dubber" and data["stage"] == "tts"
        datetime.fromisoformat(data["started"]) and datetime.fromisoformat(data["eta"])
    assert not path.exists()
    now = datetime.now(timezone.utc)
    path.write_text(json.dumps({"app": "audiobook-builder", "stage": "train", "pid": os.getpid(),
                                "started": now.isoformat(), "eta": (now + timedelta(hours=1)).isoformat()}))
    with pytest.raises(GpuLockTimeout):
        with gpu_job("movie-dubber", "tts", timeout=0.2, poll=0.05, path=path):
            pass
    assert json.loads(path.read_text())["app"] == "audiobook-builder"           # never removed by us


def test_gpu_lock_reentrant_keeps_the_outer_lock(tmp_path):
    path = tmp_path / "gpu.lock"
    with gpu_job("movie-dubber", "whole run", eta=100, path=path):
        with gpu_job("movie-dubber", "tts", eta=10, path=path):
            assert json.loads(path.read_text())["stage"] == "whole run"
        assert path.exists()                                                          # inner stage did not release it
    assert not path.exists()


def test_gpu_lock_staleness_rules():
    now = datetime.now(timezone.utc)
    old = {"app": "audiobook-builder", "pid": os.getpid(), "eta": (now - timedelta(hours=3)).isoformat()}
    assert is_stale(old, now=now)                                                    # eta older than two hours, even if the pid is alive
    recent = {"app": "audiobook-builder", "pid": os.getpid(), "eta": (now - timedelta(hours=1)).isoformat()}
    assert not is_stale(recent, now=now)
    dead = {"app": "audiobook-builder", "pid": 999999999, "eta": (now + timedelta(hours=1)).isoformat()}
    assert is_stale(dead, now=now)


def test_gpu_lock_left_by_our_crashed_run_is_taken_over(tmp_path):
    path = tmp_path / "gpu.lock"
    path.write_text(json.dumps({"app": "movie-dubber", "stage": "old", "started": "2026-01-01T00:00:00+00:00",
                                "eta": "2099-01-01T00:00:00+00:00", "pid": 999999999}))
    assert isinstance(try_acquire("movie-dubber", "new", 60, path), Acquired)
    release(path)


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


# ------------------------------------------------------------------ VRAM / RAM policy
def test_vram_budget_keeps_headroom_and_sizes_the_tts_batch():
    from dubber.infra import resources as R
    laptop = R.Snapshot("RTX 4090 Laptop", 16.0, 15.0, 32.0, 20.0, "nvidia-smi")
    # 15 free - max(2 GB, 8% of 16) = 13 GB; RAM stays at 75% of free
    assert laptop.vram_budget_gb == 13.0 and laptop.ram_budget_gb == 15.0
    assert R.TTS_ITEM_VRAM_GB == pytest.approx(1.2)
    assert R.vram_allowance_gb(30.0, 40.0) == pytest.approx(26.8)   # 8% of 40 GB is 3.2, above the 2 GB floor
    # everything fits: GPU, the big TTS model (5 + 0.5 + 1.2 = 6.7 <= 13)
    assert R.plan_stage("asr", laptop, "cuda").device == "cuda"
    assert R.plan_stage("tts", laptop, "cuda").tts_model == "tts_1_7b"
    # busy card (2.5 GB free -> 0.5 GB budget): ASR and translation go to the CPU
    busy = R.Snapshot("RTX 4090 Laptop", 16.0, 2.5, 32.0, 20.0, "nvidia-smi")
    assert busy.vram_budget_gb == 0.5
    assert R.plan_stage("asr", busy, "cuda").device == "cpu"
    assert R.plan_stage("translation", busy, "cuda").device == "cpu"
    mid = R.Snapshot("x", 16.0, 6.5, 32.0, 20.0)                    # 4.5 GB budget: 1.7B needs 6.7, 0.6B needs 4.3
    assert mid.vram_budget_gb == 4.5
    assert R.plan_stage("tts", mid, "cuda").tts_model == "tts_0_6b"
    assert R.plan_stage("tts", mid, "cuda", lighter_ok=False).tts_model == "tts_1_7b"
    # batch: what is left of the budget after the model; CUDA Graphs always 1; CPU limited by RAM
    assert R.tts_batch("cuda", laptop.vram_budget_gb - 5.0, 15.0) == 6   # (13 - 5 - 0.5) // 1.2
    assert R.tts_batch("cuda", 30.0, 15.0) == R.MAX_BATCH
    assert R.tts_batch("cuda", 0.2, 15.0) == 1
    assert R.tts_batch("cuda", laptop.vram_budget_gb - 5.0, 15.0, graphs=True) == 1
    assert R.tts_batch("cuda", laptop.vram_budget_gb - 5.0, 1.0) <= 2     # almost no free RAM
    assert R.tts_batch("cpu", 0.0, 1.5) == 1 and R.tts_batch("cpu", 0.0, 40.0) == R.MAX_CPU_BATCH
    # several voices stay loaded while there is room, else they are swapped
    assert R.keep_voices_loaded(6.0) and not R.keep_voices_loaded(1.0)
    assert "budget 13.0 GB" in laptop.describe()


def test_vram_fraction_cap_is_off_unless_set(monkeypatch):
    from dubber.infra import resources as R
    from voxprint_common import suite_settings as suite
    monkeypatch.delenv("VOXPRINT_VRAM_FRACTION", raising=False)
    monkeypatch.setattr(suite, "read", lambda: {"gpu": "auto"})
    assert R.vram_fraction() is None
    assert R.Snapshot("RTX", 16.0, 15.0, 32.0, 20.0).vram_budget_gb == 13.0
    monkeypatch.setenv("VOXPRINT_VRAM_FRACTION", "0.5")
    assert R.vram_fraction() == pytest.approx(0.5)
    assert R.vram_allowance_gb(15.0, 16.0, 0.5) == pytest.approx(8.0)     # min(13, 0.5 * 16)
    assert R.Snapshot("RTX", 16.0, 15.0, 32.0, 20.0).vram_budget_gb == 8.0
    monkeypatch.setenv("VOXPRINT_VRAM_FRACTION", "0")
    assert R.vram_fraction() is None
    monkeypatch.setenv("VOXPRINT_VRAM_FRACTION", "")
    assert R.vram_fraction() is None
    monkeypatch.delenv("VOXPRINT_VRAM_FRACTION")
    monkeypatch.setattr(suite, "read", lambda: {"gpu": {"device": "cuda:1", "vram_fraction": 0.25}})
    assert suite.gpu() == "cuda:1" and suite.normalize_gpu("cuda:0") == "cuda:0"
    assert R.vram_fraction() == pytest.approx(0.25)
    monkeypatch.setattr(suite, "read", lambda: {"gpu": "cpu", "vram_fraction": 0.4})
    assert suite.gpu() == "auto" and R.vram_fraction() == pytest.approx(0.4)
    assert R.vram_fraction({"gpu": {"vram_fraction": 0.3}}) == pytest.approx(0.3)


def test_vram_budget_remeasures_free_before_each_batch_group(monkeypatch):
    from dubber.engines import tts as tts_mod
    from dubber.infra import resources as R
    free = {"gb": 15.0}
    monkeypatch.setattr(R, "gpu_from_torch", lambda: ("Test", 16.0, free["gb"]))
    monkeypatch.delenv("VOXPRINT_VRAM_FRACTION", raising=False)
    budget = R.VramBudget()
    # TTS headroom on a 16 GB card: 8 % is 1.28 GB, raised to the 1.5 GB floor
    assert budget.budget == pytest.approx(13.5)
    free["gb"] = 9.0
    assert budget.remeasure() == pytest.approx(7.5)
    assert budget.left() == pytest.approx(7.5)

    class Eng(tts_mod.MockTTS):
        def max_batch(self) -> int:
            return 2

    engine = Eng()
    engine.budget = budget
    seen = []
    real = budget.remeasure

    def wrapped() -> float:
        seen.append(free["gb"])
        return real()

    budget.remeasure = wrapped  # type: ignore[method-assign]
    got: dict = {}
    engine.run_queue([(i, "hi") for i in range(4)], tts_mod.VoiceSpec("v", "clone"), lambda i, w: got.__setitem__(i, w))
    assert len(got) == 4 and len(seen) >= 2                          # one fresh reading before each group of 2


def test_snapshot_reads_ram_and_gpu(monkeypatch):
    from dubber.infra import resources as R
    monkeypatch.setenv("VOXPRINT_FAKE_VRAM", "RTX 5090 Laptop,24,20")
    s = R.snapshot(use_torch=False)
    assert s.gpu == "RTX 5090 Laptop" and s.vram_budget_gb == 18.0 and s.ram_total_gb > 0


def test_runner_moves_a_stage_that_does_not_fit_to_the_cpu(tmp_path, monkeypatch):
    from dubber.core.project import Project
    from dubber.pipeline import runner as RN, stages as S
    monkeypatch.setenv("VOXPRINT_FAKE_VRAM", "RTX 4090 Laptop,16,2")
    monkeypatch.setattr(S, "_device", lambda cfg: "cuda")
    logs = []
    r = RN.Runner(Project(tmp_path / "p"), {"allow_download": True}, RN.Callbacks(log=logs.append))
    assert r._stage_cfg("asr")["device"] == "cpu"
    assert any("running on the CPU" in m for m in logs)
    monkeypatch.setenv("VOXPRINT_FAKE_VRAM", "RTX 4090 Laptop,16,15")
    assert r._stage_cfg("tts")["device"] == "cuda" and r._stage_cfg("tts")["tts_model"] == "tts_1_7b"
