"""Shared suite resources: manifest refcounts, the file lock, self-test repair, and migration."""
from __future__ import annotations

import json
import os
import threading

from dubber.cli import EXIT_GPU, EXIT_OK, main
from dubber.infra import components
from dubber.infra.shared_deps import GpuStatus, Piece, migrate, run_selftest


def _ffmpeg_name() -> str:
    return "ffmpeg.exe" if os.name == "nt" else "ffmpeg"


def test_manifest_refcount_preserves_unknown_keys_and_deletes_the_last_ref(tmp_path):
    home = tmp_path / "home"
    folder = home / "shared" / "ffmpeg" / "n8.1"
    folder.mkdir(parents=True)
    (folder / _ffmpeg_name()).write_text("ffmpeg", encoding="utf-8")
    outside = tmp_path / "chosen-models"
    outside.mkdir()
    (home / "shared" / "manifest.json").write_text(json.dumps({
        "schema": 1,
        "note": "keep-me",
        "resources": [{
            "id": "ffmpeg",
            "version": "n8.1",
            "path": str(folder),
            "apps": ["movie-dubber"],
            "extra": {"kept": True},
        }],
    }), encoding="utf-8")
    components.prepare(home)
    added = components.add_component(home, "ffmpeg-8", kind="tool", version="n8.1", path=folder, app="audiobook-builder")
    assert "movie-dubber" in added.refs and "audiobook-builder" in added.refs
    try:
        components.add_component(home, "models", kind="model", version="store", path=outside, app="movie-dubber")
    except components.OutsideShared:
        pass
    else:
        raise AssertionError("a model path outside the shared models directory must be refused")
    data = json.loads((home / "shared" / "manifest.json").read_text(encoding="utf-8"))
    assert data["schema"] == 1 and data["note"] == "keep-me"
    assert data["components"]["ffmpeg-8"]["kind"] == "tool"
    assert data["components"]["ffmpeg-8"]["path"] == "ffmpeg/n8.1"
    assert set(data["components"]["ffmpeg-8"]["refs"]) == {"movie-dubber", "audiobook-builder"}
    dropped = components.release_component(home, "ffmpeg-8", "movie-dubber")
    assert dropped.deleted == () and folder.is_dir()
    assert components.component_refs(home, "ffmpeg-8") == ["audiobook-builder"]
    assert outside.is_dir()
    finished = components.release_app(home, "audiobook-builder")
    assert str(folder) in finished.deleted
    assert not folder.exists()
    assert not (home / "shared" / "manifest.json").exists()


def test_model_component_lives_under_the_models_root_and_the_last_ref_deletes_it(tmp_path, monkeypatch):
    home = tmp_path / "home"
    models = home / "models" / "sep"
    models.mkdir(parents=True)
    (models / "weights.bin").write_text("weights", encoding="utf-8")
    monkeypatch.setenv("VOXPRINT_HOME", str(home))
    monkeypatch.delenv("VOXPRINT_MODELS_DIR", raising=False)
    monkeypatch.delenv("VOXPRINT_MODEL_CACHE", raising=False)
    added = components.add_component(home, "models", kind="model", version="store", path=home / "models", app="movie-dubber")
    assert added.kind == "model"
    assert added.path == (home / "models").resolve().as_posix()
    data = json.loads((home / "shared" / "manifest.json").read_text(encoding="utf-8"))
    assert data["components"]["models"]["kind"] == "model"
    finished = components.release_app(home, "movie-dubber")
    assert str((home / "models").resolve()) not in finished.deleted
    assert (models / "weights.bin").read_text(encoding="utf-8") == "weights"
    assert (home / "models").is_dir()
    assert not (home / "shared" / "manifest.json").exists()


def test_last_ref_removes_an_empty_models_root_this_app_alone_owns(tmp_path, monkeypatch):
    home = tmp_path / "home"
    models = home / "models"
    models.mkdir(parents=True)
    (models / ".users.json").write_text('{"movie-dubber": true}\n', encoding="utf-8")
    monkeypatch.setenv("VOXPRINT_HOME", str(home))
    monkeypatch.delenv("VOXPRINT_MODELS_DIR", raising=False)
    monkeypatch.delenv("VOXPRINT_MODEL_CACHE", raising=False)
    components.add_component(home, "models", kind="model", version="store", path=models, app="movie-dubber")
    finished = components.release_app(home, "movie-dubber")
    assert str(models.resolve()) in finished.deleted
    assert not models.exists()
    assert not (home / "shared" / "manifest.json").exists()


def test_manifest_keeps_a_running_directory_for_the_uninstaller(tmp_path, monkeypatch):
    home = tmp_path / "home"
    folder = home / "shared" / "runtimes" / "py3.14-torch2.11-cu130"
    folder.mkdir(parents=True)
    (folder / "runtime-key.json").write_text("{}", encoding="utf-8")
    components.add_component(home, "python-3.14", kind="runtime", version=folder.name, path=folder, app="movie-dubber")
    monkeypatch.setattr(components, "running_from", lambda path: True)
    result = components.release_app(home, "movie-dubber")
    assert str(folder.resolve()) in result.pending
    assert folder.is_dir()
    assert not (home / "shared" / "manifest.json").exists()


def test_manifest_lock_keeps_every_concurrent_reference(tmp_path):
    home = tmp_path / "home"
    folder = home / "shared" / "ffmpeg" / "n8.1"
    folder.mkdir(parents=True)
    (folder / _ffmpeg_name()).write_text("ffmpeg", encoding="utf-8")
    errors: list[BaseException] = []

    def work(index: int) -> None:
        try:
            components.add_component(home, "ffmpeg-8", kind="tool", version="n8.1", path=folder, app=f"app-{index}")
        except BaseException as exc:  # noqa: BLE001 - the assertion below reports it
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert set(components.component_refs(home, "ffmpeg-8")) == {f"app-{index}" for index in range(8)}


def test_selftest_repairs_a_missing_piece(tmp_path):
    state = {"ok": False}
    progress: list[dict] = []

    def inspect() -> list[Piece]:
        return [Piece("ffmpeg", "n8.1", str(tmp_path), state["ok"], "" if state["ok"] else "missing")]

    def fix(resource_id: str, prog) -> None:
        assert resource_id == "ffmpeg"
        prog({"progress": True, "resource": "ffmpeg", "status": "repairing", "message": "Repairing ffmpeg"})
        state["ok"] = True

    report = run_selftest(progress.append, gpu=lambda: GpuStatus(True, "", ""), inspect=inspect, fix=fix, repair=True)
    assert report.outcome == "repair" and report.exit_code == EXIT_OK and report.repaired == ["ffmpeg"]
    assert any(event.get("status") == "repairing" for event in progress)


def test_selftest_reports_an_old_driver_and_does_not_repair(capsys, monkeypatch):
    called: list[str] = []

    def gpu() -> GpuStatus:
        return GpuStatus(False, "old_driver", "NVIDIA driver branch 550 is older than 600. NVIDIA driver downloads: https://www.nvidia.com/Download/index.aspx")

    report = run_selftest(
        lambda event: None,
        gpu=gpu,
        inspect=lambda: [],
        fix=lambda resource_id, prog: called.append(resource_id),
        repair=True,
    )
    assert report.exit_code == EXIT_GPU and report.outcome == "gpu"
    assert "https://www.nvidia.com/Download/index.aspx" in report.message
    assert called == []

    def fake(progress, *, repair=True, check_gpu=True):
        progress({"progress": True, "resource": "gpu", "status": "failed", "message": report.message})
        return report

    monkeypatch.setattr("dubber.infra.shared_deps.live_selftest", fake)
    assert main(["selftest", "--json"]) == EXIT_GPU
    captured = capsys.readouterr()
    body = json.loads(captured.out)
    assert body["outcome"] == "gpu" and body["exit_code"] == EXIT_GPU
    assert "https://www.nvidia.com/Download/index.aspx" in body["error"]
    assert '"status": "failed"' in captured.err


def test_migrate_moves_the_legacy_runtime_ffmpeg_and_models(tmp_path, monkeypatch):
    from voxprint_common.paths import home as suite_home

    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "vox"))
    home = suite_home()
    home.mkdir(parents=True)
    old = home / "runtime-0123456789ab"
    (old / "env").mkdir(parents=True)
    (old / "runtime-key.json").write_text(json.dumps({"flavor": "cu130"}), encoding="utf-8")
    (old / "env" / "marker").write_text("ok", encoding="utf-8")
    app = tmp_path / "app"
    (app / "bin").mkdir(parents=True)
    (app / "bin" / _ffmpeg_name()).write_text("ffmpeg", encoding="utf-8")
    (app / "runtime-dir.txt").write_text(str(old), encoding="utf-8")
    (home / "shared" / "models" / "kept").mkdir(parents=True)
    actions = migrate(home, app)
    dest = home / "shared" / "runtimes" / "py3.14-torch2.11-cu130"
    assert (dest / "env" / "marker").is_file()
    assert not old.exists()
    assert not (app / "runtime-dir.txt").exists()
    assert (home / "shared" / "ffmpeg" / "n8.1" / _ffmpeg_name()).read_text(encoding="utf-8") == "ffmpeg"
    assert not (app / "bin").exists()
    assert (home / "models" / "kept").is_dir()
    assert not (home / "shared" / "models").exists()
    assert actions
    assert migrate(home, app) == []
