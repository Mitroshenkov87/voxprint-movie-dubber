"""Laptop build 1011: optional models must not block startup, downloads retry, foreign models survive."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.error import URLError

import pytest

from dubber import models
from dubber.cli import EXIT_MODELS
from dubber.i18n import tr
from dubber.infra import components, model_store
from dubber.infra.model_store import ModelUnavailable
from dubber.infra.shared_deps import Piece, Report, download_resumable, live_pieces, live_selftest


def _plant_foreign_models(models_dir: Path) -> dict[str, Path]:
    """Audiobook Builder and Plotweaver weights inside the shared models directory."""
    planted = {
        "asr": models_dir / "Qwen--Qwen3-ASR-1.7B",
        "aligner": models_dir / "Qwen--Qwen3-ForcedAligner-0.6B",
        "gemma": models_dir / "llm" / "gemma-3-12b",
    }
    for folder in planted.values():
        folder.mkdir(parents=True)
        (folder / "weights.bin").write_bytes(b"keep-me")
    return planted


def _users_document() -> dict:
    return {
        "schema": 1,
        "note": {"keep": True},
        "audiobook-builder": True,
        "plotweaver": True,
        "apps": ["audiobook-builder", "plotweaver"],
    }


def _manifest(models_dir: Path, refs: list[str]) -> dict:
    return {
        "schema": 1,
        "note": "keep-me",
        "components": {
            "models": {
                "kind": "model",
                "version": "store",
                "path": models_dir.resolve().as_posix(),
                "sha256": "a" * 64,
                "installed_at": "2026-10-11T00:00:00+00:00",
                "refs": list(refs),
                "extra": {"kept": True},
            },
            "model-gemma": {
                "kind": "model",
                "version": "12b",
                "path": (models_dir / "llm" / "gemma-3-12b").resolve().as_posix(),
                "sha256": "b" * 64,
                "installed_at": "2026-10-11T00:00:00+00:00",
                "refs": ["plotweaver"],
                "vendor": "plotweaver",
            },
        },
    }


def _assert_foreign_survived(planted: dict[str, Path], models_dir: Path) -> None:
    for folder in planted.values():
        assert (folder / "weights.bin").read_bytes() == b"keep-me"
    users = json.loads((models_dir / ".users.json").read_text(encoding="utf-8"))
    assert users["note"] == {"keep": True}
    assert users["audiobook-builder"] is True
    assert users["plotweaver"] is True
    assert "audiobook-builder" in users["apps"] and "plotweaver" in users["apps"]


def test_register_user_preserves_unknown_keys_and_other_apps(tmp_path):
    raw = _users_document()
    (tmp_path / ".users.json").write_text(json.dumps(raw), encoding="utf-8")
    model_store.register_user("movie-dubber", tmp_path)
    written = json.loads((tmp_path / ".users.json").read_text(encoding="utf-8"))
    assert written["note"] == {"keep": True}
    assert written["schema"] == 1
    assert written["audiobook-builder"] is True and written["plotweaver"] is True
    assert written["movie-dubber"] is True
    assert written["apps"] == ["audiobook-builder", "plotweaver", "movie-dubber"]
    others = model_store.unregister_user("movie-dubber", tmp_path)
    assert "audiobook-builder" in others and "plotweaver" in others
    again = json.loads((tmp_path / ".users.json").read_text(encoding="utf-8"))
    assert again["note"] == {"keep": True}
    assert "movie-dubber" not in again.get("apps", [])
    assert again["audiobook-builder"] is True


def test_release_fetch_and_repair_keep_other_apps_models(tmp_path, monkeypatch):
    home = tmp_path / "home"
    models_dir = home / "models"
    planted = _plant_foreign_models(models_dir)
    (models_dir / ".users.json").write_text(json.dumps(_users_document()), encoding="utf-8")
    manifest = home / "shared" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(_manifest(models_dir, ["movie-dubber"])), encoding="utf-8")
    monkeypatch.setenv("VOXPRINT_HOME", str(home))
    monkeypatch.delenv("VOXPRINT_MODELS_DIR", raising=False)
    monkeypatch.delenv("VOXPRINT_MODEL_CACHE", raising=False)

    model_store.register_user(root=models_dir)
    info = model_store.ensure_model(
        "org/movie-dubber-tts",
        root=models_dir,
        downloader=lambda repo, dest, revision, patterns, token: (
            dest.mkdir(parents=True, exist_ok=True),
            (dest / "config.json").write_bytes(b"{}"),
            (dest / "model.safetensors").write_bytes(b"ours"),
        ),
    )
    assert info.path == models_dir / "org--movie-dubber-tts"
    assert (planted["gemma"] / "weights.bin").is_file()

    good = b"ours"
    manifest_body = {"org/movie-dubber-tts": {"revision": "c" * 40, "files": {
        "config.json": {"size": 2, "sha256": hashlib.sha256(b"{}").hexdigest()},
        "model.safetensors": {"size": len(good), "sha256": hashlib.sha256(good).hexdigest()},
    }}}
    monkeypatch.setattr(model_store, "load_manifest", lambda path=None: manifest_body)
    ours = info.path
    (ours / "sidecar.txt").write_text("foreign-extra", encoding="utf-8")
    (ours / "config.json").unlink()
    with pytest.raises(ModelUnavailable, match="checksum"):
        model_store.ensure_model(
            "org/movie-dubber-tts",
            root=models_dir,
            downloader=lambda repo, dest, revision, patterns, token: (
                dest.mkdir(parents=True, exist_ok=True),
                (dest / "config.json").write_bytes(b"{}"),
                (dest / "model.safetensors").write_bytes(b"BAD"),
            ),
        )
    assert (ours / "sidecar.txt").read_text(encoding="utf-8") == "foreign-extra"
    assert (ours / "model.safetensors").read_bytes() == b"ours"

    components.add_component(home, "models", kind="model", version="store", path=models_dir, app="movie-dubber")
    components.release_app(home, "movie-dubber")
    model_store.unregister_user(root=models_dir)

    _assert_foreign_survived(planted, models_dir)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["note"] == "keep-me"
    assert data["components"]["model-gemma"]["refs"] == ["plotweaver"]
    assert data["components"]["model-gemma"]["vendor"] == "plotweaver"
    assert "movie-dubber" not in data["components"].get("models", {}).get("refs", [])
    assert (models_dir / "org--movie-dubber-tts" / "model.safetensors").is_file()


def test_sole_user_deletes_only_models_this_app_recorded(tmp_path, monkeypatch):
    home = tmp_path / "home"
    models_dir = home / "models"
    planted = _plant_foreign_models(models_dir)
    (models_dir / ".users.json").write_text(json.dumps({"movie-dubber": True, "note": "keep"}), encoding="utf-8")
    monkeypatch.setenv("VOXPRINT_HOME", str(home))
    monkeypatch.delenv("VOXPRINT_MODELS_DIR", raising=False)
    model_store.ensure_model(
        "org/ours",
        root=models_dir,
        downloader=lambda repo, dest, revision, patterns, token: (
            dest.mkdir(parents=True, exist_ok=True),
            (dest / "config.json").write_bytes(b"{}"),
            (dest / "model.safetensors").write_bytes(b"ours"),
        ),
    )
    components.add_component(home, "models", kind="model", version="store", path=models_dir, app="movie-dubber")
    components.release_app(home, "movie-dubber")
    assert not (models_dir / "org--ours").exists()
    for folder in planted.values():
        assert (folder / "weights.bin").read_bytes() == b"keep-me"
    assert models_dir.is_dir()
    users = json.loads((models_dir / ".users.json").read_text(encoding="utf-8"))
    assert users["note"] == "keep"


def test_optional_model_repair_does_not_abort(monkeypatch, tmp_path):
    from dubber.infra.shared_deps import install_shared_models

    def ensure(repo, allow_download=True, token=None, log=lambda message: None):
        if repo in (models.SPECS["diar"].repo, models.SPECS["embed"].repo):
            raise ModelUnavailable(f"{repo} missing")
        return tmp_path / "present", {"source": "present", "download_s": 0.0}

    monkeypatch.setattr("dubber.infra.shared_deps.model_registry.ensure", ensure)
    install_shared_models(lambda event: None)


def test_models_piece_stays_ok_when_only_diar_and_embed_are_missing(monkeypatch, tmp_path):
    present = tmp_path / "present"
    present.mkdir()

    def locate(repo: str):
        if repo in (models.SPECS["diar"].repo, models.SPECS["embed"].repo):
            return None
        return present

    monkeypatch.setattr("dubber.infra.shared_deps.model_registry.locate", locate)
    monkeypatch.setattr("dubber.infra.shared_deps.model_fetch_skipped", lambda: False)
    piece = {item.id: item for item in live_pieces(tmp_path)}["models"]
    assert piece.ok, piece.detail
    assert "diar" not in piece.detail and "embed" not in piece.detail


def test_selftest_reports_optional_models_without_failing(monkeypatch):
    monkeypatch.setattr("dubber.infra.shared_deps.guard_cpu_onnxruntime", lambda: "")
    monkeypatch.setattr("dubber.infra.shared_deps.migrate", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        "dubber.infra.shared_deps.live_pieces",
        lambda home: [Piece("ffmpeg", "n8.1", str(home), True)],
    )
    monkeypatch.setattr("dubber.infra.shared_deps.missing_optional_models", lambda: ["diar", "embed"])
    report = live_selftest(lambda event: None, repair=False, check_gpu=False)
    assert report.exit_code == 0
    assert report.optional_missing == ["diar", "embed"]


def test_model_repair_dialog_never_uses_the_gpu_gate_title():
    from dubber.ui.setup_progress import startup_repair_dialog

    dialog = startup_repair_dialog(Report("models", EXIT_MODELS, "diar: download failed"))
    assert dialog is not None
    title, body = dialog
    assert title == tr("models.repair_title")
    assert title != tr("gpu.gate_title")
    assert "Этот компьютер не может запустить" not in title
    assert "diar: download failed" in body
    assert startup_repair_dialog(Report("ok", 0, "")) is None


def test_main_model_failure_does_not_show_the_gpu_gate(monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QMessageBox

    import main as app_main

    QApplication.instance() or QApplication([])
    seen: dict[str, str] = {}

    def critical(_parent, title, body):
        seen["title"] = title
        seen["body"] = body

    monkeypatch.setattr(QMessageBox, "critical", critical)
    monkeypatch.setattr(
        "dubber.ui.setup_progress.ensure_with_dialog",
        lambda: Report("models", EXIT_MODELS, "asr: download failed"),
    )
    code = app_main.main([])
    assert code == EXIT_MODELS
    assert seen["title"] == tr("models.repair_title")
    assert seen["title"] != tr("gpu.gate_title")
    assert "asr: download failed" in seen["body"]


def test_optional_banner_disables_multi_voice_and_retry_clears_it(monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from dubber.ui.window import MainWindow

    QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show_optional_models(["diar", "embed"])
    assert window.lbl_optional.isVisible()
    assert window.btn_optional_retry.isVisible()
    assert not window.film.chk_multi.isEnabled()
    assert not window.film.chk_multi.isChecked()
    assert tr("gpu.gate_title") not in window.lbl_optional.text()
    assert "diar" in window.lbl_optional.text() and "embed" in window.lbl_optional.text()

    monkeypatch.setattr("dubber.ui.window.fetch_optional_with_dialog", lambda: [])
    window.retry_optional_models()
    assert not window.lbl_optional.isVisible()
    assert window.film.chk_multi.isEnabled()
    window.close()


def test_download_resumable_retries_with_a_per_read_timeout(tmp_path, monkeypatch):
    calls = {"n": 0, "timeouts": []}
    slept: list[float] = []

    class Response:
        status = 200

        def __init__(self) -> None:
            self._sent = False

        def read(self, size):
            if self._sent:
                return b""
            self._sent = True
            return b"zip-bytes"

        def close(self):
            return None

    def opener(request, timeout=None):
        calls["n"] += 1
        calls["timeouts"].append(timeout)
        if calls["n"] < 3:
            raise URLError("timed out")
        return Response()

    monkeypatch.setattr("dubber.infra.shared_deps.time.sleep", lambda seconds: slept.append(seconds))
    events: list[str] = []
    download_resumable(
        "https://example.invalid/ffmpeg.zip",
        tmp_path / "ffmpeg.zip",
        lambda event: events.append(str(event.get("message"))),
        "ffmpeg",
        opener,
    )
    assert calls["n"] == 3
    assert calls["timeouts"] == [30, 30, 30]
    assert slept == [1, 2]
    assert any("attempt 1 of 5" in message for message in events)
    assert any("attempt 3 of 5" in message for message in events)
    assert (tmp_path / "ffmpeg.zip").read_bytes() == b"zip-bytes"


def test_default_fetch_does_not_fail_when_only_optional_models_fail():
    from dubber.cli import _blocking_model_failures

    assert _blocking_model_failures(["diar", "embed"], explicit=False) == []
    assert _blocking_model_failures(["asr", "diar"], explicit=False) == ["asr"]
    assert _blocking_model_failures(["diar"], explicit=True) == ["diar"]


def test_installer_scripts_retry_downloads_and_keep_a_fresh_setup_log():
    root = Path(__file__).resolve().parents[1]
    ps1 = (root / "installer" / "install-runtime.ps1").read_text(encoding="utf-8")
    sh = (root / "installer" / "linux" / "install.sh").read_text(encoding="utf-8")
    iss = (root / "installer" / "VoxprintMovieDubber.iss").read_text(encoding="utf-8")
    assert "function Resume-Download" in ps1
    assert ps1.count("attempt") >= 1
    assert "for ($attempt = 1; $attempt -le 5; $attempt++)" in ps1
    assert "CopyTo(" not in ps1
    assert "ReadWriteTimeout" in ps1
    assert "VoxprintMovieDubber-setup-" in ps1
    assert "UTF8Encoding $false" in ps1
    assert "Select-Object -Skip 5" in ps1
    assert "refusing to delete shared models path" in ps1
    assert "curl -fL --retry 5 --retry-all-errors --retry-delay 1 -C -" in sh
    assert "--connect-timeout 30 --speed-limit 1 --speed-time 30" in sh
    assert "refusing to delete shared models path" in sh
    assert "VoxprintMovieDubber-setup-" in iss
    assert "GetDateTimeString" in iss
