"""Headless CLI: one JSON object, dry-run does no work, exit codes stay distinct."""
from __future__ import annotations

import json
from pathlib import Path

from dubber.cli import (
    EXIT_CANCELLED,
    EXIT_GPU,
    EXIT_INPUT,
    EXIT_JOB,
    EXIT_MODELS,
    EXIT_OK,
    EXIT_USAGE,
    _job_code,
    main,
)


def _json(capsys, code: int) -> dict:
    captured = capsys.readouterr()
    text = captured.out.strip()
    decoder = json.JSONDecoder()
    data, end = decoder.raw_decode(text)
    assert text[end:].strip() == ""
    assert isinstance(data, dict)
    assert data["exit_code"] == code
    assert data["ok"] is (code == EXIT_OK)
    assert "Traceback" not in captured.err
    return data


def test_version_json_and_human_line(capsys):
    assert main(["version", "--json"]) == EXIT_OK
    data = _json(capsys, EXIT_OK)
    assert data["command"] == "version"
    assert data["dry_run"] is False
    assert "Bochan" in data["label"]
    assert "1.0.0 RC" in data["label"]
    assert main(["--version"]) == EXIT_OK
    captured = capsys.readouterr()
    assert "1.0.0 RC" in captured.out and "Bochan" in captured.out
    assert captured.out.count("\n") == 1


def test_dry_run_is_one_json_object_and_does_not_download(monkeypatch, capsys):
    def boom(*_args, **_kwargs):
        raise AssertionError("models.ensure must not run during dry-run")

    monkeypatch.setattr("dubber.models.ensure", boom)
    code = main(["--dry-run", "--json"])
    data = _json(capsys, code)
    assert data["dry_run"] is True
    assert data["command"] == "dry-run"
    assert isinstance(data["runtime"], dict) and data["runtime"]["python"]
    assert isinstance(data["models"], list) and data["models"]
    assert {"key", "available"} <= set(data["models"][0])
    gpu = data["gpu"]
    assert "tier" in gpu and "summary" in gpu
    if gpu["ok"]:
        assert code == EXIT_OK
    else:
        assert code == EXIT_GPU
        assert gpu["code"] in {"no_cuda", "low_compute"}
    assert isinstance(data["would_run"], dict)


def test_dry_run_reports_a_forced_unsupported_gpu(monkeypatch, capsys):
    monkeypatch.setattr("dubber.cli._gpu", lambda: {
        "ok": False, "code": "no_cuda", "summary": "No supported NVIDIA GPU was found.",
        "name": "", "compute": None, "vram_total_gb": 0.0, "vram_free_gb": 0.0,
        "vram_known": False, "vram_source": "none", "tier": "16gb",
    })
    code = main(["--dry-run"])
    data = _json(capsys, code)
    assert code == EXIT_GPU
    assert data["gpu"]["code"] == "no_cuda"
    assert data["error"]


def test_fetch_models_unknown_key_is_usage(capsys):
    assert main(["--fetch-models", "--models", "nope"]) == EXIT_USAGE
    captured = capsys.readouterr()
    assert "nope" in captured.out

    assert main(["fetch-models", "--models", "nope", "--json"]) == EXIT_USAGE
    data = _json(capsys, EXIT_USAGE)
    assert "nope" in data["error"]


def test_fetch_models_dry_run_does_not_download(monkeypatch, capsys):
    monkeypatch.setattr("dubber.models.ensure", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("download")))
    monkeypatch.setattr("dubber.cli._gpu", lambda: {"ok": True, "code": None, "summary": "ok", "vram_total_gb": 24.0})
    monkeypatch.setattr("dubber.cli._model_rows", lambda: [
        {"key": "sep", "repo": "x", "title": "sep", "available": False},
    ])
    code = main(["fetch-models", "--models", "sep", "--dry-run", "--json"])
    data = _json(capsys, code)
    assert code == EXIT_MODELS
    assert data["would_run"]["models"] == ["sep"]


def test_run_project_dry_run_does_not_create_a_project(monkeypatch, capsys, tmp_path):
    video = tmp_path / "clip.mkv"
    video.write_bytes(b"x")
    monkeypatch.setenv("VOXPRINT_DUBBER_HOME", str(tmp_path / "home"))
    monkeypatch.setattr("dubber.models.ensure", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("download")))
    code = main(["run-project", str(video), "--languages", "en,ru", "--dry-run", "--json"])
    data = _json(capsys, code)
    assert data["would_run"]["kind"] == "video"
    assert data["would_run"]["source_lang"] == "en"
    assert data["would_run"]["target_lang"] == "ru"
    assert "probe" in data["would_run"]["stages"]
    assert not list((tmp_path / "home").rglob("project.json"))
    if not data["gpu"]["ok"]:
        assert code == EXIT_GPU
    elif data.get("error", "").startswith("missing models"):
        assert code == EXIT_MODELS
    else:
        assert code == EXIT_OK


def test_missing_input_and_project_info(capsys, tmp_path):
    missing = tmp_path / "gone.vxdub"
    assert main(["info", str(missing), "--json"]) == EXIT_INPUT
    data = _json(capsys, EXIT_INPUT)
    assert "not found" in data["error"]

    from dubber.core.project import Project

    src = tmp_path / "film.mkv"
    src.write_bytes(b"x")
    project = Project.create(tmp_path / "proj", src, source_lang="en", target_lang="de")
    assert main(["project-info", str(project.folder), "--json"]) == EXIT_OK
    data = _json(capsys, EXIT_OK)
    assert data["project"]["kind"] == "project"
    assert data["project"]["source_lang"] == "en"
    assert data["project"]["target_lang"] == "de"
    assert data["dry_run"] is False


def test_help_json_is_one_object(capsys):
    assert main(["--help", "--json"]) == EXIT_OK
    data = _json(capsys, EXIT_OK)
    assert data["command"] == "help"
    assert {row["code"] for row in data["exit_codes"]} == {0, 2, 3, 4, 5, 6, 7}


def test_usage_and_job_codes(capsys):
    assert main(["nope", "--json"]) == EXIT_USAGE
    data = _json(capsys, EXIT_USAGE)
    assert data["command"] == "usage"
    assert _job_code(True, "") == EXIT_OK
    assert _job_code(False, "cancelled") == EXIT_CANCELLED
    assert _job_code(False, "tts failed") == EXIT_JOB


def test_docs_list_the_commands_and_exit_codes():
    text = (Path(__file__).resolve().parents[1] / "docs" / "CLI.md").read_text(encoding="utf-8")
    for name in ("version", "diagnose", "fetch-models", "run-project", "info", "--dry-run", "--json"):
        assert name in text
    for code, title in (
        (0, "ok"), (2, "usage"), (3, "gpu"), (4, "models"), (5, "input"), (6, "job"), (7, "cancelled"),
    ):
        assert f"| {code} |" in text or f"| {code} " in text
        assert title in text


def test_bad_language_is_usage(capsys, tmp_path):
    video = tmp_path / "clip.mkv"
    video.write_bytes(b"x")
    assert main(["run-project", str(video), "--languages", "en", "--json"]) == EXIT_USAGE
    data = _json(capsys, EXIT_USAGE)
    assert "languages" in data["error"]
