"""BUILD.json is the only copy of the offset and the codename."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Build installers workflow run_number was 17 on 2026-10-10 (workflow run 38057799983).
# The next run of that workflow is 18. offset 983 makes that build 1001.
SEEN_RELEASE_RUN = 17


def test_build_json_offset_makes_the_next_release_build_at_least_1001():
    data = json.loads((ROOT / "BUILD.json").read_text(encoding="utf-8"))
    assert set(data) == {"offset", "codename"}
    assert isinstance(data["codename"], str) and data["codename"].strip()
    assert isinstance(data["offset"], int)
    nxt = SEEN_RELEASE_RUN + 1
    assert nxt + data["offset"] >= 1001
    assert data["offset"] == 1001 - nxt


def test_library_build_number_is_never_invented_as_zero(tmp_path, monkeypatch):
    """voxprint_common.buildinfo reads BUILD.json and build_info.json. A missing build stays null."""
    from dubber import appinfo
    from voxprint_common.buildinfo import parse

    source = {"offset": 983, "codename": "Bochan"}
    assert parse(source, env={}).build is None
    assert parse(source, env={"GITHUB_RUN_NUMBER": "18"}).build == 1001
    merged = {"offset": 983, "codename": "Bochan", "build": 1001}
    assert parse(merged, env={"GITHUB_RUN_NUMBER": "18"}).build == 1001
    assert parse(merged, env={"VOXPRINT_BUILD": "dev", "GITHUB_RUN_NUMBER": "18"}).build == 1001
    assert parse(merged, env={"VOXPRINT_BUILD": "1004"}).build == 1004
    assert parse({}, env={}).build is None
    assert parse({"offset": 983, "build": "1001", "codename": "Bochan"}, env={}).build == 1001
    assert parse({"build": 0, "codename": "Bochan"}, env={}).build == 0

    folder = tmp_path / "app"
    folder.mkdir()
    (folder / "BUILD.json").write_text(json.dumps(source), encoding="utf-8")
    monkeypatch.setattr(appinfo, "resource_dir", lambda: folder)
    monkeypatch.delenv("VOXPRINT_BUILD", raising=False)
    monkeypatch.delenv("GITHUB_RUN_NUMBER", raising=False)
    assert appinfo.app_build() == appinfo.DEV_BUILD
    assert appinfo.app_codename() == "Bochan"
    assert appinfo.version_label() == '1.0.0 RC · build dev "Bochan"'
    monkeypatch.setenv("GITHUB_RUN_NUMBER", "18")
    assert appinfo.app_build() == 1001
    assert appinfo.version_label() == '1.0.0 RC · build 1001 "Bochan"'
    (folder / "build_info.json").write_text(
        json.dumps({"build": 1004, "codename": "Stamped"}), encoding="utf-8",
    )
    monkeypatch.delenv("GITHUB_RUN_NUMBER", raising=False)
    assert appinfo.app_build() == 1004
    assert appinfo.app_codename() == "Stamped"
    assert "build 0" not in appinfo.version_label()


def test_dev_marker_in_the_label(monkeypatch):
    from dubber import appinfo
    from voxprint_common.buildinfo import BuildInfo

    monkeypatch.setattr(
        appinfo,
        "_identity",
        lambda: BuildInfo(app="movie-dubber", version="1.0.0", rc=None, build=None, codename="Bochan", show_rc=True),
    )
    assert appinfo.app_build() == "dev"
    assert appinfo.version_label() == '1.0.0 RC · build dev "Bochan"'


def _has_cyrillic(text: str) -> bool:
    return any("\u0400" <= ch <= "\u052f" for ch in text)


def test_identity_sources_have_no_cyrillic_and_are_not_hardcoded():
    appinfo = (ROOT / "dubber" / "appinfo.py").read_text(encoding="utf-8")
    assert "Chazak" not in appinfo
    assert "APP_BUILD = 1000" not in appinfo
    assert 'CODENAME = "' not in appinfo
    for rel in (
        "BUILD.json", "dubber/appinfo.py", "dubber/cli.py", "docs/CLI.md", "docs/BUILDING.md",
        "README.md", "main.py", ".github/workflows/audit.yml", ".github/workflows/build-installer.yml",
    ):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert not _has_cyrillic(text), rel
