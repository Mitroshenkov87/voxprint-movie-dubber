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
    assert data["codename"] == "Bochan"
    assert isinstance(data["offset"], int)
    nxt = SEEN_RELEASE_RUN + 1
    assert nxt + data["offset"] >= 1001
    assert data["offset"] == 1001 - nxt


def test_resolve_build_order():
    from dubber.appinfo import DEV_BUILD, resolve_build, resolve_codename

    source = {"offset": 983, "codename": "Bochan"}
    assert resolve_build({}, baked={}, source=source) == 983
    assert resolve_build({"GITHUB_RUN_NUMBER": "18"}, baked={}, source=source) == 1001
    assert resolve_build({"GITHUB_RUN_NUMBER": "18"}, baked={"build": 1001, "codename": "Bochan"}, source=source) == 1001
    assert resolve_build({"VOXPRINT_BUILD": "dev", "GITHUB_RUN_NUMBER": "18"}, baked={"build": 1001}, source=source) == DEV_BUILD
    assert resolve_build({"VOXPRINT_BUILD": "1004"}, baked={"build": 1001}, source=source) == 1004
    assert resolve_build({}, baked={}, source={}) == DEV_BUILD
    assert resolve_build({}, baked={"build": "1001"}, source=source) == 1001
    assert resolve_codename(baked={"codename": "Bochan"}, source={"codename": "Other"}) == "Bochan"
    assert resolve_codename(baked={}, source={"codename": "Bochan"}) == "Bochan"
    assert resolve_codename(baked={}, source={}) == ""


def test_dev_marker_in_the_label(monkeypatch):
    from dubber import appinfo

    monkeypatch.setattr(appinfo, "build_info", lambda: {})
    monkeypatch.setattr(appinfo, "build_json", lambda: {"offset": 983, "codename": "Bochan"})
    monkeypatch.setenv("VOXPRINT_BUILD", "dev")
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
