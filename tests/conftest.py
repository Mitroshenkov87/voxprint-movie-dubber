"""Test isolation: private data folder and Desktop, offscreen Qt, English UI."""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    home, desk = tmp_path / "home", tmp_path / "Desktop"
    desk.mkdir()
    monkeypatch.setenv("VOXPRINT_DUBBER_HOME", str(home))
    monkeypatch.setenv("VOXPRINT_DESKTOP", str(desk))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    from dubber import i18n
    i18n.reset()
    i18n.set_language("en", save=False)
    yield
    i18n.reset()


@pytest.fixture
def clip():
    return ROOT / "assets" / "test_clip" / "voxprint-test-clip.mkv"
