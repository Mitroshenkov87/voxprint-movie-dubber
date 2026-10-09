"""The icon set and the splash are the files the app and the installer load, in the expected sizes."""
import struct
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)


def _png_size(path: Path) -> tuple:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR"
    return struct.unpack(">II", data[16:24])


def _jpeg_size(path: Path) -> tuple:
    data = path.read_bytes()
    assert data[:2] == b"\xff\xd8"
    i = 2
    while i + 9 < len(data):
        assert data[i] == 0xFF
        marker = data[i + 1]
        if marker in (0xC0, 0xC1, 0xC2):
            height, width = struct.unpack(">HH", data[i + 5:i + 9])
            return width, height
        i += 2 + struct.unpack(">H", data[i + 2:i + 4])[0]
    raise AssertionError(f"no JPEG frame in {path}")


def _ico_sizes(path: Path) -> set:
    data = path.read_bytes()
    reserved, kind, count = struct.unpack_from("<HHH", data, 0)
    assert reserved == 0 and kind == 1
    return {tuple(v or 256 for v in struct.unpack_from("<BB", data, 6 + 16 * n)) for n in range(count)}


def test_icon_and_splash_files():
    assert _png_size(ROOT / "assets" / "voxprint-dubber.png") == (1024, 1024)
    assert _jpeg_size(ROOT / "assets" / "splash.jpg") == (1024, 1024)
    assert _png_size(ROOT / "installer" / "linux" / "voxprint-dubber-256.png") == (256, 256)
    for name in ("voxprint-dubber.ico", "voxprint-dubber-setup.ico"):
        assert _ico_sizes(ROOT / "assets" / name) == {(s, s) for s in ICON_SIZES}
    assert "original artwork" in (ROOT / "assets" / "ICON-LICENSE.txt").read_text(encoding="utf-8")


def test_installer_and_window_use_the_icons():
    iss = (ROOT / "installer" / "VoxprintMovieDubber.iss").read_text(encoding="utf-8")
    assert r"SetupIconFile=..\assets\voxprint-dubber-setup.ico" in iss
    assert r"UninstallDisplayIcon={app}\assets\voxprint-dubber.ico" in iss
    main = (ROOT / "main.py").read_text(encoding="utf-8")
    assert main.index("splash_mod.show()") < main.index("from dubber.ui.window import MainWindow")


def test_splash_paints_name_and_status():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from dubber.ui import splash
    s = splash.show()
    s.loading_ui()
    assert s.message() and s._fraction > 0
    assert not s.pixmap().isNull()
    pm = s.grab()
    assert pm.width() > 200
    s.close()
