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
    assert r"..\dubber\*" in iss and "recursesubdirs" in iss
    assert r"..\dubber\assets\icons\*.svg" in iss
    dubber_line = next(line for line in iss.splitlines() if r"..\dubber\*" in line)
    assert "*.svg" not in dubber_line.split("Excludes:", 1)[-1]
    main = (ROOT / "main.py").read_text(encoding="utf-8")
    assert main.index("splash_mod.show()") < main.index("from dubber.ui.window import MainWindow")
    window = (ROOT / "dubber" / "ui" / "window.py").read_text(encoding="utf-8")
    assert "TODO(icons)" not in window


def test_suite_icons_are_tinted_from_current_color():
    pytest.importorskip("PySide6")
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from dubber.ui.icons import ICON_NAMES, icons_dir, load_svg, render_icon

    for name in ICON_NAMES:
        text = (icons_dir() / f"{name}.svg").read_text(encoding="utf-8")
        assert "currentColor" in text
    icon = render_icon(load_svg("save"), normal="#ff0000", disabled="#00aa00", active="#0000ff", size=22)
    assert _has_opaque(icon, QIcon.Mode.Normal, red=True)
    assert _has_opaque(icon, QIcon.Mode.Disabled, green=True)
    assert _has_opaque(icon, QIcon.Mode.Selected, blue=True)


def _has_opaque(icon, mode, *, red=False, green=False, blue=False) -> bool:
    image = icon.pixmap(22, 22, mode).toImage()
    for y in range(image.height()):
        for x in range(image.width()):
            color = image.pixelColor(x, y)
            if color.alpha() < 200:
                continue
            if red and color.red() > 180 and color.green() < 40 and color.blue() < 40:
                return True
            if green and color.green() > 120 and color.red() < 40:
                return True
            if blue and color.blue() > 180 and color.red() < 40 and color.green() < 40:
                return True
    return False


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
