"""Suite icons for Save, Save As, and model download.

The SVG files stroke with ``currentColor``. Qt does not map that to the widget palette, so each
pixmap is painted in a theme colour: the normal text colour, the disabled colour, and white for
the selected menu row.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

from dubber.appinfo import resource_dir
from dubber.ui.theme import TEXT, TEXT_DISABLED

ICON_NAMES = ("save", "save-as", "save-copy", "export", "download")
_SIZE = 18
_ACTIVE = "#ffffff"


def icons_dir() -> Path:
    """``dubber/assets/icons`` in a checkout and in an installed copy."""
    if getattr(sys, "_MEIPASS", None):
        return resource_dir() / "dubber" / "assets" / "icons"
    return Path(__file__).resolve().parent.parent / "assets" / "icons"


def load_svg(name: str) -> str:
    """Return the SVG source for an icon file name."""
    return (icons_dir() / f"{name}.svg").read_text(encoding="utf-8")


def suite_icon(name: str) -> QIcon:
    """A theme-tinted icon. ``name`` is the file stem (``save``, ``save-as``, ``download``, ...)."""
    return render_icon(load_svg(name))


def render_icon(svg: str, *, normal: str = TEXT, disabled: str = TEXT_DISABLED, active: str = _ACTIVE,
                size: int = _SIZE) -> QIcon:
    """Build an icon from SVG, tinted for the normal, disabled, and selected states."""
    icon = QIcon()
    for mode, color in (
        (QIcon.Mode.Normal, normal),
        (QIcon.Mode.Disabled, disabled),
        (QIcon.Mode.Active, active),
        (QIcon.Mode.Selected, active),
    ):
        for scale in (1.0, 2.0):
            icon.addPixmap(_pixmap(svg, color, size, scale), mode, QIcon.State.Off)
    return icon


def _pixmap(svg: str, color: str, size: int, scale: float) -> QPixmap:
    tinted = svg.replace("currentColor", color)
    renderer = QSvgRenderer(QByteArray(tinted.encode("utf-8")))
    pixels = max(1, int(round(size * scale)))
    pixmap = QPixmap(pixels, pixels)
    pixmap.setDevicePixelRatio(scale)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    try:
        renderer.render(painter)
    finally:
        painter.end()
    return pixmap
