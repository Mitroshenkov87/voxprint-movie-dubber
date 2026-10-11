"""Movie Dubber splash: the shared window plus the click-burst picture.

``voxprint_common.ui.splash.Splash`` paints the 960 by 540 window. Clicks arrive
through ``on_click``. The burst picture is an overlay on that window. A click
on the picture hides it and leaves the splash up.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from dubber.appinfo import _identity, resource_dir
from dubber.i18n import tr
from dubber.ui.easter_egg import ClickBurst
from voxprint_common.buildinfo import BuildInfo
from voxprint_common.ui.splash import Splash, SplashEvent

IDENTITY = "#b98cff"


class _EggOverlay(QWidget):
    """The hidden picture. A click hides it and leaves the splash up."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        self.art = QLabel()
        self.art.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.caption = QLabel()
        self.caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.caption.setWordWrap(True)
        self.caption.setStyleSheet("color: #f2f2f5; background: rgba(0, 0, 0, 160); padding: 6px;")
        lay.addWidget(self.art, 1)
        lay.addWidget(self.caption)
        self.setStyleSheet("background: rgba(0, 0, 0, 210);")
        self.hide()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Hide the picture. The splash underneath stays open."""
        self.hide()
        event.accept()

    def show_image(self, path, caption: str) -> None:
        """Show ``path`` with ``caption``. A missing file shows the caption only."""
        self.caption.setText(caption)
        pixmap = QPixmap(str(path))
        parent = self.parentWidget()
        width = parent.width() if parent is not None else 960
        height = parent.height() if parent is not None else 540
        if pixmap.isNull():
            self.art.setText(caption)
            self.art.setPixmap(QPixmap())
        else:
            self.art.setText("")
            self.art.setPixmap(pixmap.scaled(max(1, width - 16), max(1, height - 48),
                                              Qt.AspectRatioMode.KeepAspectRatio,
                                              Qt.TransformationMode.SmoothTransformation))
        self.setGeometry(0, 0, width, height)
        self.show()
        self.raise_()


def splash_build() -> BuildInfo:
    """Build identity for the splash. The second line is :func:`voxprint_common.buildinfo.label`."""
    return _identity()


def show_splash() -> Splash:
    """Show the shared splash and count clicks with ``on_click``.

    The returned object is the shared :class:`Splash`. ``_easter`` is the overlay
    widget, so a test can see that a burst did not close the window.
    """
    art = resource_dir() / "assets" / "splash.jpg"
    easter = resource_dir() / "dubber" / "assets" / "splash-easter.jpg"
    burst = ClickBurst()
    overlay_box: dict[str, _EggOverlay] = {}

    def on_click(event: SplashEvent) -> None:
        """Show the picture after six clicks in two seconds. Clicks on the picture do not count."""
        del event
        overlay = overlay_box.get("overlay")
        if overlay is None or overlay.isVisible():
            return
        if burst.click():
            overlay.show_image(easter, tr("splash.easter_egg"))

    splash = Splash(art, IDENTITY, splash_build(), on_click=on_click)
    window = splash._window
    if not isinstance(window, QWidget):
        raise TypeError("the shared splash window is not a QWidget")
    overlay = _EggOverlay(window)
    overlay_box["overlay"] = overlay
    # The shared Splash has no field for the picture. setattr keeps the attribute
    # without a type ignore that mypy flags when the library is not installed.
    setattr(splash, "_easter", overlay)
    return splash
