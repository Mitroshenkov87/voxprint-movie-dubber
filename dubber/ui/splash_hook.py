"""Movie Dubber splash: the shared window plus the click-burst picture.

``voxprint_common.ui.splash.Splash`` (0.1.0) paints the 960 by 540 window and
has no click callback. The burst is a Qt event filter on that window, which is
the smallest hook that leaves the shared splash unchanged. A real click is
delivered through ``QApplication``; calling ``mousePressEvent`` directly does
not reach the filter.
"""
from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from dubber.appinfo import app_build, app_codename, resource_dir
from dubber.i18n import tr
from dubber.ui.easter_egg import ClickBurst
from voxprint_common.buildinfo import BuildInfo
from voxprint_common.ui.splash import Splash

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


class _ClickFilter(QObject):
    """Count rapid clicks on the splash window and show the picture once."""

    def __init__(self, window: QWidget, art) -> None:
        super().__init__(window)
        self._burst = ClickBurst()
        self._window = window
        self.overlay = _EggOverlay(window)
        self._art = art

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt naming
        """Record a press on the splash. The picture's own clicks are not counted."""
        if watched is self._window and event.type() == QEvent.Type.MouseButtonPress and not self.overlay.isVisible():
            if self._burst.click():
                self.overlay.show_image(self._art, tr("splash.easter_egg"))
        return False


def splash_build() -> BuildInfo:
    """Build identity for the splash label. The version line is ``voxprint_common.buildinfo.label``."""
    number = app_build()
    build = None if isinstance(number, str) else int(number)
    return BuildInfo(app="movie-dubber", version="1.0.0", rc=None, build=build, codename=app_codename(), show_rc=True)


def show_splash() -> Splash:
    """Show the shared splash and attach the click-burst picture.

    The returned object is the shared :class:`Splash`. ``_easter`` is the overlay
    widget, so a test can see that a burst did not close the window.
    """
    art = resource_dir() / "assets" / "splash.jpg"
    splash = Splash(art, IDENTITY, splash_build())
    window = splash._window
    if not isinstance(window, QWidget):
        raise TypeError("the shared splash window is not a QWidget")
    filt = _ClickFilter(window, resource_dir() / "dubber" / "assets" / "splash-easter.jpg")
    window.installEventFilter(filt)
    # The shared Splash has no fields for the picture. setattr keeps the attributes
    # without a type ignore that mypy flags when the library is not installed.
    setattr(splash, "_easter", filt.overlay)
    setattr(splash, "_click_filter", filt)
    return splash
