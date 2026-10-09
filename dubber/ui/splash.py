"""Start-up splash screen: shown by ``main`` right after the QApplication exists, before the heavy UI modules are
imported, so something appears within about a second.  Only Qt and small modules are imported here.

Same layout as Voxprint Audiobook Builder: the square artwork (``assets/splash.jpg``) is about 480 logical pixels
(smaller on a small work area), rendered at the screen's scale; a translucent strip at the bottom carries the name,
version, a thin progress bar and the status line."""
from __future__ import annotations

from PySide6.QtCore import QRect, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QSplashScreen

from dubber.appinfo import APP_DISPLAY_NAME, APP_VERSION, resource_dir
from dubber.i18n import tr

SIDE = 480                                   # logical px; at most 60 % of the shorter side of the work area
STRIP = 92                                   # height of the bottom strip
TEXT, MUTED, BAR = "#f2f2f5", "#d0d0da", "#b98cff"


def _side() -> int:
    screen = QGuiApplication.primaryScreen()
    if screen is None:
        return SIDE
    a = screen.availableGeometry()
    return max(240, min(SIDE, int(min(a.width(), a.height()) * 0.6)))


def _pixmap(side: int) -> QPixmap:
    screen = QGuiApplication.primaryScreen()
    dpr = screen.devicePixelRatio() if screen is not None else 1.0
    art = QPixmap(str(resource_dir() / "assets" / "splash.jpg"))
    if art.isNull():
        art = QPixmap(int(side * dpr), int(side * dpr))
        art.fill(QColor("#070d1c"))
    pm = art.scaled(int(side * dpr), int(side * dpr), Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                    Qt.TransformationMode.SmoothTransformation)
    pm.setDevicePixelRatio(dpr)                # drawn in logical pixels, sharp at 150 %
    return pm


class Splash(QSplashScreen):
    """The splash; :meth:`status` sets the line and the bar and repaints at once (the event loop is not running yet)."""

    def __init__(self) -> None:
        self.side = _side()
        super().__init__(_pixmap(self.side))
        self._text, self._fraction = "", 0.0

    def drawContents(self, p: QPainter) -> None:  # noqa: N802 - Qt naming
        s = self.side
        top = s - STRIP
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(QRect(0, top, s, STRIP), QColor(10, 12, 18, 170))
        f = QFont(self.font())
        f.setPixelSize(19)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor(TEXT))
        p.drawText(QRect(18, top + 10, s - 36, 26), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, APP_DISPLAY_NAME)
        f.setPixelSize(12)
        f.setBold(False)
        p.setFont(f)
        p.setPen(QColor(MUTED))
        p.drawText(QRect(18, top + 54, s - 36, 26), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"v{APP_VERSION}")
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 255, 255, 50))
        p.drawRoundedRect(QRectF(18, top + 44, s - 36, 4), 2, 2)
        p.setBrush(QColor(BAR))
        p.drawRoundedRect(QRectF(18, top + 44, (s - 36) * self._fraction, 4), 2, 2)
        p.setPen(QColor(MUTED))
        p.drawText(QRect(18, top + 54, s - 36, 26), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self._text)

    def message(self) -> str:  # noqa: D102 - the status line (QSplashScreen.message() stays empty: we paint it ourselves)
        return self._text

    def status(self, text: str, fraction: float = -1.0) -> None:
        self._text = text
        if fraction >= 0:
            self._fraction = max(0.0, min(1.0, fraction))
        self.repaint()
        QApplication.processEvents()

    def loading_ui(self) -> None:
        self.status(tr("splash.loading_ui"), 0.3)

    def opening(self) -> None:
        self.status(tr("splash.opening"), 0.8)


def show() -> Splash:
    s = Splash()
    s.show()
    QApplication.processEvents()
    return s
