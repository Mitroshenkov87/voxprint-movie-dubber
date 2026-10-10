"""Start-up splash screen: shown by ``main`` right after the QApplication exists, before the heavy UI modules are
imported, so something appears within about a second.  Only Qt and small modules are imported here.

Same layout as Voxprint Audiobook Builder: the square artwork (``assets/splash.jpg``) is about 480 logical pixels
(smaller on a small work area), rendered at the screen's scale; a translucent strip at the bottom carries the name,
version, a thin progress bar and the status line."""
from __future__ import annotations

from PySide6.QtCore import QRect, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QGuiApplication, QMouseEvent, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QSplashScreen, QVBoxLayout, QWidget

from dubber.appinfo import APP_DISPLAY_NAME, resource_dir, version_label
from dubber.i18n import tr
from dubber.ui.easter_egg import SHOW_S, ClickBurst

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


class _EggOverlay(QWidget):
    """The hidden picture.  A click hides it.  It is a child of the splash, so loading keeps going."""

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

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt naming
        self.hide()
        event.accept()

    def show_image(self, path, side: int, caption: str) -> None:
        self.caption.setText(caption)
        pm = QPixmap(str(path))
        if pm.isNull():
            self.art.setText(caption)
            self.art.setPixmap(QPixmap())
        else:
            self.art.setText("")
            self.art.setPixmap(pm.scaled(max(1, side - 16), max(1, side - 48), Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation))
        self.setGeometry(0, 0, side, side)
        self.show()
        self.raise_()


class Splash(QSplashScreen):
    """The splash; :meth:`status` sets the line and the bar and repaints at once (the event loop is not running yet)."""

    def __init__(self) -> None:
        self.side = _side()
        super().__init__(_pixmap(self.side))
        self._text, self._fraction = "", 0.0
        self._burst = ClickBurst()
        self._egg: _EggOverlay | None = None

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
        p.drawText(QRect(18, top + 54, s - 36, 26), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, version_label())
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

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt naming
        """Six clicks in two seconds show the picture once.  A click does not close the splash."""
        if self._egg is not None and self._egg.isVisible():
            self._egg.hide()
            event.accept()
            return
        if self._burst.click():
            self._show_egg()
        event.accept()

    def _show_egg(self) -> None:
        if self._egg is None:
            self._egg = _EggOverlay(self)
        path = resource_dir() / "dubber" / "assets" / "splash-easter.jpg"
        self._egg.show_image(path, self.side, tr("splash.easter_egg"))
        QTimer.singleShot(int(SHOW_S * 1000), self._egg.hide)

    def loading_ui(self) -> None:
        self.status(tr("splash.loading_ui"), 0.3)

    def opening(self) -> None:
        self.status(tr("splash.opening"), 0.8)


def show() -> Splash:
    s = Splash()
    s.show()
    QApplication.processEvents()
    return s
