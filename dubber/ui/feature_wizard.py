"""Review wizard: the current film, characters, lines, and player.

The shared audio screen stays the main window. This dialog hosts the
existing four-step window so every control on those steps stays available.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QVBoxLayout, QWidget

from dubber.i18n import tr
from dubber.ui.window import MainWindow


class FeatureWizard(QDialog):
    """Film, characters, lines, and the player, opened from the main screen."""

    def __init__(self, parent: QWidget, studio: MainWindow) -> None:
        """Embed ``studio`` so its steps stay the same widgets.

        Args:
            parent: The main screen.
            studio: The existing four-step window.
        """
        super().__init__(parent)
        self.studio = studio
        studio.setParent(self)
        studio.setWindowFlag(Qt.WindowType.Widget, True)
        studio.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setWindowTitle(tr("shell.review_title"))
        self.resize(1180, 860)
        body = QVBoxLayout(self)
        body.setContentsMargins(0, 0, 0, 0)
        body.addWidget(studio)

    def open_step(self, index: int) -> None:
        """Show step ``index`` and bring the wizard forward.

        Args:
            index: 0 film, 1 characters, 2 lines, 3 player.
        """
        self.studio.go(index)
        self.show()
        self.raise_()
        self.activateWindow()
