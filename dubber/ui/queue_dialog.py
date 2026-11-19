"""The film queue, opened from the More menu.

Add puts films at the end. Remove drops a film that is not running. Cancel
stops the film that is dubbing. The next film does not start on its own after
Cancel.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from dubber.i18n import tr
from dubber.ui.dub_queue import QueueItem
from voxprint_common.ui.theme import ROOT_PLAIN, TEXT, TEXT_MUTED, build_shell_style

_STATUS_KEYS = {
    "queued": "shell.queue.queued",
    "running": "shell.queue.running",
    "done": "shell.queue.done",
    "failed": "shell.queue.failed",
    "cancelled": "shell.queue.cancelled",
}


class QueueHost(Protocol):
    """What the main window does for this dialog."""

    def queue_rows(self) -> tuple[QueueItem, ...]:
        """Return the saved films."""
        ...

    def queue_add(self) -> None:
        """Ask for films and append them."""
        ...

    def queue_remove(self, film: str) -> None:
        """Drop ``film`` when it is not running."""
        ...

    def queue_cancel(self) -> None:
        """Stop the film that is dubbing."""
        ...


class QueueDialog(QDialog):
    """The waiting list. It uses the same dark surface as the main window."""

    def __init__(self, host: QWidget) -> None:
        """Build the dialog.

        Args:
            host: The main window. It owns the queue.
        """
        super().__init__(host.window())
        self._host = cast(QueueHost, host)
        self.setObjectName("queueDialog")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(_style())
        self._paint_dark()
        self.setMinimumWidth(560)
        body = QVBoxLayout(self)
        body.setContentsMargins(20, 16, 20, 16)
        body.setSpacing(12)
        self._title = QLabel()
        self._title.setObjectName("queueTitle")
        body.addWidget(self._title)
        self._empty = QLabel()
        self._empty.setObjectName("queueEmpty")
        self._empty.setWordWrap(True)
        body.addWidget(self._empty)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._rows_host = QWidget()
        self._rows_host.setObjectName("queueRows")
        self._rows = QVBoxLayout(self._rows_host)
        self._rows.setContentsMargins(0, 0, 0, 0)
        self._rows.setSpacing(8)
        scroll.setWidget(self._rows_host)
        self._scroll = scroll
        body.addWidget(scroll, 1)
        buttons = QHBoxLayout()
        self._add = QPushButton()
        self._add.setObjectName("queueAdd")
        self._add.clicked.connect(self._add_clicked)
        self._close = QPushButton()
        self._close.setObjectName("queueClose")
        self._close.clicked.connect(self.close)
        buttons.addWidget(self._add)
        buttons.addStretch(1)
        buttons.addWidget(self._close)
        body.addLayout(buttons)
        self.reload()

    def reload(self) -> None:
        """Show the current films."""
        self.setWindowTitle(tr("shell.queue.title"))
        self._title.setText(tr("shell.queue.title"))
        self._add.setText(tr("shell.queue.add"))
        self._close.setText(tr("catalog.close"))
        self._empty.setText(tr("shell.queue.empty"))
        while self._rows.count():
            item = self._rows.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()
        rows = self._host.queue_rows()
        self._empty.setVisible(not rows)
        self._scroll.setVisible(bool(rows))
        for row in rows:
            self._rows.addWidget(self._row(row))
        self._rows.addStretch(1)
        self.adjustSize()

    def _add_clicked(self) -> None:
        self._host.queue_add()

    def _row(self, item: QueueItem) -> QWidget:
        row = QWidget()
        row.setObjectName("queueRow")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        name = QLabel(Path(item.path).name)
        name.setObjectName("queueName")
        name.setToolTip(item.path)
        status = QLabel(_status_text(item.status))
        status.setObjectName("queueStatus")
        remove = QPushButton(tr("shell.queue.remove"))
        remove.setObjectName("queueRemove")
        remove.setProperty("film", item.path)
        remove.setEnabled(item.status != "running")
        remove.clicked.connect(lambda _checked=False, film=item.path: self._host.queue_remove(film))
        cancel = QPushButton(tr("shell.queue.cancel"))
        cancel.setObjectName("queueCancel")
        cancel.setEnabled(item.status == "running")
        cancel.clicked.connect(self._host.queue_cancel)
        layout.addWidget(name, 1)
        layout.addWidget(status)
        layout.addWidget(remove)
        layout.addWidget(cancel)
        return row

    def _paint_dark(self) -> None:
        colour = QColor(ROOT_PLAIN)
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, colour)
        palette.setColor(QPalette.ColorRole.Base, colour)
        self.setPalette(palette)
        self.setAutoFillBackground(True)


def _status_text(status: str) -> str:
    key = _STATUS_KEYS.get(status, "")
    if not key:
        return status
    return tr(key)


def _style() -> str:
    return build_shell_style(False) + f"""
QDialog#queueDialog {{
    background: {ROOT_PLAIN};
    color: {TEXT};
}}
QDialog#queueDialog QLabel {{
    background: transparent;
    color: {TEXT};
}}
QLabel#queueTitle {{
    font-size: 18px;
    font-weight: 600;
}}
QLabel#queueEmpty, QLabel#queueStatus {{
    color: {TEXT_MUTED};
}}
QWidget#queueRows {{
    background: {ROOT_PLAIN};
}}
"""
