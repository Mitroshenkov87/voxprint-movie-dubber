"""Custom voices for the people in the film.

This is a plain dialog, not Expert mode. The shared shell has no slot for a
panel under the Voices chips. Each row is a library voice and Listen. Likeness
and the other tuning stay in Expert mode.
"""
from __future__ import annotations

from typing import Protocol, cast

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from dubber.core import voices
from dubber.core.project import Voice
from dubber.i18n import tr
from dubber.ui.pages import fill_voice_combo, voice_key
from dubber.ui.voice_label import library_voice_label
from voxprint_common.ui.theme import (
    CONTROL_BORDER,
    PRIMARY,
    PRIMARY_BORDER,
    ROOT_PLAIN,
    TEXT,
    TEXT_MUTED,
    TEXT_ON_PRIMARY,
    build_shell_style,
)


class CastHost(Protocol):
    """What the main window does for this dialog."""

    def cast_prepared(self) -> bool:
        """Return whether Prepare has finished and the film has lines."""
        ...

    def cast_characters(self) -> list[tuple[str, str, str, str]]:
        """Return ``(id, Voice N, voice key, speech detail)`` for each speaker."""
        ...

    def cast_prepare(self) -> None:
        """Start Prepare."""
        ...

    def cast_play_excerpt(self, sid: str) -> None:
        """Play this speaker's short clip from the film."""
        ...

    def cast_play_library(self, sid: str) -> None:
        """Play the library voice chosen for this speaker."""
        ...

    def cast_assign(self, sid: str, key: str) -> None:
        """Store the library voice ``key`` on this character."""
        ...


class VoiceCastDialog(QDialog):
    """Character voices on the shell's dark theme. Expert mode stays closed."""

    def __init__(self, host: QWidget) -> None:
        """Build the dialog.

        Args:
            host: The main window. It supplies the characters and saves voices.
        """
        super().__init__(host)
        self._host = cast(CastHost, host)
        self._original: list[tuple[str, str, str, str]] = []
        self.setObjectName("voiceCast")
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setStyleSheet(_style())
        self._paint_dark()
        self.setMinimumWidth(680)
        body = QVBoxLayout(self)
        body.setContentsMargins(24, 20, 24, 20)
        body.setSpacing(14)
        self._title = QLabel()
        self._title.setObjectName("voiceCastTitle")
        self._hint = QLabel()
        self._hint.setObjectName("voiceCastHint")
        self._hint.setWordWrap(True)
        body.addWidget(self._title)
        body.addWidget(self._hint)
        self._prepare = QPushButton()
        self._prepare.setObjectName("voiceCastPrepare")
        self._prepare.setAutoDefault(False)
        self._prepare.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._prepare.clicked.connect(self._prepare_clicked)
        prepare_row = QWidget()
        prepare_row.setObjectName("voiceCastPrepareRow")
        prepare_layout = QHBoxLayout(prepare_row)
        prepare_layout.setContentsMargins(0, 0, 0, 0)
        prepare_layout.addWidget(self._prepare)
        prepare_layout.addStretch(1)
        self._prepare_row = prepare_row
        body.addWidget(prepare_row)
        scroll = QScrollArea()
        scroll.setObjectName("voiceCastScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._host_rows = QWidget()
        self._host_rows.setObjectName("voiceCastRows")
        self._rows = QVBoxLayout(self._host_rows)
        self._rows.setContentsMargins(0, 0, 0, 0)
        self._rows.setSpacing(10)
        scroll.setWidget(self._host_rows)
        self._paint_dark(scroll.viewport())
        body.addWidget(scroll, 1)
        self._scroll = scroll
        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        buttons.addStretch(1)
        self._cancel = QPushButton()
        self._cancel.setObjectName("voiceCastCancel")
        self._cancel.setAutoDefault(False)
        self._cancel.clicked.connect(self._cancel_clicked)
        self._done = QPushButton()
        self._done.setObjectName("voiceCastDone")
        self._done.setAutoDefault(False)
        self._done.clicked.connect(self.accept)
        buttons.addWidget(self._cancel)
        buttons.addWidget(self._done)
        body.addLayout(buttons)
        self.reload()

    def reload(self) -> None:
        """Show the prepare line, or one row per character."""
        self.setWindowTitle(tr("shell.voice.cast_title"))
        self._title.setText(tr("shell.voice.cast_title"))
        self._cancel.setText(tr("shell.voice.cancel"))
        self._done.setText(tr("shell.voice.done"))
        self._prepare.setText(tr("shell.voice.prepare"))
        self._clear_rows()
        ready = self._host.cast_prepared()
        self._original = self._host.cast_characters() if ready else []
        self._hint.setText(tr("shell.voice.cast_hint") if ready else tr("shell.voice.cast_wait"))
        self._prepare_row.setVisible(not ready)
        self._scroll.setVisible(ready)
        self.setMinimumWidth(920 if ready else 640)
        if not ready:
            self.adjustSize()
            return
        items = _library_items()
        for sid, label, chosen, detail in self._original:
            self._rows.addWidget(self._row(sid, label, chosen, detail, items))
        self._scroll.setMinimumHeight(min(440, max(56, 64 * len(self._original))))
        self._rows.addStretch(1)
        self.adjustSize()

    def place_on_parent(self) -> None:
        """Centre the dialog on the main window."""
        parent = self.parentWidget()
        if parent is None:
            return
        host = parent.window()
        self.adjustSize()
        width = max(self.minimumWidth(), self.sizeHint().width())
        height = max(self.sizeHint().height(), 220 if not self._scroll.isVisible() else 360)
        width = min(width, max(320, host.width() - 80))
        height = min(height, max(200, host.height() - 80))
        self.resize(width, height)
        origin = host.mapToGlobal(host.rect().topLeft())
        self.move(
            origin.x() + max(0, (host.width() - self.width()) // 2),
            origin.y() + max(0, (host.height() - self.height()) // 2),
        )

    def _prepare_clicked(self) -> None:
        self._host.cast_prepare()

    def _cancel_clicked(self) -> None:
        for sid, _label, key, _detail in self._original:
            self._host.cast_assign(sid, key or "clone:")
        self.reject()

    def _row(
        self,
        sid: str,
        label: str,
        current: str,
        detail: str,
        items: list[tuple[Voice, str]],
    ) -> QWidget:
        row = QWidget()
        row.setObjectName(f"voiceRow-{sid}")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        names_host = QWidget()
        names_host.setFixedWidth(136)
        names = QVBoxLayout(names_host)
        names.setContentsMargins(0, 0, 0, 0)
        names.setSpacing(0)
        title = QLabel(label)
        title.setObjectName(f"voiceName-{sid}")
        meta = QLabel(detail)
        meta.setObjectName(f"voiceMeta-{sid}")
        meta.setProperty("voiceMeta", "true")
        names.addWidget(title)
        names.addWidget(meta)
        excerpt = QPushButton(tr("shell.voice.play_excerpt"))
        excerpt.setObjectName(f"voiceExcerpt-{sid}")
        excerpt.setAutoDefault(False)
        excerpt.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        excerpt.clicked.connect(lambda _checked=False, speaker=sid: self._host.cast_play_excerpt(speaker))
        picker = QComboBox()
        picker.setObjectName(f"voicePick-{sid}")
        picker.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        picker.setMinimumContentsLength(16)
        picker.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        if items:
            fill_voice_combo(picker, _voice_from_key(current), items)
            picker.currentIndexChanged.connect(
                lambda _index, speaker=sid, box=picker: self._assigned(speaker, box)
            )
        else:
            picker.addItem(tr("shell.voice.none"))
            picker.setEnabled(False)
        preview = QPushButton(tr("shell.voice.play_library"))
        preview.setObjectName(f"voiceLibrary-{sid}")
        preview.setAutoDefault(False)
        preview.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        preview.clicked.connect(lambda _checked=False, speaker=sid: self._host.cast_play_library(speaker))
        layout.addWidget(names_host)
        layout.addWidget(excerpt)
        layout.addWidget(picker, 1)
        layout.addWidget(preview)
        return row

    def _assigned(self, sid: str, picker: QComboBox) -> None:
        data = picker.currentData()
        if not isinstance(data, str) or not data:
            return
        self._host.cast_assign(sid, data)

    def _clear_rows(self) -> None:
        while self._rows.count():
            item = self._rows.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()
                continue
            layout = item.layout()
            if layout is None:
                continue
            while layout.count():
                child = layout.takeAt(0)
                if child is None:
                    continue
                child_widget = child.widget()
                if child_widget is not None:
                    child_widget.hide()
                    child_widget.setParent(None)
                    child_widget.deleteLater()

    def _paint_dark(self, widget: QWidget | None = None) -> None:
        target = self if widget is None else widget
        colour = QColor(ROOT_PLAIN)
        palette = target.palette()
        palette.setColor(QPalette.ColorRole.Window, colour)
        palette.setColor(QPalette.ColorRole.Base, colour)
        target.setPalette(palette)
        target.setAutoFillBackground(True)


def current_voice_key(voice: Voice) -> str:
    """Return the combo value for ``voice`` when it is a library voice."""
    if voice.kind != "library" or not voice.id:
        return ""
    return voice_key(voice)


def _style() -> str:
    return build_shell_style(False) + f"""
QDialog#voiceCast {{
    background: {ROOT_PLAIN};
    color: {TEXT};
    border: 1px solid {CONTROL_BORDER};
    border-radius: 12px;
}}
QDialog#voiceCast QLabel {{
    background: transparent;
    color: {TEXT};
}}
QLabel#voiceCastTitle {{
    font-size: 18px;
    font-weight: 600;
    background: transparent;
}}
QLabel#voiceCastHint, QLabel[voiceMeta="true"] {{
    color: {TEXT_MUTED};
    background: transparent;
}}
QLabel[voiceMeta="true"] {{
    font-size: 11px;
}}
QWidget#voiceCastRows, QWidget#voiceCastPrepareRow, QScrollArea#voiceCastScroll {{
    background: {ROOT_PLAIN};
    border: none;
}}
QScrollArea#voiceCastScroll > QWidget > QWidget {{
    background: {ROOT_PLAIN};
}}
QPushButton#voiceCastDone {{
    background: {PRIMARY};
    color: {TEXT_ON_PRIMARY};
    border: 1px solid {PRIMARY_BORDER};
    font-weight: 600;
}}
"""


def _library_items() -> list[tuple[Voice, str]]:
    """Return the installed library voices as combo entries."""
    return [
        (Voice("library", voice.id), library_voice_label(voice))
        for voice in voices.list_library()
    ]


def _voice_from_key(key: str) -> Voice:
    """Return a voice from a combo value, or a blank library voice."""
    if not key or ":" not in key:
        return Voice("library", "")
    kind, _, voice_id = key.partition(":")
    if kind != "library":
        return Voice("library", "")
    return Voice("library", voice_id)
