"""Custom voices for the people in the film.

This is a plain dialog, not Expert mode. The shared shell has no slot for a
panel under the Voices chips. Each row is a library voice and Listen. Likeness
and the other tuning stay in Expert mode.
"""
from __future__ import annotations

from typing import Protocol, cast

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from dubber.core import voices
from dubber.core.project import Voice
from dubber.i18n import tr
from dubber.ui.pages import fill_voice_combo, voice_key
from voxprint_common.ui.theme import build_shell_style


class CastHost(Protocol):
    """What the main window does for this dialog."""

    def cast_prepared(self) -> bool:
        """Return whether Prepare has finished and the film has lines."""
        ...

    def cast_characters(self) -> list[tuple[str, str, str]]:
        """Return ``(id, name, voice key)`` for each character."""
        ...

    def cast_prepare(self) -> None:
        """Start Prepare."""
        ...

    def cast_listen(self, sid: str) -> None:
        """Play a sample of this character's voice."""
        ...

    def cast_assign(self, sid: str, key: str) -> None:
        """Store the library voice ``key`` on this character."""
        ...


class VoiceCastDialog(QDialog):
    """Character voices. Prepare comes first. Expert mode stays closed."""

    def __init__(self, host: QWidget) -> None:
        """Build the dialog.

        Args:
            host: The main window. It supplies the characters and saves voices.
        """
        super().__init__(host)
        self._host = cast(CastHost, host)
        self.setObjectName("voiceCast")
        self.setWindowTitle(tr("shell.voice.custom"))
        self.setStyleSheet(build_shell_style(False))
        self.setMinimumWidth(560)
        body = QVBoxLayout(self)
        body.setContentsMargins(16, 16, 16, 16)
        body.setSpacing(12)
        self._wait = QLabel(tr("shell.voice.cast_wait"))
        self._wait.setObjectName("voiceCastWait")
        self._wait.setWordWrap(True)
        self._prepare = QPushButton(tr("shell.voice.prepare"))
        self._prepare.setObjectName("voiceCastPrepare")
        self._prepare.clicked.connect(self._prepare_clicked)
        body.addWidget(self._wait)
        body.addWidget(self._prepare, 0)
        scroll = QScrollArea()
        scroll.setObjectName("voiceCastScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._host_rows = QWidget()
        self._rows = QVBoxLayout(self._host_rows)
        self._rows.setContentsMargins(0, 0, 0, 0)
        self._rows.setSpacing(8)
        scroll.setWidget(self._host_rows)
        body.addWidget(scroll, 1)
        self._scroll = scroll
        self.reload()

    def reload(self) -> None:
        """Show the prepare line, or one row per character."""
        self.setWindowTitle(tr("shell.voice.custom"))
        self._wait.setText(tr("shell.voice.cast_wait"))
        self._prepare.setText(tr("shell.voice.prepare"))
        self._clear_rows()
        ready = self._host.cast_prepared()
        self._wait.setVisible(not ready)
        self._prepare.setVisible(not ready)
        self._scroll.setVisible(ready)
        if not ready:
            return
        items = _library_items()
        for sid, name, current in self._host.cast_characters():
            self._rows.addLayout(self._row(sid, name, current, items))
        self._rows.addStretch(1)

    def _prepare_clicked(self) -> None:
        self._host.cast_prepare()

    def _row(self, sid: str, name: str, current: str, items: list[tuple[Voice, str]]) -> QHBoxLayout:
        row = QHBoxLayout()
        label = QLabel(name)
        label.setObjectName(f"voiceName-{sid}")
        label.setMinimumWidth(120)
        picker = QComboBox()
        picker.setObjectName(f"voicePick-{sid}")
        if items:
            fill_voice_combo(picker, _voice_from_key(current), items)
            picker.currentIndexChanged.connect(
                lambda _index, speaker=sid, box=picker: self._assigned(speaker, box)
            )
        else:
            picker.addItem(tr("shell.voice.none"))
            picker.setEnabled(False)
        listen = QPushButton(tr("chars.listen"))
        listen.setObjectName(f"voiceListen-{sid}")
        listen.clicked.connect(lambda _checked=False, speaker=sid: self._host.cast_listen(speaker))
        row.addWidget(label)
        row.addWidget(picker, 1)
        row.addWidget(listen)
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
            layout = item.layout()
            if layout is not None:
                while layout.count():
                    child = layout.takeAt(0)
                    if child is None:
                        continue
                    widget = child.widget()
                    if widget is not None:
                        widget.deleteLater()
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()


def _library_items() -> list[tuple[Voice, str]]:
    """Return the installed library voices as combo entries."""
    items: list[tuple[Voice, str]] = []
    for voice in voices.list_library():
        extra = ", ".join(part for part in (voice.language, voice.gender) if part)
        label = tr("voice.library", name=voice.name)
        if extra:
            label = f"{label} ({extra})"
        items.append((Voice("library", voice.id), label))
    return items


def _voice_from_key(key: str) -> Voice:
    """Return a voice from a combo value, or a blank library voice."""
    if not key or ":" not in key:
        return Voice("library", "")
    kind, _, voice_id = key.partition(":")
    if kind != "library":
        return Voice("library", "")
    return Voice("library", voice_id)


def current_voice_key(voice: Voice) -> str:
    """Return the combo value for ``voice`` when it is a library voice."""
    if voice.kind != "library" or not voice.id:
        return ""
    return voice_key(voice)
