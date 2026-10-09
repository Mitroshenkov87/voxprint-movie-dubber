"""Built-in player for the preview fragment and Watch mode (Qt Multimedia, FFmpeg backend - no VLC needed).

The film's video plays without its own sound; the dubbed sound (Watch chunks or the preview WAV) is pushed into a ``QAudioSink``
in step with the video position.  If the two drift apart by more than ``RESYNC_S`` the sound is restarted at the video position.
In Watch mode a :class:`~dubber.core.watch.WatchState` decides when the film must wait for the dubbing ("buffering").
If Qt Multimedia cannot play the file, ``failed`` is emitted and the window offers the external player instead.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QTimer, QUrl, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget

from dubber.core.watch import WatchState
from dubber.i18n import tr
from dubber.ui.dub_audio import SR

RESYNC_S = 0.12
TICK_MS = 20
BUFFER_S = 0.35

try:                                                  # Qt Multimedia is in PySide6-Addons; it may be missing in a broken install
    from PySide6.QtMultimedia import QAudioFormat, QAudioSink, QMediaDevices, QMediaPlayer
    from PySide6.QtMultimediaWidgets import QVideoWidget
    HAVE_MULTIMEDIA = True
except ImportError:                                   # pragma: no cover - depends on the installation
    HAVE_MULTIMEDIA = False


class Player(QWidget):
    failed = Signal(str)
    buffering = Signal(bool)
    finished = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.source = None                    # ChunkSource | WavSource
        self.watch: Optional[WatchState] = None
        self.end_s: Optional[float] = None
        self._cursor = 0.0                    # film time of the next frame written to the sink
        self._sink = None
        self._io = None
        self._seeking = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        if HAVE_MULTIMEDIA:
            self.video = QVideoWidget()
            self.video.setMinimumHeight(220)
            self.media = QMediaPlayer(self)
            self.media.setVideoOutput(self.video)       # no QAudioOutput: the film's own sound stays silent
            self.media.errorOccurred.connect(lambda _e, msg: self.failed.emit(msg or "playback error"))
            self.media.mediaStatusChanged.connect(self._on_status)
            lay.addWidget(self.video, 1)
        else:
            self.video, self.media = None, None
            lay.addWidget(QLabel(tr("player.unavailable")))
        row = QHBoxLayout()
        self.btn_play = QPushButton()
        self.btn_play.clicked.connect(self.toggle)
        self.btn_stop = QPushButton()
        self.btn_stop.clicked.connect(self.stop)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.sliderReleased.connect(self._seek_from_slider)
        self.lbl_time = QLabel("0:00")
        self.lbl_time.setObjectName("hint")
        self.lbl_state = QLabel()
        self.lbl_state.setObjectName("warn")
        for w in (self.btn_play, self.btn_stop):
            row.addWidget(w)
        row.addWidget(self.slider, 1)
        row.addWidget(self.lbl_time)
        lay.addLayout(row)
        lay.addWidget(self.lbl_state)
        self.timer = QTimer(self)
        self.timer.setInterval(TICK_MS)
        self.timer.timeout.connect(self._tick)
        self.retranslate()

    # ------------------------------------------------------------------ texts
    def retranslate(self) -> None:
        playing = self.media is not None and self.media.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        self.btn_play.setText(tr("player.pause") if playing else tr("player.play"))
        self.btn_stop.setText(tr("player.stop"))

    # ------------------------------------------------------------------ public
    def open(self, film: Path, source, start: float = 0.0, end: Optional[float] = None, watch: Optional[WatchState] = None) -> bool:
        """Load the film and the dub source; ``start``/``end`` limit playback (preview); ``watch`` enables the buffering logic."""
        if self.media is None:
            self.failed.emit(tr("player.unavailable"))
            return False
        self.stop()
        self.source, self.watch, self.end_s = source, watch, end
        self.media.setSource(QUrl.fromLocalFile(str(film)))
        self.media.setPosition(int(start * 1000))
        self._cursor = start
        self._open_sink()
        return True

    def play(self) -> None:
        if self.media is None or self.source is None:
            return
        if self.watch is not None and self.watch.buffering:
            return
        self._restart_sound(self.position())
        self.media.play()
        self.timer.start()
        self.retranslate()

    def pause(self) -> None:
        if self.media is not None:
            self.media.pause()
        if self._sink is not None:
            self._sink.suspend()
        self.retranslate()

    def toggle(self) -> None:
        if self.media is None:
            return
        if self.media.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.pause()
        else:
            self.play()

    def stop(self) -> None:
        self.timer.stop()
        if self.media is not None:
            self.media.stop()
        if self._sink is not None:
            self._sink.stop()
            self._io = None
        if self.watch is not None:
            self.watch.stop()
        self.lbl_state.setText("")
        self.retranslate()

    def position(self) -> float:
        return (self.media.position() / 1000.0) if self.media is not None else 0.0

    def dubbed_until(self, seconds: float, finished: bool = False) -> None:
        """Watch mode: the dubbing moved on (called from the window)."""
        if self.watch is not None:
            self._apply(self.watch.update(self.position(), seconds, finished))

    # ------------------------------------------------------------------ sound
    def _open_sink(self) -> None:
        if self._sink is not None:
            return
        fmt = QAudioFormat()
        fmt.setSampleRate(SR)
        fmt.setChannelCount(2)
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Float)
        dev = QMediaDevices.defaultAudioOutput()
        if dev.isNull():
            self.lbl_state.setText(tr("player.no_audio_device"))
            return
        self._sink = QAudioSink(dev, fmt, self)
        self._sink.setBufferSize(int(SR * 2 * 4 * BUFFER_S))

    def _restart_sound(self, t: float) -> None:
        if self._sink is None:
            return
        self._sink.stop()
        self._cursor = t
        self._io = self._sink.start()

    def _sound_position(self) -> float:
        """Film time that is audible now = written cursor minus what still waits in the sink's buffer."""
        if self._sink is None:
            return self.position()
        queued = (self._sink.bufferSize() - self._sink.bytesFree()) / (SR * 2 * 4)
        return self._cursor - max(0.0, queued)

    def _feed(self) -> None:
        if self._io is None or self._sink is None or self.source is None:
            return
        free = self._sink.bytesFree() // 8
        if free <= 0:
            return
        x = self.source.read(self._cursor, int(free))
        self._io.write(x.tobytes())
        self._cursor += free / SR

    # ------------------------------------------------------------------ clock
    def _tick(self) -> None:
        pos = self.position()
        dur = (self.media.duration() / 1000.0) if self.media is not None else 0.0
        if dur > 0 and not self.slider.isSliderDown():
            self.slider.setValue(int(1000 * pos / dur))
        self.lbl_time.setText(f"{int(pos // 3600)}:{int(pos % 3600 // 60):02d}:{int(pos % 60):02d}")
        if self.end_s is not None and pos >= self.end_s:
            self.stop()
            self.finished.emit()
            return
        if self.watch is not None:
            self._apply(self.watch.update(pos, self.watch.dubbed_until, self.watch.finished))
            if self.watch.buffering:
                return
        if self.media is not None and self.media.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            if abs(self._sound_position() - pos) > RESYNC_S:
                self._restart_sound(pos)
            self._feed()

    def _apply(self, action: str) -> None:
        if action == "pause":
            if self.media is not None:
                self.media.pause()
            if self._sink is not None:
                self._sink.suspend()
            self.lbl_state.setText(tr("player.buffering"))
            self.buffering.emit(True)
        elif action == "play":
            self.lbl_state.setText("")
            self.buffering.emit(False)
            self.play()

    def _seek_from_slider(self) -> None:
        if self.media is None or self.media.duration() <= 0:
            return
        t = self.slider.value() / 1000.0 * self.media.duration() / 1000.0
        if self.watch is not None and not self.watch.finished:
            t = min(t, max(0.0, self.watch.dubbed_until - 5.0))     # never seek into the part that is not dubbed yet
        self.media.setPosition(int(t * 1000))
        self._restart_sound(t)

    def _on_status(self, status) -> None:
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.stop()
            self.finished.emit()
        elif status == QMediaPlayer.MediaStatus.InvalidMedia:
            self.failed.emit(tr("player.invalid"))
