"""The main window: one dark glass window with four steps - Film, Characters, Script, Dub.

Same shell as the Voxprint Audiobook Builder: translucent root, Acrylic backdrop on Windows 11, cards with rounded corners,
a gear menu for the UI language.  Heavy work never runs in the GUI thread: the pipeline runs in a QThread that starts one worker
*process* per model stage (``dubber.pipeline.runner``); diagnostics live in their own dialog.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from PySide6.QtCore import QTimer, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QIcon
from PySide6.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel, QMenu, QPushButton, QStackedWidget, QVBoxLayout,
                               QWidget)

from dubber import i18n, paths, platform_win, settings
from dubber.appinfo import APP_DISPLAY_NAME, APP_VERSION, resource_dir
from dubber.core import audio, media, voices
from dubber.core.project import Project, Voice
from dubber.core.watch import WatchState
from dubber.i18n import tr
from dubber.pipeline import runner as R
from dubber.pipeline import stages as S
from dubber.ui import dialogs
from dubber.ui.dub_audio import ChunkSource, WavSource
from dubber.ui.jobs import PipelineThread
from dubber.ui.pages import CharactersPage, DubPage, FilmPage, ScriptPage, fmt_eta, fmt_time
from dubber.ui.player import HAVE_MULTIMEDIA, Player
from dubber.ui.theme import build_style

MIN_W, MIN_H = 720, 520
VIDEO_EXTS = {".mkv", ".mp4", ".avi", ".mov", ".webm", ".m4v", ".ts", ".m2ts"}
SUB_EXTS = {".srt", ".vtt", ".ass", ".ssa"}
PREPARE_UNTIL = "translation"
PREVIEW_S = 60.0
STEPS = ("film", "characters", "script", "dub")


def engine_cfg() -> Dict[str, Any]:
    """Engines from Settings; ``VOXPRINT_ENGINES=mock`` runs the whole flow with the CPU stand-ins (no models needed)."""
    if os.environ.get("VOXPRINT_ENGINES", "").strip().lower() == "mock":
        return dict(S.MOCK_CFG)
    return settings.engine_cfg()


class MainWindow(QWidget):
    def __init__(self, autorun: bool = False, options_hook=None, cfg: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.backdrop = "plain"
        self.project: Optional[Project] = None
        self.info: Optional[media.MediaInfo] = None
        self.job: Optional[PipelineThread] = None
        self.job_kind = ""
        self.watch_state: Optional[WatchState] = None
        self.preview_start: Optional[float] = None
        self.diag_dialog: Optional[dialogs.DiagnosticsDialog] = None
        self._cfg = cfg
        self._options_hook = options_hook
        self.setObjectName("root")
        self.setAcceptDrops(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._build()
        self.setStyleSheet(build_style(False))
        self.retranslate()
        self._update_steps()
        self._fit()
        if autorun:
            QTimer.singleShot(500, lambda: self.open_diagnostics(True))

    # ------------------------------------------------------------------ construction
    def _build(self) -> None:
        body = QVBoxLayout(self)
        body.setContentsMargins(24, 18, 24, 18)
        body.setSpacing(12)
        head = QHBoxLayout()
        self.lbl_title = QLabel(APP_DISPLAY_NAME)
        self.lbl_title.setObjectName("title")
        head.addWidget(self.lbl_title)
        head.addStretch(1)
        self.btn_settings = QPushButton()
        self.btn_settings.clicked.connect(self.open_settings)
        self.btn_diag = QPushButton()
        self.btn_diag.clicked.connect(lambda: self.open_diagnostics(False))
        self.btn_gear = QPushButton("\u2699")
        self.btn_gear.setObjectName("gear")
        self.btn_gear.clicked.connect(self._language_menu)
        for b in (self.btn_settings, self.btn_diag, self.btn_gear):
            head.addWidget(b)
        body.addLayout(head)
        self.lbl_tagline = QLabel()
        self.lbl_tagline.setObjectName("subtitle")
        self.lbl_tagline.setWordWrap(True)
        body.addWidget(self.lbl_tagline)

        bar = QHBoxLayout()
        bar.setSpacing(6)
        self.step_buttons = []
        for i, key in enumerate(STEPS):
            b = QPushButton()
            b.setObjectName("step")
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, n=i: self.go(n))
            self.step_buttons.append(b)
            bar.addWidget(b, 1)
        body.addLayout(bar)

        self.player = Player()
        self.player.failed.connect(self._on_player_failed)
        self.film = FilmPage()
        self.chars = CharactersPage()
        self.script = ScriptPage()
        self.dubp = DubPage(self.player)
        self.stack = QStackedWidget()
        for w in (self.film, self.chars, self.script, self.dubp):
            self.stack.addWidget(w)
        body.addWidget(self.stack, 1)
        self.lbl_footer = QLabel()
        self.lbl_footer.setObjectName("footer")
        body.addWidget(self.lbl_footer)

        self.film.choose_file.connect(self.choose_file)
        self.film.choose_subtitles.connect(self.choose_subtitles)
        self.film.prepare.connect(self.prepare)
        self.film.cmb_target.currentIndexChanged.connect(lambda _i: self._film_changed())
        self.film.cmb_subs.currentIndexChanged.connect(lambda _i: self._film_changed())
        self.chars.changed.connect(self._chars_changed)
        self.chars.find_speakers.connect(lambda: self.run_pipeline("find", PREPARE_UNTIL))
        self.chars.listen.connect(self.listen)
        self.chars.next_step.connect(lambda: self.go(2))
        self.script.next_step.connect(lambda: self.go(3))
        self.dubp.dub.connect(lambda: self.run_pipeline("dub", "mux"))
        self.dubp.cancel.connect(self.cancel_job)
        self.dubp.preview.connect(self.start_preview)
        self.dubp.watch.connect(self.start_watch)
        self.dubp.open_result.connect(self.show_result)
        self.dubp.external.connect(self.open_external)
        self.dubp.volume.connect(self.set_original_volume)
        self._sample_player = None

    def _fit(self) -> None:
        w, h = 1040, 760
        try:
            avail = (self.screen() or QApplication.primaryScreen()).availableGeometry()
            w, h = min(w, int(avail.width() * 0.94)), min(h, int(avail.height() * 0.90))
        except Exception:  # noqa: BLE001 - no screen (tests)
            pass
        self.setMinimumSize(MIN_W, MIN_H)
        self.resize(max(w, MIN_W), max(h, MIN_H))

    # ------------------------------------------------------------------ texts
    def retranslate(self) -> None:
        self.setWindowTitle(APP_DISPLAY_NAME)
        icon = resource_dir() / "assets" / "voxprint-dubber.ico"
        if icon.is_file():
            self.setWindowIcon(QIcon(str(icon)))
        self.lbl_tagline.setText(tr("ui.tagline"))
        self.btn_settings.setText(tr("ui.settings"))
        self.btn_diag.setText(tr("ui.btn_diag"))
        for i, key in enumerate(STEPS):
            self.step_buttons[i].setText(f"{i + 1}. {tr('step.' + key)}")
        for page in (self.film, self.chars, self.script, self.dubp, self.player):
            page.retranslate()
        self.lbl_footer.setText(f"{tr('ui.footer_main')}  ·  {APP_VERSION}")

    def _language_menu(self) -> None:
        menu = QMenu(self)
        for code in i18n.LANGS:
            act = menu.addAction(i18n.LANG_NAMES[code])
            act.setCheckable(True)
            act.setChecked(code == i18n.current())
            act.triggered.connect(lambda _=False, c=code: self.set_language(c))
        menu.exec(self.btn_gear.mapToGlobal(self.btn_gear.rect().bottomLeft()))

    def set_language(self, code: str) -> None:
        i18n.set_language(code)
        self.retranslate()

    # ------------------------------------------------------------------ steps
    def _prepared(self) -> bool:
        return bool(self.project and (self.project.stages.get(PREPARE_UNTIL) or {}).get("done") and self.project.lines)

    def _update_steps(self) -> None:
        busy = self.job is not None and self.job.isRunning()
        ready = self._prepared()
        allowed = [True, ready, ready, ready]
        for i, b in enumerate(self.step_buttons):
            b.setEnabled(allowed[i])
            b.setChecked(self.stack.currentIndex() == i)
        self.film.btn_prepare.setEnabled(bool(self.project) and not busy)
        self.film.btn_file.setEnabled(not busy)
        for w in (self.chars.chk_multi, self.chars.cmb_single, self.chars.btn_find, self.chars.btn_merge):
            w.setEnabled(not busy)
        for c in self.chars.cards.values():
            c.setEnabled(not busy)
        self.script.set_editable(not busy)
        self.dubp.running(busy)
        self.dubp.btn_dub.setEnabled(ready and not busy)
        self.dubp.btn_preview.setEnabled(ready and not busy)

    def go(self, index: int) -> None:
        if index > 0 and not self._prepared():
            index = 0
        self.stack.setCurrentIndex(index)
        self._update_steps()

    def target_lang(self) -> str:
        return self.film.target_lang()

    # ------------------------------------------------------------------ film
    def dragEnterEvent(self, e) -> None:  # noqa: N802
        urls = e.mimeData().urls() if e.mimeData().hasUrls() else []
        if any(Path(u.toLocalFile()).suffix.lower() in VIDEO_EXTS | SUB_EXTS for u in urls):
            e.acceptProposedAction()

    def dropEvent(self, e) -> None:  # noqa: N802
        for u in e.mimeData().urls():
            p = Path(u.toLocalFile())
            if p.suffix.lower() in VIDEO_EXTS:
                self.set_source(p)
            elif p.suffix.lower() in SUB_EXTS and self.project is not None:
                self.film.set_subs_file(str(p))
                self._film_changed()

    def choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, tr("ui.file_dialog"), str(Path.home()), tr("ui.file_filter"))
        if path:
            self.set_source(Path(path))

    def choose_subtitles(self) -> None:
        start = str(self.project.source.parent) if self.project else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(self, tr("film.subs_choose"), start, tr("film.subs_filter"))
        if path:
            self.film.set_subs_file(path)
            self._film_changed()

    def projects_root(self) -> Path:
        custom = str(settings.load().get("projects_dir") or "").strip()
        return Path(custom) if custom else paths.projects_dir()

    def set_source(self, path: Path) -> None:
        """Open a film: read what is inside, open (or resume) its project folder."""
        if self.job is not None and self.job.isRunning():
            return
        self.player.stop()
        path = Path(path)
        try:
            self.info = media.probe(path)
            error = ""
        except Exception as exc:  # noqa: BLE001 - shown on the page
            self.info, error = None, str(exc)
        self.film.set_media(path, self.info, error)
        if self.info is None:
            self.project = None
            self._update_steps()
            return
        folder = Project.folder_for(path, self.projects_root())
        if (folder / "project.json").exists():
            self.project = Project(folder)
        else:
            self.project = Project.create(folder, path, target_lang=self.film.target_lang(),
                                          output_format=settings.load().get("output_format", "same"))
        self.film.load(self.project)
        self.watch_state = WatchState(self.info.duration)
        self._reload_pages()
        self.dubp.lbl_stage.setText("")
        self.dubp.lbl_watch.setText("")
        self.go(0)

    def _film_changed(self) -> None:
        if self.project is None:
            return
        self.film.store(self.project)
        self.project.save()
        s = settings.load()
        self.film.show_subtitle_status(self.project, bool(s.get("subdl_key") or s.get("opensubtitles_key")))

    def prepare(self) -> None:
        if self.project is None:
            self.film.lbl_prepare.setText(tr("ui.start_need_file"))
            return
        self._film_changed()
        self.run_pipeline("prepare", PREPARE_UNTIL)

    # ------------------------------------------------------------------ characters / script
    def _chars_changed(self, what: str) -> None:
        if what in ("multi", "merge"):
            self.script.load(self.project)
        if what == "name":
            self.script.load(self.project)

    def listen(self, sid: str) -> None:
        """Play what a voice will sound like: the library sample, the reference clip, or a few of the speaker's lines."""
        p = self.project
        if p is None:
            return
        v = Voice.from_dict(p.settings.get("single_voice")) if not sid else (p.speaker(sid).voice if p.speaker(sid) else Voice())
        clip: Optional[Path] = None
        if v.kind == "library":
            lv = voices.get_library_voice(v.id)
            clip = lv.preview if lv else None
        else:
            ref = p.speaker(sid).ref_audio if sid and p.speaker(sid) else (p.settings.get("single_ref") or {}).get("audio", "")
            if ref and p.abs(ref).exists():
                clip = p.abs(ref)
            else:
                clip = self._speaker_sample(sid)
        if clip is None:
            self.chars.lbl_multi_status.setText(tr("chars.no_sample"))
            return
        self._play_clip(clip)

    def _speaker_sample(self, sid: str) -> Optional[Path]:
        p = self.project
        src = p.folder / "audio" / "mix44.wav"
        if not src.exists():
            return None
        chosen = voices.pick_reference_lines(p.lines, sid or None, max_s=10.0) or [ln for ln in p.lines if not sid or ln.speaker == sid][:3]
        if not chosen:
            return None
        import numpy as np

        parts, sr = [], 44100
        for ln in chosen:
            x, sr = audio.read_range(src, ln.start, ln.end)
            parts += [x, np.zeros((int(0.3 * sr),) + x.shape[1:], x.dtype)]
        out = p.path("voices", f"sample_{sid or 'single'}.wav")
        audio.write(out, np.concatenate(parts), sr)
        return out

    def _play_clip(self, path: Path) -> None:
        if not HAVE_MULTIMEDIA:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
            return
        from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

        if self._sample_player is None:
            self._sample_player = QMediaPlayer(self)
            self._sample_out = QAudioOutput(self)
            self._sample_player.setAudioOutput(self._sample_out)
        self._sample_player.stop()
        self._sample_player.setSource(QUrl.fromLocalFile(str(path)))
        self._sample_player.play()

    # ------------------------------------------------------------------ pipeline
    def cfg(self) -> Dict[str, Any]:
        return dict(self._cfg) if self._cfg is not None else engine_cfg()

    def run_pipeline(self, kind: str, until: str, preview: Optional[float] = None) -> bool:
        if self.project is None or (self.job is not None and self.job.isRunning()):
            return False
        self.project.save()
        self.job_kind = kind
        self.job = PipelineThread(self.project.folder, self.cfg(), until, preview, PREVIEW_S)
        self.job.stage.connect(self._on_stage)
        self.job.log.connect(self.dubp.log.appendPlainText)
        self.job.progress.connect(self._on_progress)
        self.job.until.connect(self._on_until)
        self.job.done.connect(self._on_done)
        if kind in ("prepare", "find"):
            self.film.progress.setValue(0)
            self.film.progress.show()
            self.film.lbl_prepare.setText(tr("film.preparing"))
            if kind == "find":
                self.chars.lbl_multi_status.setText(tr("chars.finding"))
        else:
            self.dubp.progress.setValue(0)
            self.dubp.lbl_stage.setText(tr("dub.starting"))
            if kind == "dub" and self.watch_state is not None:
                self.watch_state.finished = False
        self.job.start()
        self._update_steps()
        return True

    def cancel_job(self) -> None:
        if self.job is not None:
            self.job.cancel()

    def _on_stage(self, key: str, status: str, msg: str) -> None:
        text = tr("run.stage", stage=tr(f"stage.{key}"), status=tr(f"run.{status}")) + (f" — {msg}" if msg and status != "running" else "")
        self.dubp.log.appendPlainText(text)
        if self.job_kind in ("prepare", "find"):
            self.film.lbl_prepare.setText(text)
            if self.job_kind == "find":
                self.chars.lbl_multi_status.setText(text)
        else:
            self.dubp.lbl_stage.setText(text)

    def _on_progress(self, frac: float, eta: float) -> None:
        bar = self.film.progress if self.job_kind in ("prepare", "find") else self.dubp.progress
        bar.setValue(int(frac * 1000))
        if self.job_kind == "dub":
            self.dubp.lbl_eta.setText(fmt_eta(eta))

    def _on_until(self, seconds: float) -> None:
        if self.job_kind != "dub" or self.watch_state is None:
            return
        total = self.watch_state.total
        finished = seconds >= total - 0.5
        self.watch_state.update(self.player.position(), seconds, finished)
        self.player.dubbed_until(seconds, finished)
        self.dubp.btn_watch.setEnabled(self.watch_state.ready)
        self.dubp.lbl_watch.setText(tr("dub.dubbed_until", t=fmt_time(seconds), total=fmt_time(total))
                                    + ("" if self.watch_state.ready else "  " + tr("dub.watch_wait")))

    def _on_done(self, ok: bool, message: str) -> None:
        kind = self.job_kind
        folder = self.job.result_folder if self.job else None
        self.job_kind = ""
        if self.project is not None:
            self.project = Project(self.project.folder)
        self._reload_pages()
        if kind in ("prepare", "find"):
            self.film.progress.hide()
            self.film.lbl_prepare.setText(tr("film.prepared", n=len(self.project.lines)) if ok else tr("run.failed", error=message))
            if ok and kind == "prepare":
                self.go(1)
        elif kind == "dub":
            self.dubp.lbl_stage.setText(tr("dub.done", path=message) if ok else tr("run.failed", error=message))
            if ok and self.watch_state is not None:
                self.watch_state.finished = True
                self.player.dubbed_until(self.watch_state.total, True)
                self.dubp.btn_watch.setEnabled(True)
                self.dubp.lbl_eta.setText("")
        elif kind == "preview":
            if ok and folder is not None:
                self.dubp.lbl_stage.setText(tr("dub.preview_ready"))
                self._play_preview(folder)
            else:
                self.dubp.lbl_stage.setText(tr("run.failed", error=message))
        self._update_steps()

    def _reload_pages(self) -> None:
        if self.project is None:
            return
        self.chars.load(self.project)
        self.script.load(self.project)
        self.dubp.load(self.project)
        s = settings.load()
        self.film.show_subtitle_status(self.project, bool(s.get("subdl_key") or s.get("opensubtitles_key")))

    def set_original_volume(self, value: float) -> None:
        if self.project is not None:
            self.project.settings["original_volume"] = round(value, 2)
            self.project.save()

    # ------------------------------------------------------------------ preview / watch / result
    def start_preview(self) -> None:
        if self.project is None:
            return
        self.player.stop()
        self.preview_start = R.best_preview_start(self.project, PREVIEW_S)
        self.run_pipeline("preview", "mix", preview=self.preview_start)

    def _play_preview(self, folder: Path) -> None:
        wav = Path(folder) / "out" / "dub_track.wav"
        if not wav.exists() or self.project is None:
            return
        start = float(self.preview_start or 0.0)
        if self.player.open(self.project.source, WavSource(wav, start), start, start + PREVIEW_S):
            self.player.play()
        self.dubp.btn_external.setEnabled(True)

    def watch_source(self):
        """Finished dub -> the whole dub track; while dubbing -> the Watch chunks."""
        p = self.project
        track = p.folder / "out" / "dub_track.wav"
        if (p.stages.get("mix") or {}).get("done") and track.exists():
            return WavSource(track, 0.0)
        return ChunkSource(p.folder / "watch")

    def start_watch(self) -> None:
        if self.project is None or self.watch_state is None or not self.watch_state.ready:
            return
        if self.player.open(self.project.source, self.watch_source(), 0.0, None, self.watch_state):
            self.watch_state.start()
            self.player.play()

    def show_result(self) -> None:
        out = (self.project.settings.get("output_file") or "") if self.project else ""
        if out and Path(out).exists():
            dialogs.show_in_folder(Path(out))

    def open_external(self) -> None:
        """External player fallback: the finished file, else a short clip of the preview fragment."""
        if self.project is None:
            return
        out = self.project.settings.get("output_file") or ""
        if out and Path(out).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(out))
            return
        wav = self.project.folder / "preview" / "out" / "dub_track.wav"
        if wav.exists() and self.preview_start is not None:
            clip = self.project.folder / "preview" / "out" / "preview.mp4"
            try:
                media.preview_clip(self.project.source, self.preview_start, PREVIEW_S, wav, clip)
            except Exception as exc:  # noqa: BLE001
                self.dubp.lbl_stage.setText(tr("run.failed", error=str(exc)))
                return
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(clip)))

    def _on_player_failed(self, msg: str) -> None:
        self.dubp.lbl_watch.setText(tr("player.failed", error=msg))
        self.dubp.btn_external.setEnabled(True)

    # ------------------------------------------------------------------ settings / diagnostics
    def open_settings(self) -> None:
        dlg = dialogs.SettingsDialog(self)
        dlg.setStyleSheet(build_style(False))
        if dlg.exec() and self.project is not None:
            self._film_changed()

    def open_diagnostics(self, start: bool = False) -> None:
        if self.diag_dialog is None:
            self.diag_dialog = dialogs.DiagnosticsDialog(self, self.target_lang, self._options_hook)
            self.diag_dialog.setStyleSheet(build_style(False))
        self.diag_dialog.show()
        self.diag_dialog.raise_()
        if start and not self.diag_dialog.running():
            self.diag_dialog.start()

    # ------------------------------------------------------------------ window effects
    def showEvent(self, e) -> None:  # noqa: N802
        """Acrylic backdrop once the native window exists (Windows 11), else the plain dark look."""
        super().showEvent(e)
        if sys.platform == "win32" and self.backdrop == "plain":
            self.backdrop = platform_win.apply_backdrop(int(self.winId()))
            self.setStyleSheet(build_style(self.backdrop == "acrylic"))
            if self.backdrop != "acrylic":
                self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)

    def closeEvent(self, e) -> None:  # noqa: N802
        self.player.stop()
        if self.job is not None and self.job.isRunning():
            self.job.cancel()
            self.job.wait(30000)
        if self.diag_dialog is not None:
            self.diag_dialog.close()
        super().closeEvent(e)
