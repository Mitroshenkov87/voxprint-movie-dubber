"""The main window: one dark glass window with four steps - Film, Characters, Lines, Dub.

Film is all most people need: drop a movie, press Dub, get ``<name>.dub-<lang>.mkv`` next to it.  Characters and
Lines are optional review tabs; one voice still shows its voice and likeness. After the analysis they get a badge only when something deserves a look.

Same shell as the Voxprint Audiobook Builder: translucent root, Acrylic backdrop on Windows 11, cards with rounded corners,
a gear menu for the UI language.  Heavy work never runs in the GUI thread: the pipeline runs in a QThread that starts one worker
process for the model stages (``dubber.pipeline.runner``); diagnostics live in their own dialog.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from PySide6.QtCore import QTimer, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QIcon
from PySide6.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel, QMenu, QMessageBox, QPushButton, QStackedWidget,
                               QVBoxLayout, QWidget)

from dubber import i18n, paths, platform_win, settings
from dubber.appinfo import APP_DISPLAY_NAME, resource_dir, version_label
from dubber.core import audio, media, review, voices, vxdub
from dubber.core.project import Project, Voice
from dubber.core.watch import WatchState
from dubber.i18n import tr
from dubber.pipeline import runner as R
from dubber.pipeline import stages as S
from dubber.ui import dialogs
from dubber.ui.dub_audio import ChunkSource, WavSource
from dubber.ui.icons import suite_icon
from dubber.ui.jobs import PipelineThread
from dubber.ui.pages import TARGET_LANGS, CharactersPage, DubPage, FilmPage, LinesPage, fmt_eta, fmt_time
from dubber.ui.player import HAVE_MULTIMEDIA, Player
from dubber.ui.theme import build_style

MIN_W, MIN_H = 720, 520
VIDEO_EXTS = {".mkv", ".mp4", ".avi", ".mov", ".webm", ".m4v", ".ts", ".m2ts"}
PROJECT_EXT = vxdub.EXTENSION
SUB_EXTS = {".srt", ".vtt", ".ass", ".ssa"}
PREPARE_UNTIL = "translation"
PREVIEW_S = 60.0
STEPS = ("film", "characters", "lines", "dub")
FILM_KINDS = ("prepare", "find", "auto")          # jobs whose progress shows on the Film screen


def engine_cfg() -> Dict[str, Any]:
    """Engines from Settings; ``VOXPRINT_ENGINES=mock`` runs the whole flow with the CPU stand-ins (no models needed)."""
    if os.environ.get("VOXPRINT_ENGINES", "").strip().lower() == "mock":
        return dict(S.MOCK_CFG)
    return settings.engine_cfg()


class MainWindow(QWidget):
    """Main window for choosing a film, reviewing the dub, and running the pipeline. ``autorun`` opens diagnostics and starts them."""

    def __init__(self, autorun: bool = False, options_hook: Optional[Callable[..., None]] = None,
                 cfg: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.backdrop = "plain"
        self.project: Optional[Project] = None
        self.info: Optional[media.MediaInfo] = None
        self.job: Optional[PipelineThread] = None
        self.job_kind = ""
        self.watch_state: Optional[WatchState] = None
        self.preview_start: Optional[float] = None
        self.diag_dialog: Optional[dialogs.DiagnosticsDialog] = None
        self.vxdub_path: Optional[Path] = None
        self.vxdub_doc: Optional[vxdub.Document] = None
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
        self.btn_file = QPushButton()
        self.btn_file.clicked.connect(self._file_menu)
        self.file_menu = QMenu(self)
        self.act_open = self.file_menu.addAction("")
        self.act_save = self.file_menu.addAction(suite_icon("save"), "")
        self.act_save_as = self.file_menu.addAction(suite_icon("save-as"), "")
        self.act_open.triggered.connect(lambda _checked=False: self.choose_vxdub())
        self.act_save.triggered.connect(lambda _checked=False: self.save_vxdub())
        self.act_save_as.triggered.connect(lambda _checked=False: self.save_vxdub_as())
        self.btn_settings = QPushButton()
        self.btn_settings.clicked.connect(self.open_settings)
        self.btn_diag = QPushButton()
        self.btn_diag.clicked.connect(lambda: self.open_diagnostics(False))
        self.btn_gear = QPushButton("\u2699")
        self.btn_gear.setObjectName("gear")
        self.btn_gear.clicked.connect(self._language_menu)
        for b in (self.btn_file, self.btn_settings, self.btn_diag, self.btn_gear):
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
        self.lines = LinesPage()
        self.dubp = DubPage(self.player)
        self.stack = QStackedWidget()
        for w in (self.film, self.chars, self.lines, self.dubp):
            self.stack.addWidget(w)
        body.addWidget(self.stack, 1)
        self.lbl_footer = QLabel()
        self.lbl_footer.setObjectName("footer")
        body.addWidget(self.lbl_footer)

        self.film.choose_file.connect(self.choose_file)
        self.film.choose_subtitles.connect(self.choose_subtitles)
        self.film.prepare.connect(self.prepare)
        self.film.dub.connect(self.dub_all)
        self.film.cancel.connect(self.cancel_job)
        self.film.show_result.connect(self.show_result)
        def _on_watch() -> None:
            self.go(3)
            self.start_watch()

        self.film.watch.connect(_on_watch)
        self.film.option_changed.connect(self._option_changed)
        self.film.options_toggled.connect(lambda on: settings.save({"options_open": bool(on)}))
        self.film.set_options_open(bool(settings.load().get("options_open")))
        self.chars.changed.connect(self._chars_changed)
        self.chars.find_speakers.connect(lambda: self.run_pipeline("find", PREPARE_UNTIL))
        self.chars.listen.connect(self.listen)
        self.chars.open_catalog.connect(self.open_voice_catalog)
        self.chars.save_actor.connect(self.save_actor_voice)
        self.chars.next_step.connect(lambda: self.go(2))
        self.lines.next_step.connect(lambda: self.go(3))
        self.dubp.dub.connect(lambda: self.run_pipeline("dub", "mux"))
        self.dubp.cancel.connect(self.cancel_job)
        self.dubp.preview.connect(self.start_preview)
        self.dubp.watch.connect(self.start_watch)
        self.dubp.open_result.connect(self.show_result)
        self.dubp.external.connect(self.open_external)
        self.dubp.volume.connect(self.set_original_volume)
        self._sample_player: Any = None

    def _fit(self) -> None:
        w, h = 1180, 900
        try:
            avail = (self.screen() or QApplication.primaryScreen()).availableGeometry()
            w, h = min(w, int(avail.width() * 0.94)), min(h, int(avail.height() * 0.90))
        except Exception:  # noqa: BLE001 - no screen (tests)
            pass
        self.setMinimumSize(MIN_W, MIN_H)
        self.resize(max(w, MIN_W), max(h, MIN_H))

    # ------------------------------------------------------------------ texts
    def retranslate(self) -> None:
        """Refresh the window and every page for the current language."""
        self.setWindowTitle(APP_DISPLAY_NAME)
        icon = resource_dir() / "assets" / "voxprint-dubber.ico"
        if icon.is_file():
            self.setWindowIcon(QIcon(str(icon)))
        self.lbl_tagline.setText(tr("ui.tagline"))
        self.btn_file.setText(tr("vxdub.file"))
        self.act_open.setText(tr("vxdub.open"))
        self.act_save.setText(tr("vxdub.save"))
        self.act_save_as.setText(tr("vxdub.save_as"))
        self.btn_settings.setText(tr("ui.settings"))
        self.btn_diag.setText(tr("ui.btn_diag"))
        for i, key in enumerate(STEPS):
            self.step_buttons[i].setText(f"{i + 1}. {tr('step.' + key)}")
        for page in (self.film, self.chars, self.lines, self.dubp, self.player):
            page.retranslate()
        self._update_steps()
        self._refresh_plan()
        self.lbl_footer.setText(f"{tr('ui.footer_main')}  ·  {version_label()}")

    def _language_menu(self) -> None:
        menu = QMenu(self)
        for code in i18n.LANGS:
            act = menu.addAction(i18n.LANG_NAMES[code])
            act.setCheckable(True)
            act.setChecked(code == i18n.current())
            act.triggered.connect(lambda _=False, c=code: self.set_language(c))
        menu.exec(self.btn_gear.mapToGlobal(self.btn_gear.rect().bottomLeft()))

    def set_language(self, code: str) -> None:
        """Switch the interface language and refresh the window."""
        i18n.set_language(code)
        self.retranslate()

    # ------------------------------------------------------------------ steps
    def _prepared(self) -> bool:
        return bool(self.project and (self.project.stages.get(PREPARE_UNTIL) or {}).get("done") and self.project.lines)

    def _multi(self) -> bool:
        return bool(self.project and self.project.settings.get("multi_voice"))

    def _update_steps(self) -> None:
        busy = self.job is not None and self.job.isRunning()
        ready = self._prepared()
        allowed = [True, ready, ready, ready]
        project = self.project
        hints = review.attention(project) if project is not None and ready and not busy else {"characters": [], "lines": []}
        if not self._multi():
            hints["characters"] = []
        for i, b in enumerate(self.step_buttons):
            key = STEPS[i]
            items = [tr(k, **kw) for k, kw in hints.get(key, [])]
            b.setEnabled(allowed[i])
            b.setChecked(self.stack.currentIndex() == i)
            b.setText(f"{i + 1}. {tr('step.' + key)}" + ("  \u25cf" if items else ""))
            b.setProperty("attention", "true" if items else "false")
            b.setToolTip("\n".join(items) if items else "")
            b.style().unpolish(b)
            b.style().polish(b)
        self.film.set_attention([f"{tr('step.' + k)} - {tr(m, **kw)}" for k in ("characters", "lines") for m, kw in hints[k]])
        self.film.running(busy)
        self.film.btn_dub.setEnabled(bool(self.project) and not busy)
        self.film.btn_prepare.setEnabled(bool(self.project) and not busy)
        for w in (self.chars.btn_find, self.chars.btn_merge):
            w.setEnabled(not busy)
        for c in self.chars.cards.values():
            c.setEnabled(not busy)
        self.lines.set_editable(not busy)
        self.dubp.running(busy)
        self.dubp.btn_dub.setEnabled(ready and not busy)
        self.dubp.btn_preview.setEnabled(ready and not busy)
        self.act_save.setEnabled(bool(self.project) and not busy)
        self.act_save_as.setEnabled(bool(self.project) and not busy)

    def go(self, index: int) -> None:
        """Show a wizard step, staying on Film until the project is prepared."""
        if index > 0 and not self._prepared():
            index = 0
        self.stack.setCurrentIndex(index)
        self._update_steps()

    def target_lang(self) -> str:
        """Return the dub language selected on the Film page."""
        return self.film.target_lang()

    # ------------------------------------------------------------------ film
    def dragEnterEvent(self, e) -> None:  # noqa: N802
        """Accept a dragged video, subtitle, or project file."""
        urls = e.mimeData().urls() if e.mimeData().hasUrls() else []
        if any(Path(u.toLocalFile()).suffix.lower() in VIDEO_EXTS | SUB_EXTS | {PROJECT_EXT} for u in urls):
            e.acceptProposedAction()

    def dropEvent(self, e) -> None:  # noqa: N802
        """Open a dropped film or project, or attach a dropped subtitle file."""
        for u in e.mimeData().urls():
            p = Path(u.toLocalFile())
            if p.suffix.lower() == PROJECT_EXT:
                self.open_vxdub(p)
            elif p.suffix.lower() in VIDEO_EXTS:
                self.set_source(p)
            elif p.suffix.lower() in SUB_EXTS and self.project is not None:
                self.film.set_subs_file(str(p))
                self._film_changed()

    def choose_file(self) -> None:
        """Ask for a film file and open it."""
        path, _ = QFileDialog.getOpenFileName(self, tr("ui.file_dialog"), str(Path.home()), tr("ui.file_filter"))
        if path:
            self.set_source(Path(path))

    def _file_menu(self) -> None:
        self.file_menu.exec(self.btn_file.mapToGlobal(self.btn_file.rect().bottomLeft()))

    def choose_vxdub(self) -> None:
        """Ask for a project file and open it."""
        start = str(self.vxdub_path.parent) if self.vxdub_path else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(self, tr("vxdub.open_title"), start, tr("vxdub.filter"))
        if path:
            self.open_vxdub(Path(path))

    def save_vxdub_as(self) -> None:
        """Ask where to save the project file, then save it."""
        if self.project is None:
            return
        start = self.vxdub_path or self.project.source.with_suffix(PROJECT_EXT)
        path, _ = QFileDialog.getSaveFileName(self, tr("vxdub.save_title"), str(start), tr("vxdub.filter"))
        if not path:
            return
        dest = Path(path)
        if dest.suffix.lower() != PROJECT_EXT:
            dest = dest.with_suffix(PROJECT_EXT)
        self.save_vxdub(dest)

    def save_vxdub(self, path: Optional[Path] = None) -> bool:
        """Write the open dub to ``path`` (or the current project file).  Returns whether a file was written."""
        if self.project is None or (self.job is not None and self.job.isRunning()):
            return False
        if path is None:
            if self.vxdub_path is None:
                self.save_vxdub_as()
                return self.vxdub_path is not None
            path = self.vxdub_path
        self.film.store(self.project)
        self.project.save()
        try:
            self.vxdub_doc = vxdub.write_project(self.project, path, previous=self.vxdub_doc)
        except vxdub.VxdubError as exc:
            self._vxdub_message(tr("vxdub.save_title"), tr("vxdub.save_failed", error=str(exc)))
            return False
        self.vxdub_path = Path(path)
        self.film.lbl_prepare.setText(tr("vxdub.saved", name=self.vxdub_path.name))
        return True

    def open_vxdub(self, path: Path, locate=None, notify=None) -> bool:
        """Open a ``.vxdub`` file, relink its video, and show that dub.

        ``locate(saved_path)`` asks for the video when the saved path and the relative hint both miss.
        ``notify(title, text)`` reports a problem.  Both default to dialogs.
        """
        if self.job is not None and self.job.isRunning():
            return False
        self.player.stop()
        path = Path(path)
        ask = locate if locate is not None else self._locate_video
        try:
            project, document = vxdub.open_project(path, self.projects_root(), ask)
        except vxdub.SchemaTooNew as exc:
            self._vxdub_message(tr("vxdub.schema_title"), tr("vxdub.schema_new", found=exc.found, supported=vxdub.SCHEMA), notify)
            return False
        except vxdub.FingerprintMismatch:
            self._vxdub_message(tr("vxdub.moved_title"), tr("vxdub.fingerprint"), notify)
            return False
        except vxdub.SourceMissing:
            self._vxdub_message(tr("vxdub.moved_title"), tr("vxdub.missing"), notify)
            return False
        except vxdub.VxdubError as exc:
            self._vxdub_message(tr("vxdub.open_title"), tr("vxdub.open_failed", error=str(exc)), notify)
            return False
        try:
            info = media.probe(project.source)
        except Exception as exc:  # noqa: BLE001 - ffprobe failures are shown on the page
            self.project = None
            self.vxdub_path = None
            self.vxdub_doc = None
            self.film.set_media(project.source, None, str(exc))
            self._update_steps()
            return False
        self.info = info
        self.project = project
        self.vxdub_path = path
        self.vxdub_doc = document
        self.film.set_media(project.source, info, "")
        self.film.load(project)
        self.watch_state = WatchState(info.duration)
        self._reload_pages()
        self.dubp.lbl_stage.setText("")
        self.dubp.lbl_watch.setText("")
        self.film.lbl_prepare.setText("")
        out = project.settings.get("output_file") or ""
        self.film.btn_show.setVisible(bool(out) and Path(out).exists())
        self.film.btn_watch.hide()
        self._refresh_plan()
        self.go(0)
        return True

    def _locate_video(self, saved: str) -> Optional[Path]:
        name = Path(saved).name if saved else ""
        QMessageBox.information(self, tr("vxdub.moved_title"), tr("vxdub.moved", name=name or saved))
        start = str(Path(saved).parent) if saved else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(self, tr("vxdub.moved_title"), start, tr("ui.file_filter"))
        return Path(path) if path else None

    def _vxdub_message(self, title: str, text: str, notify=None) -> None:
        if notify is not None:
            notify(title, text)
            return
        QMessageBox.warning(self, title, text)

    def choose_subtitles(self) -> None:
        """Ask for a subtitle file and use it for this film."""
        start = str(self.project.source.parent) if self.project else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(self, tr("film.subs_choose"), start, tr("film.subs_filter"))
        if path:
            self.film.set_subs_file(path)
            self._film_changed()

    def projects_root(self) -> Path:
        """Return the folder that holds film projects."""
        custom = str(settings.load().get("projects_dir") or "").strip()
        return Path(custom) if custom else paths.projects_dir()

    def set_source(self, path: Path) -> None:
        """Open a film: read what is inside, open (or resume) its project folder."""
        if self.job is not None and self.job.isRunning():
            return
        self.vxdub_path = None
        self.vxdub_doc = None
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
            self.project = Project.create(folder, path, **self.new_project_settings(self.info))
        self.apply_auto(self.project)
        self.project.save()
        self.film.load(self.project)
        self.watch_state = WatchState(self.info.duration)
        self._reload_pages()
        self.dubp.lbl_stage.setText("")
        self.dubp.lbl_watch.setText("")
        self.film.lbl_prepare.setText("")
        self.film.btn_show.setVisible(bool(self.project.settings.get("output_file")) and Path(self.project.settings["output_file"]).exists())
        self.film.btn_watch.hide()
        self._refresh_plan()
        self.go(0)

    # ------------------------------------------------------------------ the best defaults
    @staticmethod
    def default_target(info: Optional[media.MediaInfo]) -> str:
        """Remembered dub language, else the UI language (Russian when the UI language is not a dub language), never the film's own."""
        s = settings.load()
        want = str(s.get("dub_target_lang") or "")
        ui = i18n.current()
        cands = [want] if want in TARGET_LANGS else []
        cands += [c for c in (ui, "ru", "en") if c in TARGET_LANGS]
        orig = ""
        if info is not None and info.audio:
            t = next((t for t in info.audio if t.index == media.pick_original_track(info, cands[0])), info.audio[0])
            orig = t.lang2
        return next((c for c in cands if c != orig), cands[0])

    def new_project_settings(self, info: Optional[media.MediaInfo]) -> Dict[str, Any]:
        """A new film starts with the remembered Options (or the defaults: original track, subtitles automatic, profanity as in the
        original, one voice, voice-over volume, MKV)."""
        s = settings.load()
        target = self.default_target(info)
        out: Dict[str, Any] = {"target_lang": target, "audio_track_auto": True, "audio_track": media.pick_original_track(info, target),
                               "output_format": settings.output_format()}
        for key, proj_key in settings.DUB_DEFAULTS.items():
            if key in ("dub_target_lang", "output_format"):
                continue
            out[proj_key] = s[key]
        if out.get("subtitle_choice") not in ("auto", "none"):
            out["subtitle_choice"] = "auto"
        return out

    def apply_auto(self, p: Project) -> None:
        """Automatic choices the user has not overridden: the original audio track, and one library voice in the dub language
        (a voice made or downloaded with the Audiobook Builder) instead of a clone, when there is one."""
        if p.settings.get("audio_track_auto", True):
            p.settings["audio_track"] = media.pick_original_track(self.info, p.settings.get("target_lang", "ru"))
        if not p.settings.get("single_voice_user"):
            prev = dict(p.settings.get("single_voice") or {})
            weight = prev.get("actor_weight", p.settings.get("actor_weight", 0.5))
            lv = voices.default_single_voice(p.settings.get("target_lang", "ru"))
            chosen = {"kind": "library", "id": lv.id} if lv else {"kind": "clone", "id": ""}
            chosen["actor_weight"] = weight
            p.settings["single_voice"] = chosen

    def _refresh_plan(self) -> None:
        if self.project is None:
            self.film.set_plan(None)
            return
        from dubber.pipeline.stages import output_path

        self.film.set_plan(self.project, output_path(self.project).name)

    def _option_changed(self, what: str) -> None:
        """An Option changed: store it in the film's project and remember it for the next film."""
        if self.project is None or (self.job is not None and self.job.isRunning()):
            return
        if what == "voice":
            self.project.settings["single_voice_user"] = True
        self._film_changed()
        if what in ("target", "voice") and not self.project.settings.get("single_voice_user"):
            self.apply_auto(self.project)
            self.project.save()
            self.film.load(self.project)
        p = self.project.settings
        remember = {"target": {"dub_target_lang": p["target_lang"]}, "profanity": {"dub_profanity": p["profanity"]},
                    "multi": {"dub_multi_voice": bool(p["multi_voice"])}, "volume": {"dub_original_volume": p["original_volume"]},
                    "format": {"output_format": p["output_format"]},
                    "subs": {"dub_subtitles": p["subtitle_choice"]} if p["subtitle_choice"] in ("auto", "none") else {}}.get(what, {})
        if remember:
            settings.save(remember)
        if what == "profanity":
            self._profanity_changed()
        if what in ("multi", "volume"):
            self.chars.load(self.project)
            self.lines.load(self.project)
            self.dubp.load(self.project)
        self._refresh_plan()
        self._update_steps()

    def _film_changed(self) -> None:
        if self.project is None:
            return
        self.film.store(self.project)
        self.project.save()
        s = settings.load()
        self.film.show_subtitle_status(self.project, bool(s.get("subdl_key") or s.get("opensubtitles_key")))

    def _profanity_changed(self) -> None:
        """Apply the profanity mode to the prepared script at once (the pipeline does the same on its next run)."""
        if self.project is None or (self.job is not None and self.job.isRunning()):
            return
        from dubber.core import profanity

        if self._prepared():
            n = profanity.apply_to_lines(self.project.lines, self.project.settings["profanity"], self.project.settings["target_lang"])
            self.project.save()
            self.lines.load(self.project)
            self.film.lbl_prepare.setText(tr("film.softened", n=n) if self.project.settings["profanity"] == "soften" else "")

    def prepare(self) -> None:
        """The optional review path: make the lines (and speakers), then open the review tabs."""
        if self.project is None:
            self.film.lbl_prepare.setText(tr("ui.start_need_file"))
            return
        self._film_changed()
        self.run_pipeline("prepare", PREPARE_UNTIL)

    def dub_all(self) -> bool:
        """The main action: the whole pipeline with the current (preset) Options, ending with the dubbed file next to the film."""
        if self.project is None:
            self.film.lbl_prepare.setText(tr("ui.start_need_file"))
            return False
        self._film_changed()
        self.apply_auto(self.project)
        self.project.save()
        return self.run_pipeline("auto", "mux")

    # ------------------------------------------------------------------ characters / lines
    def _chars_changed(self, what: str) -> None:
        project = self.project
        if project is None:
            return
        if what == "voice":
            self.film.load(project)
            self._refresh_plan()
        if what in ("merge", "name", "voice"):
            self.lines.load(project)

    def listen(self, sid: str) -> None:
        """Play what a voice will sound like: the library sample, the reference clip, or a few of the speaker's lines."""
        p = self.project
        if p is None:
            return
        speaker = p.speaker(sid) if sid else None
        if not sid:
            v = Voice.from_dict(p.settings.get("single_voice"))
        else:
            v = speaker.voice if speaker is not None else Voice()
        clip: Optional[Path] = None
        if v.kind == "library":
            lv = voices.get_library_voice(v.id)
            clip = lv.preview if lv else None
        else:
            ref = speaker.ref_audio if speaker is not None else (p.settings.get("single_ref") or {}).get("audio", "")
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
        if p is None:
            return None
        src = p.folder / "audio" / "mix44.wav"
        if not src.exists():
            return None
        chosen = voices.pick_reference_lines(p.lines, sid or None, max_s=10.0, wav=src) or [ln for ln in p.lines if not sid or ln.speaker == sid][:3]
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

        player = self._sample_player
        if player is None:
            player = QMediaPlayer(self)
            self._sample_out = QAudioOutput(self)
            player.setAudioOutput(self._sample_out)
            self._sample_player = player
        player.stop()
        player.setSource(QUrl.fromLocalFile(str(path)))
        player.play()

    # ------------------------------------------------------------------ pipeline
    def cfg(self) -> Dict[str, Any]:
        """Return the engine settings for a pipeline run."""
        return dict(self._cfg) if self._cfg is not None else engine_cfg()

    def run_pipeline(self, kind: str, until: str, preview: Optional[float] = None) -> bool:
        """Start a background pipeline job and return whether it started. ``prepare``, ``find``, and ``auto`` use the Film page."""
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
        if kind in FILM_KINDS:
            self.film.progress.setValue(0)
            self.film.progress.show()
            self.film.btn_watch.hide()
            self.film.lbl_prepare.setText(tr("film.dubbing") if kind == "auto" else tr("film.preparing"))
            if kind == "find":
                self.chars.lbl_multi_status.setText(tr("chars.finding"))
        else:
            self.dubp.progress.setValue(0)
            self.dubp.lbl_stage.setText(tr("dub.starting"))
        if kind in ("dub", "auto") and self.watch_state is not None:
            self.watch_state.finished = False
        self.job.start()
        self._update_steps()
        return True

    def cancel_job(self) -> None:
        """Ask the running pipeline job to stop."""
        if self.job is not None:
            self.job.cancel()

    def _on_stage(self, key: str, status: str, msg: str) -> None:
        text = tr("run.stage", stage=tr(f"stage.{key}"), status=tr(f"run.{status}")) + (f" — {msg}" if msg and status != "running" else "")
        self.dubp.log.appendPlainText(text)
        if self.job_kind in FILM_KINDS:
            self.film.lbl_prepare.setText(text)
            if self.job_kind == "find":
                self.chars.lbl_multi_status.setText(text)
        else:
            self.dubp.lbl_stage.setText(text)

    def _on_progress(self, frac: float, eta: float) -> None:
        bar = self.film.progress if self.job_kind in FILM_KINDS else self.dubp.progress
        bar.setValue(int(frac * 1000))
        if self.job_kind == "dub":
            self.dubp.lbl_eta.setText(fmt_eta(eta))
        elif self.job_kind == "auto":
            self.film.lbl_eta.setText(fmt_eta(eta))

    def _on_until(self, seconds: float) -> None:
        if self.job_kind not in ("dub", "auto") or self.watch_state is None:
            return
        total = self.watch_state.total
        finished = seconds >= total - 0.5
        self.watch_state.update(self.player.position(), seconds, finished)
        self.player.dubbed_until(seconds, finished)
        self.dubp.btn_watch.setEnabled(self.watch_state.ready)
        self.film.btn_watch.setVisible(self.watch_state.ready)
        self.dubp.lbl_watch.setText(tr("dub.dubbed_until", t=fmt_time(seconds), total=fmt_time(total))
                                    + ("" if self.watch_state.ready else "  " + tr("dub.watch_wait")))

    def _on_done(self, ok: bool, message: str) -> None:
        from dubber.engines.asr import is_cuda_fallback_failure

        kind = self.job_kind
        cuda_stop = (not ok) and is_cuda_fallback_failure(message)
        if cuda_stop:
            message = tr("asr.cuda_fallback")
        folder = self.job.result_folder if self.job else None
        self.job_kind = ""
        project = self.project
        if project is not None:
            project = Project(project.folder)
            self.project = project
        self._reload_pages()
        if kind in ("prepare", "find"):
            self.film.progress.hide()
            n_lines = len(project.lines) if project is not None else 0
            self.film.lbl_prepare.setText(tr("film.prepared", n=n_lines) if ok else tr("run.failed", error=message))
            if ok and kind == "prepare":
                self.go(1 if self._multi() else 2)
        elif kind == "auto":
            self.film.progress.hide()
            self.film.lbl_eta.setText("")
            out = Path(message) if ok and message else None
            self.film.lbl_prepare.setText(tr("film.done", name=out.name) if out else tr("run.failed", error=message))
            self.film.btn_show.setVisible(bool(out))
            if ok and self.watch_state is not None:
                self.watch_state.finished = True
                self.player.dubbed_until(self.watch_state.total, True)
                self.dubp.btn_watch.setEnabled(True)
                self.film.btn_watch.show()
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
        if cuda_stop:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.warning(self, tr("asr.cuda_fallback_title"), tr("asr.cuda_fallback"))
        self._update_steps()

    def _reload_pages(self) -> None:
        if self.project is None:
            return
        self.chars.load(self.project)
        self.lines.load(self.project)
        self.dubp.load(self.project)
        self._refresh_plan()
        s = settings.load()
        self.film.show_subtitle_status(self.project, bool(s.get("subdl_key") or s.get("opensubtitles_key")))

    def set_original_volume(self, value: float) -> None:
        """Save how loud the original audio stays under the dub."""
        if self.project is not None:
            self.project.settings["original_volume"] = round(value, 2)
            self.project.save()
            settings.save({"dub_original_volume": round(value, 2)})
            self.film.load(self.project)

    # ------------------------------------------------------------------ preview / watch / result
    def start_preview(self) -> None:
        """Dub a one-minute fragment and play it."""
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
        opened = self.player.open(self.project.source, WavSource(wav, start), start, start + PREVIEW_S)
        poster = Path(tempfile.gettempdir()) / "voxprint-dubber-poster.jpg"
        if media.poster_frame(self.project.source, start, poster):
            self.player.show_poster(poster)
        self.dubp.btn_external.setEnabled(bool(opened))

    def watch_source(self):
        """Finished dub -> the whole dub track; while dubbing -> the Watch chunks."""
        p = self.project
        track = p.folder / "out" / "dub_track.wav"
        if (p.stages.get("mix") or {}).get("done") and track.exists():
            return WavSource(track, 0.0)
        return ChunkSource(p.folder / "watch")

    def start_watch(self) -> None:
        """Play the film alongside the dubbed audio that is already ready."""
        if self.project is None or self.watch_state is None or not self.watch_state.ready:
            return
        if self.player.open(self.project.source, self.watch_source(), 0.0, None, self.watch_state):
            self.watch_state.start()
            self.player.play()

    def show_result(self) -> None:
        """Show the finished dubbed file in its folder."""
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
    def save_actor_voice(self, sid: str) -> None:
        """Keep an actor-like voice in the shared library - only after the user confirms he may use this person's voice."""
        from PySide6.QtWidgets import QInputDialog, QMessageBox

        from dubber.core import actor_voice
        from dubber.infra import shared_paths

        project = self.project
        sp = project.speaker(sid) if project else None
        if project is None or sp is None:
            return
        name, ok = QInputDialog.getText(self, tr("actor.save_title"), tr("actor.name"), text=sp.name or sid)
        if not ok or not name.strip():
            return
        if QMessageBox.question(self, tr("actor.save_title"), tr("actor.consent"),
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            return
        try:
            path = actor_voice.save_to_library(actor_voice.folder(project.folder, sid), name.strip(), tr("actor.consent_note"),
                                               shared_paths.voices_dir(), project.settings.get("target_lang", ""))
            self.chars.lbl_multi_status.setText(tr("actor.saved", path=str(path)))
            self.chars.load(project)
        except (OSError, ValueError) as exc:
            self.chars.lbl_multi_status.setText(tr("run.failed", error=str(exc)))

    def open_voice_catalog(self) -> None:
        """Open the shared voice catalog and reload characters after an install."""
        dlg = dialogs.VoiceCatalogDialog(self)
        dlg.setStyleSheet(build_style(False))
        dlg.exec()
        if dlg.changed and self.project is not None:
            self.chars.load(self.project)

    def open_settings(self) -> None:
        """Open the settings dialog and refresh the plan when they are saved."""
        dlg = dialogs.SettingsDialog(self)
        dlg.setStyleSheet(build_style(False))
        if dlg.exec() and self.project is not None:
            self._film_changed()
            self._refresh_plan()

    def open_diagnostics(self, start: bool = False) -> None:
        """Show the diagnostics dialog and start the checks when asked."""
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
        """Stop playback and background work, then close the window."""
        self.player.stop()
        if self.job is not None and self.job.isRunning():
            self.job.cancel()
            self.job.wait(30000)
        if self.diag_dialog is not None:
            self.diag_dialog.close()
        super().closeEvent(e)
