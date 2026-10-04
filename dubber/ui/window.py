"""The main window: file picker, target language, pipeline steps and the diagnostics card.

Same shell as the Voxprint Audiobook Builder windows: translucent root + scroll area (content may be taller than a small screen),
Acrylic backdrop on Windows 11, cards with rounded corners, an accent-bordered primary card, a gear menu for the UI language.
Heavy work never runs in the GUI thread: diagnostics run in a QThread that starts worker *processes* (see ``dubber.diag``).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QThread, QTimer, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QGuiApplication, QIcon
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                               QMenu, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QVBoxLayout, QWidget)

from dubber import i18n, models, paths, platform_win
from dubber.appinfo import APP_DISPLAY_NAME, APP_VERSION, resource_dir
from dubber.diag.constants import LANG_NAMES
from dubber.diag.report import Status
from dubber.diag.runner import DiagnosticRunner, DiagOptions
from dubber.i18n import tr
from dubber.pipeline import stages as pst
from dubber.ui.theme import build_style

MIN_W, MIN_H = 520, 360
TARGET_LANGS = ("ru", "en", "de")


class DiagThread(QThread):
    """Runs :class:`DiagnosticRunner` off the GUI thread."""
    progress = Signal(float, str)
    result = Signal(str, str, str)       # check id, status, summary
    finished_path = Signal(str, bool)    # report path, cancelled
    failed = Signal(str)

    def __init__(self, options: DiagOptions) -> None:
        super().__init__()
        self.options = options
        self.runner: Optional[DiagnosticRunner] = None

    def run(self) -> None:
        try:
            self.runner = DiagnosticRunner(self.options, on_progress=lambda f, t: self.progress.emit(f, t),
                                           on_result=lambda r: self.result.emit(r.id, r.status.value, r.summary))
            path = self.runner.run()
            self.finished_path.emit(str(path), self.runner.cancelled)
        except Exception as exc:  # noqa: BLE001 - the runner already swallows its own errors; this is the last safety net
            self.failed.emit(f"{type(exc).__name__}: {exc}")

    def cancel(self) -> None:
        if self.runner:
            self.runner.cancel()


class DubThread(QThread):
    """Runs the dubbing stages (several are still in development) on the chosen movie."""
    stage_done = Signal(str, bool, str, bool)      # key, ok, message, implemented
    finished_ok = Signal()

    def __init__(self, source: Path, target: str) -> None:
        super().__init__()
        self.source, self.target = source, target

    def run(self) -> None:
        work = paths.new_work_dir("vmd-dub-")
        try:
            ctx = pst.StageContext(self.source, self.target, work, out=self.source.with_name(self.source.stem + ".dub-preview.mkv"))
            pst.run_all(ctx, lambda st, res: self.stage_done.emit(st.key, res.ok, res.message, res.implemented))
        finally:
            import shutil

            shutil.rmtree(work, ignore_errors=True)
            self.finished_ok.emit()


class MainWindow(QWidget):
    """Single-window UI."""

    def __init__(self, autorun: bool = False, options_hook=None) -> None:
        super().__init__()
        self.backdrop = "plain"
        self.source: Optional[Path] = None
        self.report_path: Optional[Path] = None
        self.diag: Optional[DiagThread] = None
        self.dub: Optional[DubThread] = None
        self._chips: dict = {}
        self._counts = {"OK": 0, "WARN": 0, "FAIL": 0, "SKIP": 0, "INFO": 0}
        self._options_hook = options_hook          # tests/CLI can adjust the DiagOptions before they run
        self.setObjectName("root")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._build()
        self.setStyleSheet(build_style(False))
        self.retranslate()
        self._fit()
        if autorun:
            QTimer.singleShot(500, self.start_diagnostics)

    # ------------------------------------------------------------------ construction
    def _card(self, primary: bool = False) -> QFrame:
        f = QFrame()
        f.setObjectName("card")
        f.setProperty("primary", "true" if primary else "false")
        return f

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.content = QWidget()
        self.content.setObjectName("content")
        self.scroll.setWidget(self.content)
        self.scroll.viewport().setAutoFillBackground(False)
        outer.addWidget(self.scroll)
        body = QVBoxLayout(self.content)
        body.setContentsMargins(28, 24, 28, 24)
        body.setSpacing(14)
        self.body = body

        head = QHBoxLayout()
        self.lbl_title = QLabel(APP_DISPLAY_NAME)
        self.lbl_title.setObjectName("title")
        head.addWidget(self.lbl_title)
        head.addStretch(1)
        self.btn_gear = QPushButton("\u2699")
        self.btn_gear.setObjectName("gear")
        self.btn_gear.clicked.connect(self._language_menu)
        head.addWidget(self.btn_gear)
        body.addLayout(head)
        self.lbl_tagline = QLabel()
        self.lbl_tagline.setObjectName("subtitle")
        self.lbl_tagline.setWordWrap(True)
        body.addWidget(self.lbl_tagline)
        self.lbl_note = QLabel()
        self.lbl_note.setObjectName("warn")
        self.lbl_note.setWordWrap(True)
        body.addWidget(self.lbl_note)

        # --- 1. movie
        c = self._card()
        lay = QVBoxLayout(c)
        lay.setContentsMargins(18, 14, 18, 14)
        self.lbl_movie = QLabel()
        self.lbl_movie.setObjectName("sectiontitle")
        row = QHBoxLayout()
        self.btn_file = QPushButton()
        self.btn_file.clicked.connect(self.choose_file)
        self.lbl_file = QLabel()
        self.lbl_file.setObjectName("fileLabel")
        self.lbl_file.setWordWrap(True)
        row.addWidget(self.btn_file)
        row.addWidget(self.lbl_file, 1)
        lay.addWidget(self.lbl_movie)
        lay.addLayout(row)
        body.addWidget(c)

        # --- 2. language
        c = self._card()
        lay = QVBoxLayout(c)
        lay.setContentsMargins(18, 14, 18, 14)
        self.lbl_lang = QLabel()
        self.lbl_lang.setObjectName("sectiontitle")
        row = QHBoxLayout()
        self.lbl_target = QLabel()
        self.lbl_target.setObjectName("fileLabel")
        self.cmb_target = QComboBox()
        for code in TARGET_LANGS:
            self.cmb_target.addItem("", code)
        row.addWidget(self.lbl_target)
        row.addWidget(self.cmb_target)
        row.addStretch(1)
        self.lbl_source = QLabel()
        self.lbl_source.setObjectName("hint")
        lay.addWidget(self.lbl_lang)
        lay.addLayout(row)
        lay.addWidget(self.lbl_source)
        body.addWidget(c)

        # --- 3. steps
        c = self._card()
        lay = QVBoxLayout(c)
        lay.setContentsMargins(18, 14, 18, 14)
        self.lbl_steps = QLabel()
        self.lbl_steps.setObjectName("sectiontitle")
        lay.addWidget(self.lbl_steps)
        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)
        for i, st in enumerate(pst.STAGES):
            chip = QLabel()
            chip.setObjectName("chip")
            self._chips[st.key] = chip
            grid.addWidget(chip, i // 5, i % 5)
        lay.addLayout(grid)
        row = QHBoxLayout()
        self.btn_start = QPushButton()
        self.btn_start.clicked.connect(self.start_dubbing)
        self.lbl_start = QLabel()
        self.lbl_start.setObjectName("status")
        self.lbl_start.setWordWrap(True)
        row.addWidget(self.btn_start)
        row.addWidget(self.lbl_start, 1)
        lay.addLayout(row)
        body.addWidget(c)

        # --- diagnostics (primary card)
        c = self._card(primary=True)
        lay = QVBoxLayout(c)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.setSpacing(8)
        self.lbl_diag = QLabel()
        self.lbl_diag.setObjectName("sectiontitle")
        self.lbl_diag_desc = QLabel()
        self.lbl_diag_desc.setObjectName("fileLabel")
        self.lbl_diag_desc.setWordWrap(True)
        self.chk_download = QCheckBox()
        self.chk_download.setChecked(True)
        self.chk_quick = QCheckBox()
        self.lbl_token = QLabel()
        self.lbl_token.setObjectName("fileLabel")
        self.lbl_token.setWordWrap(True)
        self.edt_token = QLineEdit()
        self.edt_token.setEchoMode(QLineEdit.EchoMode.Password)
        self.lbl_token_hint = QLabel()
        self.lbl_token_hint.setObjectName("hint")
        self.lbl_token_hint.setWordWrap(True)
        self.btn_diag = QPushButton()
        self.btn_diag.setObjectName("primary")
        self.btn_diag.clicked.connect(self.start_diagnostics)
        self.btn_cancel = QPushButton()
        self.btn_cancel.clicked.connect(self.cancel_diagnostics)
        self.btn_cancel.hide()
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.hide()
        self.lbl_status = QLabel()
        self.lbl_status.setObjectName("status")
        self.lbl_status.setWordWrap(True)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setFixedHeight(130)
        self.log.hide()
        self.row_result = QHBoxLayout()
        self.btn_open = QPushButton()
        self.btn_folder = QPushButton()
        self.btn_copy = QPushButton()
        self.btn_open.clicked.connect(self.open_report)
        self.btn_folder.clicked.connect(self.show_in_folder)
        self.btn_copy.clicked.connect(self.copy_report)
        for b in (self.btn_open, self.btn_folder, self.btn_copy):
            self.row_result.addWidget(b)
            b.hide()
        self.row_result.addStretch(1)
        for w in (self.lbl_diag, self.lbl_diag_desc, self.chk_download, self.chk_quick, self.lbl_token, self.edt_token, self.lbl_token_hint):
            lay.addWidget(w)
        lay.addWidget(self.btn_diag)
        lay.addWidget(self.btn_cancel)
        lay.addWidget(self.progress)
        lay.addWidget(self.lbl_status)
        lay.addWidget(self.log)
        lay.addLayout(self.row_result)
        body.addWidget(c)

        body.addStretch(1)
        self.lbl_footer = QLabel()
        self.lbl_footer.setObjectName("footer")
        self.lbl_footer.setWordWrap(True)
        body.addWidget(self.lbl_footer)

    def _fit(self) -> None:
        """Natural size, limited to ~94 % x ~90 % of the screen; the scroll area takes over on small screens."""
        hint = self.content.sizeHint()
        w, h = max(780, hint.width()), max(560, hint.height() + 8)
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
        self.lbl_note.setText(tr("ui.preview_note") + f"  [{APP_VERSION}]")
        self.lbl_movie.setText(tr("card.movie"))
        self.btn_file.setText(tr("ui.choose_file"))
        self.lbl_file.setText(str(self.source) if self.source else tr("ui.no_file"))
        self.lbl_lang.setText(tr("card.lang"))
        self.lbl_target.setText(tr("ui.lang_target"))
        for i, code in enumerate(TARGET_LANGS):
            self.cmb_target.setItemText(i, tr(f"lang.{code}"))
        self.lbl_source.setText(tr("ui.lang_source"))
        self.lbl_steps.setText(tr("card.steps"))
        for st in pst.STAGES:
            chip = self._chips[st.key]
            chip.setText(f"{tr(st.title_key)} · {tr('ui.chip_ready') if st.implemented else tr('ui.chip_soon')}")
            chip.setProperty("state", "ready" if st.implemented else "soon")
            chip.style().unpolish(chip)
            chip.style().polish(chip)
        self.btn_start.setText(tr("ui.btn_start"))
        self.lbl_diag.setText(tr("card.diag"))
        self.lbl_diag_desc.setText(tr("ui.diag_desc"))
        self._retranslate_download()
        self.chk_quick.setText(tr("ui.diag_quick"))
        self.lbl_token.setText(tr("ui.diag_token"))
        self.lbl_token_hint.setText(tr("ui.diag_token_hint"))
        self.btn_diag.setText(tr("ui.btn_diag"))
        self.btn_cancel.setText(tr("ui.btn_cancel"))
        self.btn_open.setText(tr("ui.btn_open"))
        self.btn_folder.setText(tr("ui.btn_folder"))
        self.btn_copy.setText(tr("ui.btn_copy"))
        self.lbl_footer.setText(tr("ui.footer"))

    def _retranslate_download(self) -> None:
        try:
            keys = ["tts_1_7b", "asr", "sep", "mt_en_ru", "diar"]
            gb = models.total_download_gb(keys)
        except Exception:  # noqa: BLE001
            gb = 8.0
        self.chk_download.setText(tr("ui.diag_download", gb=gb) if gb > 0 else tr("ui.diag_download_none"))

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
        for th in (self.diag, self.dub):
            if th is not None and th.isRunning():
                if isinstance(th, DiagThread):
                    th.cancel()
                th.wait(15000)
        super().closeEvent(e)

    # ------------------------------------------------------------------ actions
    def choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, tr("ui.file_dialog"), str(Path.home()), tr("ui.file_filter"))
        if path:
            self.set_source(Path(path))

    def set_source(self, path: Path) -> None:
        self.source = path
        self.lbl_file.setText(str(path))

    def target_lang(self) -> str:
        return self.cmb_target.currentData() or "ru"

    def start_dubbing(self) -> None:
        if not self.source:
            self.lbl_start.setText(tr("ui.start_need_file"))
            return
        self.btn_start.setEnabled(False)
        self.lbl_start.setText(tr("ui.start_running"))
        for st in pst.STAGES:
            self._chips[st.key].setToolTip("")
        self.dub = DubThread(self.source, self.target_lang())
        self.dub.stage_done.connect(self._on_stage_done)
        self.dub.finished_ok.connect(self._on_dub_finished)
        self.dub.start()

    def _on_stage_done(self, key: str, ok: bool, message: str, implemented: bool) -> None:
        self._chips[key].setToolTip(message)
        self.lbl_start.setText(f"{key}: {message}")

    def _on_dub_finished(self) -> None:
        self.btn_start.setEnabled(True)
        self.lbl_start.setText(tr("ui.start_done"))

    def build_options(self) -> DiagOptions:
        opt = DiagOptions(allow_download=self.chk_download.isChecked(), quick=self.chk_quick.isChecked(), target_lang=self.target_lang(),
                          hf_token=self.edt_token.text().strip())
        if self._options_hook:
            self._options_hook(opt)
        return opt

    def start_diagnostics(self) -> None:
        if self.diag is not None and self.diag.isRunning():
            return
        self.btn_diag.setEnabled(False)
        self.btn_cancel.show()
        for b in (self.btn_open, self.btn_folder, self.btn_copy):
            b.hide()
        self.log.clear()
        self.log.show()
        self.progress.setValue(0)
        self.progress.show()
        self._counts = {k: 0 for k in self._counts}
        self.lbl_status.setText(tr("ui.diag_running"))
        self.diag = DiagThread(self.build_options())
        self.diag.progress.connect(self._on_progress)
        self.diag.result.connect(self._on_result)
        self.diag.finished_path.connect(self._on_finished)
        self.diag.failed.connect(self._on_failed)
        self.diag.start()

    def cancel_diagnostics(self) -> None:
        if self.diag is not None:
            self.diag.cancel()
            self.btn_cancel.setEnabled(False)

    def _on_progress(self, frac: float, text: str) -> None:
        if frac >= 0:
            self.progress.setValue(int(frac * 1000))
        self.lbl_status.setText(f"{tr('ui.diag_running')} {text}")

    def _on_result(self, check_id: str, status: str, summary: str) -> None:
        self._counts[status] = self._counts.get(status, 0) + 1
        self.log.appendPlainText(f"[{status:<4}] {check_id}  {summary[:110]}")

    def _finish_ui(self) -> None:
        self.btn_diag.setEnabled(True)
        self.btn_cancel.hide()
        self.btn_cancel.setEnabled(True)
        self.progress.hide()

    def _on_finished(self, path: str, cancelled: bool) -> None:
        self._finish_ui()
        self.report_path = Path(path)
        c = self._counts
        counts = tr("ui.counts", ok=c.get("OK", 0), warn=c.get("WARN", 0), fail=c.get("FAIL", 0), skip=c.get("SKIP", 0))
        self.lbl_status.setText(tr("ui.diag_cancelled" if cancelled else "ui.diag_done", path=path) + "\n" + counts)
        for b in (self.btn_open, self.btn_folder, self.btn_copy):
            b.show()
        self._retranslate_download()

    def _on_failed(self, error: str) -> None:
        self._finish_ui()
        self.lbl_status.setText(tr("ui.diag_failed", error=error))

    # ------------------------------------------------------------------ report actions
    def open_report(self) -> None:
        if self.report_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.report_path)))

    def show_in_folder(self) -> None:
        if not self.report_path:
            return
        if sys.platform == "win32":
            try:
                subprocess.Popen(["explorer", "/select,", str(self.report_path)])
                return
            except OSError:
                pass
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.report_path.parent)))

    def copy_report(self) -> None:
        if not self.report_path:
            return
        try:
            QGuiApplication.clipboard().setText(self.report_path.read_text(encoding="utf-8", errors="replace"))
            self.lbl_status.setText(tr("ui.copied"))
        except OSError as exc:
            self.lbl_status.setText(str(exc))
