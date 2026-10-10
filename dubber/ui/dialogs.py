"""Settings and diagnostics dialogs."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QGuiApplication
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem, QPlainTextEdit, QProgressBar, QPushButton, QVBoxLayout,
                               QWidget)

from dubber import models, settings
from dubber.appinfo import version_label
from dubber.diag.runner import DiagnosticRunner, DiagOptions
from dubber.i18n import tr
from dubber.infra import shared_paths


def _combo(items, current: str) -> QComboBox:
    c = QComboBox()
    for value, label in items:
        c.addItem(label, value)
    i = c.findData(current)
    c.setCurrentIndex(max(0, i))
    return c


def _secret() -> QLineEdit:
    e = QLineEdit()
    e.setEchoMode(QLineEdit.EchoMode.Password)
    return e


class SettingsDialog(QDialog):
    """Subtitle services, Hugging Face token, models folder, engines and device."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("settings.title"))
        self.setMinimumWidth(560)
        s = settings.load()
        lay = QVBoxLayout(self)
        ver = QLabel(version_label())
        ver.setObjectName("hint")
        lay.addWidget(ver)
        form = QFormLayout()
        self.edt_subdl = _secret()
        self.edt_subdl.setText(s["subdl_key"])
        self.edt_os_key = _secret()
        self.edt_os_key.setText(s["opensubtitles_key"])
        self.edt_os_user = QLineEdit(s["opensubtitles_user"])
        self.edt_os_pass = _secret()
        self.edt_os_pass.setText(s["opensubtitles_password"])
        self.edt_hf = _secret()
        self.edt_hf.setText(s["hf_token"])
        row = QHBoxLayout()
        self.edt_models = QLineEdit(str(shared_paths.models_dir()))
        self.edt_models.setReadOnly(True)
        btn = QPushButton(tr("settings.browse"))
        btn.clicked.connect(self._choose_models)
        row.addWidget(self.edt_models, 1)
        row.addWidget(btn)
        self.cmb_sep = _combo([("tiger", "TIGER-DnR"), ("roformer", "Mel-Band RoFormer"), ("none", tr("settings.off"))], s["separation"])
        self.cmb_diar = _combo([("pyannote", "pyannote"), ("cluster", tr("settings.diar_simple"))], s["diarization"])
        self.cmb_tts = _combo([("tts_1_7b", "Qwen3-TTS 1.7B"), ("tts_0_6b", "Qwen3-TTS 0.6B")], s["tts_model"])
        self.cmb_backend = _combo([("auto", tr("settings.auto")), ("standard/flash_attention_2", "FlashAttention 2"),
                                   ("graphs/sdpa", "CUDA Graphs"), ("standard/sdpa", "SDPA")], s["tts_backend"])
        self.cmb_device = _combo([("auto", tr("settings.auto")), ("cuda", "CUDA")], settings.device())
        self.chk_download = QCheckBox(tr("settings.allow_download"))
        self.chk_download.setChecked(bool(s["allow_download"]))
        for label, w in (("settings.subdl", self.edt_subdl), ("settings.os_key", self.edt_os_key), ("settings.os_user", self.edt_os_user),
                         ("settings.os_pass", self.edt_os_pass), ("settings.hf", self.edt_hf)):
            form.addRow(tr(label), w)
        form.addRow(tr("settings.models_dir"), row)
        for label, box in (("settings.separation", self.cmb_sep), ("settings.diarization", self.cmb_diar), ("settings.tts", self.cmb_tts),
                           ("settings.backend", self.cmb_backend), ("settings.device", self.cmb_device)):
            form.addRow(tr(label), box)
        form.addRow("", self.chk_download)
        lay.addLayout(form)
        hint = QLabel(tr("settings.hint"))
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        lay.addWidget(hint)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self._models_choice: Optional[Path] = None

    def _choose_models(self) -> None:
        d = QFileDialog.getExistingDirectory(self, tr("settings.models_dir"), self.edt_models.text())
        if d:
            self._models_choice = Path(d)
            self.edt_models.setText(d)

    def values(self) -> dict:
        return {"subdl_key": self.edt_subdl.text().strip(), "opensubtitles_key": self.edt_os_key.text().strip(),
                "opensubtitles_user": self.edt_os_user.text().strip(), "opensubtitles_password": self.edt_os_pass.text(),
                "hf_token": self.edt_hf.text().strip(), "separation": self.cmb_sep.currentData(),
                "diarization": self.cmb_diar.currentData(), "tts_model": self.cmb_tts.currentData(),
                "tts_backend": self.cmb_backend.currentData(), "device": self.cmb_device.currentData(),
                "allow_download": self.chk_download.isChecked()}

    def accept(self) -> None:
        vals = self.values()
        settings.save(vals)
        if vals["device"] != settings.device():
            settings.set_device(vals["device"])          # shared with the other Voxprint programs (suite.json "gpu")
        if self._models_choice is not None:
            try:
                shared_paths.set_models_dir(self._models_choice)
            except (OSError, ValueError):
                pass
        super().accept()


class DiagThread(QThread):
    """Runs :class:`DiagnosticRunner` off the GUI thread (the checks themselves run in worker processes)."""
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
        except Exception as exc:  # noqa: BLE001 - the runner swallows its own errors; this is the last safety net
            self.failed.emit(f"{type(exc).__name__}: {exc}")

    def cancel(self) -> None:
        if self.runner:
            self.runner.cancel()


class DiagnosticsDialog(QDialog):
    """Hardware and model checks with a plain-text report on the Desktop."""

    def __init__(self, parent: Optional[QWidget] = None, target_lang: Callable[[], str] = lambda: "ru", options_hook=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("card.diag"))
        self.setMinimumWidth(620)
        self._target_lang, self._options_hook = target_lang, options_hook
        self.report_path: Optional[Path] = None
        self.diag: Optional[DiagThread] = None
        self._counts = {"OK": 0, "WARN": 0, "FAIL": 0, "SKIP": 0, "INFO": 0}
        lay = QVBoxLayout(self)
        lay.setSpacing(8)
        self.lbl_desc = QLabel(tr("ui.diag_desc"))
        self.lbl_desc.setObjectName("fileLabel")
        self.lbl_desc.setWordWrap(True)
        self.chk_download = QCheckBox()
        self.chk_download.setChecked(True)
        self.chk_quick = QCheckBox(tr("ui.diag_quick"))
        self.btn_diag = QPushButton(tr("ui.btn_diag"))
        self.btn_diag.setObjectName("primary")
        self.btn_diag.clicked.connect(self.start)
        self.btn_cancel = QPushButton(tr("ui.btn_cancel"))
        self.btn_cancel.clicked.connect(self.cancel)
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
        self.log.setFixedHeight(170)
        self.log.hide()
        row = QHBoxLayout()
        self.btn_open = QPushButton(tr("ui.btn_open"))
        self.btn_folder = QPushButton(tr("ui.btn_folder"))
        self.btn_copy = QPushButton(tr("ui.btn_copy"))
        self.btn_open.clicked.connect(self.open_report)
        self.btn_folder.clicked.connect(self.show_in_folder)
        self.btn_copy.clicked.connect(self.copy_report)
        for b in (self.btn_open, self.btn_folder, self.btn_copy):
            row.addWidget(b)
            b.hide()
        row.addStretch(1)
        foot = QLabel(tr("ui.footer"))
        foot.setObjectName("hint")
        foot.setWordWrap(True)
        for w in (self.lbl_desc, self.chk_download, self.chk_quick, self.btn_diag, self.btn_cancel, self.progress, self.lbl_status, self.log):
            lay.addWidget(w)
        lay.addLayout(row)
        lay.addWidget(foot)
        self._download_text()

    def _download_text(self) -> None:
        try:
            gb = models.total_download_gb(["tts_1_7b", "asr", "sep", "mt_en_ru", "diar"])
        except Exception:  # noqa: BLE001
            gb = 8.0
        self.chk_download.setText(tr("ui.diag_download", gb=gb) if gb > 0 else tr("ui.diag_download_none"))

    def build_options(self) -> DiagOptions:
        s = settings.load()
        opt = DiagOptions(allow_download=self.chk_download.isChecked(), quick=self.chk_quick.isChecked(), target_lang=self._target_lang(),
                          hf_token=str(s.get("hf_token") or ""), tts_model=str(s.get("tts_model") or "tts_1_7b"))
        if self._options_hook:
            self._options_hook(opt)
        return opt

    def start(self) -> None:
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

    def cancel(self) -> None:
        if self.diag is not None:
            self.diag.cancel()
            self.btn_cancel.setEnabled(False)

    def running(self) -> bool:
        return self.diag is not None and self.diag.isRunning()

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
        self._download_text()

    def _on_failed(self, error: str) -> None:
        self._finish_ui()
        self.lbl_status.setText(tr("ui.diag_failed", error=error))

    def open_report(self) -> None:
        if self.report_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.report_path)))

    def show_in_folder(self) -> None:
        if self.report_path:
            show_in_folder(self.report_path)

    def copy_report(self) -> None:
        if not self.report_path:
            return
        try:
            QGuiApplication.clipboard().setText(self.report_path.read_text(encoding="utf-8", errors="replace"))
            self.lbl_status.setText(tr("ui.copied"))
        except OSError as exc:
            self.lbl_status.setText(str(exc))

    def closeEvent(self, e) -> None:  # noqa: N802
        diag = self.diag
        if self.running() and diag is not None:
            diag.cancel()
            diag.wait(15000)
        super().closeEvent(e)


def show_in_folder(path: Path) -> None:
    if sys.platform == "win32":
        try:
            subprocess.Popen(["explorer", "/select,", str(path)])
            return
        except OSError:
            pass
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path).parent)))


class CatalogThread(QThread):
    """Lists (``ids`` empty) or installs voices of the Voxprint voice repositories off the GUI thread."""
    listed = Signal(list, str)              # voices, error
    progress = Signal(float, str)
    installed = Signal(str, str)            # voice id, error ("" = ok)
    finished_all = Signal()

    def __init__(self, voices_to_install=None) -> None:
        super().__init__()
        self.todo = list(voices_to_install or [])
        self.cancelled = False

    def run(self) -> None:
        from dubber.core import voice_catalog as vc

        if not self.todo:
            found, errors = [], []
            for repo in vc.repos():
                try:
                    found += vc.list_remote(repo)
                except vc.VoiceCatalogError as exc:
                    errors.append(str(exc))
            self.listed.emit(found, "; ".join(errors))
            return
        for v in self.todo:
            if self.cancelled:
                break
            try:
                voice_name = v.name

                def _progress(frac: float, shown: str = voice_name) -> None:
                    self.progress.emit(frac, shown)

                vc.install(v, progress=_progress, cancel=lambda: self.cancelled)
                self.installed.emit(v.id, "")
            except Exception as exc:  # noqa: BLE001 - shown in the dialog, the next voice is tried
                self.installed.emit(v.id, str(exc))
        self.finished_all.emit()


class VoiceCatalogDialog(QDialog):
    """Open voices published by Voxprint (CC0) - downloaded once into the shared voice library."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("catalog.title"))
        self.setMinimumWidth(560)
        self.voices: list = []
        self.job: Optional[CatalogThread] = None
        self.changed = False
        lay = QVBoxLayout(self)
        self.lbl = QLabel(tr("catalog.loading"))
        self.lbl.setObjectName("status")
        self.lbl.setWordWrap(True)
        self.list = QListWidget()
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.hide()
        row = QHBoxLayout()
        self.btn_install = QPushButton(tr("catalog.download"))
        self.btn_install.setObjectName("primary")
        self.btn_install.setEnabled(False)
        self.btn_install.clicked.connect(self.install)
        self.btn_close = QPushButton(tr("catalog.close"))
        self.btn_close.clicked.connect(self.close)
        row.addWidget(self.btn_install, 1)
        row.addWidget(self.btn_close)
        hint = QLabel(tr("catalog.hint"))
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        for w in (self.lbl, self.list, self.progress):
            lay.addWidget(w)
        lay.addLayout(row)
        lay.addWidget(hint)
        self.job = CatalogThread()
        self.job.listed.connect(self._on_listed)
        self.job.start()

    def _on_listed(self, voices: list, error: str) -> None:
        from dubber.core import voice_catalog as vc

        self.voices = voices
        have = vc.installed_ids()
        self.list.clear()
        for v in voices:
            mb = f", {v.size_bytes / 1e6:.0f} MB" if v.size_bytes else ""
            it = QListWidgetItem(f"{v.name} — {v.language or '?'}, {v.gender or '?'}, {v.license or '?'}{mb}")
            it.setData(Qt.ItemDataRole.UserRole, v.id)
            if v.id in have:
                it.setText(it.text() + "  ✓ " + tr("catalog.installed"))
                it.setFlags(Qt.ItemFlag.ItemIsEnabled)
            else:
                it.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
                it.setCheckState(Qt.CheckState.Checked)
            self.list.addItem(it)
        self.lbl.setText(tr("catalog.error", error=error) if error and not voices else tr("catalog.found", n=len(voices)))
        self.btn_install.setEnabled(any(self.list.item(i).flags() & Qt.ItemFlag.ItemIsUserCheckable for i in range(self.list.count())))

    def selected(self) -> list:
        ids = {self.list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.list.count())
               if self.list.item(i).flags() & Qt.ItemFlag.ItemIsUserCheckable and self.list.item(i).checkState() == Qt.CheckState.Checked}
        return [v for v in self.voices if v.id in ids]

    def install(self) -> None:
        todo = self.selected()
        if not todo or (self.job is not None and self.job.isRunning()):
            return
        self.btn_install.setEnabled(False)
        self.progress.show()
        self.job = CatalogThread(todo)
        def show_progress(frac: float, name: str) -> None:
            self.progress.setValue(int(frac * 1000))
            self.lbl.setText(tr("catalog.downloading", name=name))

        self.job.progress.connect(show_progress)
        self.job.installed.connect(self._on_installed)
        self.job.finished_all.connect(self._on_done)
        self.job.start()

    def _on_installed(self, vid: str, error: str) -> None:
        self.changed = self.changed or not error
        if error:
            self.lbl.setText(tr("catalog.error", error=error))

    def _on_done(self) -> None:
        self.progress.hide()
        self._on_listed(self.voices, "")

    def closeEvent(self, e) -> None:  # noqa: N802
        if self.job is not None and self.job.isRunning():
            self.job.cancelled = True
            self.job.wait(30000)
        super().closeEvent(e)
