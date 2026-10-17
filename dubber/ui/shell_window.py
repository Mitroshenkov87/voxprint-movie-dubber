"""The movie window on the shared video screen.

Simple mode is the video, the language, Quality, Voices, and Dub. Expert mode
adds the side panel. Both read one settings state. Manage models opens
VOXPrint MDL. The older four-step window stays in :mod:`dubber.ui.window` for
the checks that still build it, and nothing in this window opens it.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QEvent, QObject, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QFileDialog, QPushButton, QVBoxLayout, QWidget

from dubber import i18n, paths, settings
from dubber.appinfo import APP_DISPLAY_NAME
from dubber.core import voices, vxdub
from dubber.core.project import Project
from dubber.i18n import tr
from dubber.models import launch_mdl
from dubber.pipeline.stages import MOCK_CFG, ORDER
from dubber.ui import dialogs
from dubber.ui.jobs import PipelineThread
from dubber.ui.shell_state import (
    DUB_SURFACE,
    apply_engine,
    apply_project,
    build_layout,
    dub_settings,
)
from voxprint_common.i18n import add_catalogs
from voxprint_common.i18n import set_language as set_shell_language
from voxprint_common.paths import state_dir
from voxprint_common.suite_settings import set_window_geometry, window_geometry
from voxprint_common.ui.expert import ExpertMode
from voxprint_common.ui.simple_image import Phase, SimpleLayout, SimpleWidget
from voxprint_common.ui.theme import build_shell_style, build_style
from voxprint_common.ui.variants import WINDOW_HEIGHT, WINDOW_MIN_HEIGHT, WINDOW_MIN_WIDTH, WINDOW_WIDTH

VIDEO_EXTS = {".mkv", ".mp4", ".avi", ".mov", ".webm", ".m4v", ".ts", ".m2ts"}


class _VideoPick(QObject):
    """Click or drop chooses a video and keeps the full path."""

    def __init__(self, owner: ShellWindow) -> None:
        super().__init__(owner)
        self._owner = owner

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        """Accept a click or a dropped file.

        Args:
            watched: The drop label.
            event: The Qt event.

        Returns:
            True when this filter handled the event.
        """
        del watched
        kind = event.type()
        if kind == QEvent.Type.MouseButtonRelease:
            self._owner.browse()
            return True
        if kind == QEvent.Type.DragEnter:
            mime = event.mimeData()  # type: ignore[attr-defined]
            if mime.hasUrls():
                event.acceptProposedAction()  # type: ignore[attr-defined]
                return True
            return False
        if kind == QEvent.Type.Drop:
            urls = event.mimeData().urls()  # type: ignore[attr-defined]
            if not urls:
                return False
            path = urls[0].toLocalFile()
            if path:
                self._owner.set_video(Path(path))
            event.acceptProposedAction()  # type: ignore[attr-defined]
            return True
        return False


class ShellWindow(QWidget):
    """Simple video screen. Expert mode only opens the side panel."""

    def __init__(
        self,
        autorun: bool = False,
        options_hook: Callable[..., None] | None = None,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        """Build the screen and restore the last size when one was stored.

        Args:
            autorun: Open diagnostics and start them.
            options_hook: Extra diagnostic options from the command line.
            cfg: Engine config for a run. None reads the saved engines.
        """
        super().__init__()
        add_catalogs(i18n.locales_dir())
        set_shell_language(i18n.current())
        self._engine = cfg
        self._options_hook = options_hook
        self.project: Project | None = None
        self.job: PipelineThread | None = None
        self._run_id = 0
        self._eta: int | None = None
        self._outputs: dict[str, Path] = {}
        self._fitted = False
        self.diag_dialog: dialogs.DiagnosticsDialog | None = None
        self.expert = ExpertMode()
        self.shell: SimpleLayout = build_layout(state_dir() / "movie-dubber")
        self.settings_model = dub_settings()
        self.view = SimpleWidget(
            self.shell,
            self.settings_model,
            on_generate=self._started,
            on_command=self._command,
            on_history=self._history,
            surface=DUB_SURFACE,
        )
        self.setObjectName("shell")
        self.setWindowTitle(APP_DISPLAY_NAME)
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self.setStyleSheet(build_shell_style(False))
        body = QVBoxLayout(self)
        body.setContentsMargins(0, 0, 0, 0)
        body.addWidget(self.view.root)
        self._install_video_pick()
        self._add_play()
        self._add_diagnostics()
        self._stop_expert = self.expert.subscribe(self.view.set_expert)
        self.view.set_expert(self.expert.enabled())
        self._apply_saved_language()
        if autorun:
            self.open_diagnostics(True)

    def set_video(self, path: Path) -> None:
        """Show ``path`` as the chosen video.

        Args:
            path: A film the user dropped or picked.
        """
        if path.suffix.lower() not in VIDEO_EXTS or not path.is_file():
            self.view.set_note(tr("shell.need_file"))
            return
        self.view.set_note("")
        self.view.set_prompt_text(str(path.resolve()))
        self.view.refresh(prompt=True)

    def browse(self) -> None:
        """Ask for a video file."""
        picked, _selected = QFileDialog.getOpenFileName(
            self,
            tr("shell.browse"),
            str(self._projects_root()),
            tr("shell.browse_filter"),
        )
        if picked:
            self.set_video(Path(picked))

    def show_optional_models(self, missing: list[str]) -> None:
        """Tell the user that optional models are absent. MDL installs them.

        Args:
            missing: Registry keys that are not on disk. The ids are not shown.
        """
        if missing:
            self.view.set_note(tr("shell.models_missing"))

    def open_vxdub(self, path: Path) -> bool:
        """Open a saved dub and show its video.

        Args:
            path: A ``.vxdub`` file.

        Returns:
            True when the video is on screen.
        """
        try:
            project, _document = vxdub.open_project(path, self._projects_root(), self._ask_video)
        except vxdub.VxdubError as exc:
            self.view.set_note(tr("vxdub.open_failed", error=str(exc)))
            return False
        self.project = project
        source = project.source
        if source.is_file():
            self.set_video(source)
        return source.is_file()

    def open_diagnostics(self, start: bool = False) -> None:
        """Show the diagnostics dialog.

        Args:
            start: Start the checks at once.
        """
        if self.diag_dialog is None:
            self.diag_dialog = dialogs.DiagnosticsDialog(
                self,
                lambda: str(self.shell.get("dub.target_lang")),
                self._options_hook,
            )
            self.diag_dialog.setStyleSheet(build_style(False))
        self.diag_dialog.show()
        self.diag_dialog.raise_()
        if start and not self.diag_dialog.running():
            self.diag_dialog.start()

    def showEvent(self, event) -> None:  # noqa: N802
        """Use 1200 by 900 the first time, then the stored rectangle."""
        super().showEvent(event)
        if self._fitted:
            return
        self._fitted = True
        stored = window_geometry()
        if stored is None:
            self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
            return
        self.resize(
            max(WINDOW_MIN_WIDTH, int(stored["width"])),
            max(WINDOW_MIN_HEIGHT, int(stored["height"])),
        )
        self.move(int(stored["x"]), int(stored["y"]))

    def closeEvent(self, event) -> None:  # noqa: N802
        """Stop a running dub and remember the window size."""
        self._stop_job()
        if self.job is not None and self.job.isRunning():
            self.job.wait(30000)
        if self.isVisible():
            set_window_geometry(self.width(), self.height(), self.x(), self.y())
        if self.diag_dialog is not None:
            self.diag_dialog.close()
        self._stop_expert()
        super().closeEvent(event)

    def _install_video_pick(self) -> None:
        drop = self.view.root.findChild(QWidget, "fileDrop")
        if drop is None:
            return
        self._picker = _VideoPick(self)
        drop.installEventFilter(self._picker)

    def _add_play(self) -> None:
        button = QPushButton(tr("shell.play"))
        button.setObjectName("playResult")
        button.setAutoDefault(False)
        button.setMinimumHeight(34)
        button.clicked.connect(self._play)
        bar = self.view.root.findChild(QWidget, "actionBar")
        if bar is None or bar.layout() is None:
            return
        bar.layout().insertWidget(0, button)

    def _add_diagnostics(self) -> None:
        overflow = self.view.root.findChild(QPushButton, "overflow")
        menu = None if overflow is None else overflow.menu()
        if menu is None:
            return
        menu.addAction(tr("card.diag"), lambda: self.open_diagnostics(False))
        cancel = self.view.root.findChild(QPushButton, "cancel")
        if cancel is not None:
            cancel.clicked.connect(self._stop_job)

    def _apply_saved_language(self) -> None:
        saved = str(settings.load().get("dub_target_lang") or "")
        if saved in {"ru", "en", "de"} and saved != str(self.shell.get("dub.target_lang")):
            self.shell.choose("language", saved)
            self.view.refresh()

    def _command(self, command_id: str) -> None:
        if command_id == "models.manage":
            self._manage_models()
            return
        if command_id == "file.open_folder":
            self._open_folder()
            return
        if command_id == "file.save":
            self._open_folder()
            return
        if command_id == "view.toggle_expert_mode":
            self.expert.toggle()
            return
        if command_id == "view.settings":
            self._open_settings()

    def _history(self, action: str, item_id: str) -> None:
        path = self._outputs.get(item_id)
        if path is None:
            return
        if action == "open":
            self._open_file(path)
            return
        if action == "show_folder":
            self._reveal(path)

    def _started(self) -> None:
        path = Path(self.shell.prompt.strip())
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTS:
            self.shell.block(tr("shell.need_file"))
            self.view.refresh()
            return
        self._run_id += 1
        run_id = self._run_id
        self._eta = None
        try:
            self.project = self._open_project(path)
            self._remember()
            cfg = self._cfg()
            cfg.update(apply_engine(self.shell))
            self.job = PipelineThread(self.project.folder, cfg, "mux")
        except OSError as exc:
            self.shell.block(tr("shell.failed", error=str(exc)))
            self.view.refresh()
            return
        self.job.stage.connect(lambda key, status, message, token=run_id: self._on_stage(token, key, status, message))
        self.job.progress.connect(lambda frac, eta, token=run_id: self._on_progress(token, frac, eta))
        self.job.done.connect(lambda ok, message, token=run_id: self._on_done(token, ok, message))
        self.shell.progress(0, 100, tr("shell.dubbing"))
        self.view.refresh()
        self.job.start()

    def _on_stage(self, run_id: int, key: str, status: str, message: str) -> None:
        if run_id != self._run_id or self.shell.phase is not Phase.BUSY:
            return
        index = ORDER.index(key) + 1 if key in ORDER else 0
        label = tr("run.stage", stage=tr(f"stage.{key}"), status=tr(f"run.{status}"))
        if message and status != "running":
            label = f"{label} — {message}"
        if self._eta is None:
            self.shell.progress(index, len(ORDER), label)
        else:
            self.shell.progress(index, len(ORDER), label, eta_seconds=self._eta)
        self.view.refresh()

    def _on_progress(self, run_id: int, frac: float, eta: float) -> None:
        if run_id != self._run_id or self.shell.phase is not Phase.BUSY:
            return
        self._eta = int(eta) if eta >= 0 else None
        step = max(0, min(100, round(float(frac) * 100)))
        label = self.shell.stage_label or tr("shell.dubbing")
        if self._eta is None:
            self.shell.progress(step, 100, label)
        else:
            self.shell.progress(step, 100, label, eta_seconds=self._eta)
        self.view.refresh()

    def _on_done(self, run_id: int, ok: bool, message: str) -> None:
        if run_id != self._run_id or self.shell.phase is not Phase.BUSY:
            return
        if not ok:
            self.shell.block(tr("shell.failed", error=message))
            self.view.refresh()
            return
        out = Path(message) if message else None
        item_id = f"dub-{run_id}"
        if out is not None:
            self._outputs[item_id] = out
        self.shell.finish(item_id)
        self.view.set_note(tr("shell.done", name=out.name if out is not None else ""))
        self.view.refresh()

    def _stop_job(self) -> None:
        if self.job is not None and self.job.isRunning():
            self.job.cancel()

    def _open_project(self, source: Path) -> Project:
        folder = Project.folder_for(source, self._projects_root())
        values = apply_project(self.shell)
        raw_voice = values["single_voice"]
        if not isinstance(raw_voice, dict):
            raise TypeError("single voice is not a map")
        single = dict(raw_voice)
        if single.get("kind") == "library" and not single.get("id"):
            voice = voices.default_single_voice(str(values["target_lang"]))
            if voice is not None:
                single["id"] = voice.id
            else:
                single["kind"] = "clone"
        values["single_voice"] = single
        if (folder / "project.json").is_file():
            project = Project(folder)
            project.settings.update(values)
            project.settings["source"] = str(source.resolve())
            project.save()
            return project
        return Project.create(folder, source, **values)

    def _remember(self) -> None:
        settings.save(
            {
                "dub_target_lang": str(self.shell.get("dub.target_lang")),
                "dub_profanity": str(self.shell.get("dub.profanity")),
                "dub_multi_voice": bool(self.shell.get("dub.multi_voice")),
                "dub_original_volume": _mix_level(self.shell.get("dub.mix_original")),
                "output_format": str(self.shell.get("dub.output_format")),
                "dub_subtitles": str(self.shell.get("dub.subtitles")) if str(self.shell.get("dub.subtitles")) in {"auto", "none"} else "auto",
                "vram_tier": str(self.shell.get("dub.vram_tier")),
                "device": str(self.shell.get("dub.device")),
            }
        )
        settings.set_device(str(self.shell.get("dub.device")))

    def _cfg(self) -> dict[str, Any]:
        if self._engine is not None:
            return dict(self._engine)
        if os.environ.get("VOXPRINT_ENGINES", "").strip().lower() == "mock":
            return dict(MOCK_CFG)
        return settings.engine_cfg()

    def _manage_models(self) -> None:
        try:
            launch_mdl()
        except OSError:
            self.view.set_note(tr("shell.mdl_failed"))

    def _open_settings(self) -> None:
        dialogs.SettingsDialog(self).exec()

    def _play(self) -> None:
        path = self._current_output()
        if path is None:
            self.view.set_note(tr("shell.need_result"))
            return
        self._open_file(path)

    def _open_folder(self) -> None:
        path = self._current_output() or self._video_path()
        if path is None:
            self.view.set_note(tr("shell.need_file"))
            return
        self._reveal(path)

    def _current_output(self) -> Path | None:
        selected = self.shell.selected
        if selected and selected in self._outputs:
            return self._outputs[selected]
        if self._outputs:
            return next(reversed(self._outputs.values()))
        return None

    def _video_path(self) -> Path | None:
        text = self.shell.prompt.strip()
        if not text:
            return None
        path = Path(text)
        return path if path.is_file() else None

    def _open_file(self, path: Path) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _reveal(self, path: Path) -> None:
        folder = path if path.is_dir() else path.parent
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _projects_root(self) -> Path:
        custom = str(settings.load().get("projects_dir") or "").strip()
        return Path(custom) if custom else paths.projects_dir()

    def _ask_video(self, saved: str) -> Path | None:
        del saved
        picked, _selected = QFileDialog.getOpenFileName(self, tr("vxdub.moved_title"), "", tr("shell.browse_filter"))
        return Path(picked) if picked else None


def _mix_level(value: object) -> float:
    """Return a mix level, or the usual 0.15 when the value is not a number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.15
    return float(value)
