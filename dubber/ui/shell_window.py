"""The movie window on the shared audio screen.

Simple mode is the video, the language, Quality, Voices, and Dub. Expert mode
adds the side panel. Both read one settings state. The screen is
``surface_for("movie-dubber")`` from voxprint-common. The overflow menu opens
the existing film, character, line, and player steps. Manage models opens
VOXPrint MDL.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, cast

from PySide6.QtCore import QEvent, QObject, QUrl
from PySide6.QtGui import QDesktopServices, QDropEvent
from PySide6.QtWidgets import QFileDialog, QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget

from dubber import i18n, paths, settings
from dubber.appinfo import APP_DISPLAY_NAME
from dubber.core.project import Project
from dubber.i18n import tr
from dubber.models import launch_mdl
from dubber.pipeline.stages import MOCK_CFG, ORDER
from dubber.ui import dialogs
from dubber.ui.feature_wizard import FeatureWizard
from dubber.ui.jobs import PipelineThread
from dubber.ui.setup_progress import ensure_with_dialog, startup_repair_dialog
from dubber.ui.pages import voice_from_key
from dubber.ui.shell_state import apply_engine, apply_project, build_layout, dub_settings
from dubber.ui.voice_cast import VoiceCastDialog, current_voice_key
from dubber.ui.window import MainWindow
from voxprint_common.i18n import add_catalogs
from voxprint_common.i18n import set_language as set_shell_language
from voxprint_common.paths import state_dir
from voxprint_common.suite_settings import set_window_geometry, window_geometry
from voxprint_common.ui.expert import ExpertMode
from voxprint_common.ui.simple_image import Phase, SimpleLayout, SimpleWidget
from voxprint_common.ui.theme import build_shell_style, build_style
from voxprint_common.ui.variants import (
    WINDOW_HEIGHT,
    WINDOW_MIN_HEIGHT,
    WINDOW_MIN_WIDTH,
    WINDOW_WIDTH,
    surface_for,
)

VIDEO_EXTS = {".mkv", ".mp4", ".avi", ".mov", ".webm", ".m4v", ".ts", ".m2ts"}
# Object name of the optional video result region. Components adds it in the Yitro tag.
VIDEO_RESULT_NAME = "videoResult"


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
        if kind == QEvent.Type.DragEnter or kind == QEvent.Type.Drop:
            dropped = cast(QDropEvent, event)
            if kind == QEvent.Type.DragEnter:
                if dropped.mimeData().hasUrls():
                    dropped.acceptProposedAction()
                    return True
                return False
            urls = dropped.mimeData().urls()
            if not urls:
                return False
            path = urls[0].toLocalFile()
            if path:
                self._owner.set_video(Path(path))
            dropped.acceptProposedAction()
            return True
        return False


class ShellWindow(QWidget):
    """Simple audio screen. Expert mode only opens the side panel."""

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
        self._studio: MainWindow | None = None
        self._wizard: FeatureWizard | None = None
        self._optional_missing: list[str] = []
        self._run_id = 0
        self._eta: int | None = None
        self._outputs: dict[str, Path] = {}
        self._source: Path | None = None
        self._fitted = False
        self._cast: VoiceCastDialog | None = None
        self._video_result: QWidget | None = None
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
            on_refresh=self._settle_screen,
            surface=surface_for("movie-dubber"),
        )
        self._settle_screen()
        self.setObjectName("shell")
        self.setWindowTitle(APP_DISPLAY_NAME)
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self.setStyleSheet(build_shell_style(False))
        body = QVBoxLayout(self)
        body.setContentsMargins(0, 0, 0, 0)
        body.addWidget(self.view.root)
        self._install_video_pick()
        self._hide_narrator_voice()
        self._add_menu()
        self._watch_custom_voices()
        self._stop_expert = self.expert.subscribe(self._set_expert)
        self._set_expert(self.expert.enabled())
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
        chosen = path.resolve()
        self._source = chosen
        self.view.set_note("")
        self.view.set_prompt_text(str(chosen))
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
        self._optional_missing = list(missing)
        if missing:
            self.view.set_note(tr("shell.models_missing"))
        if self._studio is not None:
            self._studio.show_optional_models(list(missing))

    def open_vxdub(self, path: Path) -> bool:
        """Open a saved dub in the review wizard and show its video.

        Args:
            path: A ``.vxdub`` file.

        Returns:
            True when the project opened.
        """
        studio = self._ensure_studio(sync_file=False, push=False)
        opened = studio.open_vxdub(path)
        self.project = studio.project
        if opened and studio.project is not None and studio.project.source.is_file():
            self._source = studio.project.source.resolve()
            self.view.set_note("")
            self.view.set_prompt_text(str(self._source))
            self.view.refresh(prompt=True)
            if self._wizard is not None:
                self._wizard.open_step(0)
        return opened

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
        if self._wizard is not None:
            self._wizard.close()
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

    def _add_menu(self) -> None:
        overflow = self.view.root.findChild(QPushButton, "overflow")
        menu = None if overflow is None else overflow.menu()
        if menu is None:
            return
        menu.addAction(tr("step.film"), lambda: self._open_review(0))
        menu.addAction(tr("step.characters"), lambda: self._open_review(1))
        menu.addAction(tr("step.lines"), lambda: self._open_review(2))
        menu.addAction(tr("shell.menu_player"), lambda: self._open_review(3))
        menu.addAction(tr("vxdub.open"), self._open_project_file)
        menu.addAction(tr("vxdub.save"), self._save_project)
        menu.addAction(tr("vxdub.save_as"), self._save_project_as)
        menu.addAction(tr("shell.menu_repair"), self._repair_shared)
        menu.addAction(tr("card.diag"), lambda: self.open_diagnostics(False))
        cancel = self.view.root.findChild(QPushButton, "cancel")
        if cancel is not None:
            cancel.clicked.connect(self._stop_job)

    def _ensure_studio(self, *, sync_file: bool = True, push: bool = True) -> MainWindow:
        """Return the four-step window, creating it on the first call.

        Args:
            sync_file: Open the film chosen on the main screen.
            push: Copy the shared settings onto that film.
        """
        if self._studio is None:
            self._studio = MainWindow(cfg=self._cfg(), options_hook=self._options_hook)
            if self._optional_missing:
                self._studio.show_optional_models(list(self._optional_missing))
            self._wizard = FeatureWizard(self, self._studio)
        if sync_file:
            self._sync_studio_file()
        if push:
            self._push_shared_settings(self._studio)
        self.project = self._studio.project
        return self._studio

    def _sync_studio_file(self) -> None:
        studio = self._studio
        source = self._source
        if studio is None or source is None:
            return
        if studio.job is not None and studio.job.isRunning():
            return
        project = studio.project
        if project is not None and project.source.resolve() == source.resolve():
            return
        studio.set_source(source)

    def _push_shared_settings(self, studio: MainWindow) -> None:
        """Copy the main-screen settings onto the open film.

        A subtitle file chosen in the wizard stays. The shell only stores
        automatic or none.

        Args:
            studio: The four-step window.
        """
        project = studio.project
        if project is None or (studio.job is not None and studio.job.isRunning()):
            return
        values = apply_project(self.shell)
        current = project.settings.get("subtitle_choice")
        if isinstance(current, str) and current not in {"auto", "none"}:
            values["subtitle_choice"] = current
        project.settings.update(values)
        project.save()
        studio.film.load(project)
        cfg = self._cfg()
        cfg.update(apply_engine(self.shell))
        studio._cfg = cfg
        self._remember()

    def _open_review(self, index: int) -> None:
        """Open the review wizard on a step.

        Characters, lines, and the player stay on the film step until the
        film has been prepared, which is the same rule as the old step buttons.

        Args:
            index: 0 film, 1 characters, 2 lines, 3 player.
        """
        studio = self._ensure_studio()
        if index > 0 and not studio._prepared():
            self.view.set_note(tr("shell.prepare_first"))
            index = 0
        if self._wizard is not None:
            self._wizard.open_step(index)

    def _open_project_file(self) -> None:
        studio = self._ensure_studio(sync_file=False, push=False)
        studio.choose_vxdub()
        self.project = studio.project
        if studio.project is not None and studio.project.source.is_file():
            self._source = studio.project.source.resolve()
            self.view.set_prompt_text(str(self._source))
            self.view.refresh(prompt=True)
        if self._wizard is not None:
            self._wizard.open_step(0)

    def _save_project(self) -> None:
        studio = self._ensure_studio()
        if studio.save_vxdub():
            self.view.set_note(tr("vxdub.saved", name=studio.vxdub_path.name if studio.vxdub_path else ""))

    def _save_project_as(self) -> None:
        studio = self._ensure_studio()
        studio.save_vxdub_as()
        self.project = studio.project

    def _repair_shared(self) -> None:
        """Check the shared runtime and models, and repair what can be repaired."""
        report = ensure_with_dialog()
        if report.exit_code == 0:
            self.view.set_note(tr("shell.repair_ok"))
            return
        dialog = startup_repair_dialog(report)
        title, body = dialog if dialog is not None else (tr("models.repair_title"), report.message or "")
        QMessageBox.critical(self, title, body or report.message or "")

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

    def _watch_custom_voices(self) -> None:
        """Open character voices when the Custom chip is chosen."""
        chip = self.view.root.findChild(QPushButton, "chip-voices-pick")
        if chip is not None:
            chip.clicked.connect(self._open_voice_cast)

    def _open_voice_cast(self) -> VoiceCastDialog:
        """Choose Custom voices and show the character list.

        The shell has no inline panel under a chip row, so this is a dialog.

        Returns:
            The dialog.
        """
        if self.shell.presets.get("voices") != "pick":
            self.shell.choose("voices", "pick")
            self.view.refresh()
        if self._cast is None:
            self._cast = VoiceCastDialog(self)
        self._cast.reload()
        self._cast.show()
        self._cast.raise_()
        return self._cast

    def cast_prepared(self) -> bool:
        """Return whether the open film has been prepared."""
        studio = self._studio
        return studio is not None and studio._prepared()

    def cast_characters(self) -> list[tuple[str, str, str]]:
        """Return each character's id, name, and library voice key."""
        studio = self._studio
        project = None if studio is None else studio.project
        if project is None:
            return []
        people = sorted(project.speakers, key=lambda speaker: -speaker.seconds)
        return [
            (speaker.id, speaker.name or speaker.id, current_voice_key(speaker.voice))
            for speaker in people
        ]

    def cast_prepare(self) -> None:
        """Run Prepare, then fill the character list."""
        studio = self._ensure_studio()
        if studio.project is None:
            self.view.set_note(tr("shell.need_file"))
            return
        if studio.job is not None and studio.job.isRunning():
            return
        studio.prepare()
        if studio.job is None:
            return
        studio.job.done.connect(self._cast_finished)

    def _cast_finished(self, ok: bool, message: str) -> None:
        if not ok:
            self.view.set_note(tr("run.failed", error=message))
        if self._cast is not None and self._cast.isVisible():
            self._cast.reload()

    def cast_listen(self, sid: str) -> None:
        """Play the sample for one character.

        Args:
            sid: Speaker id.
        """
        studio = self._ensure_studio(sync_file=False, push=False)
        studio.listen(sid)

    def cast_assign(self, sid: str, key: str) -> None:
        """Save one character's library voice on the shared project.

        Args:
            sid: Speaker id.
            key: Combo value, ``library:<id>``.
        """
        studio = self._ensure_studio(sync_file=False, push=False)
        project = studio.project
        speaker = None if project is None else project.speaker(sid)
        if project is None or speaker is None:
            return
        speaker.voice = voice_from_key(key)
        project.settings["multi_voice"] = True
        project.save()
        if self.shell.get("dub.multi_voice") is not True:
            self.shell.set_value("dub.multi_voice", True)
        studio.chars.load(project)

    def _settle_screen(self) -> None:
        """Hide an empty history strip and remember a future video region."""
        if not hasattr(self, "view"):
            return
        self._settle_history()
        self._bind_video_result()

    def _settle_history(self) -> None:
        """Show the history strip only when a finished dub is in it.

        The shell keeps a 104 px strip even when it is empty. An empty strip
        is a blank band, so it stays hidden until there is something to show.
        """
        strip = self.view.root.findChild(QWidget, "historyStrip")
        if strip is None:
            return
        if self.shell.history:
            strip.setFixedHeight(104)
            strip.show()
            return
        strip.hide()
        strip.setFixedHeight(0)

    def _bind_video_result(self) -> None:
        """Remember the shell video result region when the Yitro tag adds it.

        This app does not draw a preview. The region stays the shell's widget.
        """
        self._video_result = self.view.root.findChild(QWidget, VIDEO_RESULT_NAME)

    def _set_expert(self, enabled: bool) -> None:
        """Open or close the side panel, then mark likeness as a percent.

        Args:
            enabled: True shows the expert settings beside the result.
        """
        self.view.set_expert(enabled)
        self._show_likeness_percent()

    def _hide_narrator_voice(self) -> None:
        """Hide the audiobook narrator row on the audio surface.

        The shell always adds a Warm, Clear, or Low combo for this surface
        and has no switch to leave it out. Film voices are the Voices chips,
        the Characters step, and the Expert likeness field.
        """
        for name in ("voiceLabel", "voicePicker"):
            widget = self.view.root.findChild(QWidget, name)
            if widget is not None:
                widget.hide()

    def _show_likeness_percent(self) -> None:
        """Draw the likeness number with a percent sign.

        The shell integer field has no unit. The project still stores the
        share from 0 to 1.
        """
        spin = self.view.root.findChild(QSpinBox, "dub.actor_weight")
        if spin is not None:
            spin.setSuffix("%")

    def _restore_source(self) -> Path | None:
        """Put the chosen film back into the prompt.

        Dub appends the shell voice label to the prompt even when that combo
        is hidden. The film path stays on this window.

        Returns:
            The film, or None when none is chosen.
        """
        path = self._source
        if path is None or not path.is_file() or path.suffix.lower() not in VIDEO_EXTS:
            return None
        self.shell.set_prompt(str(path))
        return path

    def _started(self) -> None:
        path = self._restore_source()
        if path is None:
            self.shell.block(tr("shell.need_file"))
            self.view.refresh()
            return
        studio = self._ensure_studio()
        if studio.project is None:
            self.shell.block(tr("shell.need_file"))
            self.view.refresh()
            return
        if not studio.dub_all() or studio.job is None:
            self.shell.block(tr("shell.dubbing"))
            self.view.refresh()
            return
        self.job = studio.job
        self.project = studio.project
        self._run_id += 1
        run_id = self._run_id
        self._eta = None
        self.job.stage.connect(lambda key, status, message, token=run_id: self._on_stage(token, key, status, message))
        self.job.progress.connect(lambda frac, eta, token=run_id: self._on_progress(token, frac, eta))
        self.job.done.connect(lambda ok, message, token=run_id: self._on_done(token, ok, message))
        self.shell.progress(0, 100, tr("shell.dubbing"))
        self.view.refresh()

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
        if self._studio is not None:
            self._studio.cancel_job()
        if self.job is not None and self.job.isRunning():
            self.job.cancel()

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
        if self._source is not None and self._source.is_file():
            return self._source
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


def _mix_level(value: object) -> float:
    """Return a mix level, or the usual 0.15 when the value is not a number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.15
    return float(value)
