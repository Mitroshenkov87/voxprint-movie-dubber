"""Background jobs of the window: the dubbing pipeline (and the preview fragment) in a QThread around the Qt-free runner.

Model stages still run as separate worker processes (see ``dubber.pipeline.runner``); this thread only waits for them and turns
the runner's callbacks into Qt signals.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, Optional

from PySide6.QtCore import QThread, Signal

from dubber.core.project import Project
from dubber.pipeline import runner as R


class PipelineThread(QThread):
    """Run the dubbing pipeline, or a preview fragment, on a worker thread.

    Args:
        folder: Project folder the runner reads and writes.
        cfg: Engine settings passed to the runner.
        until_stage: Last pipeline stage to run.
        preview: Film time where a preview fragment starts, or None for a full run.
        preview_len: Length of that fragment in seconds.
    """
    stage = Signal(str, str, str)          # key, status, message
    log = Signal(str)
    progress = Signal(float, float)        # overall 0..1, ETA seconds (-1 unknown)
    until = Signal(float)                  # Watch mode: dub final up to here
    done = Signal(bool, str)               # ok, output file / error

    def __init__(self, folder: Path, cfg: Dict[str, Any], until_stage: str = "mux", preview: Optional[float] = None,
                 preview_len: float = 60.0) -> None:
        super().__init__()
        self.folder, self.cfg, self.until_stage = Path(folder), dict(cfg), until_stage
        self.preview, self.preview_len = preview, preview_len
        self.cancel_event = threading.Event()
        self.result_folder = self.folder

    def run(self) -> None:
        """Run the pipeline to the chosen stage and emit whether it finished."""
        try:
            p = Project(self.folder)
            if self.preview is not None:
                p = R.preview_project(p, self.preview, self.preview_len)
                self.result_folder = p.folder
            cb = R.Callbacks(stage=lambda k, s, m: self.stage.emit(k, s, m), log=self.log.emit,
                             progress=lambda f, e: self.progress.emit(f, e), dubbed_until=self.until.emit)
            res = R.Runner(p, self.cfg, cb, self.cancel_event).run(until_stage=self.until_stage)
            self.done.emit(res.ok, res.message)
        except Exception as exc:  # noqa: BLE001 - last safety net: the window shows the message
            self.done.emit(False, f"{type(exc).__name__}: {exc}")

    def cancel(self) -> None:
        """Ask the pipeline to stop."""
        self.cancel_event.set()
