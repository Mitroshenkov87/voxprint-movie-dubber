"""Progress dialog for the shared-component check that runs before the window opens."""
from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QApplication, QProgressDialog

from dubber.infra.shared_deps import Report, live_selftest


def ensure_with_dialog() -> Report:
    """Check the shared runtime, wheels, ffmpeg and models, and repair a missing piece.

    The dialog stays hidden on a healthy launch and appears while a piece is being repaired or has failed.
    The window path already refused an unsupported GPU, so this check does not repeat that gate.
    """
    app = QApplication.instance()
    dialog = QProgressDialog("Checking shared components...", "", 0, 0)
    dialog.setWindowTitle("Voxprint AI Movie Dubber")
    dialog.setCancelButton(None)
    dialog.setMinimumDuration(0)
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)
    shown = False

    def progress(event: dict[str, Any]) -> None:
        nonlocal shown
        status = str(event.get("status") or "")
        message = str(event.get("message") or "Checking shared components...")
        if status in {"repairing", "failed"} or shown:
            shown = True
            dialog.setLabelText(message)
            if not dialog.isVisible():
                dialog.show()
            if app is not None:
                app.processEvents()

    try:
        return live_selftest(progress, repair=True, check_gpu=False)
    finally:
        dialog.close()
