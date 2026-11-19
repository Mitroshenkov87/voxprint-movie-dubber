"""Progress dialog for the shared-component check that runs before the window opens."""
from __future__ import annotations

import subprocess
from typing import Any

from PySide6.QtWidgets import QApplication, QProgressDialog

from dubber import models as model_registry
from dubber.appinfo import APP_DISPLAY_NAME
from dubber.i18n import tr
from dubber.infra.shared_deps import Report, live_selftest, missing_optional_models


def startup_repair_dialog(report: Report) -> tuple[str, str] | None:
    """Title and body for a failed startup repair. A healthy report returns ``None``.

    The title is the model-download title. It is never the graphics-card gate title.
    """
    if report.exit_code == 0:
        return None
    title = tr("models.repair_title")
    body = tr("models.repair_body")
    if report.message:
        body = f"{body}\n\n{report.message}"
    return title, body


def fetch_optional_with_dialog() -> list[str]:
    """Ask VOXPrint MDL for the speaker models and return the keys still missing.

    The dialog is not modal. A failure leaves the window open. This does not
    download inside the process.
    """
    app = QApplication.instance()
    dialog = QProgressDialog("Checking multi-voice models...", "", 0, 0)
    dialog.setWindowTitle(tr("models.repair_title"))
    dialog.setCancelButton(None)
    dialog.setMinimumDuration(0)
    dialog.setModal(False)
    dialog.show()
    try:
        ids = [model_registry.mdl_id(key) for key in model_registry.OPTIONAL_MODELS]
        env = model_registry.mdl_env()
        dialog.setLabelText("voxprint-mdl enqueue --app movie-dubber")
        if app is not None:
            app.processEvents()
        subprocess.run(model_registry.mdl_argv("enqueue", model_ids=ids), env=env, check=False)
        dialog.setLabelText("voxprint-mdl status --wait")
        if app is not None:
            app.processEvents()
        subprocess.run(model_registry.mdl_argv("status", "--wait"), env=env, check=False)
    finally:
        dialog.close()
    return missing_optional_models()


def ensure_with_dialog() -> Report:
    """Check the shared runtime, wheels, ffmpeg and models, and repair a missing piece.

    The dialog stays hidden on a healthy launch and appears while a piece is being repaired or has failed.
    The window path already refused an unsupported GPU, so this check does not repeat that gate.
    """
    app = QApplication.instance()
    dialog = QProgressDialog("Checking shared components...", "", 0, 0)
    dialog.setWindowTitle(APP_DISPLAY_NAME)
    dialog.setCancelButton(None)
    dialog.setMinimumDuration(0)
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)
    shown = False

    def progress(event: dict[str, Any]) -> None:
        nonlocal shown
        if str(event.get("resource") or "") == "models":
            return
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
