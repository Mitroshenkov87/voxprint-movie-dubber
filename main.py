"""Voxprint AI Movie Dubber - entry point.

    main.py                       start the window
    main.py PROJECT.vxdub         start the window and open that dubbing project
    main.py --diagnose            start the window and run the diagnostics at once (the report goes to the Desktop)
    main.py --diagnose-cli        run the diagnostics without a window (console output), same report
    main.py --worker NAME ARGS    internal: one heavy step in its own process (used by the diagnostics and the pipeline)
    main.py --fetch-models        download the AI models (used by the installer); --models KEY,KEY limits the set
    main.py --register-models-user      (installer) add this program to the shared models folder's .users.json
    main.py --unregister-models-user --out FILE   (uninstaller) remove it; FILE gets "<other users>\n<models folder>"
    main.py --register-runtime-user / --unregister-runtime-user --out FILE   the same for the shared runtime-<key>\.users.json
    main.py --sync-suite-settings (installer) write the shared suite.json (models folder, UI language) if it has none yet
    main.py --run-project DIR [--stages a,b]      run (or resume) a dubbing project without a window (detached long run)
    main.py --selftest            create the window, process events briefly, exit 0 (smoke test)
    main.py --version

Options for the diagnostics:  --out FILE  --quick  --no-download  --target ru|en|de  --skip system,gpu,network,models,tts,stages
--tts-model tts_1_7b|tts_0_6b  --tts-modes standard_sdpa,graphs_sdpa,...  --cpu-tts  --cpu-sep  --clip FILE  --asr-repo REPO
(the Hugging Face token, if needed, is read from the HF_TOKEN environment variable - never from the command line)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _arg(argv, name, default=None):
    if name in argv:
        i = argv.index(name)
        if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            return argv[i + 1]
    return default


def _options_from_argv(argv):
    from dubber.diag.runner import ALL_TTS_MODES, DiagOptions

    o = DiagOptions()
    o.quick = "--quick" in argv
    o.allow_download = "--no-download" not in argv
    o.target_lang = _arg(argv, "--target", "ru")
    o.hf_token = os.environ.get("HF_TOKEN", "")
    o.skip = tuple(x for x in (_arg(argv, "--skip", "") or "").split(",") if x)
    o.tts_model = _arg(argv, "--tts-model", o.tts_model)
    modes = _arg(argv, "--tts-modes")
    o.tts_modes = tuple(m for m in modes.split(",") if m in ALL_TTS_MODES) if modes else ALL_TTS_MODES
    o.cpu_tts, o.cpu_sep = "--cpu-tts" in argv, "--cpu-sep" in argv
    if _arg(argv, "--out"):
        o.out_path = Path(_arg(argv, "--out"))
    if _arg(argv, "--clip"):
        o.clip = Path(_arg(argv, "--clip"))
    o.asr_repo = _arg(argv, "--asr-repo", "") or ""
    return o


def _setup_logging() -> None:
    import logging

    try:
        from dubber import paths

        logging.basicConfig(filename=str(paths.logs_dir() / "dubber.log"), level=logging.INFO, encoding="utf-8",
                            format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    except Exception:  # noqa: BLE001 - logging must never block the start
        pass


def project_file_from_argv(argv) -> str | None:
    """A ``.vxdub`` path passed as the first argument, or ``None``."""
    if not argv or str(argv[0]).startswith("-"):
        return None
    if Path(argv[0]).suffix.lower() == ".vxdub":
        return str(argv[0])
    return None


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    from dubber.infra.stdio_guard import guard_stdio

    guard_stdio()                                            # pythonw.exe: stdout may be None or a dead handle
    if "--worker" in argv:                                   # heavy step in its own process: no Qt, no logging setup
        i = argv.index("--worker")
        from dubber.infra.quiet import mute_library_noise
        from dubber.workers.common import worker_main

        mute_library_noise()

        return worker_main(argv[i + 1], argv[i + 2])
    if "--version" in argv:
        from dubber.appinfo import version_line

        print(version_line())
        return 0
    _setup_logging()
    if "--register-models-user" in argv or "--unregister-models-user" in argv:
        return _models_users(argv)
    if "--register-runtime-user" in argv or "--unregister-runtime-user" in argv:
        from dubber.infra import runtime

        if "--register-runtime-user" in argv:
            runtime.register_user()
            return 0
        return runtime.unregister_cli(_arg(argv, "--out"))
    if "--sync-suite-settings" in argv:
        from dubber.infra import suite

        return suite.sync_cli()
    if "--run-project" in argv:
        from dubber.pipeline.runner import run_project_cli

        return run_project_cli(Path(_arg(argv, "--run-project")), _arg(argv, "--stages", "") or "")
    if "--fetch-models" in argv:
        return _fetch_models(argv)
    if "--diagnose-cli" in argv:
        from dubber.diag.runner import DiagnosticRunner

        def prog(f, t):
            if f >= 0:
                print(f"[{f * 100:5.1f}%] {t}", flush=True)

        runner = DiagnosticRunner(_options_from_argv(argv), on_progress=prog,
                                  on_result=lambda r: print(f"  [{r.status.value:<4}] {r.id}: {r.summary[:120]}", flush=True))
        path = runner.run()
        print(f"Report saved: {path}", flush=True)
        return 0
    from PySide6.QtCore import QTimer
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    from dubber.appinfo import resource_dir

    running = QApplication.instance()
    app = running if isinstance(running, QApplication) else QApplication(sys.argv)
    app.setApplicationName("Voxprint Movie Dubber")
    icon = resource_dir() / "assets" / "voxprint-dubber.ico"
    if icon.is_file():
        app.setWindowIcon(QIcon(str(icon)))
    from dubber.ui import splash as splash_mod

    splash = splash_mod.show()                               # the first thing on screen: before the heavy UI imports below
    splash.loading_ui()
    from dubber.ui.window import MainWindow

    splash.opening()
    win = MainWindow(autorun="--diagnose" in argv, options_hook=(lambda o: _apply_cli(o, argv)) if "--diagnose" in argv else None)
    win.show()
    opened = project_file_from_argv(argv)
    if opened:
        win.open_vxdub(Path(opened))
    splash.finish(win)                                       # closes once the main window is on screen
    if "--selftest" in argv:
        QTimer.singleShot(400, app.quit)
    return app.exec()


def _fetch_models(argv) -> int:
    """Download the models into the app's models folder (resumable: finished models are skipped).  Exit 0 = all present."""
    from dubber import models

    spec = _arg(argv, "--models", "")
    keys = [k for k in spec.split(",") if k] or list(models.INSTALL_MODELS)
    unknown = [k for k in keys if k not in models.SPECS]
    if unknown:
        print(f"Unknown model key(s): {', '.join(unknown)}.  Known: {', '.join(models.SPECS)}", flush=True)
        return 2
    print(f"Downloading AI models (about {models.total_download_gb(keys)} GB, finished ones are skipped) to {models.paths.models_dir()}", flush=True)
    failed = []
    for key in keys:
        s = models.SPECS[key]
        print(f"- {s.title} [{s.repo}] ...", flush=True)
        try:
            models.ensure(s.repo, True, log=lambda m: print("    " + str(m), flush=True))
            print("  done", flush=True)
        except Exception as exc:  # noqa: BLE001 - report and continue with the next model
            failed.append(key)
            print(f"  FAILED: {' '.join(str(exc).split())[:300]}", flush=True)
    print("All models are ready." if not failed else f"Not downloaded: {', '.join(failed)}.  Run this step again or start the diagnostics later.", flush=True)
    return 1 if failed else 0


def _models_users(argv) -> int:
    """Installer / uninstaller helper for the shared ``models/.users.json`` (never fails the setup: exit 0 unless writing failed)."""
    from dubber.infra import model_store, shared_paths

    try:
        model_store.sweep_stale_locks()
        if "--register-models-user" in argv:
            model_store.register_user()
            return 0
        others = model_store.unregister_user()
        out = _arg(argv, "--out")
        if out:
            Path(out).write_text(f"{len(others)}\n{shared_paths.models_dir()}\n", encoding="utf-8")
        print(f"other programs using the models: {', '.join(others) or 'none'}", flush=True)
        return 0
    except OSError as exc:
        print(f"models users file: {exc}", flush=True)
        return 1


def _apply_cli(opt, argv) -> None:
    """Command-line values override the window's check boxes for an auto-started run."""
    cli = _options_from_argv(argv)
    for name in ("out_path", "clip", "tts_model", "tts_modes", "skip", "cpu_tts", "cpu_sep", "asr_repo"):
        setattr(opt, name, getattr(cli, name))
    if "--quick" in argv:
        opt.quick = True
    if "--no-download" in argv:
        opt.allow_download = False
    if "--target" in argv:
        opt.target_lang = cli.target_lang
    opt.hf_token = opt.hf_token or cli.hf_token


if __name__ == "__main__":
    raise SystemExit(main())
