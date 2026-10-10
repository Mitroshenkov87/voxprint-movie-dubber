"""Headless commands for other programs and scripts.

    python main.py version|diagnose|fetch-models|run-project|info ...
    python main.py --dry-run --json

``--json`` prints one JSON object on stdout. Logs and errors go to stderr.
``--dry-run`` only discovers the machine and the job: no model load and no download.
Exit codes are :data:`EXIT_CODES`. See ``docs/CLI.md``.
"""
from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import platform
import signal
import sys
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Never, Sequence

from dubber import models, paths
from dubber import settings as app_settings
from dubber.appinfo import APP_VERSION, app_build, app_codename, version_label, version_line
from dubber.core import vxdub
from dubber.core.project import Project
from dubber.diag.runner import ALL_TTS_MODES, DiagOptions, DiagnosticRunner
from dubber.infra.gpu_policy import gate_skipped, probe_torch, startup_block
from dubber.infra.resources import snapshot
from dubber.infra.vram_tier import detect, install_models
from dubber.pipeline.runner import Callbacks, Runner
from dubber.pipeline.stages import ORDER

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_GPU = 3
EXIT_MODELS = 4
EXIT_INPUT = 5
EXIT_JOB = 6
EXIT_CANCELLED = 7

#: (code, name, when). Code 1 is not used.
EXIT_CODES: tuple[tuple[int, str, str], ...] = (
    (EXIT_OK, "ok", "The command finished."),
    (EXIT_USAGE, "usage", "Unknown command or flag, or a missing value, language, stage, or model key."),
    (EXIT_GPU, "gpu", "No supported NVIDIA GPU (compute capability 8.9 or newer, RTX 40-series or newer)."),
    (EXIT_MODELS, "models", "A model this command needs is not on disk. Dry-run does not download it."),
    (EXIT_INPUT, "input", "The video, project folder, or .vxdub file is missing or cannot be read."),
    (EXIT_JOB, "job", "The dubbing job failed."),
    (EXIT_CANCELLED, "cancelled", "The job was cancelled."),
)

CommandName = Literal["version", "diagnose", "fetch-models", "run-project", "info", "dry-run"]
VIDEO_SUFFIXES = {".mkv", ".mp4", ".avi", ".mov", ".webm", ".m4v", ".ts", ".m2ts"}
SOURCE_LANGS = {"auto", "en", "ru", "de"}
TARGET_LANGS = {"en", "ru", "de"}
_VALUE_FLAGS = {
    "--models", "--stages", "--target", "--source-lang", "--target-lang", "--languages",
    "--out", "--skip", "--tts-model", "--tts-modes", "--clip", "--asr-repo",
    "--run-project", "--project-info",
}
_SWITCHES = {"--json", "--dry-run", "--quick", "--no-download", "--cpu-tts", "--cpu-sep", "--version", "--diagnose-cli", "--fetch-models", "--help", "-h"}
_COMMANDS = {"version", "diagnose", "fetch-models", "run-project", "info", "project-info"}


class UsageError(Exception):
    """The command line is not valid. Exit :data:`EXIT_USAGE`."""


class InputError(Exception):
    """The file or project cannot be used. Exit :data:`EXIT_INPUT`."""


@dataclass
class Request:
    command: CommandName
    json_mode: bool
    dry_run: bool
    path: str = ""
    values: dict[str, str] = field(default_factory=dict)
    switches: set[str] = field(default_factory=set)
    argv: list[str] = field(default_factory=list)


def _help_text() -> str:
    lines = [
        "Voxprint AI Movie Dubber commands (see docs/CLI.md)",
        "",
        "  version",
        "  diagnose",
        "  fetch-models [--models a,b]",
        "  run-project PATH [--languages SRC,TGT] [--source-lang L] [--target-lang L] [--stages a,b]",
        "  info PATH",
        "  --dry-run",
        "",
        "Global: --json  --dry-run  --help",
        "",
        "Exit codes:",
    ]
    for code, name, when in EXIT_CODES:
        lines.append(f"  {code}  {name:<10} {when}")
    return "\n".join(lines)


def parse_args(argv: Sequence[str]) -> Request:
    """Split ``argv`` (no program name) into a command. Raises :class:`UsageError`."""
    values: dict[str, str] = {}
    switches: set[str] = set()
    positionals: list[str] = []
    items = [str(a) for a in argv]
    i = 0
    while i < len(items):
        token = items[i]
        if token in _SWITCHES:
            switches.add(token)
            i += 1
            continue
        if token in _VALUE_FLAGS:
            if i + 1 >= len(items) or items[i + 1].startswith("-"):
                raise UsageError(f"{token} needs a value")
            values[token] = items[i + 1]
            i += 2
            continue
        if token.startswith("-"):
            raise UsageError(f"unknown flag {token}")
        positionals.append(token)
        i += 1
    if "--help" in switches or "-h" in switches:
        raise UsageError("help")
    command, path = _command_from(positionals, values, switches)
    return Request(command, "--json" in switches or "--dry-run" in switches, "--dry-run" in switches,
                   path, values, switches, items)


def _command_from(positionals: list[str], values: dict[str, str], switches: set[str]) -> tuple[CommandName, str]:
    path = ""
    if positionals and positionals[0] in _COMMANDS:
        name = "info" if positionals[0] == "project-info" else positionals[0]
        rest = positionals[1:]
        if name in ("run-project", "info"):
            if not rest:
                raise UsageError(f"{positionals[0]} needs a path")
            if len(rest) > 1:
                raise UsageError("unexpected arguments: " + " ".join(rest[1:]))
            path = rest[0]
        elif rest:
            raise UsageError("unexpected arguments: " + " ".join(rest))
        return _as_command(name), path
    if positionals and not (positionals[0] in _COMMANDS):
        raise UsageError("unexpected arguments: " + " ".join(positionals))
    if values.get("--run-project"):
        return "run-project", values["--run-project"]
    if values.get("--project-info"):
        return "info", values["--project-info"]
    if "--version" in switches:
        return "version", ""
    if "--diagnose-cli" in switches:
        return "diagnose", ""
    if "--fetch-models" in switches:
        return "fetch-models", ""
    if "--dry-run" in switches:
        return "dry-run", ""
    if positionals:
        raise UsageError(f"unknown command {positionals[0]}")
    raise UsageError("give a command (version, diagnose, fetch-models, run-project, info) or --dry-run")


def _as_command(name: str) -> CommandName:
    match name:
        case "version":
            return "version"
        case "diagnose":
            return "diagnose"
        case "fetch-models":
            return "fetch-models"
        case "run-project":
            return "run-project"
        case "info":
            return "info"
        case "dry-run":
            return "dry-run"
        case _:
            raise UsageError(f"unknown command {name}")


def _log(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def _human(text: str) -> None:
    print(text, file=sys.stdout, flush=True)


def _envelope(req: Request, code: int, extra: dict[str, Any] | None = None, error: str = "") -> dict[str, Any]:
    body: dict[str, Any] = {
        "ok": code == EXIT_OK,
        "exit_code": code,
        "command": req.command,
        "dry_run": req.dry_run,
        "version": APP_VERSION,
        "label": version_label(),
        "build": app_build(),
        "codename": app_codename(),
    }
    if error:
        body["error"] = error
    if extra:
        body.update(extra)
    return body


def _finish(req: Request, code: int, extra: dict[str, Any] | None = None, error: str = "", human: str = "") -> int:
    if error and req.json_mode:
        _log(error)
    payload = _envelope(req, code, extra, error)
    if req.json_mode:
        print(json.dumps(payload, ensure_ascii=True), flush=True)
    elif human:
        _human(human)
    elif error:
        _human(error)
    return code


def _torch_version() -> str | None:
    """Installed PyTorch version. Does not import torch (the GPU probe imports it itself)."""
    loaded = sys.modules.get("torch")
    if loaded is not None:
        version = str(getattr(loaded, "__version__", "") or "")
        return version or None
    try:
        return importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return None


def _runtime() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "frozen": bool(getattr(sys, "frozen", False)),
        "torch": _torch_version(),
    }


def _gpu() -> dict[str, Any]:
    """Gate and VRAM tier. Does not load a model. ``VOXPRINT_SKIP_GPU_GATE`` is not applied."""
    available, capability, name = probe_torch()
    block = startup_block(available, capability, name)
    snap = snapshot(use_torch=False)
    known = snap.vram_total_gb > 0
    tier = detect(snap.vram_total_gb)
    compute = None if capability is None else f"{int(capability[0])}.{int(capability[1])}"
    if block is None:
        summary = name.strip() or snap.gpu or "supported NVIDIA GPU"
        code = None
    else:
        code = block.code
        if code == "low_compute":
            summary = f"{block.name or 'NVIDIA GPU'} compute capability {block.capability} is below 8.9"
        elif code == "no_cuda":
            summary = "No supported NVIDIA GPU was found (PyTorch CUDA is not available)."
        else:
            unexpected: Never = code
            summary = str(unexpected)
    return {
        "ok": block is None,
        "code": code,
        "summary": summary,
        "name": name.strip() or snap.gpu,
        "compute": compute,
        "vram_total_gb": snap.vram_total_gb,
        "vram_free_gb": snap.vram_free_gb,
        "vram_known": known,
        "vram_source": snap.source,
        "tier": tier,
    }


def _model_rows() -> list[dict[str, Any]]:
    rows = []
    for spec in models.SPECS.values():
        found = models.locate(spec.repo)
        rows.append({"key": spec.key, "repo": spec.repo, "title": spec.title, "available": found is not None})
    return rows


def _discovery(would: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"runtime": _runtime(), "gpu": _gpu(), "models": _model_rows(), "would_run": would}


def _gpu_code(gpu: dict[str, Any]) -> int | None:
    if gpu.get("ok"):
        return None
    return EXIT_GPU


def _languages(req: Request, source: str = "auto", target: str = "ru") -> tuple[str, str]:
    pair = req.values.get("--languages", "")
    src, tgt = source, target
    if pair:
        parts = [p.strip().lower() for p in pair.split(",")]
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise UsageError("--languages needs SRC,TGT (for example en,ru)")
        src, tgt = parts
    if "--source-lang" in req.values:
        src = req.values["--source-lang"].strip().lower()
    if "--target-lang" in req.values:
        tgt = req.values["--target-lang"].strip().lower()
    if "--target" in req.values and "--target-lang" not in req.values and not pair:
        tgt = req.values["--target"].strip().lower()
    if src not in SOURCE_LANGS:
        raise UsageError(f"unknown source language {src}")
    if tgt not in TARGET_LANGS:
        raise UsageError(f"unknown target language {tgt}")
    return src, tgt


def _stages(req: Request) -> list[str]:
    raw = req.values.get("--stages", "")
    if not raw:
        return list(ORDER)
    names = [part.strip() for part in raw.split(",") if part.strip()]
    if not names:
        raise UsageError("--stages needs a stage name")
    unknown = [name for name in names if name not in ORDER]
    if unknown:
        raise UsageError(f"unknown stage(s): {', '.join(unknown)}. Known: {', '.join(ORDER)}")
    last = names[-1]
    return ORDER[: ORDER.index(last) + 1]


def _setup_logging() -> None:
    try:
        logging.basicConfig(filename=str(paths.logs_dir() / "dubber.log"), level=logging.INFO, encoding="utf-8",
                            format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    except Exception:  # noqa: BLE001 - logging must never block the command
        pass


def _cmd_version(req: Request) -> int:
    extra = _discovery() if req.dry_run else None
    if req.dry_run and extra is not None:
        gpu_exit = _gpu_code(extra["gpu"])
        if gpu_exit is not None:
            return _finish(req, gpu_exit, extra, error=str(extra["gpu"]["summary"]))
    return _finish(req, EXIT_OK, extra, human=version_line())


def _cmd_dry(req: Request) -> int:
    extra = _discovery({
        "kind": "discovery",
        "stages": list(ORDER),
        "note": "No project was given. Dry-run does not start a dub and does not load models.",
    })
    gpu_exit = _gpu_code(extra["gpu"])
    if gpu_exit is not None:
        return _finish(req, gpu_exit, extra, error=str(extra["gpu"]["summary"]))
    return _finish(req, EXIT_OK, extra, human=version_line())


def _requested_models(req: Request) -> list[str]:
    spec = req.values.get("--models", "")
    keys = [key for key in spec.split(",") if key]
    if keys:
        unknown = [key for key in keys if key not in models.SPECS]
        if unknown:
            known = ", ".join(models.SPECS)
            raise UsageError(f"Unknown model key(s): {', '.join(unknown)}.  Known: {known}")
        return keys
    vram = snapshot(use_torch=False).vram_total_gb
    return install_models(vram, models.INSTALL_MODELS)


def _cmd_fetch(req: Request) -> int:
    keys = _requested_models(req)
    rows = _model_rows()
    by_key = {row["key"]: row for row in rows}
    chosen = [by_key[key] for key in keys if key in by_key]
    would = {"kind": "fetch-models", "models": keys}
    if req.dry_run:
        extra = _discovery(would)
        gpu = extra["gpu"]
        gpu_exit = _gpu_code(gpu)
        if gpu_exit is not None:
            return _finish(req, gpu_exit, extra, error=str(gpu["summary"]))
        missing = [row["key"] for row in chosen if not row["available"]]
        if missing:
            return _finish(req, EXIT_MODELS, extra, error="missing models: " + ", ".join(missing))
        return _finish(req, EXIT_OK, extra, human="models present: " + ", ".join(keys))
    return _fetch_for_real(req, keys, chosen)


def _fetch_for_real(req: Request, keys: list[str], chosen: list[dict[str, Any]]) -> int:
    _setup_logging()
    failed: list[str] = []
    notes: list[str] = []

    def emit(text: str) -> None:
        notes.append(text)
        _log(text) if req.json_mode else _human(text)

    emit(f"Downloading AI models (about {models.total_download_gb(keys)} GB, finished ones are skipped) to {models.paths.models_dir()}")
    for key in keys:
        spec = models.SPECS[key]
        emit(f"- {spec.title} [{spec.repo}] ...")
        try:
            models.ensure(spec.repo, True, log=lambda message: emit("    " + str(message)))
            emit("  done")
        except Exception as exc:  # noqa: BLE001 - report and continue with the next model
            failed.append(key)
            emit(f"  FAILED: {' '.join(str(exc).split())[:300]}")
    if failed:
        message = f"Not downloaded: {', '.join(failed)}.  Run this step again or start the diagnostics later."
        return _finish(req, EXIT_MODELS, {"models": chosen, "failed": failed, "log": notes}, error=message)
    return _finish(req, EXIT_OK, {"models": chosen, "failed": [], "log": notes}, human="All models are ready.")


def _classify_path(path: Path) -> str:
    if path.suffix.lower() == ".vxdub":
        return "vxdub"
    if path.is_dir():
        return "project"
    if path.suffix.lower() in VIDEO_SUFFIXES:
        return "video"
    raise InputError(f"not a video, a project folder, or a .vxdub file: {path}")


def _describe_input(req: Request) -> dict[str, Any]:
    """What ``run-project`` or ``info`` would use. Does not create a project folder."""
    if not req.path:
        raise UsageError("a path is required")
    path = Path(req.path)
    if not path.exists():
        raise InputError(f"not found: {path}")
    kind = _classify_path(path)
    source, target = "auto", "ru"
    lines = 0
    if kind == "vxdub":
        try:
            doc = vxdub.read(path)
        except vxdub.VxdubError as exc:
            raise InputError(str(exc)) from exc
        manifest = doc.manifest
        source = str(manifest.get("source_lang") or source)
        target = str(manifest.get("target_lang") or target)
        transcript = doc.transcript.get("lines") if isinstance(doc.transcript, dict) else None
        lines = len(transcript) if isinstance(transcript, list) else 0
    elif kind == "project":
        if not (path / "project.json").is_file():
            raise InputError(f"not a project folder: {path}")
        project = Project(path)
        source = str(project.settings.get("source_lang") or source)
        target = str(project.settings.get("target_lang") or target)
        lines = len(project.lines)
    source, target = _languages(req, source, target)
    stages = _stages(req) if req.command == "run-project" or req.command == "dry-run" else list(ORDER)
    return {
        "kind": kind,
        "path": str(path),
        "source_lang": source,
        "target_lang": target,
        "lines": lines,
        "stages": stages,
    }


def _needed_models(source: str, target: str, total_gb: float) -> list[str]:
    keys = list(install_models(total_gb, models.INSTALL_MODELS))
    if source != "auto":
        spec = models.mt_spec(source, target)
        if spec is not None and spec.key not in keys:
            keys.append(spec.key)
    return keys


def _missing(keys: list[str], rows: list[dict[str, Any]]) -> list[str]:
    have = {row["key"]: row["available"] for row in rows}
    return [key for key in keys if not have.get(key, False)]


def _cmd_info(req: Request) -> int:
    try:
        described = _describe_input(req)
    except UsageError:
        raise
    except InputError as exc:
        extra = _discovery() if req.dry_run else None
        return _finish(req, EXIT_INPUT, extra, error=str(exc))
    project = {"project": described}
    if not req.dry_run:
        return _finish(req, EXIT_OK, project, human=_info_line(described))
    extra = _discovery(described)
    extra.update(project)
    gpu_exit = _gpu_code(extra["gpu"])
    if gpu_exit is not None:
        return _finish(req, gpu_exit, extra, error=str(extra["gpu"]["summary"]))
    return _finish(req, EXIT_OK, extra, human=_info_line(described))


def _info_line(described: dict[str, Any]) -> str:
    return (f"{described['kind']}: {described['path']}  {described['source_lang']} -> {described['target_lang']}"
            f"  lines {described['lines']}")


def _cmd_run(req: Request) -> int:
    try:
        described = _describe_input(req)
    except UsageError:
        raise
    except InputError as exc:
        extra = _discovery() if req.dry_run else None
        return _finish(req, EXIT_INPUT, extra, error=str(exc))
    if req.dry_run:
        extra = _discovery(described)
        vram = float(extra["gpu"].get("vram_total_gb") or 0.0)
        needed = _needed_models(described["source_lang"], described["target_lang"], vram)
        described = {**described, "models": needed}
        extra["would_run"] = described
        gpu_exit = _gpu_code(extra["gpu"])
        if gpu_exit is not None:
            return _finish(req, gpu_exit, extra, error=str(extra["gpu"]["summary"]))
        missing = _missing(needed, extra["models"])
        if missing:
            return _finish(req, EXIT_MODELS, extra, error="missing models: " + ", ".join(missing))
        return _finish(req, EXIT_OK, extra, human="would run " + ", ".join(described["stages"]))
    gpu = _gpu()
    if not gpu["ok"] and not gate_skipped():
        return _finish(req, EXIT_GPU, {"gpu": gpu, "would_run": described}, error=str(gpu["summary"]))
    return _run_for_real(req, described)


def _open_for_run(described: dict[str, Any]) -> Project:
    path = Path(described["path"])
    kind = described["kind"]
    if kind == "project":
        return Project(path)
    if kind == "vxdub":
        try:
            project, _doc = vxdub.open_project(path, paths.projects_dir())
        except vxdub.VxdubError as exc:
            raise InputError(str(exc)) from exc
        return project
    folder = Project.folder_for(path, paths.projects_dir())
    if (folder / "project.json").is_file():
        project = Project(folder)
    else:
        project = Project.create(folder, path)
    project.settings["source_lang"] = described["source_lang"]
    project.settings["target_lang"] = described["target_lang"]
    project.save()
    return project


def _job_code(ok: bool, message: str) -> int:
    if ok:
        return EXIT_OK
    if message.strip().lower() == "cancelled":
        return EXIT_CANCELLED
    return EXIT_JOB


def _run_for_real(req: Request, described: dict[str, Any]) -> int:
    _setup_logging()
    try:
        project = _open_for_run(described)
    except InputError as exc:
        return _finish(req, EXIT_INPUT, {"would_run": described}, error=str(exc))
    cancel = threading.Event()

    def _stop(_signum: int, _frame: object) -> None:
        cancel.set()

    previous = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _stop)
    logs: list[str] = []

    def stage(key: str, status: str, message: str) -> None:
        line = f"[{status:>7}] {key}: {message}"
        logs.append(line)
        _log(line) if req.json_mode else _human(line)

    def log(text: str) -> None:
        logs.append(text)
        _log(text) if req.json_mode else _human("    " + text)

    try:
        result = Runner(project, app_settings.engine_cfg(), Callbacks(stage=stage, log=log), cancel).run(
            until_stage=described["stages"][-1])
    except KeyboardInterrupt:
        return _finish(req, EXIT_CANCELLED, {"would_run": described, "log": logs}, error="cancelled")
    finally:
        signal.signal(signal.SIGINT, previous)
    code = _job_code(result.ok, result.message)
    summary = ("done: " + result.message) if result.ok else ("FAILED: " + result.message)
    if not req.json_mode:
        _human(summary)
        return code
    error = "" if result.ok else result.message or "job failed"
    return _finish(req, code, {"would_run": described, "message": result.message, "stages": result.stages, "log": logs},
                   error=error)


def _diag_options(req: Request) -> DiagOptions:
    options = DiagOptions()
    options.quick = "--quick" in req.switches
    options.allow_download = "--no-download" not in req.switches
    if "--target" in req.values:
        target = req.values["--target"].strip().lower()
        if target not in TARGET_LANGS:
            raise UsageError(f"unknown target language {target}")
        options.target_lang = target
    options.hf_token = os.environ.get("HF_TOKEN", "")
    options.skip = tuple(part for part in req.values.get("--skip", "").split(",") if part)
    if "--tts-model" in req.values:
        options.tts_model = req.values["--tts-model"]
    modes = req.values.get("--tts-modes")
    options.tts_modes = tuple(mode for mode in modes.split(",") if mode in ALL_TTS_MODES) if modes else ALL_TTS_MODES
    options.cpu_tts = "--cpu-tts" in req.switches
    options.cpu_sep = "--cpu-sep" in req.switches
    if "--out" in req.values:
        options.out_path = Path(req.values["--out"])
    if "--clip" in req.values:
        options.clip = Path(req.values["--clip"])
    options.asr_repo = req.values.get("--asr-repo", "") or ""
    return options


def _cmd_diagnose(req: Request) -> int:
    would = {"kind": "diagnose", "checks": ["system", "gpu", "models", "tts", "stages"]}
    if req.dry_run:
        extra = _discovery(would)
        gpu_exit = _gpu_code(extra["gpu"])
        if gpu_exit is not None:
            return _finish(req, gpu_exit, extra, error=str(extra["gpu"]["summary"]))
        missing = [row["key"] for row in extra["models"] if not row["available"]]
        if missing:
            return _finish(req, EXIT_MODELS, extra, error="missing models: " + ", ".join(missing))
        return _finish(req, EXIT_OK, extra, human="diagnose would run: " + ", ".join(would["checks"]))
    gpu = _gpu()
    if not gpu["ok"] and not gate_skipped():
        return _finish(req, EXIT_GPU, {"gpu": gpu, "would_run": would}, error=str(gpu["summary"]))
    return _diagnose_for_real(req)


def _diagnose_for_real(req: Request) -> int:
    _setup_logging()
    options = _diag_options(req)

    def prog(frac: float, title: str) -> None:
        if frac >= 0:
            line = f"[{frac * 100:5.1f}%] {title}"
            _log(line) if req.json_mode else _human(line)

    def on_result(result: Any) -> None:
        line = f"  [{result.status.value:<4}] {result.id}: {result.summary[:120]}"
        _log(line) if req.json_mode else _human(line)

    try:
        runner = DiagnosticRunner(options, on_progress=prog, on_result=on_result)
        path = runner.run()
    except KeyboardInterrupt:
        return _finish(req, EXIT_CANCELLED, error="cancelled")
    message = f"Report saved: {path}"
    if not req.json_mode:
        _human(message)
    return _finish(req, EXIT_OK, {"report": str(path)}, human="")


def _dispatch(req: Request) -> int:
    command = req.command
    match command:
        case "version":
            return _cmd_version(req)
        case "diagnose":
            return _cmd_diagnose(req)
        case "fetch-models":
            return _cmd_fetch(req)
        case "run-project":
            return _cmd_run(req)
        case "info":
            return _cmd_info(req)
        case "dry-run":
            return _cmd_dry(req)
        case _ as unexpected:
            never: Never = unexpected
            raise AssertionError(never)


def _usage_finish(json_mode: bool, text: str, dry_run: bool = False) -> int:
    if json_mode:
        _log(text)
        print(json.dumps({
            "ok": False,
            "exit_code": EXIT_USAGE,
            "command": "usage",
            "dry_run": dry_run,
            "error": text,
            "version": APP_VERSION,
            "label": version_label(),
            "build": app_build(),
            "codename": app_codename(),
        }, ensure_ascii=True), flush=True)
    else:
        _human(text)
    return EXIT_USAGE


def _help_finish(json_mode: bool, dry_run: bool = False) -> int:
    if not json_mode:
        _human(_help_text())
        return EXIT_OK
    print(json.dumps({
        "ok": True,
        "exit_code": EXIT_OK,
        "command": "help",
        "dry_run": dry_run,
        "version": APP_VERSION,
        "label": version_label(),
        "build": app_build(),
        "codename": app_codename(),
        "commands": ["version", "diagnose", "fetch-models", "run-project", "info"],
        "exit_codes": [{"code": code, "name": name, "when": when} for code, name, when in EXIT_CODES],
    }, ensure_ascii=True), flush=True)
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Run one headless command. ``argv`` is the argument list without the program name."""
    items = list(sys.argv[1:] if argv is None else argv)
    json_mode = "--json" in items or "--dry-run" in items
    try:
        dry_run = "--dry-run" in items
        if "--help" in items or "-h" in items:
            return _help_finish(json_mode, dry_run)
        req = parse_args(items)
    except UsageError as exc:
        if str(exc) == "help":
            return _help_finish(json_mode, "--dry-run" in items)
        return _usage_finish(json_mode, str(exc), "--dry-run" in items)
    try:
        return _dispatch(req)
    except UsageError as exc:
        return _finish(req, EXIT_USAGE, error=str(exc))
    except InputError as exc:
        return _finish(req, EXIT_INPUT, error=str(exc))
    except KeyboardInterrupt:
        return _finish(req, EXIT_CANCELLED, error="cancelled")
    except Exception as exc:  # noqa: BLE001 - one JSON object, the traceback stays on stderr
        traceback.print_exc(file=sys.stderr)
        return _finish(req, EXIT_JOB, error=f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
