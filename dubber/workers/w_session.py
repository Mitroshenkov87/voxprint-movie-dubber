"""Worker ``session``: several model stages, one after another, in this process.

The parent writes one JSON command per line on stdin (``{"stage", "project", "cfg"}``) and ``{"cmd": "quit"}`` to stop.
Each stage answers with the usual ``@@VX@@`` result line, including a ``stage`` field, so the parent can keep the
process.  Models stay in :mod:`dubber.infra.resident` until VRAM pressure or the quit.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Tuple

from dubber.workers.common import WorkerContext, _emit


def _resolved_device(cfg: Dict[str, Any]) -> str:
    from dubber.workers.common import torch_device

    device = str(cfg.get("device") or "auto")
    if device == "auto":
        device = torch_device("auto")
    return device if device in ("cuda", "cpu") else "cpu"


def _install_loaders(cfg: Dict[str, Any]) -> None:
    """Register raw constructors.  They must not call ``resident.slot`` (that would recurse).

    Translation and speech are not registered here.  A stand-in object would occupy the slot the real engine
    builds, and the language pair is not known until that stage runs.  Prefetch only notes those names."""
    if not cfg.get("allow_download", True):
        return
    from dubber.infra import resident

    device = _resolved_device(cfg)
    if cfg.get("separation") == "tiger":
        def separation() -> Any:
            from dubber.engines.separation import build_tiger

            return build_tiger(device, True, lambda _m: None)

        resident.register_loader("separation", separation)
    if cfg.get("asr") == "whisper":
        def asr() -> Any:
            from dubber import models
            from dubber.engines.asr import load_whisper
            from dubber.pipeline.stages import DEFAULT_CFG

            repo = str(cfg.get("asr_repo") or DEFAULT_CFG["asr_repo"])
            folder, _ = models.ensure(repo, True, log=lambda _m: None)
            return load_whisper(str(folder), device)

        resident.register_loader("asr", asr)


def _run_one(folder: str, key: str, cfg: Dict[str, Any], ctx: WorkerContext) -> Tuple[str, Dict[str, Any]]:
    from dubber.core.project import Project
    from dubber.diag.procs import GpuSampler
    from dubber.pipeline import stages
    from dubber.pipeline.prefetch import overlap, start_model

    sampler = GpuSampler()
    sampler.start()

    def emit(kind: str, **kw: Any) -> None:
        ctx.log(json.dumps({"kind": kind, **kw}, ensure_ascii=False))

    def gpu() -> str:
        project = Project(Path(folder))
        summary = stages.FUNCS[key](project, {**stages.DEFAULT_CFG, **cfg}, emit)
        project.save()
        return summary

    def cpu() -> None:
        start_model(key)

    try:
        summary = overlap(gpu, cpu)
    finally:
        sampler.stop()
    return summary, sampler.summary()


def run(args: Dict[str, Any], ctx: WorkerContext) -> Dict[str, Any]:
    """Run JSON stage commands from stdin in this process until ``quit``, keeping loaded models between stages."""
    from dubber.infra import resident

    resident.enable()
    resident.reset()
    base = dict(args.get("cfg") or {})
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        try:
            cmd = json.loads(line)
        except ValueError:
            continue
        if cmd.get("cmd") == "quit":
            break
        key = str(cmd.get("stage") or "")
        folder = str(cmd.get("project") or args.get("project") or "")
        cfg = {**base, **dict(cmd.get("cfg") or {})}
        _install_loaders(cfg)
        try:
            summary, gpu = _run_one(folder, key, cfg, ctx)
            _emit({"t": "result", "status": "OK", "summary": summary, "stage": key, "gpu": gpu})
        except Exception as exc:  # noqa: BLE001 - one stage fails, the process stays up for the next command
            _emit({"t": "result", "status": "FAIL", "summary": f"{type(exc).__name__}: {exc}", "stage": key,
                   "traceback": traceback.format_exc()})
    resident.reset()
    return {"status": "OK", "summary": "session closed"}
