"""Worker ``stage``: one model stage of a dubbing project in its own process (see ``dubber.pipeline.stages``).

Args: ``project`` (folder), ``stage`` (key), ``cfg`` (engine settings).  Progress goes to the parent as JSON log lines
(``{"kind": "log|progress|until", ...}``); the stage saves the project files itself before the result line.
"""
from __future__ import annotations

import json
from typing import Any, Dict

from dubber.workers.common import WorkerContext


def run(args: Dict[str, Any], ctx: WorkerContext) -> Dict[str, Any]:
    from dubber.core.project import Project
    from dubber.pipeline import stages

    p = Project(args["project"])
    key = args["stage"]

    def emit(kind: str, **kw: Any) -> None:
        ctx.log(json.dumps({"kind": kind, **kw}, ensure_ascii=False))

    summary = stages.FUNCS[key](p, {**stages.DEFAULT_CFG, **(args.get("cfg") or {})}, emit)
    p.save()
    return {"status": "OK", "summary": summary}
