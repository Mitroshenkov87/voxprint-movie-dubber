"""Worker ``fetch``: make sure the models are on disk (download when allowed) and report where each came from.

Args: ``keys`` (model keys from ``dubber.models.SPECS``), ``allow_download``, ``repos`` (optional {key: repo} overrides).
Separated from the model checks so that download time never pollutes load-time numbers.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

from dubber import models
from dubber.workers.common import WorkerContext


def run(args: Dict[str, Any], ctx: WorkerContext) -> Dict[str, Any]:
    lines: List[str] = []
    bad, ok_n, dl_total = [], 0, 0.0
    overrides = args.get("repos") or {}
    results = {}
    for key in args.get("keys", []):
        spec = models.SPECS.get(key)
        repo = overrides.get(key) or (spec.repo if spec else key)
        title = spec.title if spec else repo
        t = time.time()
        try:
            ctx.log(f"models: checking {repo}")
            folder, info = models.ensure(repo, args.get("allow_download", True), log=ctx.log)
            size = models._dir_size(folder) / 1024 ** 3
            lines.append(f"{title:<46} OK    [{repo}] {info['source']}" + (f", downloaded in {info['download_s']} s" if info["download_s"] else "") + f"  ({size:.2f} GB)")
            ok_n += 1
            dl_total += float(info["download_s"])
            results[key] = {"source": info["source"], "download_s": info["download_s"], "gb": round(size, 2)}
        except models.ModelUnavailable as exc:
            lines.append(f"{title:<46} MISSING  [{repo}] {' '.join(str(exc).split())[:220]}")
            bad.append(key)
            results[key] = {"error": str(exc)}
    status = "OK" if not bad else "WARN"
    summary = f"{ok_n} present, {len(bad)} missing" + (f" ({', '.join(bad)})" if bad else "") + (f"; downloaded in {dl_total:.0f} s" if dl_total else "")
    return {"status": status, "summary": summary, "details": lines, "metrics": {"models": results, "download_s": dl_total, "missing": bad}}
