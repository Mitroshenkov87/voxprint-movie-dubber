"""Worker ``fetch``: report models that are already on disk. It does not download.

Args: ``keys`` (model keys from ``dubber.models.SPECS``), ``repos`` (optional {key: repo} overrides).
``allow_download`` is accepted and ignored.
"""
from __future__ import annotations

from typing import Any, Dict, List

from dubber import models
from dubber.infra import model_store
from dubber.workers.common import WorkerContext


def run(args: Dict[str, Any], ctx: WorkerContext) -> Dict[str, Any]:
    """Ensure each requested model is on disk and report its source, size, and download time."""
    lines: List[str] = []
    bad, ok_n, dl_total = [], 0, 0.0
    overrides = args.get("repos") or {}
    results = {}
    for key in args.get("keys", []):
        spec = models.SPECS.get(key)
        repo = overrides.get(key) or (spec.repo if spec else key)
        title = spec.title if spec else repo
        try:
            ctx.log(f"models: checking {repo}")
            folder, info = models.ensure(repo, args.get("allow_download", True), log=ctx.log)
            size = model_store.folder_bytes(folder) / 1024 ** 3
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
