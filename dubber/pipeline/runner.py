"""Runs a project through the stages: cache check, one worker process per model stage, the shared GPU lock, progress + ETA,
Watch-mode chunks while the dub is being made, and the 1-minute preview fragment.

The runner is Qt-free; the window drives it from a QThread and gets callbacks.  ``main.py --run-project DIR`` runs it without a
window (a detached process for very long jobs: closing the window does not stop it, a crash of a model stage does not kill it).
"""
from __future__ import annotations

import json
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from dubber.core import audio, mixing
from dubber.core.project import CHUNK_S, Line, Project, read_json, write_json
from dubber.core.watch import Eta
from dubber.pipeline import stages as S

#: rough share of the total time per stage (for the overall progress bar)
WEIGHTS = {"probe": 0.2, "extract": 1, "subtitles": 0.5, "vad": 1, "separation": 8, "asr": 6, "script": 0.2, "diarization": 3,
           "translation": 2, "voices": 0.3, "tts": 70, "mix": 4, "mux": 3}
TIMEOUT_PER_FILM_S = {"tts": 6.0, "separation": 2.0, "asr": 2.0, "diarization": 1.5, "translation": 0.5, "vad": 0.3}


class Cancelled(Exception):
    pass


@dataclass
class Callbacks:
    stage: Callable[[str, str, str], None] = lambda key, status, msg: None        # status: running | done | cached | skipped | failed
    log: Callable[[str], None] = lambda text: None
    progress: Callable[[float, float], None] = lambda frac, eta_s: None           # overall 0..1, ETA seconds (-1 unknown)
    dubbed_until: Callable[[float], None] = lambda seconds: None                 # Watch mode: final dub available up to here


@dataclass
class RunResult:
    ok: bool
    message: str = ""
    stages: Dict[str, str] = field(default_factory=dict)


class Runner:
    def __init__(self, project: Project, cfg: Optional[Dict[str, Any]] = None, cb: Optional[Callbacks] = None,
                 cancel: Optional[threading.Event] = None) -> None:
        self.p = project
        self.cfg = {**S.DEFAULT_CFG, **(cfg or {})}
        self.cb = cb or Callbacks()
        self.cancel = cancel or threading.Event()
        self.eta = Eta()
        self._done_w = 0.0
        self._watch_written: set = set()

    # ------------------------------------------------------------------ public
    def run(self, until_stage: str = "mux", from_stage: Optional[str] = None) -> RunResult:
        keys = S.ORDER[: S.ORDER.index(until_stage) + 1]
        total_w = sum(WEIGHTS[k] for k in keys)
        prev = ""
        res = RunResult(True)
        t_start = time.time()
        for key in keys:
            if self.cancel.is_set():
                return RunResult(False, "cancelled", res.stages)
            inputs = S.inputs_for(key, self.p, self.cfg, prev)
            prev = inputs
            if self.p.stage_fresh(key, inputs) and key != from_stage:
                self.cb.stage(key, "cached", self.p.stages[key].get("summary", ""))
                res.stages[key] = "cached"
                if key == "tts":
                    self.cb.dubbed_until(float(self.p.settings.get("duration") or 0))
                self._advance(WEIGHTS[key], total_w, t_start)
                continue
            self.cb.stage(key, "running", "")
            t0 = time.time()
            try:
                summary = self._run_stage(key, lambda frac: self._report(frac, WEIGHTS[key], total_w, t_start))
            except Cancelled:
                self.cb.stage(key, "failed", "cancelled")
                return RunResult(False, "cancelled", res.stages)
            except Exception as exc:  # noqa: BLE001 - one stage failing ends the run with a clear message
                msg = f"{type(exc).__name__}: {exc}"
                self.cb.stage(key, "failed", msg)
                res.stages[key] = "failed"
                return RunResult(False, f"{key}: {msg}", res.stages)
            if self._isolated(key):
                self.p = Project(self.p.folder)               # the worker saved its results
            self.p.mark_done(key, inputs, time.time() - t0, summary=summary)
            self.cb.stage(key, "done", summary)
            res.stages[key] = "done"
            self._advance(WEIGHTS[key], total_w, t_start)
        self.cb.progress(1.0, 0.0)
        return RunResult(True, self.p.settings.get("output_file", ""), res.stages)

    # ------------------------------------------------------------------ progress
    def _advance(self, w: float, total_w: float, t_start: float) -> None:
        self._done_w += w
        self._report(0.0, 0.0, total_w, t_start)

    def _report(self, stage_frac: float, w: float, total_w: float, t_start: float) -> None:
        frac = min(1.0, (self._done_w + w * stage_frac) / max(total_w, 1e-6))
        eta = self.eta.update(time.time(), frac, 1.0)
        self.cb.progress(frac, eta)

    # ------------------------------------------------------------------ stage execution
    def _isolated(self, key: str) -> bool:
        return key in S.HEAVY and not self.cfg.get("inprocess")

    def _run_stage(self, key: str, on_frac: Callable[[float], None]) -> str:
        def emit(kind: str, **kw: Any) -> None:
            if kind == "log":
                self.cb.log(str(kw.get("text", "")))
            elif kind == "progress":
                on_frac(float(kw.get("value", 0.0)))
            elif kind == "until":
                self._on_until(float(kw["seconds"]))
            if self.cancel.is_set() and not self._isolated(key):
                raise Cancelled()

        if key == "tts":
            shutil.rmtree(self.p.folder / "watch", ignore_errors=True)
            self._watch_written = set()
        if not self._isolated(key):
            summary = S.FUNCS[key](self.p, self.cfg, emit)
            self.p.save()
            return summary
        self.p.save()
        from dubber.infra import gpu_lock

        cfg = self._stage_cfg(key)
        if key in S.GPU and cfg.get("device") == "cuda":
            eta = TIMEOUT_PER_FILM_S.get(key, 1.0) * float(self.p.settings.get("duration") or 600) / 4
            with gpu_lock.gpu_job(f"dubbing:{key}", eta_s=eta, cancel=self.cancel.is_set,
                                  on_wait=lambda h, s: self.cb.log(f"waiting for the GPU: {h.get('owner', '?')} runs {h.get('job', '?')}")):
                return self._worker(key, emit, cfg)
        return self._worker(key, emit, cfg)

    def _stage_cfg(self, key: str) -> Dict[str, Any]:
        """The VRAM / RAM policy for this stage, measured now (``dubber.infra.resources``): a model that does not fit the VRAM
        budget (free memory minus headroom) runs on the CPU or, for speech, as the lighter model.  The worker sizes its TTS
        batch from the same rule, and measures free VRAM again before each batch."""
        from dubber.infra import resources

        device = S._device(self.cfg) if key in S.GPU else "cpu"
        snap = resources.snapshot(use_torch=False)
        lighter_ok = bool(self.cfg.get("allow_download", True))
        if not lighter_ok:
            from dubber import models

            lighter_ok = models.locate(models.SPECS["tts_0_6b"].repo) is not None
        plan = resources.plan_stage(key, snap, device, self.cfg.get("tts_model", "tts_1_7b"), lighter_ok)
        if key in S.GPU:
            self.cb.log(f"{key}: {snap.describe()} -> {plan.device}" + (f", {plan.tts_model}" if key == "tts" else ""))
        for note in plan.notes:
            self.cb.log(note)
        cfg = dict(self.cfg)
        if key in S.GPU:
            cfg["device"] = plan.device
        if key == "tts" and plan.tts_model:
            cfg["tts_model"] = plan.tts_model
        return cfg

    def _worker(self, key: str, emit: Callable[..., None], cfg: Optional[Dict[str, Any]] = None) -> str:
        from dubber.diag.procs import run_worker

        def on_log(line: str) -> None:
            try:
                msg = json.loads(line)
            except ValueError:
                emit("log", text=line)
                return
            kind = msg.pop("kind", "log")
            try:
                emit(kind, **msg)
            except Cancelled:
                pass

        dur = float(self.p.settings.get("duration") or 600)
        cfg = cfg if cfg is not None else self.cfg
        timeout = max(600.0, TIMEOUT_PER_FILM_S.get(key, 1.0) * dur * (4 if cfg.get("device") == "cpu" else 1))
        env = {"HF_TOKEN": cfg["hf_token"]} if cfg.get("hf_token") else {}
        out = run_worker("stage", {"project": str(self.p.folder), "stage": key, "cfg": {k: v for k, v in cfg.items() if k != "hf_token"}},
                         timeout=timeout, env=env, on_log=on_log, cancel=self.cancel, sample_gpu=False)
        if out.cancelled:
            raise Cancelled()
        r = out.result or {}
        if r.get("status") == "OK":
            return str(r.get("summary", ""))
        if out.timed_out:
            raise RuntimeError(f"timed out after {out.seconds:.0f} s")
        from dubber.diag.procs import describe_exit_code

        raise RuntimeError(r.get("summary") or f"worker ended without a result ({describe_exit_code(out.returncode)}): "
                           + out.stderr_tail[-400:])

    # ------------------------------------------------------------------ Watch mode
    def _on_until(self, seconds: float) -> None:
        """New dub is final up to ``seconds``: mix the Watch chunks that are complete now."""
        try:
            self._write_watch_chunks(seconds)
        except Exception as exc:  # noqa: BLE001 - Watch mode must never break the dubbing
            self.cb.log(f"watch chunk: {type(exc).__name__}: {exc}")
        self.cb.dubbed_until(seconds)

    def _write_watch_chunks(self, until: float) -> None:
        total = float(self.p.settings.get("duration") or 0)
        lines = block_lines(self.p)
        keep = [ln for ln in self.p.lines if ln.keep_original]
        cache: Dict[int, Tuple[Any, int]] = {}
        done = []
        for i, (a, b) in enumerate(mixing.chunk_bounds(total, CHUNK_S)):
            if i in self._watch_written or b > until + 1e-6:
                continue
            x, sr = S.mix_into(self.p, a, b, lines + keep, cache)
            audio.write(self.p.path("watch", f"chunk_{i:05d}.wav"), x, sr)
            self._watch_written.add(i)
            done.append(i)
        idx = {"chunk_s": CHUNK_S, "until": until, "total": total, "chunks": sorted(self._watch_written)}
        write_json(self.p.path("watch", "index.json"), idx)


def block_lines(p: Project) -> List[Line]:
    """Placed lines from the TTS block records (valid while the TTS worker still owns lines.json)."""
    out = []
    for f in sorted((p.folder / "tts" / "blocks").glob("*.json")):
        for r in (read_json(f, {}) or {}).get("lines", []):
            ln = Line(int(r["id"]), float(r["start"]), float(r["end"]))
            ln.audio, ln.place_start = r["audio"], float(r["start"])
            try:
                ln.audio_s = audio.duration(p.abs(r["audio"]))
            except Exception:  # noqa: BLE001
                ln.audio_s = 0.0
            out.append(ln)
    return out


# ---------------------------------------------------------------------------------------------- preview fragment
def best_preview_start(p: Project, length: float = 60.0) -> float:
    """Start of the minute with the most dialogue (from the windows, if known), else 10 % into the film."""
    total = float(p.settings.get("duration") or (p.settings.get("media") or {}).get("duration") or 0)
    windows = read_json(p.folder / "analysis" / "windows.json", []) or []
    if windows and total > length:
        best, best_s = 0.0, -1.0
        for w in windows:
            t0 = max(0.0, min(w[0] - 1.0, total - length))
            s = sum(max(0.0, min(b, t0 + length) - max(a, t0)) for a, b in windows)
            if s > best_s:
                best, best_s = t0, s
        return round(best, 2)
    return round(min(max(0.0, total * 0.1), max(0.0, total - length)), 2)


def preview_project(p: Project, start: float, length: float = 60.0) -> Project:
    """A sub-project ``<project>/preview`` for ``[start, start+length)`` with the same settings and voices."""
    folder = p.folder / "preview"
    sub = Project(folder)
    keep = {k: v for k, v in p.settings.items() if k not in ("range", "output", "output_file", "duration")}
    if sub.settings.get("range") != [start, start + length]:
        shutil.rmtree(folder, ignore_errors=True)
        sub = Project(folder)
    sub.settings.update(keep)
    sub.settings["range"] = [round(start, 3), round(start + length, 3)]
    if p.settings.get("multi_voice") and p.speakers:
        sub.settings["voice_hint"] = {s.id: {"kind": s.voice.kind, "id": s.voice.id} for s in p.speakers}
    sub.save()
    return sub


def run_project_cli(folder: Path, stages_arg: str = "") -> int:
    """``main.py --run-project DIR``: run / resume a project without a window; progress on stdout, settings from the app settings."""
    from dubber import settings as app_settings

    p = Project(folder)
    if not p.settings.get("source"):
        print(f"not a project folder: {folder}", flush=True)
        return 2
    cfg = app_settings.engine_cfg()
    last = stages_arg.split(",")[-1] if stages_arg else "mux"
    cb = Callbacks(stage=lambda k, s, m: print(f"[{s:>7}] {k}: {m}", flush=True), log=lambda t: print("    " + t, flush=True),
                   progress=lambda f, e: None)
    res = Runner(p, cfg, cb).run(until_stage=last)
    print("done: " + res.message if res.ok else "FAILED: " + res.message, flush=True)
    return 0 if res.ok else 1
