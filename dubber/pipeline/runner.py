"""Runs a project through the stages: cache check, one worker process for the model stages, the shared GPU lock, progress + ETA,
Watch-mode chunks while the dub is being made, and the 1-minute preview fragment.

The runner is Qt-free; the window drives it from a QThread and gets callbacks. ``main.py run-project`` (see ``dubber.cli``)
runs it without a window.
"""
from __future__ import annotations

import copy
import json
import logging
import shutil
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from dubber.core import audio, mixing, subtitles
from dubber.core.project import CHUNK_S, Line, Project, read_json, write_json
from dubber.core.watch import Eta
from dubber.pipeline import stages as S

#: rough share of the total time per stage (for the overall progress bar)
WEIGHTS = {"probe": 0.2, "extract": 1, "subtitles": 0.5, "vad": 1, "separation": 8, "asr": 6, "script": 0.2, "diarization": 3,
           "translation": 2, "voices": 0.3, "tts": 70, "mix": 4, "mux": 3}
TIMEOUT_PER_FILM_S = {"tts": 6.0, "separation": 2.0, "asr": 2.0, "diarization": 1.5, "translation": 0.5, "vad": 0.3}


class Cancelled(Exception):
    """Raised when the user cancels a stage that is running in this process."""


@dataclass
class Callbacks:
    """Progress hooks the window or the CLI receives while a project runs.

    Args:
        stage: ``(key, status, message)``. Status is ``running``, ``done``, ``cached``, ``skipped``, or ``failed``.
        progress: ``(fraction, eta_seconds)``. Fraction runs from 0 to 1; ETA is seconds, or -1 when unknown.
        dubbed_until: Seconds of film whose dub is already final.
    """
    stage: Callable[[str, str, str], None] = lambda key, status, msg: None        # status: running | done | cached | skipped | failed
    log: Callable[[str], None] = lambda text: None
    progress: Callable[[float, float], None] = lambda frac, eta_s: None           # overall 0..1, ETA seconds (-1 unknown)
    dubbed_until: Callable[[float], None] = lambda seconds: None                 # Watch mode: final dub available up to here


@dataclass
class RunResult:
    """Outcome of one pipeline run.

    ``message`` is the output path, ``cancelled``, or the failing stage and its short error.
    ``stages`` maps each key to ``cached``, ``done``, or ``failed``. ``debug`` is the traceback
    for a failure and stays empty on success. The user-facing message does not include it.
    """
    ok: bool
    message: str = ""
    stages: Dict[str, str] = field(default_factory=dict)
    debug: str = ""


def stage_error(summary: str, debug: str = "") -> RuntimeError:
    """A stage failure. ``summary`` is shown to the user; ``debug`` keeps the traceback.

    Args:
        summary: Short error, usually ``TypeName: message``.
        debug: Full traceback from the worker, or empty when the caller will format one.

    Returns:
        RuntimeError with a ``debug`` attribute.
    """
    err = RuntimeError(summary)
    err.debug = debug  # type: ignore[attr-defined]
    return err


class Runner:
    """Runs one project through the stage order and reports progress on the callbacks. ``cancel`` stops before the next stage, and during a stage in this process."""

    def __init__(self, project: Project, cfg: Optional[Dict[str, Any]] = None, cb: Optional[Callbacks] = None,
                 cancel: Optional[threading.Event] = None) -> None:
        self.p = project
        self.cfg = {**S.DEFAULT_CFG, **(cfg or {})}
        self.cb = cb or Callbacks()
        self.cancel = cancel or threading.Event()
        self.eta = Eta()
        self._done_w = 0.0
        self._watch_written: set = set()
        self._gpu_cm: Any = None
        self._t_start = 0.0

    def _seed_preview(self) -> None:
        """When this project is a fragment of a finished film, reuse that film's stage files."""
        raw = self.p.settings.get("preview_of")
        if not raw:
            return
        reuse_preview(Project(Path(str(raw))), self.p, self.cfg)

    # ------------------------------------------------------------------ public
    def run(self, until_stage: str = "mux", from_stage: Optional[str] = None) -> RunResult:
        """Run every stage from probe through ``until_stage``, re-running ``from_stage`` even when its cache is fresh, and release the GPU lock if this run held it."""
        self._seed_preview()
        keys = S.ORDER[: S.ORDER.index(until_stage) + 1]
        total_w = sum(WEIGHTS[k] for k in keys)
        prev = ""
        res = RunResult(True)
        t_start = time.time()
        self._t_start = t_start
        try:
            return self._run_keys(keys, total_w, prev, res, t_start, from_stage)
        finally:
            self._release_gpu()

    def _run_keys(self, keys: List[str], total_w: float, prev: str, res: RunResult, t_start: float,
                  from_stage: Optional[str]) -> RunResult:
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
                debug = str(getattr(exc, "debug", "") or "") or traceback.format_exc()
                logging.getLogger("dubber.pipeline").error("stage %s failed: %s\n%s", key, msg, debug)
                self.cb.stage(key, "failed", msg)
                res.stages[key] = "failed"
                return RunResult(False, f"{key}: {msg}", res.stages, debug)
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
            self._log_gpu(key, emit)
            self.p.save()
            return summary
        self.p.save()
        cfg = self._stage_cfg(key)
        if self.cfg.get("resident_models", True):
            self._hold_gpu(key, cfg)
            return self._session_stage(key, emit, cfg)
        from dubber.infra import gpu_lock

        if key in S.GPU and cfg.get("device") == "cuda":
            eta = TIMEOUT_PER_FILM_S.get(key, 1.0) * float(self.p.settings.get("duration") or 600) / 4
            with gpu_lock.gpu_job(f"dubbing:{key}", eta_s=eta, cancel=self.cancel.is_set,
                                  on_wait=lambda h, s: self.cb.log(f"waiting for the GPU: {h.get('owner', '?')} runs {h.get('job', '?')}")):
                return self._worker(key, emit, cfg)
        return self._worker(key, emit, cfg)

    def _hold_gpu(self, key: str, cfg: Dict[str, Any]) -> None:
        """One lock for every CUDA stage of this run.  Released in :meth:`run` so the next film can take the GPU."""
        if self._gpu_cm is not None or key not in S.GPU or cfg.get("device") != "cuda":
            return
        from dubber.infra import gpu_lock

        eta = TIMEOUT_PER_FILM_S.get(key, 1.0) * float(self.p.settings.get("duration") or 600)
        cm = gpu_lock.gpu_job(f"dubbing:{key}", eta_s=eta, cancel=self.cancel.is_set,
                              on_wait=lambda h, s: self.cb.log(f"waiting for the GPU: {h.get('owner', '?')} runs {h.get('job', '?')}"))
        cm.__enter__()
        self._gpu_cm = cm

    def _release_gpu(self) -> None:
        cm = self._gpu_cm
        self._gpu_cm = None
        if cm is not None:
            cm.__exit__(None, None, None)

    def _log_gpu(self, key: str, emit: Callable[..., None]) -> None:
        if key not in S.GPU:
            return
        from dubber.infra.thermal import _mask_text, read_nvidia, stage_telemetry_line

        temp, power, mask = read_nvidia()
        if temp is None and power is None and not mask:
            return
        emit("log", text=stage_telemetry_line(key, temp, power, _mask_text(mask)))

    def _session_stage(self, key: str, emit: Callable[..., None], cfg: Dict[str, Any]) -> str:
        """Run the stage in the shared model process, and do pending CPU work beside it."""
        from dubber.pipeline import prefetch
        from dubber.pipeline.session import get_session

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

        def gpu() -> str:
            return get_session(self.cfg).run_stage(self.p.folder, key, cfg, on_log, cancel=self.cancel.is_set)

        def cpu() -> None:
            prefetch.run_cpu(prefetch.plan(key).cpu, self.p, cfg, emit)

        return prefetch.overlap(gpu, cpu)

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
        from dubber.infra import vram_tier

        tier = vram_tier.resolve(snap.vram_total_gb, str(self.cfg.get("vram_tier") or vram_tier.AUTO))
        preferred = str(self.cfg.get("tts_model", "tts_1_7b"))
        tts_model = vram_tier.tts_model_for(snap.vram_total_gb, preferred) if snap.has_gpu else preferred
        if key == "tts":
            self.cb.log(vram_tier.describe(snap.vram_total_gb, str(self.cfg.get("vram_tier") or vram_tier.AUTO)))
            if tts_model != preferred:
                self.cb.log(f"tts: {preferred} does not fit this card with the VRAM headroom - using {tts_model}")
        plan = resources.plan_stage(key, snap, device, tts_model, lighter_ok)
        if key in S.GPU:
            self.cb.log(f"{key}: {snap.describe()} -> {plan.device}" + (f", {plan.tts_model}" if key == "tts" else ""))
        for note in plan.notes:
            self.cb.log(note)
        cfg = dict(self.cfg)
        cfg["job_started"] = self._t_start or time.time()      # the thermal rule's full-speed window counts from here
        cfg["vram_tier"] = tier
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
            raise stage_error(f"timed out after {out.seconds:.0f} s", str(r.get("traceback") or ""))
        from dubber.diag.procs import describe_exit_code

        raise stage_error(r.get("summary") or f"worker ended without a result ({describe_exit_code(out.returncode)}): "
                          + out.stderr_tail[-400:], str(r.get("traceback") or ""))

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
_REUSE_STOP = ("tts", "mix", "mux")


def _seed_span(start: float, end: float, t0: float, t1: float) -> Optional[Tuple[float, float]]:
    """Clip ``[start, end)`` to ``[t0, t1)`` and shift it so ``t0`` is zero. None when the ranges miss."""
    a, b = max(float(start), t0), min(float(end), t1)
    if b <= a:
        return None
    return a - t0, b - t0


def reuse_preview(parent: Project, sub: Project, cfg: Dict[str, Any]) -> None:
    """Copy a finished film's stage outputs into a preview and mark that prefix fresh.

    The preview range is already on ``sub``. Audio, windows, subtitles, recognition and the
    script are sliced to that range and shifted to start at zero. Voice references are copied
    as they are (they are short clips, not a timeline). TTS, mix and mux stay undone so only
    the fragment is synthesised. A second run of the same preview leaves edits in place.

    Args:
        parent: The full film, with stages already marked done.
        sub: The preview project. ``settings["range"]`` is ``[start, end]``.
        cfg: The engine config the preview run will use, so the cache hashes match.
    """
    if sub.settings.get("preview_seeded"):
        return
    rng = sub.settings.get("range") or [0.0, 0.0]
    t0, t1 = float(rng[0]), float(rng[1])
    if t1 <= t0:
        return
    prefix: List[str] = []
    for key in S.ORDER:
        if key in _REUSE_STOP:
            break
        if not (parent.stages.get(key) or {}).get("done"):
            break
        prefix.append(key)
    if not prefix:
        return
    sub.settings["duration"] = round(t1 - t0, 3)
    if "extract" in prefix:
        for name in ("mix16.wav", "mix44.wav", "mix48s.wav"):
            src = parent.folder / "audio" / name
            if src.is_file():
                data, sr = audio.read_range(src, t0, t1, mono=False)
                audio.write(sub.path("audio", name), data, sr)
    if "subtitles" in prefix:
        subs_dir = parent.folder / "subs"
        if subs_dir.is_dir():
            for src in sorted(subs_dir.iterdir()):
                if not src.is_file():
                    continue
                try:
                    cues = subtitles.load_file(src)
                except (OSError, ValueError):
                    continue
                kept = []
                for cue in cues:
                    span = _seed_span(cue.start, cue.end, t0, t1)
                    if span is not None:
                        kept.append(subtitles.Cue(span[0], span[1], cue.text))
                subtitles.write_srt(kept, sub.path("subs", src.name))
    if "vad" in prefix:
        windows = []
        for raw in read_json(parent.folder / "analysis" / "windows.json", []) or []:
            span = _seed_span(float(raw[0]), float(raw[1]), t0, t1)
            if span is not None:
                windows.append([span[0], span[1]])
        write_json(sub.path("analysis", "windows.json"), windows)
    if "separation" in prefix:
        for name in ("speech.wav", "background.wav", "speech16.wav"):
            src = parent.folder / "stems" / name
            if src.is_file():
                data, sr = audio.read_range(src, t0, t1, mono=False)
                audio.write(sub.path("stems", name), data, sr)
    if "asr" in prefix:
        asr = read_json(parent.folder / "analysis" / "asr.json", None)
        if isinstance(asr, dict):
            segs = []
            for seg in asr.get("segments") or []:
                if not isinstance(seg, dict) or "start" not in seg or "end" not in seg:
                    continue
                span = _seed_span(float(seg["start"]), float(seg["end"]), t0, t1)
                if span is None:
                    continue
                item = dict(seg)
                item["start"], item["end"] = span
                segs.append(item)
            shifted = dict(asr)
            shifted["segments"] = segs
            write_json(sub.path("analysis", "asr.json"), shifted)
    if "script" in prefix:
        lines: List[Line] = []
        for ln in parent.lines:
            span = _seed_span(ln.start, ln.end, t0, t1)
            if span is None:
                continue
            copied = copy.deepcopy(ln)
            copied.start, copied.end = span
            copied.audio, copied.audio_s = "", 0.0
            copied.place_start, copied.stretch, copied.fit, copied.spoken = -1.0, 1.0, "", ""
            lines.append(copied)
        sub.lines = lines
        sub.speakers = [copy.deepcopy(sp) for sp in parent.speakers]
    if "voices" in prefix:
        src_voices = parent.folder / "voices"
        if src_voices.is_dir():
            shutil.copytree(src_voices, sub.folder / "voices", dirs_exist_ok=True)
    sub.settings["preview_seeded"] = True
    sub.save()
    prev = ""
    for key in S.ORDER:
        inputs = S.inputs_for(key, sub, cfg, prev)
        prev = inputs
        if key not in prefix:
            continue
        sub.mark_done(key, inputs, 0.0, summary="reused from the full film")


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
    sub.settings["preview_of"] = str(p.folder.resolve())
    if p.settings.get("multi_voice") and p.speakers:
        sub.settings["voice_hint"] = {s.id: {"kind": s.voice.kind, "id": s.voice.id} for s in p.speakers}
    sub.save()
    return sub
