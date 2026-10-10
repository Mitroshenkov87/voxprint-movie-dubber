"""The diagnostic runner: plans the checks, runs each one guarded, saves the report after every step.

Order (cheap and safe first, expensive last; the report file is rewritten after every step, so a crash keeps everything measured so far):

1. system facts (OS, hardware, disk, packages, network, ffmpeg), nvidia-smi, Windows adapters
2. ``gpu.torch``: PyTorch/CUDA, FlashAttention, CUDA Graphs micro-tests (worker process)
3. ``models.fetch``: make sure the models are on disk (download time reported separately from load time)
4. ``tts.*``: Qwen3-TTS speed in 4 modes (standard/graphs x SDPA/FlashAttention-2), one worker process each
5. ``stage.*``: the pipeline on the bundled test clip: extract -> VAD -> separation -> ASR -> diarization -> translation -> TTS -> fit -> mix -> mux
"""
from __future__ import annotations

import json
import shutil
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from dubber import ffmpeg, models, paths
from dubber.appinfo import version_line
from dubber.diag import checks_system as cs
from dubber.diag.constants import LANG_NAMES, TTS_PHRASES, WARMUP_PHRASE, test_clip_dir
from dubber.diag.procs import GpuSampler, WorkerOutcome, describe_exit_code, run_worker
from dubber.diag.report import CheckResult, Report, Status, report_filename, save_report
from dubber.diag import clip_stages as pst

ALL_TTS_MODES = ("standard_sdpa", "standard_fa2", "graphs_sdpa", "graphs_fa2")


@dataclass
class DiagOptions:
    """What to run.  ``hf_token`` is passed to workers through the environment only and is scrubbed from the report."""
    allow_download: bool = True
    quick: bool = False                       # one TTS run per phrase, only the SDPA modes
    target_lang: str = "ru"
    hf_token: str = ""
    clip: Optional[Path] = None               # a user clip instead of the bundled one
    out_path: Optional[Path] = None           # report path (default: Desktop)
    tts_model: str = "tts_1_7b"               # key in dubber.models.SPECS
    tts_modes: Tuple[str, ...] = ALL_TTS_MODES
    skip: Tuple[str, ...] = ()                # group names to skip: system, gpu, tts, stages, network
    cpu_tts: bool = False                     # run TTS on the CPU when there is no GPU (slow: minutes per phrase)
    cpu_sep: bool = False                     # run separation on the CPU (very slow)
    asr_repo: str = ""                        # override the ASR model repo (tests use a tiny one)
    timeout_scale: float = 1.0

    def as_report_options(self) -> Dict[str, Any]:
        """Header fields for the report. The token is recorded only as ``given`` or ``none``."""
        return {"target": self.target_lang, "quick": "yes" if self.quick else "no", "downloads": "allowed" if self.allow_download else "off",
                "hf_token": "given" if self.hf_token else "none", "tts_model": models.SPECS[self.tts_model].repo.split("/")[-1],
                "clip": "bundled synthetic" if not self.clip else self.clip.name}


@dataclass
class Step:
    """One diagnostic check in the run plan.

    Args:
        id: Stable check id written into the report.
        title: Label shown while the check runs.
        fn: Zero-argument function that returns a check result.
        weight: Share of the overall progress bar.
        group: Skip name. ``DiagOptions.skip`` drops a whole group before it runs.
    """
    id: str
    title: str
    fn: Callable[[], CheckResult]
    weight: float = 1.0
    group: str = "misc"


def guarded(check_id: str, title: str, fn: Callable[[], CheckResult]) -> CheckResult:
    """Run one check; an exception becomes a FAIL result with the traceback (the report goes on)."""
    t0 = time.time()
    try:
        res = fn()
    except Exception as exc:  # noqa: BLE001
        res = CheckResult(check_id, title, Status.FAIL, f"{type(exc).__name__}: {str(exc)[:200]}", traceback=traceback.format_exc())
    if res.seconds is None:
        res.seconds = time.time() - t0
    res.id = res.id or check_id
    res.title = res.title or title
    return res


class DiagnosticRunner:
    """Runs the diagnostics; thread-safe to cancel from another thread (``cancel()``)."""

    def __init__(self, options: Optional[DiagOptions] = None, on_progress: Optional[Callable[[float, str], None]] = None,
                 on_result: Optional[Callable[[CheckResult], None]] = None) -> None:
        self.opt = options or DiagOptions()
        self.on_progress = on_progress or (lambda f, t: None)
        self.on_result = on_result or (lambda r: None)
        self._cancel = threading.Event()
        self.report = Report(app_line=version_line(), options=self.opt.as_report_options())
        if self.opt.hf_token:
            self.report.secrets.append(self.opt.hf_token)
        self.path: Path = self.opt.out_path or (paths.desktop_dir() / report_filename())
        self.saved_to: Optional[Path] = None
        self.work: Optional[Path] = None
        self.data: Dict[str, Any] = {}              # stage outputs (paths etc.)
        self.clip_expected: Dict[str, Any] = {}
        self.has_gpu = False
        self.best_tts_mode: Optional[str] = None

    # ------------------------------------------------------------------ control
    def cancel(self) -> None:
        """Stop before the next check and cancel a worker that is still running."""
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        """True after :meth:`cancel` has been called."""
        return self._cancel.is_set()

    # ------------------------------------------------------------------ main loop
    def run(self) -> Path:
        """Run everything, write the report and return its path (never raises)."""
        rep = self.report
        self.work = paths.new_work_dir()
        try:
            steps = self._plan()
            total = sum(s.weight for s in steps) or 1.0
            done = 0.0
            self._save()
            for st in steps:
                if self.cancelled:
                    rep.note("cancelled by the user before: " + st.id)
                    break
                if st.group in self.opt.skip:
                    done += st.weight
                    continue
                rep.current_step = st.id
                self.on_progress(done / total, st.title)
                self._save()                               # so a crash inside the step leaves "Now running: <id>" in the file
                res = guarded(st.id, st.title, st.fn)
                rep.add(res)
                try:
                    self.on_result(res)
                except Exception:  # noqa: BLE001
                    pass
                done += st.weight
                self._save()
        except Exception:  # noqa: BLE001 - the runner itself must never lose the report
            rep.add(CheckResult("runner", "Diagnostic runner", Status.FAIL, "internal error in the runner", traceback=traceback.format_exc()))
        finally:
            rep.current_step = ""
            rep.finish(complete=not self.cancelled)
            self._save(final=True)
            if self.work:
                shutil.rmtree(self.work, ignore_errors=True)
            self.on_progress(1.0, "done")
        return self.saved_to or self.path

    def _save(self, final: bool = False) -> None:
        try:
            self.saved_to = save_report(self.report, self.path, paths.reports_dir())
            if final and self.saved_to.parent != paths.reports_dir():
                try:
                    shutil.copyfile(self.saved_to, paths.reports_dir() / self.saved_to.name)
                except OSError:
                    pass
        except OSError as exc:
            if final:
                self.report.note(f"could not save the report file: {exc}")

    # ------------------------------------------------------------------ plan
    def _plan(self) -> List[Step]:
        o = self.opt
        steps = [
            Step("system.app", "Program", cs.check_app, 0.2, "system"),
            Step("system.os", "Operating system", cs.check_os, 0.5, "system"),
            Step("system.hw", "Hardware", cs.check_hardware, 0.5, "system"),
            Step("system.disk", "Disk", cs.check_disk, 0.2, "system"),
            Step("system.packages", "Packages", cs.check_packages, 0.3, "system"),
            Step("system.network", "Network", cs.check_network, 1.0, "network"),
            Step("system.ffmpeg", "ffmpeg", cs.check_ffmpeg, 0.5, "system"),
            Step("gpu.smi", "NVIDIA driver", cs.check_gpu_smi, 0.5, "gpu"),
            Step("gpu.ctranslate2", "CTranslate2 CUDA", cs.check_ctranslate2, 0.5, "gpu"),
            Step("gpu.adapters", "Video adapters", cs.check_gpu_wmi, 0.5, "gpu"),
            Step("gpu.torch", "PyTorch / CUDA / FlashAttention / CUDA Graphs", self._check_gpu_torch, 3.0, "gpu"),
            Step("models.fetch", "Models on disk", self._check_models, 6.0, "models"),
        ]
        for mode in o.tts_modes:
            steps.append(Step(f"tts.{mode}", f"TTS speed: {mode}", self._tts_fn(mode), 5.0, "tts"))
        steps += [
            Step("stage.extract", "Extract audio", self._stage_extract, 0.5, "stages"),
            Step("stage.vad", "VAD", self._stage_vad, 0.5, "stages"),
            Step("stage.separation", "Separation", self._stage_sep, 3.0, "stages"),
            Step("stage.asr", "ASR", self._stage_asr, 3.0, "stages"),
            Step("stage.diarization", "Diarization", self._stage_diar, 2.0, "stages"),
            Step("stage.translation", "Translation", self._stage_mt, 2.0, "stages"),
            Step("stage.tts", "TTS of the clip lines", self._stage_tts, 4.0, "stages"),
            Step("stage.fit", "Fit", self._stage_inproc("fit"), 0.2, "stages"),
            Step("stage.mix", "Mix", self._stage_inproc("mix"), 0.5, "stages"),
            Step("stage.mux", "Mux", self._stage_inproc("mux"), 0.5, "stages"),
        ]
        return steps

    # ------------------------------------------------------------------ worker plumbing
    def _env(self) -> Dict[str, str]:
        env = {}
        if self.opt.hf_token:
            env["HF_TOKEN"] = self.opt.hf_token
        return env

    def _worker(self, check_id: str, title: str, name: str, args: Dict[str, Any], timeout: float) -> CheckResult:
        """Run a worker and turn whatever happened into a :class:`CheckResult`."""
        args = dict(args)
        args.setdefault("allow_download", self.opt.allow_download)
        from dubber.infra import gpu_lock

        def waiting(holder: Dict[str, Any], seconds: float) -> None:
            self.on_progress(-1.0, f"waiting for the GPU: {holder.get('owner', '?')} is running {holder.get('job', '?')} ({seconds:.0f} s)")

        try:
            # the shared GPU lock: never run a GPU check while Voxprint Audiobook Builder trains or narrates
            with gpu_lock.gpu_job(f"diagnostics:{check_id}", eta_s=timeout, on_wait=waiting, cancel=self._cancel.is_set):
                out: WorkerOutcome = run_worker(name, args, timeout=timeout * self.opt.timeout_scale, env=self._env(),
                                                on_log=lambda m: self.on_progress(-1.0, m), cancel=self._cancel)
        except gpu_lock.GpuLockTimeout as exc:
            r = CheckResult(check_id, title)
            r.status, r.summary = Status.SKIP, str(exc)
            return r
        return self.result_from_outcome(check_id, title, out)

    @staticmethod
    def result_from_outcome(check_id: str, title: str, out: WorkerOutcome) -> CheckResult:
        """Turn a finished, crashed, timed-out, or cancelled worker into a check result, and warn when a finished worker then exits non-zero."""
        r = CheckResult(check_id, title)
        r.seconds = out.seconds
        tail = "\n".join(out.stderr_tail.splitlines()[-30:])
        if out.start_error:
            r.status, r.summary, r.traceback = Status.FAIL, f"could not start the worker process: {out.start_error}", out.start_error
            return r
        if out.cancelled:
            r.status, r.summary = Status.SKIP, "cancelled"
            return r
        if out.timed_out:
            r.status = Status.FAIL
            r.summary = f"TIMEOUT after {out.seconds:.0f} s - the step hung or is far too slow (process was killed)"
            r.traceback = "worker stderr (last lines):\n" + tail
            return r
        res = out.result
        if res is None:
            r.status = Status.FAIL
            r.summary = f"worker CRASHED: {describe_exit_code(out.returncode)}"
            r.traceback = "worker stderr (last lines):\n" + (tail or "(empty)")
            return r
        try:
            r.status = Status(res.get("status", "OK"))
        except ValueError:
            r.status = Status.WARN
        r.summary = str(res.get("summary", ""))
        r.details = [str(x) for x in res.get("details", [])]
        r.metrics = dict(res.get("metrics", {}))
        r.traceback = str(res.get("traceback", "") or "")
        if r.status == Status.FAIL and tail and not r.traceback:
            r.traceback = "worker stderr (last lines):\n" + tail
        elif r.status == Status.FAIL and tail:
            r.traceback += "\n\nworker stderr (last lines):\n" + tail
        gpu_line = GpuSampler.describe(out.gpu)
        if gpu_line:
            r.details.append(gpu_line)
        r.details.append(f"{'worker wall time':<28}: {out.seconds:.1f} s (inside the worker: {res.get('total_s', '?')} s)")
        if out.returncode not in (0, None) and r.status in (Status.OK, Status.INFO):
            r.status = Status.WARN
            r.details.append(f"NOTE: the worker process exited with {describe_exit_code(out.returncode)} AFTER finishing the work (crash on shutdown)")
            r.traceback = (r.traceback + "\n" if r.traceback else "") + "worker stderr (last lines):\n" + tail
        return r

    # ------------------------------------------------------------------ steps
    def _check_gpu_torch(self) -> CheckResult:
        r = self._worker("gpu.torch", "PyTorch / CUDA / FlashAttention / CUDA Graphs", "gpu", {}, 300)
        self.has_gpu = bool(r.metrics.get("name"))
        return r

    def _required_models(self) -> List[str]:
        """Models the selected steps need (tests skip groups to keep downloads small)."""
        o = self.opt
        keys: List[str] = []
        if "tts" not in o.skip:
            keys.append(o.tts_model)
        if "stages" not in o.skip:
            keys += ["asr", "sep", "diar"] + (["mt_en_" + o.target_lang] if o.target_lang != "en" else [])
            if "tts" in o.skip and o.tts_model not in keys:
                keys.append(o.tts_model)                 # the stage that synthesises the clip lines needs it too
        return [k for k in dict.fromkeys(keys) if k in models.SPECS]

    def _check_models(self) -> CheckResult:
        keys = self._required_models()
        missing = [k for k in keys if not models.locate(models.SPECS[k].repo)]
        need_gb = models.total_download_gb(keys)
        if not keys:
            return CheckResult("models.fetch", "Models on disk", Status.SKIP, "no models needed for the selected steps")
        repos = {"asr": self.opt.asr_repo} if self.opt.asr_repo else {}
        r = self._worker("models.fetch", "Models on disk", "fetch", {"keys": keys, "repos": repos}, 3 * 3600)
        r.details.insert(0, f"needed: {', '.join(keys)}; missing before this step: {', '.join(missing) or 'none'}; about {need_gb} GB to download")
        r.details.append(f"models folder: {paths.models_dir()}")
        # the gated diarization model is optional: do not alarm the user about it
        if r.status == Status.WARN and set(r.metrics.get("missing", [])) <= {"diar"}:
            r.summary += " (diarization model is gated: needs a Hugging Face token; that step will be skipped)"
        return r

    def _tts_args(self) -> Dict[str, Any]:
        ex = self._expected()
        ref = ex.get("reference", {})
        lang = self.opt.target_lang
        phrases = TTS_PHRASES.get(lang, TTS_PHRASES["en"])
        return {"model_repo": models.SPECS[self.opt.tts_model].repo, "ref_audio": str(test_clip_dir() / ref.get("file", "ref_voice.wav")),
                "ref_text": ref.get("text", ""), "language": LANG_NAMES.get(lang, "English"), "warmup_text": WARMUP_PHRASE.get(lang, "Hello."),
                "phrases": phrases[:2] if self.opt.quick else phrases, "runs": 1 if self.opt.quick else 2,
                "device": "auto"}

    def _tts_fn(self, mode: str) -> Callable[[], CheckResult]:
        """A zero-arg check for one TTS mode. The mode is bound here so the plan does not share one loop variable."""
        def run() -> CheckResult:
            return self._check_tts(mode)
        return run

    def _check_tts(self, mode: str) -> CheckResult:
        cid = f"tts.{mode}"
        title = f"Qwen3-TTS speed: {mode}"
        if self.opt.quick and mode.endswith("fa2"):
            return CheckResult(cid, title, Status.SKIP, "skipped in quick mode")
        if not self.has_gpu and not self.opt.cpu_tts:
            return CheckResult(cid, title, Status.SKIP, "no NVIDIA GPU visible to PyTorch; the CPU speed test is off (enable 'cpu_tts' to run it; takes minutes)")
        if not self.has_gpu and mode != "standard_sdpa":
            return CheckResult(cid, title, Status.SKIP, "needs a GPU")
        if not models.locate(models.SPECS[self.opt.tts_model].repo):
            return CheckResult(cid, title, Status.SKIP, "TTS model is not on this computer (downloads off or failed, see models.fetch)")
        args = self._tts_args()
        args["mode"] = mode
        r = self._worker(cid, title, "tts", args, 1800 if self.has_gpu else 3600)
        if r.status in (Status.OK, Status.WARN) and r.metrics.get("rtf") is not None:
            best = self.best_tts_mode
            prev = self.report.get(f"tts.{best}") if best is not None else None
            prev_rtf = prev.metrics.get("rtf", 1e9) if prev is not None else 1e9
            if best is None or r.metrics["rtf"] < prev_rtf:
                self.best_tts_mode = mode
        return r

    # ---- stages on the clip
    def _expected(self) -> Dict[str, Any]:
        if not self.clip_expected:
            try:
                self.clip_expected = json.loads((test_clip_dir() / "expected.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self.clip_expected = {}
        return self.clip_expected

    def _clip_path(self) -> Path:
        return self.opt.clip or (test_clip_dir() / "voxprint-test-clip.mkv")

    def _stage_extract(self) -> CheckResult:
        r = CheckResult("stage.extract", "Extract audio with ffmpeg", Status.OK)
        clip = self._clip_path()
        if not clip.is_file():
            return CheckResult("stage.extract", "Extract audio", Status.FAIL, f"test clip not found: {clip}")
        info = ffmpeg.probe(clip)
        streams = info["streams"]
        r.kv("clip", f"{clip.name} ({clip.stat().st_size / 1024:.0f} KB, {info['duration']} s)")
        for stream in streams:
            r.line(f"  stream {stream['index']}: {stream['type']} {stream['codec']} {stream.get('lang', '')} "
                   f"{stream.get('channels') or ''} {stream.get('rate') or ''}")
        ctx = pst.StageContext(clip, self.opt.target_lang, self.work / "stages", log=lambda m: None)         # type: ignore[operator]
        res = pst.ExtractStage().run(ctx)
        self.data.update(ctx.data)
        self.data["clip"] = str(clip)
        r.summary = f"{res.message} in {res.seconds:.2f} s"
        r.metrics.update(run_s=round(res.seconds, 3), load_s=0.0, device="cpu (ffmpeg)")
        r.kv("duration", f"{info['duration']} s")
        return r

    def _need(self, check_id: str, title: str, *keys: str) -> Optional[CheckResult]:
        miss = [k for k in keys if not self.data.get(k)]
        if miss:
            return CheckResult(check_id, title, Status.SKIP, f"needs the result of an earlier step that did not succeed ({', '.join(miss)})")
        return None

    def _stage_vad(self) -> CheckResult:
        skip = self._need("stage.vad", "VAD", "wav16")
        if skip:
            return skip
        out = self.work / "vad.json"                                              # type: ignore[operator]
        exp = self._expected().get("lines") if not self.opt.clip else None
        r = self._worker("stage.vad", "Voice activity detection (Silero)", "vad", {"wav16": self.data["wav16"], "expected": exp, "out_json": str(out)}, 300)
        if out.is_file():
            self.data["vad"] = json.loads(out.read_text(encoding="utf-8"))["segments"]
        return r

    def _stage_sep(self) -> CheckResult:
        skip = self._need("stage.separation", "Separation", "wav44")
        if skip:
            return skip
        if not self.has_gpu and not self.opt.cpu_sep:
            return CheckResult("stage.separation", "Separation (TIGER-DnR)", Status.SKIP,
                               "no GPU: TIGER-DnR takes ~100 s per 4 s of audio on a CPU; skipped (the next stages use the full mix instead)")
        outdir = self.work / "sep"                                                # type: ignore[operator]
        r = self._worker("stage.separation", "Dialogue/effects/music separation (TIGER-DnR)", "sep",
                         {"wav": self.data["wav44"], "out_dir": str(outdir), "segments": self.data.get("vad")}, 900)
        if (outdir / "dialog16k.wav").is_file():
            self.data["dialog16"] = str(outdir / "dialog16k.wav")
        return r

    def _stage_asr(self) -> CheckResult:
        wav = self.data.get("dialog16") or self.data.get("wav16")
        if not wav:
            return CheckResult("stage.asr", "ASR", Status.SKIP, "needs extracted audio")
        out = self.work / "asr.json"                                              # type: ignore[operator]
        ex = self._expected()
        expected_text = " ".join(l["text"] for l in ex.get("lines", [])) if not self.opt.clip else ""
        args = {"wav16": wav, "out_json": str(out), "expected_text": expected_text, "expect_gpu": self.has_gpu}
        if self.opt.asr_repo:
            args["repo"] = self.opt.asr_repo
        r = self._worker("stage.asr", "Speech recognition (faster-whisper)", "asr", args, 1200)
        if out.is_file():
            self.data["asr"] = json.loads(out.read_text(encoding="utf-8"))
        r.details.append(f"input audio                 : {'dialogue stem from separation' if self.data.get('dialog16') else 'full mix (separation did not run)'}")
        return r

    def _stage_diar(self) -> CheckResult:
        wav = self.data.get("dialog16") or self.data.get("wav16")
        if not wav:
            return CheckResult("stage.diarization", "Diarization", Status.SKIP, "needs extracted audio")
        out = self.work / "diar.json"                                             # type: ignore[operator]
        r = self._worker("stage.diarization", "Speaker diarization (pyannote community-1)", "diar",
                         {"wav16": wav, "out_json": str(out), "expected_speakers": None if self.opt.clip else 2}, 900)
        if out.is_file():
            self.data["diar"] = json.loads(out.read_text(encoding="utf-8"))
        return r

    def _lines_for_translation(self) -> List[Dict[str, Any]]:
        asr = self.data.get("asr")
        if asr and asr.get("segments"):
            return [{"id": i, "start": s["start"], "end": s["end"], "text": s["text"]} for i, s in enumerate(asr["segments"]) if s["text"].strip()]
        return [{"id": i, "start": l["start"], "end": l["end"], "text": l["text"]} for i, l in enumerate(self._expected().get("lines", []))]

    def _stage_mt(self) -> CheckResult:
        lines = self._lines_for_translation()
        src = (self.data.get("asr") or {}).get("language") or self._expected().get("language", "en")
        out = self.work / "mt.json"                                               # type: ignore[operator]
        r = self._worker("stage.translation", "Translation (Opus-MT)", "mt",
                         {"source": src, "target": self.opt.target_lang, "sentences": [l["text"] for l in lines], "out_json": str(out)}, 600)
        self.data["lines"] = lines
        if out.is_file():
            pairs = json.loads(out.read_text(encoding="utf-8"))["pairs"]
            for l, p in zip(lines, pairs):
                l["translation"] = p["tgt"]
        return r

    def _stage_tts(self) -> CheckResult:
        lines = [l for l in self.data.get("lines", []) if l.get("translation")]
        if not lines:
            return CheckResult("stage.tts", "TTS of the clip lines", Status.SKIP, "no translated lines (translation did not run)")
        mode = self.best_tts_mode or ("graphs_sdpa" if self.has_gpu else "standard_sdpa")
        if not self.has_gpu and not self.opt.cpu_tts:
            return CheckResult("stage.tts", "TTS of the clip lines", Status.SKIP, "no GPU and the CPU TTS test is off")
        if not models.locate(models.SPECS[self.opt.tts_model].repo):
            return CheckResult("stage.tts", "TTS of the clip lines", Status.SKIP, "TTS model is not on this computer")
        args = self._tts_args()
        args.update(mode=mode, phrases=[], lines=[{"id": l["id"], "text": l["translation"]} for l in lines],
                    out_dir=str(self.work / "tts"))                              # type: ignore[operator]
        r = self._worker("stage.tts", f"Dubbed lines with Qwen3-TTS ({mode})", "tts", args, 1800)
        r.details.insert(0, f"mode used                   : {mode} (fastest working mode from the speed tests, else default)")
        made = {m["id"]: m for m in r.metrics.pop("made", [])}
        self.data["dub_lines"] = [made[l["id"]] | {"id": l["id"]} for l in lines if l["id"] in made]
        self.data["slots"] = [{"start": l["start"], "end": l["end"]} for l in lines if l["id"] in made]
        return r

    def _stage_inproc(self, which: str) -> Callable[[], CheckResult]:
        stage = {"fit": pst.FitStage, "mix": pst.MixStage, "mux": pst.MuxStage}[which]()
        title = {"fit": "Time-fit plan", "mix": "Mix dub over original", "mux": "Mux the new audio track"}[which]

        def go() -> CheckResult:
            r = CheckResult(f"stage.{which}", title, Status.OK)
            need = {"fit": ("dub_lines",), "mix": ("dub_lines", "wav44"), "mux": ("dub_track", "clip")}[which]
            miss = [k for k in need if not self.data.get(k)]
            if miss:
                return CheckResult(f"stage.{which}", title, Status.SKIP, "needs data from steps that did not run: " + ", ".join(miss))
            ctx = pst.StageContext(Path(self.data["clip"]), self.opt.target_lang, self.work / "stages", out=self.work / "dubbed.mkv",   # type: ignore[operator]
                                   data=self.data)
            res = stage.run(ctx)
            r.summary = res.message
            r.metrics.update(run_s=round(res.seconds, 3), load_s=0.0, device="cpu")
            if which == "fit":
                for it in self.data.get("fit_plan", {}).get("items", []):
                    r.line(f"  line {it['id']}: dub/slot = {it['ratio']}  -> {it['verdict']}")
                if self.data.get("fit_plan", {}).get("too_long"):
                    r.status = Status.INFO
            if which == "mux" and ctx.out and ctx.out.is_file():
                r.kv("output size", f"{ctx.out.stat().st_size / 1024:.0f} KB")
            return r
        return go
