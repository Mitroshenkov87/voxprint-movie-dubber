"""Pipeline stages of the dubber, in processing order (research note 06 section 1).

Each stage is a small object with ``key``, ``title_key`` (i18n), ``implemented`` and ``run(ctx)``.  The heavy model stages are
*workers* (``dubber/workers/w_*.py``, one process per model); here they are registered as stubs for the product pipeline - the
diagnostic runner (``dubber.diag.runner``) already drives the same workers on the bundled test clip, so the code paths that matter
for speed are real.  Implemented in-process: ``extract`` (ffmpeg), ``fit`` (plan only, no stretching yet), ``mix``, ``mux``.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dubber import ffmpeg


@dataclass
class StageContext:
    """Shared state of one dubbing run."""
    source: Path
    target_lang: str
    work: Path
    out: Optional[Path] = None
    log: Callable[[str], None] = lambda m: None
    data: Dict[str, Any] = field(default_factory=dict)       # stage outputs by key


@dataclass
class StageResult:
    key: str
    ok: bool
    message: str = ""
    seconds: float = 0.0
    implemented: bool = True


class Stage:
    key = ""
    title_key = ""
    implemented = False

    def run(self, ctx: StageContext) -> StageResult:
        t = time.time()
        ctx.log(f"[{self.key}] stub - not implemented in the skeleton")
        return StageResult(self.key, True, "stub (not implemented yet)", time.time() - t, implemented=False)


class ExtractStage(Stage):
    key, title_key, implemented = "extract", "stage.extract", True

    def run(self, ctx: StageContext) -> StageResult:
        t = time.time()
        ctx.work.mkdir(parents=True, exist_ok=True)
        ffmpeg.extract_audio(ctx.source, ctx.work / "mix16.wav", 16000, 1)
        ffmpeg.extract_audio(ctx.source, ctx.work / "mix44.wav", 44100, 1)
        ctx.data["wav16"], ctx.data["wav44"] = str(ctx.work / "mix16.wav"), str(ctx.work / "mix44.wav")
        return StageResult(self.key, True, "mono 16 kHz and 44.1 kHz WAV extracted", time.time() - t)


class FitStage(Stage):
    """Plan only: compares the length of each synthesised line with its slot and says which would need squeezing (no audio is changed)."""
    key, title_key, implemented = "fit", "stage.fit", True

    def run(self, ctx: StageContext) -> StageResult:
        t = time.time()
        plan = fit_plan(ctx.data.get("slots", []), ctx.data.get("dub_lines", []))
        ctx.data["fit_plan"] = plan
        msg = f"{plan['ok']} fit, {plan['squeeze']} need squeezing (<=1.15x), {plan['too_long']} too long" if plan["total"] else "nothing to fit"
        return StageResult(self.key, True, msg, time.time() - t)


class MixStage(Stage):
    key, title_key, implemented = "mix", "stage.mix", True

    def run(self, ctx: StageContext) -> StageResult:
        t = time.time()
        out = ctx.work / "dub_track.wav"
        info = mix_track(ctx.data["wav44"], ctx.data.get("slots", []), ctx.data.get("dub_lines", []), out)
        ctx.data["dub_track"] = str(out)
        return StageResult(self.key, True, f"{info['lines']} lines mixed, ducking {info['duck_db']} dB", time.time() - t)


class MuxStage(Stage):
    key, title_key, implemented = "mux", "stage.mux", True

    def run(self, ctx: StageContext) -> StageResult:
        t = time.time()
        out = ctx.out or (ctx.work / "dubbed.mkv")
        lang = {"ru": "rus", "en": "eng", "de": "deu"}.get(ctx.target_lang, ctx.target_lang)
        idx = ffmpeg.add_dub_track(ctx.source, Path(ctx.data["dub_track"]), out, language=lang, title=f"AI dub ({ctx.target_lang})")
        ctx.data["output"] = str(out)
        streams = ffmpeg.probe(out)["streams"]
        n_a = sum(1 for s in streams if s["type"] == "audio")                 # type: ignore[index]
        return StageResult(self.key, True, f"{out.name}: {n_a} audio tracks (new track #{idx}), video copied", time.time() - t)


def _stub(key: str, title_key: str) -> Stage:
    s = Stage()
    s.key, s.title_key = key, title_key
    return s


STAGES: List[Stage] = [ExtractStage(), _stub("vad", "stage.vad"), _stub("separation", "stage.separation"), _stub("asr", "stage.asr"),
                       _stub("diarization", "stage.diarization"), _stub("translation", "stage.translation"),
                       _stub("tts", "stage.tts"), FitStage(), MixStage(), MuxStage()]


def run_all(ctx: StageContext, on_stage: Optional[Callable[[Stage, StageResult], None]] = None) -> List[StageResult]:
    """Run every stage in order; a stage that needs missing data (e.g. mix without synthesised lines) reports an error and the run goes on."""
    results = []
    for st in STAGES:
        try:
            res = st.run(ctx)
        except Exception as exc:  # noqa: BLE001 - one broken stage must not crash the UI
            res = StageResult(st.key, False, f"{type(exc).__name__}: {exc}", 0.0, st.implemented)
        results.append(res)
        if on_stage:
            on_stage(st, res)
        if st.key == "extract" and not res.ok:
            break
    return results


# ---------------------------------------------------------------------------------------------- pure helpers (tested)
MAX_SQUEEZE = 1.15          # research note 04: time-stretch up to 1.15x is inaudible; beyond that shorten the translation


def fit_plan(slots: List[Dict[str, float]], lines: List[Dict[str, Any]]) -> Dict[str, Any]:
    """For each line: ratio = synthesised seconds / slot seconds.  <=1 fits; <=1.15 can be squeezed; more = translation must be shortened."""
    items, ok, squeeze, too_long = [], 0, 0, 0
    for slot, ln in zip(slots, lines):
        slot_s = max(1e-3, slot["end"] - slot["start"])
        ratio = float(ln["seconds"]) / slot_s
        verdict = "fits" if ratio <= 1.0 else ("squeeze" if ratio <= MAX_SQUEEZE else "too_long")
        ok += verdict == "fits"
        squeeze += verdict == "squeeze"
        too_long += verdict == "too_long"
        items.append({"id": ln.get("id"), "ratio": round(ratio, 2), "verdict": verdict})
    return {"total": len(items), "ok": ok, "squeeze": squeeze, "too_long": too_long, "items": items}


def mix_track(original_wav: str, slots: List[Dict[str, float]], lines: List[Dict[str, Any]], out_wav: Path, duck_db: float = -14.0) -> Dict[str, Any]:
    """Overlay the dubbed lines on the original soundtrack; the original is ducked by ``duck_db`` under each line (50 ms ramps).

    The skeleton keeps the original under the dub (the real product will use the separated background stem instead).
    Lines longer than their slot are placed at the slot start and may run into the next pause (fit/stretch comes later).
    """
    import numpy as np
    import soundfile as sf

    base, sr = sf.read(original_wav, dtype="float32", always_2d=True)
    base = base.mean(axis=1)
    mix = base.copy()
    gain = np.ones(len(base), dtype=np.float32)
    dub = np.zeros(len(base), dtype=np.float32)
    duck = 10 ** (duck_db / 20)
    ramp = int(0.05 * sr)
    n = 0
    for slot, ln in zip(slots, lines):
        w, wsr = sf.read(ln["path"], dtype="float32", always_2d=True)
        w = w.mean(axis=1)
        if wsr != sr:
            idx = np.linspace(0, len(w) - 1, int(len(w) * sr / wsr))
            w = np.interp(idx, np.arange(len(w)), w).astype(np.float32)
        s = max(0, int(slot["start"] * sr))
        e = min(len(base), s + len(w))
        if e <= s:
            continue
        dub[s:e] += w[: e - s]
        g0, g1 = max(0, s - ramp), min(len(base), e + ramp)
        seg = np.full(g1 - g0, duck, dtype=np.float32)
        r = min(ramp, len(seg) // 2)
        if r:
            seg[:r] = np.linspace(1.0, duck, r)
            seg[-r:] = np.linspace(duck, 1.0, r)
        gain[g0:g1] = np.minimum(gain[g0:g1], seg)
        n += 1
    mix = base * gain + dub
    peak = float(np.abs(mix).max()) or 1.0
    if peak > 0.98:
        mix = mix * (0.98 / peak)
    sf.write(str(out_wav), mix, sr, subtype="PCM_16")
    return {"lines": n, "duck_db": duck_db, "samples": len(mix), "sr": sr}
