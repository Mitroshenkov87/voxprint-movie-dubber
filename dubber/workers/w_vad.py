"""Worker ``vad``: Silero VAD (MIT, ~2 MB, runs on CPU) - finds the speech windows of the soundtrack.

Input  (args): ``wav16`` mono 16 kHz WAV; optional ``expected`` (list of {start,end}) for a sanity check; ``out_json``.
Output: ``out_json`` = ``{"segments": [{"start","end"}...]}`` (seconds).
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict

from dubber.workers.common import WorkerContext, read_wav_mono


def run(args: Dict[str, Any], ctx: WorkerContext) -> Dict[str, Any]:
    """Find speech windows in ``wav16`` with Silero VAD, writing ``out_json`` when that path is set."""
    t = time.time()
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    import_s = time.time() - t
    t = time.time()
    model = load_silero_vad()
    load_s = time.time() - t
    audio, sr = read_wav_mono(args["wav16"], 16000)
    wav = torch.from_numpy(audio)
    t = time.time()
    ts = get_speech_timestamps(wav, model, sampling_rate=16000, return_seconds=True, min_silence_duration_ms=300)
    run_s = time.time() - t
    # second pass = steady-state speed (the first call includes lazy initialisation)
    t = time.time()
    get_speech_timestamps(wav, model, sampling_rate=16000, return_seconds=True, min_silence_duration_ms=300)
    run2_s = time.time() - t
    segs = [{"start": round(float(s["start"]), 3), "end": round(float(s["end"]), 3)} for s in ts]
    if args.get("out_json"):
        with open(args["out_json"], "w", encoding="utf-8") as fh:
            json.dump({"segments": segs}, fh)
    dur = len(audio) / sr
    speech = sum(s["end"] - s["start"] for s in segs)
    details = [f"{'import silero_vad+torch':<28}: {import_s:.2f} s", f"{'model load':<28}: {load_s:.2f} s (device cpu)",
               f"{'processing':<28}: {run_s:.3f} s first call, {run2_s:.3f} s second call for {dur:.1f} s of audio "
               f"(x{dur / max(run2_s, 1e-6):.0f} real time)", f"{'speech windows':<28}: {len(segs)}, {speech:.1f} s of speech"]
    for s in segs:
        details.append(f"  {s['start']:7.2f} - {s['end']:7.2f} s")
    status, note = "OK", ""
    exp = args.get("expected")
    if exp:
        hit = 0
        for e in exp:
            cov = sum(max(0.0, min(e["end"], s["end"]) - max(e["start"], s["start"])) for s in segs)
            if cov >= 0.7 * (e["end"] - e["start"]):
                hit += 1
        note = f"; {hit}/{len(exp)} known lines covered"
        details.append(f"{'known lines covered (>=70%)':<28}: {hit}/{len(exp)}")
        if hit < len(exp):
            status = "WARN"
    return {"status": status, "summary": f"{len(segs)} speech windows, {speech:.1f} s of {dur:.1f} s{note}", "details": details,
            "metrics": {"load_s": round(load_s, 3), "run_s": round(run2_s, 3), "device": "cpu", "n_segments": len(segs)}}
