"""Worker ``diar``: speaker diarization with pyannote ``speaker-diarization-community-1`` (gated: needs a Hugging Face token).

Input (args): ``wav16``, ``expected_speakers`` (int, optional), ``out_json``, ``allow_download``, ``device``.
Without a token (or without pyannote.audio installed) the result is SKIP with the exact reason - not a failure of the app.
The audio is passed to pyannote as an in-memory waveform, so torchcodec/ffmpeg shared libraries (a frequent Windows problem) are not needed.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

from dubber import models
from dubber.workers.common import WorkerContext, cuda_sync, free_gpu, peak_vram_gb, read_wav_mono, reset_peak, torch_device


def run(args: Dict[str, Any], ctx: WorkerContext) -> Dict[str, Any]:
    """Diarize ``wav16`` with pyannote and write turns to ``out_json`` when set, or skip when the model cannot be loaded."""
    import importlib.util

    try:
        have = importlib.util.find_spec("pyannote.audio") is not None
    except (ImportError, ValueError):          # the parent package "pyannote" is missing altogether
        have = False
    if not have:
        return {"status": "SKIP", "summary": "pyannote.audio is not installed in this environment (pip install pyannote.audio)"}
    repo = args.get("repo") or models.SPECS["diar"].repo
    ctx.log("diarization: locating the model")
    try:
        folder, info = models.ensure(repo, args.get("allow_download", True), log=ctx.log)
    except models.ModelUnavailable as exc:
        return {"status": "SKIP", "summary": str(exc)[:200], "details": [str(exc)]}
    t = time.time()
    import torch
    from pyannote.audio import Pipeline

    import_s = time.time() - t
    device = torch_device(args.get("device", "auto"))
    reset_peak()
    t = time.time()
    token = os.environ.get("HF_TOKEN") or None
    try:
        pipe = Pipeline.from_pretrained(str(folder))
    except Exception:  # noqa: BLE001 - some versions want the repo id + token instead of a folder
        pipe = Pipeline.from_pretrained(repo, token=token)
    if pipe is None:
        return {"status": "FAIL", "summary": "Pipeline.from_pretrained returned None (access to the gated repo was not granted?)"}
    if hasattr(pipe, "embedding_batch_size"):
        try:
            pipe.embedding_batch_size = 4          # pyannote 4.0.3 defaults to a huge batch (~10 GB VRAM); research note 02
        except Exception:  # noqa: BLE001
            pass
    pipe.to(torch.device(device))
    cuda_sync()
    load_s = time.time() - t

    audio, sr = read_wav_mono(args["wav16"], 16000)
    wave = {"waveform": torch.from_numpy(audio)[None, :], "sample_rate": 16000}
    ctx.log("diarization: running")

    def go():
        out = pipe(wave)
        cuda_sync()
        return out

    t = time.time()
    out = go()
    first_s = time.time() - t
    t = time.time()
    out = go()
    run_s = time.time() - t
    ann = getattr(out, "exclusive_speaker_diarization", None) or getattr(out, "speaker_diarization", None) or out
    turns: List[Dict[str, Any]] = []
    for turn, _, label in ann.itertracks(yield_label=True):
        turns.append({"start": round(turn.start, 3), "end": round(turn.end, 3), "speaker": str(label)})
    speakers = sorted({t_["speaker"] for t_ in turns})
    if args.get("out_json"):
        with open(args["out_json"], "w", encoding="utf-8") as fh:
            json.dump({"turns": turns, "speakers": speakers}, fh)
    dur = len(audio) / sr
    details = [f"{'model':<28}: {repo} ({info['source']})", f"{'import pyannote':<28}: {import_s:.2f} s", f"{'device':<28}: {device}",
               f"{'pipeline load':<28}: {load_s:.2f} s",
               f"{'processing':<28}: {first_s:.2f} s first call, {run_s:.2f} s second call for {dur:.1f} s (x{dur / max(run_s, 1e-6):.1f} real time)",
               f"{'speakers found':<28}: {len(speakers)} ({', '.join(speakers)}), {len(turns)} turns"]
    for t_ in turns:
        details.append(f"  {t_['start']:6.2f}-{t_['end']:6.2f}  {t_['speaker']}")
    status, note = "OK", ""
    exp = args.get("expected_speakers")
    if exp:
        note = f" (expected {exp})"
        if len(speakers) != exp:
            status = "WARN"
    free_gpu()
    return {"status": status, "summary": f"{len(speakers)} speakers{note}, {len(turns)} turns, x{dur / max(run_s, 1e-6):.1f} real time",
            "details": details, "metrics": {"load_s": round(load_s, 3), "run_s": round(run_s, 3), "device": device,
                                             "peak_vram_gb": peak_vram_gb(), "download_s": info["download_s"]}}
