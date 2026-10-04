"""Worker ``sep``: dialogue / effects / music separation with TIGER-DnR (vendored code, Apache-2.0 weights).

Input (args): ``wav`` 44.1 kHz mono WAV (the mix), ``out_dir``, ``repo`` (model repo), ``allow_download``, ``device`` (auto|cpu).
Output files in ``out_dir``: ``dialog.wav`` (44.1 kHz), ``dialog16k.wav`` (for ASR / diarization), ``music.wav``, ``effect.wav``.
Status OK when the three stems were produced; the details tell how much dialogue leaked into the pauses (a quality hint).
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict

from dubber import models
from dubber.workers.common import (WorkerContext, cuda_sync, free_gpu, peak_vram_gb, read_wav_mono, reset_peak, resample, rms_db,
                                   torch_device)


def run(args: Dict[str, Any], ctx: WorkerContext) -> Dict[str, Any]:
    import numpy as np
    import soundfile as sf
    import torch
    from safetensors.torch import load_file

    from dubber.third_party.look2hear.models import TIGERDNR

    repo = args.get("repo", "JusperLee/TIGER-DnR")
    ctx.log("separation: locating the model")
    folder, info = models.ensure(repo, args.get("allow_download", True), log=ctx.log)
    device = torch_device(args.get("device", "auto"))
    reset_peak()
    t = time.time()
    cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    model = TIGERDNR(**{k: v for k, v in cfg.items() if k in TIGERDNR.__init__.__code__.co_varnames})
    state = load_file(str(folder / "model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval().to(device)
    cuda_sync()
    load_s = time.time() - t

    audio, sr = read_wav_mono(args["wav"], 44100)
    x = torch.from_numpy(audio)[None, None, :].to(device)            # [batch=1, channels=1, samples]
    ctx.log("separation: running")

    def once():
        with torch.no_grad():
            d, e, m = model(x)
        cuda_sync()
        return d, e, m

    t = time.time()
    d, e, m = once()
    first_s = time.time() - t
    run_s = first_s
    if device == "cuda" or first_s < 20:            # second call = steady-state speed (skipped for a very slow CPU run)
        t = time.time()
        d, e, m = once()
        run_s = time.time() - t

    def to_np(tensor):
        return tensor.detach().float().cpu().numpy().reshape(-1)[: len(audio)]

    dialog, effect, music = to_np(d), to_np(e), to_np(m)
    out_dir = args["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    sf.write(os.path.join(out_dir, "dialog.wav"), dialog, 44100, subtype="PCM_16")
    sf.write(os.path.join(out_dir, "effect.wav"), effect, 44100, subtype="PCM_16")
    sf.write(os.path.join(out_dir, "music.wav"), music, 44100, subtype="PCM_16")
    sf.write(os.path.join(out_dir, "dialog16k.wav"), resample(dialog, 44100, 16000), 16000, subtype="PCM_16")

    dur = len(audio) / sr
    details = [f"{'model':<28}: {repo} ({folder}; {info['source']}" + (f", downloaded in {info['download_s']} s)" if info["download_s"] else ")"),
               f"{'device':<28}: {device}",
               f"{'weights load':<28}: {load_s:.2f} s" + (f" (state-dict mismatch: {len(missing)} missing, {len(unexpected)} unexpected keys)" if missing or unexpected else ""),
               f"{'processing':<28}: {first_s:.2f} s first call, {run_s:.2f} s second call for {dur:.1f} s of audio (x{dur / max(run_s, 1e-6):.1f} real time)",
               f"{'stem level (RMS dB)':<28}: mix {rms_db(audio):.1f}, dialog {rms_db(dialog):.1f}, effect {rms_db(effect):.1f}, music {rms_db(music):.1f}"]
    status, note = "OK", ""
    segs = args.get("segments")
    if segs:
        mask = np.zeros(len(audio), dtype=bool)
        for s in segs:
            mask[int(s["start"] * sr): int(s["end"] * sr)] = True
        if mask.any() and (~mask).any():
            speech_db, gap_db = rms_db(dialog[mask]), rms_db(dialog[~mask])
            details.append(f"{'dialog stem: speech vs gaps':<28}: {speech_db:.1f} dB in speech windows, {gap_db:.1f} dB in the gaps "
                           f"(difference {speech_db - gap_db:.1f} dB; bigger is cleaner)")
            note = f"; dialog/gap contrast {speech_db - gap_db:.0f} dB"
            if speech_db - gap_db < 10:
                status = "WARN"
    if missing or unexpected:
        status = "WARN"
    free_gpu()
    return {"status": status, "summary": f"3 stems in {run_s:.1f} s for {dur:.0f} s of audio (x{dur / max(run_s, 1e-6):.1f} real time){note}",
            "details": details, "metrics": {"load_s": round(load_s, 3), "run_s": round(run_s, 3), "device": device,
                                             "peak_vram_gb": peak_vram_gb(), "download_s": info["download_s"]}}
