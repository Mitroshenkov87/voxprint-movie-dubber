"""Worker ``mt``: offline machine translation with Opus-MT tc-big (Marian).

Input (args): ``source``, ``target`` (en/ru/de), ``sentences`` (list of str), ``out_json``, ``allow_download``, ``device``.
Direct tc-big pairs are en<->ru and ru<->de. en->de and de->en have no tc-big model and are SKIP here
(the pipeline pivots through English only when both sides are non-English and a direct model is missing).
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List

from dubber import models
from dubber.workers.common import WorkerContext, cuda_sync, free_gpu, peak_vram_gb, reset_peak, torch_device


def run(args: Dict[str, Any], ctx: WorkerContext) -> Dict[str, Any]:
    src, tgt = args.get("source", "en"), args.get("target", "ru")
    sentences: List[str] = list(args.get("sentences") or [])
    if src == tgt:
        return {"status": "SKIP", "summary": f"source and target language are the same ({src}); nothing to translate"}
    spec = models.mt_spec(src, tgt)
    if spec is None:
        return {"status": "SKIP", "summary": f"no direct Opus-MT tc-big model for {src}->{tgt}"}
    ctx.log(f"translation {src}->{tgt}: locating the model")
    folder, info = models.ensure(spec.repo, args.get("allow_download", True), log=ctx.log)
    t = time.time()
    import torch
    from transformers import MarianMTModel, MarianTokenizer

    import_s = time.time() - t
    device = torch_device(args.get("device", "auto"))
    reset_peak()
    t = time.time()
    tok = MarianTokenizer.from_pretrained(str(folder))
    model = MarianMTModel.from_pretrained(str(folder)).to(device).eval()
    cuda_sync()
    load_s = time.time() - t

    def translate(batch: List[str]) -> List[str]:
        enc = tok(models.mt_inputs(spec.target_token, batch), return_tensors="pt", padding=True, truncation=True,
                  max_length=256).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, num_beams=int(args.get("beams", 4)), max_new_tokens=256)
        cuda_sync()
        return tok.batch_decode(gen, skip_special_tokens=True)

    t = time.time()
    out = translate(sentences) if sentences else []
    first_s = time.time() - t
    run_s = first_s
    if sentences and (device == "cuda" or first_s < 20):
        t = time.time()
        out = translate(sentences)
        run_s = time.time() - t
    if args.get("out_json"):
        with open(args["out_json"], "w", encoding="utf-8") as fh:
            json.dump({"source": src, "target": tgt, "pairs": [{"src": a, "tgt": b} for a, b in zip(sentences, out)]}, fh, ensure_ascii=False)
    details = [f"{'model':<28}: {spec.repo} ({info['source']}" + (f", downloaded in {info['download_s']} s)" if info["download_s"] else ")"),
               f"{'import transformers':<28}: {import_s:.2f} s", f"{'device':<28}: {device}", f"{'model load':<28}: {load_s:.2f} s",
               f"{'translation':<28}: {first_s:.2f} s first call, {run_s:.2f} s second call for {len(sentences)} sentences "
               f"({len(sentences) / max(run_s, 1e-6):.1f} sentences/s)"]
    for a, b in zip(sentences, out):
        details.append(f"  {a}")
        details.append(f"  -> {b}")
    ok = all(o.strip() for o in out)
    free_gpu()
    return {"status": "OK" if ok else "WARN", "summary": f"{src}->{tgt}, {len(out)} sentences, {len(sentences) / max(run_s, 1e-6):.1f} sentences/s on {device}",
            "details": details, "metrics": {"load_s": round(load_s, 3), "run_s": round(run_s, 3), "device": device,
                                             "peak_vram_gb": peak_vram_gb(), "download_s": info["download_s"]}}
