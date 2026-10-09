"""Offline translation with Opus-MT (Marian models through transformers); direct pairs en<->ru, en<->de, else pivot via English."""
from __future__ import annotations

from typing import Callable, List

from dubber import models


def opus_translate(texts: List[str], src: str, tgt: str, device: str, allow_download: bool, log: Callable[[str], None],
                   batch: int = 16) -> List[str]:
    if src == tgt or not texts:
        return list(texts)
    if models.mt_spec(src, tgt) is None:
        if src != "en" and tgt != "en":
            return opus_translate(opus_translate(texts, src, "en", device, allow_download, log), "en", tgt, device, allow_download, log)
        raise RuntimeError(f"no Opus-MT model for {src}->{tgt}")
    import torch
    from transformers import MarianMTModel, MarianTokenizer

    spec = models.mt_spec(src, tgt)
    folder, _ = models.ensure(spec.repo, allow_download, log=log)          # type: ignore[union-attr]
    tok = MarianTokenizer.from_pretrained(str(folder))
    model = MarianMTModel.from_pretrained(str(folder)).to(device).eval()
    out: List[str] = []
    for i in range(0, len(texts), batch):
        part = texts[i:i + batch]
        enc = tok(part, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, num_beams=4, max_new_tokens=256)
        out += tok.batch_decode(gen, skip_special_tokens=True)
        log(f"translation: {min(len(texts), i + batch)}/{len(texts)} lines")
    return out


def mock_translate(texts: List[str], src: str, tgt: str) -> List[str]:
    """Stand-in for tests and CPU-only trials: keeps the text, marks the language."""
    return [f"[{tgt}] {t}" for t in texts]
