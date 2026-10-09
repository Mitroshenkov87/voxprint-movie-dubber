"""Offline translation with Opus-MT (Marian models through transformers); direct pairs en<->ru, en<->de, else pivot via English.

Completeness first (real case: "Yeah, that sounds cool! Learning sucks!" came back as "Да, звучит круто!" and "...Destroy it all.
Destroy." lost "Destroy"): Marian models tend to drop a short trailing sentence when given several at once.  So every line is
split into sentences, all sentences are translated in one batch, and each result is checked (:func:`suspicious`: empty, much
shorter than the source, or left untranslated).  Suspicious sentences are retried (greedy decoding, then clause by clause) and
the most complete candidate wins.  The sentences of a line are then joined again.
"""
from __future__ import annotations

import re
from typing import Callable, List, Optional, Sequence, Tuple

from dubber import models

#: target/source length in characters below which a translation is treated as incomplete (ru/de/en are all >= ~0.8 normally)
MIN_RATIO = 0.5
Translate = Callable[[List[str]], List[str]]


def split_sentences(text: str) -> List[str]:
    from dubber.core.segment import is_sentence_end

    toks = text.split()
    out: List[str] = []
    cur: List[str] = []
    for i, t in enumerate(toks):
        cur.append(t)
        if is_sentence_end(t, toks[i + 1] if i + 1 < len(toks) else None):
            out.append(" ".join(cur))
            cur = []
    if cur:
        out.append(" ".join(cur))
    return out


def _letters(s: str) -> int:
    return sum(c.isalnum() for c in s)


def suspicious(src: str, out: str, tgt: str = "") -> bool:
    """True when ``out`` looks like an incomplete (or missing) translation of ``src``."""
    a, b = _letters(src), _letters(out)
    if a == 0:
        return False
    if b == 0:
        return True
    if a >= 6 and b < MIN_RATIO * a:
        return True
    if tgt in ("ru", "uk", "bg") and not re.search(r"[А-Яа-яЁё]", out) and re.search(r"[A-Za-z]{3,}", src):
        return True                                       # left in Latin letters: not translated
    return False


def _score(src: str, out: str) -> float:
    a, b = max(1, _letters(src)), _letters(out)
    return -abs(b / a - 1.1)                             # closest to the usual length ratio


def complete_translate(texts: Sequence[str], fn: Translate, tgt: str = "", retry_fn: Optional[Translate] = None,
                       log: Callable[[str], None] = lambda m: None) -> Tuple[List[str], int]:
    """Translate ``texts`` sentence by sentence with ``fn`` (one batch); retry suspicious sentences.  Returns
    ``(translations, number of sentences that needed a retry)``."""
    sents: List[str] = []
    owner: List[int] = []
    for i, t in enumerate(texts):
        for s in split_sentences(t) or [t]:
            sents.append(s)
            owner.append(i)
    outs = list(fn(sents)) if sents else []
    retried = 0
    bad = [k for k, (s, o) in enumerate(zip(sents, outs)) if suspicious(s, o, tgt)]
    if bad:
        retried = len(bad)
        alt = list((retry_fn or fn)([sents[k] for k in bad]))
        for k, a in zip(bad, alt):
            cands = [outs[k], a]
            clauses = [c for c in re.split(r"(?<=[,;:])\s+", sents[k]) if c.strip()]
            if len(clauses) > 1:
                cands.append(" ".join(x.strip() for x in fn(clauses)))
            ok = [c for c in cands if not suspicious(sents[k], c, tgt)]
            outs[k] = max(ok or cands, key=lambda c: (_score(sents[k], c) if ok else _letters(c)))
            if not ok:
                log(f"translation may be incomplete: {sents[k]!r} -> {outs[k]!r}")
    res = [""] * len(texts)
    for k, o in enumerate(outs):
        i = owner[k]
        res[i] = (res[i] + " " + o.strip()).strip()
    return res, retried


class OpusMT:
    def __init__(self, src: str, tgt: str, device: str, allow_download: bool, log: Callable[[str], None]) -> None:
        import torch  # noqa: F401
        from transformers import MarianMTModel, MarianTokenizer

        spec = models.mt_spec(src, tgt)
        if spec is None:
            raise RuntimeError(f"no Opus-MT model for {src}->{tgt}")
        folder, _ = models.ensure(spec.repo, allow_download, log=log)
        self.tok = MarianTokenizer.from_pretrained(str(folder))
        self.model = MarianMTModel.from_pretrained(str(folder)).to(device).eval()
        self.device, self.log = device, log

    def __call__(self, texts: List[str], beams: int = 4, batch: int = 16) -> List[str]:
        import torch

        out: List[str] = []
        for i in range(0, len(texts), batch):
            part = texts[i:i + batch]
            enc = self.tok(part, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)
            with torch.no_grad():
                gen = self.model.generate(**enc, num_beams=beams, max_new_tokens=256)
            out += self.tok.batch_decode(gen, skip_special_tokens=True)
            if len(texts) > batch:
                self.log(f"translation: {min(len(texts), i + batch)}/{len(texts)} sentences")
        return out


def opus_translate(texts: List[str], src: str, tgt: str, device: str, allow_download: bool, log: Callable[[str], None],
                   batch: int = 16) -> List[str]:
    if src == tgt or not texts:
        return list(texts)
    if models.mt_spec(src, tgt) is None:
        if src != "en" and tgt != "en":
            return opus_translate(opus_translate(texts, src, "en", device, allow_download, log), "en", tgt, device, allow_download, log)
        raise RuntimeError(f"no Opus-MT model for {src}->{tgt}")
    mt = OpusMT(src, tgt, device, allow_download, log)
    out, retried = complete_translate(texts, lambda xs: mt(xs, 4, batch), tgt, retry_fn=lambda xs: mt(xs, 1, batch), log=log)
    log(f"translation: {len(texts)} lines" + (f", {retried} sentence(s) re-translated (incomplete first result)" if retried else ""))
    return out


def mock_translate(texts: List[str], src: str, tgt: str) -> List[str]:
    """Stand-in for tests and CPU-only trials: keeps the text, marks the language."""
    return [f"[{tgt}] {t}" for t in texts]
