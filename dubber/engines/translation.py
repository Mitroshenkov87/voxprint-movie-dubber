"""Offline translation with Opus-MT (Marian models through transformers).

Direct pairs among en/ru/de are en<->ru, en<->de and ru<->de. A pair of two non-English languages with no
direct model still pivots through English.

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
#: Opus-MT leaves this token alone; it stands in for AI / A.I. / A. I. and is put back after translation
AI_PLACEHOLDER = "XQZ1"
# "A. I. Pushkin" is a person's initials, not the acronym, so a following capital name is not replaced.
# The capital-name lookahead must stay case-sensitive (IGNORECASE would treat "to" as a name).
_AI_ACRONYM = re.compile(r"(?<![\w])(?:AI|A\.I\.?)(?![\w])|(?<![\w])A\.\s+I\.(?!\s*[A-Z])")
_AI_TOKEN = re.compile(r"XQZ1\w*", re.IGNORECASE)
_PROPER = {"Beavis", "Butthead", "Butt-Head", "I", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
           "Sunday", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December", "English", "Russian", "American", "America", "God", "Christmas"}
_WORD = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
_CYRILLIC_TARGETS = {"ru", "uk", "be", "bg", "sr", "kk"}
_LETTER_RUN = re.compile(r"([^\W\d_])\1{2,}")


def split_sentences(text: str) -> List[str]:
    """Split ``text`` on the same sentence-ending tokens the script uses."""
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
    """One Opus-MT model loaded for a single language pair. Raises RuntimeError when no model exists for that pair."""

    def __init__(self, src: str, tgt: str, device: str, allow_download: bool, log: Callable[[str], None]) -> None:
        import torch  # noqa: F401
        from transformers import MarianMTModel, MarianTokenizer

        spec = models.mt_spec(src, tgt)
        if spec is None:
            raise RuntimeError(f"no Opus-MT model for {src}->{tgt}")
        folder, _ = models.ensure(spec.repo, allow_download, log=log)
        self.tok = MarianTokenizer.from_pretrained(str(folder))
        self.model = MarianMTModel.from_pretrained(str(folder)).to(device).eval()
        self.device, self.log, self.prefix = device, log, spec.target_token

    def __call__(self, texts: List[str], beams: int = 4, batch: int = 16) -> List[str]:
        """Translate ``texts`` with Marian beam search, ``batch`` sentences per generate call. The completeness retry uses ``beams`` 1."""
        import torch

        out: List[str] = []
        for i in range(0, len(texts), batch):
            part = models.mt_inputs(self.prefix, texts[i:i + batch])
            enc = self.tok(part, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)
            with torch.no_grad():
                gen = self.model.generate(**enc, num_beams=beams, max_new_tokens=256)
            out += self.tok.batch_decode(gen, skip_special_tokens=True)
            if len(texts) > batch:
                self.log(f"translation: {min(len(texts), i + batch)}/{len(texts)} sentences")
        return out


def source_has_ai(text: str) -> bool:
    """True when ``text`` says AI / A.I. / A. I. as a word (not inside AIM, not "A. I. Surname")."""
    return _AI_ACRONYM.search(text or "") is not None


def protect_ai(text: str) -> str:
    """Replace the acronym with ``AI_PLACEHOLDER`` so Opus-MT copies it instead of guessing."""
    return _AI_ACRONYM.sub(AI_PLACEHOLDER, text or "")


def restore_ai(text: str, tgt: str) -> str:
    """Put the acronym back: "ИИ" for Russian, "AI" otherwise. Inflected leftovers (``XQZ1а``) count too."""
    rep = "ИИ" if (tgt or "").lower()[:2] == "ru" else "AI"
    return _AI_TOKEN.sub(rep, text or "")


def lowercase_mid_title(text: str) -> str:
    """Lowercase mid-sentence Title Case common nouns ("Artificial Intelligence" -> "artificial intelligence").

    The first word and a word after ``.!?`` stay, as do ALL CAPS and a small proper-noun list. Opus-MT otherwise
    reads the capitals as a name ("Искусственная Разведка" for Artificial Intelligence)."""
    if not text:
        return text
    out: List[str] = []
    idx = 0
    sentence_start = True
    for m in _WORD.finditer(text):
        out.append(text[idx:m.start()])
        word = m.group(0)
        if any(c.isalpha() for c in word):
            if sentence_start or word in _PROPER or (word.isupper() and len(word) > 1):
                out.append(word)
            elif _is_title_word(word):
                out.append(_lower_title(word))
            else:
                out.append(word)
            sentence_start = False
        else:
            out.append(word)
        idx = m.end()
        if re.match(r"\s*[.!?…]", text[m.end():]):
            sentence_start = True
    out.append(text[idx:])
    return "".join(out)


def _is_title_word(word: str) -> bool:
    parts = [p for p in re.split(r"[-']", word) if p]
    return bool(parts) and all(p[:1].isupper() and (len(p) == 1 or p[1:].islower()) for p in parts)


def _lower_title(word: str) -> str:
    return re.sub(r"[A-Za-z]+", lambda m: m.group(0)[:1].lower() + m.group(0)[1:], word)


def collapse_letter_runs(text: str) -> str:
    """Collapse a run of 3 or more of the same letter to 2 ("Ewwwww." -> "Eww.") before MT sees it."""
    return _LETTER_RUN.sub(r"\1\1", text or "")


def _proper_names(source: str) -> set:
    """Capitalised words and ALL-CAPS brands in the source, compared case-insensitively."""
    names = {m.group(0).lower() for m in re.finditer(r"\b[A-Z][A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*\b", source or "")}
    names |= {m.group(0).lower() for m in re.finditer(r"\b[A-Z]{2,}\b", source or "")}
    return names


def drop_latin_leftovers(text: str, source: str, tgt: str) -> str:
    """Drop Latin-only tokens a Cyrillic translation kept ("Фу, www." -> "Фу."), except names and brands from the source."""
    if (tgt or "").lower()[:2] not in _CYRILLIC_TARGETS:
        return text or ""
    keep = _proper_names(source)

    def repl(m: "re.Match[str]") -> str:
        return m.group(0) if m.group(0).lower() in keep else ""

    cleaned = re.sub(r"\b[A-Za-z]+(?:'[A-Za-z]+)?\b", repl, text or "")
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?…])", r"\1", cleaned)
    cleaned = re.sub(r",\s*(?=[.!?…])", "", cleaned)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip(" ,;")


def prepare_mt(text: str) -> str:
    """Source text as Opus-MT should see it: letter runs collapsed, AI protected, mid-sentence capitals lowered."""
    return lowercase_mid_title(protect_ai(collapse_letter_runs(text)))


def finish_mt(text: str, source: str, tgt: str) -> str:
    """Translation after Opus-MT: AI restored, then stray Latin tokens removed for a Cyrillic target."""
    return drop_latin_leftovers(restore_ai(text, tgt), source, tgt)


def opus_translate(texts: List[str], src: str, tgt: str, device: str, allow_download: bool, log: Callable[[str], None],
                   batch: int = 16) -> List[str]:
    """Translate lines with Opus-MT, retrying incomplete sentences and pivoting through English when no direct model exists. Raises RuntimeError when the pair cannot pivot."""
    if src == tgt or not texts:
        return list(texts)
    if models.mt_spec(src, tgt) is None:
        if src != "en" and tgt != "en":
            return opus_translate(opus_translate(texts, src, "en", device, allow_download, log), "en", tgt, device, allow_download, log)
        raise RuntimeError(f"no Opus-MT model for {src}->{tgt}")
    mt = OpusMT(src, tgt, device, allow_download, log)
    prepared = [prepare_mt(t) for t in texts]
    out, retried = complete_translate(prepared, lambda xs: mt(xs, 4, batch), tgt, retry_fn=lambda xs: mt(xs, 1, batch), log=log)
    out = [finish_mt(o, s, tgt) for o, s in zip(out, texts)]
    log(f"translation: {len(texts)} lines" + (f", {retried} sentence(s) re-translated (incomplete first result)" if retried else ""))
    return out


def mock_translate(texts: List[str], src: str, tgt: str) -> List[str]:
    """Stand-in for tests and CPU-only trials: keeps the text, marks the language."""
    return [f"[{tgt}] {t}" for t in texts]
