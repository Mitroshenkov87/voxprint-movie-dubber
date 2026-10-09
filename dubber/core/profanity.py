"""Profanity mode of the translation (per project): ``keep`` (as in the original, the default) or ``soften``.

``soften`` replaces obscene words (Russian *mat*) with non-obscene equivalents that may still be rude ("пиздец" -> "капец",
"охуел" -> "обалдел", "блядь" -> "чёрт").  It runs on the final translation - downloaded subtitles and Opus-MT output alike -
before the speech synthesis; every changed line keeps its previous text in ``Line.softened`` so the Script table can highlight
it and the user can edit it.  Switching back to ``keep`` restores the lines the user did not edit.

Filters are pluggable per language (:func:`register`): a dictionary + rules filter for Russian now; another language or an LLM
rewrite later only needs a ``(text) -> text`` function.
"""
from __future__ import annotations

import re
from typing import Callable, Dict, Iterable, List, Tuple

MODES = ("keep", "soften")
DEFAULT_MODE = "keep"

TextFilter = Callable[[str], str]
_FILTERS: Dict[str, TextFilter] = {}


def register(lang: str, fn: TextFilter) -> None:
    _FILTERS[lang] = fn


def supported(lang: str) -> bool:
    return lang in _FILTERS


def soften(text: str, lang: str) -> str:
    fn = _FILTERS.get(lang)
    return fn(text) if fn else text


# ---------------------------------------------------------------------------------------------- Russian
_W = r"[а-яёА-ЯЁ-]"
#: phrases first (lower case, ё/е both accepted by the patterns)
_RU_PHRASES: List[Tuple[str, str]] = [
    (r"(?:ё|е)б\s+тво(?:ю\s+мать|я\s+мать|ю)", "чёрт побери"),
    (r"на\s*хуй|нахуй", "к чёрту"),
    (r"в\s+пизду", "к чёрту"),
    (r"ни\s*хуя", "ни фига"),
    (r"до\s*хуя", "до фига"),
    (r"по\s*хуй", "плевать"),
    (r"на\s*хуя", "на фига"),
]
#: whole words (lower case, ё written as е - the lookup normalises)
_RU_WORDS: Dict[str, str] = {
    "бля": "чёрт", "блядь": "чёрт", "блять": "чёрт", "блядина": "дрянь", "блядский": "чёртов",
    "блядская": "чёртова", "блядское": "чёртово", "блядские": "чёртовы", "блядство": "безобразие",
    "хуй": "чёрт", "хуя": "фига", "хуе": "фиге", "хуем": "фигом", "хуйня": "фигня", "хуйню": "фигню", "хуйни": "фигни",
    "хуево": "хреново", "хуевый": "хреновый", "хуевая": "хреновая", "хуевое": "хреновое", "хуевые": "хреновые",
    "охуеть": "обалдеть", "охуел": "обалдел", "охуела": "обалдела", "охуели": "обалдели", "охуенно": "офигенно",
    "охуенный": "офигенный", "охуенная": "офигенная", "охуенное": "офигенное", "охуенные": "офигенные",
    "охуительно": "офигительно", "нихуя": "ни фига", "дохуя": "до фига", "похуй": "плевать", "нахуя": "на фига",
    "хуесос": "урод", "хуило": "урод", "хуила": "урод",
    "пиздец": "капец", "пиздеца": "капеца", "пизда": "хана", "пизде": "хане", "пизду": "хану", "пиздато": "классно",
    "пиздатый": "классный", "пиздатая": "классная", "пиздеть": "трепаться", "пиздит": "врёт", "пиздишь": "врёшь",
    "пиздят": "врут", "пиздел": "трепался", "пиздела": "трепалась", "спиздил": "стащил", "спиздила": "стащила",
    "спиздили": "стащили", "спиздить": "стащить", "распиздяй": "разгильдяй", "пиздюлей": "люлей", "пиздобол": "трепло",
    "ебать": "чёрт", "ебаный": "долбаный", "ебаная": "долбаная", "ебаное": "долбаное", "ебаные": "долбаные",
    "ебаного": "долбаного", "ебаной": "долбаной", "ебаным": "долбаным", "ебанутый": "долбанутый", "ебанутая": "долбанутая",
    "ебанулся": "свихнулся", "ебанулась": "свихнулась", "ебало": "рожа", "ебальник": "рожа", "ебет": "волнует",
    "ебут": "волнуют", "заебал": "достал", "заебала": "достала", "заебали": "достали", "заебало": "достало",
    "заебись": "отлично", "заебался": "замотался", "заебалась": "замоталась", "наебал": "надул", "наебала": "надула",
    "наебали": "надули", "наебать": "надуть", "уебок": "урод", "уебки": "уроды", "уебать": "врезать", "въебать": "врезать",
    "уебал": "врезал", "въебал": "врезал", "отъебись": "отвяжись", "отъебитесь": "отвяжитесь", "съебись": "проваливай",
    "съебался": "смылся", "съебалась": "смылась", "съебались": "смылись", "выебываться": "выпендриваться",
    "выебывается": "выпендривается", "выебываешься": "выпендриваешься", "долбоеб": "придурок", "долбоебы": "придурки",
    "распиздяйство": "разгильдяйство", "пидор": "урод", "пидорас": "урод", "пидоры": "уроды", "пидарас": "урод",
    "мудозвон": "трепло", "залупа": "ерунда",
}
#: anything left that is built on a mat root (with the usual prefixes)
_RU_ROOT = re.compile(r"^(?:на|по|ни|за|от|отъ|о|об|рас|раз|разъ|до|при|у|вы|пере|недо|под|подъ|въ|съ|вз|рос)?"
                      r"(?:ху[йеёяию]|пизд|еб|ёб|бляд|блят)", re.IGNORECASE)
_SAFE = {"ебола", "хуан", "хуанита", "хулиган"}


def _case_like(src: str, repl: str) -> str:
    if src.isupper() and len(src) > 1:
        return repl.upper()
    if src[:1].isupper():
        return repl[:1].upper() + repl[1:]
    return repl


def _fallback(word: str) -> str:
    w = word.lower()
    for end, adj in (("ые", "долбаные"), ("ый", "долбаный"), ("ий", "долбаный"), ("ая", "долбаная"), ("ое", "долбаное"),
                     ("ого", "долбаного"), ("ой", "долбаной")):
        if w.endswith(end) and len(w) > 5:
            return adj
    return "чёрт"


_RU_SEND = re.compile(rf"(?<!{_W})(?P<v>иди|пошёл|пошел|пошла|пошли|идите)\s+(?:на\s*хуй|нахуй|в\s+пизду|нахер)(?!{_W})",
                      re.IGNORECASE)
_RU_PHRASE_RX = [(re.compile(rf"(?<!{_W})(?:{pat})(?!{_W})", re.IGNORECASE), repl) for pat, repl in _RU_PHRASES]


def soften_ru(text: str) -> str:
    out = _RU_SEND.sub(lambda m: m.group("v") + " к чёрту", text)
    for rx, repl in _RU_PHRASE_RX:
        out = rx.sub(lambda m, r=repl: _case_like(m.group(0), r), out)

    def word(m: "re.Match[str]") -> str:
        w = m.group(0)
        key = w.lower().replace("ё", "е")
        if key in _RU_WORDS:
            return _case_like(w, _RU_WORDS[key])
        if key in _SAFE or not _RU_ROOT.match(w):
            return w
        return _case_like(w, _fallback(w))

    return re.sub(r"[а-яёА-ЯЁ]+", word, out)


register("ru", soften_ru)


# ---------------------------------------------------------------------------------------------- lines
def apply_to_lines(lines: Iterable, mode: str, lang: str) -> int:
    """Soften (or restore) the translations; returns how many lines are softened now.  Lines the user edited stay as they are."""
    n = 0
    for ln in lines:
        if ln.edited:
            n += bool(ln.softened)
            continue
        if mode != "soften":
            if ln.softened:
                ln.translation, ln.softened = ln.softened, ""
            continue
        base = ln.softened or ln.translation
        new = soften(base, lang)
        if new != base:
            ln.translation, ln.softened = new, base
            n += 1
        elif ln.softened:
            ln.translation, ln.softened = base, ""
    return n
