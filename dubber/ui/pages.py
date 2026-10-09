"""The four steps of the window: Film, Characters, Script, Dub.

Pages only show and edit the project (``dubber.core.project.Project``); the window owns the project, starts the pipeline and
switches pages.  Every edit is written to the project folder at once, so a closed window loses nothing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSlider, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from dubber.core import actor_voice, media, script, subtitles, voices
from dubber.core.project import Project, Voice
from dubber.i18n import tr

TARGET_LANGS = ("ru", "en", "de")


def card(primary: bool = False) -> QFrame:
    f = QFrame()
    f.setObjectName("card")
    f.setProperty("primary", "true" if primary else "false")
    return f


def label(obj: str = "", wrap: bool = True) -> QLabel:
    lb = QLabel()
    if obj:
        lb.setObjectName(obj)
    lb.setWordWrap(wrap)
    return lb


def fmt_time(t: float, ms: bool = False) -> str:
    t = max(0.0, t)
    h, m, s = int(t // 3600), int(t % 3600 // 60), t % 60
    body = f"{m:02d}:{s:06.3f}" if ms else f"{m:02d}:{int(s):02d}"
    return f"{h}:{body}" if h or not ms else body


def fmt_eta(seconds: float) -> str:
    if seconds < 0:
        return tr("dub.eta_unknown")
    m = int(round(seconds / 60))
    return tr("dub.eta_hm", h=m // 60, m=m % 60) if m >= 60 else tr("dub.eta_m", m=max(1, m))


def voice_items() -> List[tuple]:
    """(Voice, label) choices: a clone from the film, then the voices of the shared Voxprint voice library (read-only)."""
    out = [(Voice("clone", ""), tr("voice.clone"))]
    for v in voices.list_library():
        extra = ", ".join(x for x in (v.language, v.gender) if x)
        out.append((Voice("library", v.id), tr("voice.library", name=v.name) + (f" ({extra})" if extra else "")))
    return out


def voice_key(v: Voice) -> str:
    return f"{v.kind}:{v.id}"


def voice_from_key(key: str) -> Voice:
    kind, _, vid = str(key).partition(":")
    return Voice(kind or "clone", vid)


def fill_voice_combo(cmb: QComboBox, current: Voice, items: Optional[List[tuple]] = None) -> None:
    cmb.blockSignals(True)
    cmb.clear()
    found = False
    for v, text in items if items is not None else voice_items():
        cmb.addItem(text, voice_key(v))
        found = found or (v.kind, v.id) == (current.kind, current.id)
    if not found and current.kind == "library":
        cmb.addItem(tr("voice.missing", id=current.id), voice_key(current))
    i = cmb.findData(voice_key(current))
    cmb.setCurrentIndex(max(0, i))
    cmb.blockSignals(False)


# ================================================================================================ 1. Film
class FilmPage(QWidget):
    choose_file = Signal()
    choose_subtitles = Signal()
    prepare = Signal()

    def __init__(self) -> None:
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        c = card()
        cl = QVBoxLayout(c)
        cl.setContentsMargins(18, 14, 18, 14)
        self.lbl_drop = label("sectiontitle")
        row = QHBoxLayout()
        self.btn_file = QPushButton()
        self.btn_file.clicked.connect(self.choose_file.emit)
        self.lbl_file = label("fileLabel")
        row.addWidget(self.btn_file)
        row.addWidget(self.lbl_file, 1)
        self.lbl_info = label("status")
        cl.addWidget(self.lbl_drop)
        cl.addLayout(row)
        cl.addWidget(self.lbl_info)
        lay.addWidget(c)

        c = card()
        g = QGridLayout(c)
        g.setContentsMargins(18, 14, 18, 14)
        g.setHorizontalSpacing(12)
        self.lbl_target = label("fileLabel", False)
        self.cmb_target = QComboBox()
        for code in TARGET_LANGS:
            self.cmb_target.addItem("", code)
        self.lbl_audio = label("fileLabel", False)
        self.cmb_audio = QComboBox()
        self.lbl_subs = label("fileLabel", False)
        self.cmb_subs = QComboBox()
        self.cmb_subs.addItem("", "auto")
        self.cmb_subs.addItem("", "none")
        self.btn_subs_file = QPushButton()
        self.btn_subs_file.clicked.connect(self.choose_subtitles.emit)
        self.lbl_subs_status = label("hint")
        self.lbl_profanity = label("fileLabel", False)
        self.cmb_profanity = QComboBox()
        self.cmb_profanity.addItem("", "keep")
        self.cmb_profanity.addItem("", "soften")
        g.addWidget(self.lbl_target, 0, 0)
        g.addWidget(self.cmb_target, 0, 1)
        g.addWidget(self.lbl_audio, 1, 0)
        g.addWidget(self.cmb_audio, 1, 1, 1, 2)
        g.addWidget(self.lbl_subs, 2, 0)
        g.addWidget(self.cmb_subs, 2, 1)
        g.addWidget(self.btn_subs_file, 2, 2)
        g.addWidget(self.lbl_subs_status, 3, 0, 1, 3)
        g.addWidget(self.lbl_profanity, 4, 0)
        g.addWidget(self.cmb_profanity, 4, 1)
        g.setColumnStretch(2, 1)
        lay.addWidget(c)

        c = card(primary=True)
        cl = QVBoxLayout(c)
        cl.setContentsMargins(18, 14, 18, 14)
        self.btn_prepare = QPushButton()
        self.btn_prepare.setObjectName("primary")
        self.btn_prepare.clicked.connect(self.prepare.emit)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.hide()
        self.lbl_prepare = label("status")
        cl.addWidget(self.btn_prepare)
        cl.addWidget(self.progress)
        cl.addWidget(self.lbl_prepare)
        lay.addWidget(c)
        lay.addStretch(1)
        self.info: Optional[media.MediaInfo] = None
        self.subs_file: str = ""

    def retranslate(self) -> None:
        self.lbl_drop.setText(tr("film.title"))
        self.btn_file.setText(tr("ui.choose_file"))
        if not self.info:
            self.lbl_file.setText(tr("film.drop_hint"))
        self.lbl_target.setText(tr("ui.lang_target"))
        for i, code in enumerate(TARGET_LANGS):
            self.cmb_target.setItemText(i, tr(f"lang.{code}"))
        self.lbl_audio.setText(tr("film.audio_track"))
        self.lbl_subs.setText(tr("film.subtitles"))
        self.cmb_subs.setItemText(0, tr("film.subs_auto"))
        self.cmb_subs.setItemText(1, tr("film.subs_none"))
        if self.cmb_subs.count() > 2:
            self.cmb_subs.setItemText(2, tr("film.subs_file", name=Path(self.subs_file).name))
        self.btn_subs_file.setText(tr("film.subs_choose"))
        self.lbl_profanity.setText(tr("film.profanity"))
        self.cmb_profanity.setItemText(0, tr("film.profanity_keep"))
        self.cmb_profanity.setItemText(1, tr("film.profanity_soften"))
        self.btn_prepare.setText(tr("film.prepare"))

    def target_lang(self) -> str:
        return self.cmb_target.currentData() or "ru"

    def set_media(self, path: Path, info: Optional[media.MediaInfo], error: str = "") -> None:
        self.info = info
        self.lbl_file.setText(str(path))
        self.cmb_audio.clear()
        if info is None:
            self.lbl_info.setText(tr("film.probe_failed", error=error))
            return
        subs_txt = ", ".join(t.label() for t in info.subtitles) or tr("film.none")
        self.lbl_info.setText(tr("film.info", duration=fmt_time(info.duration), video=info.video_codec or "?",
                                 audio=len(info.audio), subs=subs_txt))
        for t in info.audio:
            self.cmb_audio.addItem(t.label(), t.index)

    def set_subs_file(self, path: str) -> None:
        self.subs_file = path
        while self.cmb_subs.count() > 2:
            self.cmb_subs.removeItem(2)
        if path:
            self.cmb_subs.addItem(tr("film.subs_file", name=Path(path).name), path)
            self.cmb_subs.setCurrentIndex(2)

    def subtitle_choice(self) -> str:
        return self.cmb_subs.currentData() or "auto"

    def load(self, p: Project) -> None:
        i = self.cmb_target.findData(p.settings.get("target_lang"))
        self.cmb_target.setCurrentIndex(max(0, i))
        i = self.cmb_audio.findData(int(p.settings.get("audio_track") or 0))
        self.cmb_audio.setCurrentIndex(max(0, i))
        self.cmb_profanity.blockSignals(True)
        self.cmb_profanity.setCurrentIndex(max(0, self.cmb_profanity.findData(p.settings.get("profanity", "keep"))))
        self.cmb_profanity.blockSignals(False)
        ch = str(p.settings.get("subtitle_choice") or "auto")
        if ch not in ("auto", "none"):
            self.set_subs_file(ch)
        else:
            self.cmb_subs.setCurrentIndex(0 if ch == "auto" else 1)
        self.show_subtitle_status(p)

    def store(self, p: Project) -> None:
        p.settings["target_lang"] = self.target_lang()
        p.settings["audio_track"] = int(self.cmb_audio.currentData() or 0)
        p.settings["subtitle_choice"] = self.subtitle_choice()
        p.settings["profanity"] = self.cmb_profanity.currentData() or "keep"

    def show_subtitle_status(self, p: Project, online_keys: bool = False) -> None:
        """Where the translation will come from (before the run: a local look; after: what the subtitles stage found)."""
        found = p.settings.get("subs_found") or {}
        if p.stages.get("subtitles", {}).get("done"):
            tgt = found.get("target")
            text = (tr("film.subs_found", source=tr(f"film.src_{tgt['source']}"), label=tgt.get("label", "")) if tgt
                    else tr("film.subs_offline"))
        elif self.subtitle_choice() == "none":
            text = tr("film.subs_offline")
        elif self.subtitle_choice() != "auto":
            text = tr("film.subs_found", source=tr("film.src_file"), label=Path(self.subtitle_choice()).name)
        else:
            lang = self.target_lang()
            emb = media.embedded_for(self.info, lang) if self.info else None
            side = subtitles.sidecar_candidates(p.source, lang) if p.settings.get("source") else []
            if emb is not None:
                text = tr("film.subs_found", source=tr("film.src_embedded"), label=emb.label())
            elif side:
                text = tr("film.subs_found", source=tr("film.src_sidecar"), label=side[0].name)
            elif online_keys:
                text = tr("film.subs_will_search")
            else:
                text = tr("film.subs_no_keys")
        self.lbl_subs_status.setText(text)


# ================================================================================================ 2. Characters
class SpeakerCard(QFrame):
    def __init__(self, page: "CharactersPage", sp, samples: List[str], items: List[tuple]) -> None:
        super().__init__()
        self.setObjectName("card")
        self.sid = sp.id
        lay = QGridLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        self.chk = QCheckBox()
        self.edt_name = QLineEdit(sp.name or sp.id)
        self.edt_name.editingFinished.connect(lambda: page.rename(self.sid, self.edt_name.text()))
        self.chk_key = QCheckBox(tr("chars.key"))
        self.chk_key.setToolTip(tr("chars.key_tip"))
        self.chk_key.setChecked(page.project.is_key(sp) if page.project is not None else False)
        self.chk_key.toggled.connect(lambda on: page.set_key(self.sid, on))
        self.lbl_secs = label("hint", False)
        self.lbl_secs.setText(tr("chars.seconds", s=int(sp.seconds), n=page.count_lines(sp.id)))
        self.cmb_voice = QComboBox()
        fill_voice_combo(self.cmb_voice, sp.voice, items)
        self.cmb_voice.currentIndexChanged.connect(lambda _i: page.set_voice(self.sid, self.cmb_voice.currentData()))
        self.btn_play = QPushButton(tr("chars.listen"))
        self.btn_play.clicked.connect(lambda: page.listen.emit(self.sid))
        self.btn_save = QPushButton(tr("chars.save_actor"))
        self.btn_save.clicked.connect(lambda: page.save_actor.emit(self.sid))
        made = page.project is not None and (actor_voice.folder(page.project.folder, sp.id) / actor_voice.RECORD).is_file()
        self.btn_save.setVisible(sp.voice.kind == "actor" and made)
        self.lbl_samples = label("fileLabel")
        self.lbl_samples.setText("\n".join(f"“{s}”" for s in samples) or tr("chars.no_lines"))
        lay.addWidget(self.chk, 0, 0)
        self.chk.setToolTip(tr("chars.merge_hint"))
        lay.addWidget(self.edt_name, 0, 1)
        lay.addWidget(self.lbl_secs, 0, 2)
        lay.addWidget(self.cmb_voice, 0, 3)
        lay.addWidget(self.btn_play, 0, 4)
        lay.addWidget(self.chk_key, 1, 0, 1, 1, Qt.AlignmentFlag.AlignTop)
        lay.addWidget(self.lbl_samples, 1, 1, 1, 3)
        lay.addWidget(self.btn_save, 1, 4)
        lay.setColumnStretch(1, 1)


class CharactersPage(QWidget):
    changed = Signal(str)                  # what changed: voice | multi | merge | name
    listen = Signal(str)                   # speaker id ('' = the single voice)
    save_actor = Signal(str)               # speaker id: keep his actor-like voice in the library
    find_speakers = Signal()
    open_catalog = Signal()
    next_step = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.project: Optional[Project] = None
        self.cards: Dict[str, SpeakerCard] = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        c = card()
        cl = QVBoxLayout(c)
        cl.setContentsMargins(18, 14, 18, 14)
        self.lbl_title = label("sectiontitle")
        self.chk_multi = QCheckBox()
        self.chk_multi.toggled.connect(self._on_multi)
        self.lbl_multi_hint = label("hint")
        cl.addWidget(self.lbl_title)
        cl.addWidget(self.chk_multi)
        cl.addWidget(self.lbl_multi_hint)
        self.btn_catalog = QPushButton()
        self.btn_catalog.clicked.connect(self.open_catalog.emit)
        cl.addWidget(self.btn_catalog, 0, Qt.AlignmentFlag.AlignLeft)
        lay.addWidget(c)

        self.box_single = card()
        sl = QHBoxLayout(self.box_single)
        sl.setContentsMargins(18, 14, 18, 14)
        self.lbl_single = label("fileLabel", False)
        self.cmb_single = QComboBox()
        self.cmb_single.currentIndexChanged.connect(self._on_single)
        self.btn_single_listen = QPushButton()
        self.btn_single_listen.clicked.connect(lambda: self.listen.emit(""))
        sl.addWidget(self.lbl_single)
        sl.addWidget(self.cmb_single, 1)
        sl.addWidget(self.btn_single_listen)
        lay.addWidget(self.box_single)

        self.box_multi = QWidget()
        ml = QVBoxLayout(self.box_multi)
        ml.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.btn_find = QPushButton()
        self.btn_find.clicked.connect(self.find_speakers.emit)
        self.btn_merge = QPushButton()
        self.btn_merge.clicked.connect(self.merge_selected)
        self.lbl_multi_status = label("status")
        self.lbl_actor = label("fileLabel", False)
        self.sld_actor = QSlider(Qt.Orientation.Horizontal)
        self.sld_actor.setRange(0, 100)
        self.sld_actor.setFixedWidth(140)
        self.sld_actor.sliderReleased.connect(self._on_actor_weight)
        row.addWidget(self.btn_find)
        row.addWidget(self.btn_merge)
        row.addWidget(self.lbl_multi_status, 1)
        row.addWidget(self.lbl_actor)
        row.addWidget(self.sld_actor)
        ml.addLayout(row)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.cards_host = QWidget()
        self.cards_host.setObjectName("content")
        self.cards_lay = QVBoxLayout(self.cards_host)
        self.cards_lay.setContentsMargins(0, 0, 0, 0)
        self.cards_lay.addStretch(1)
        self.scroll.setWidget(self.cards_host)
        ml.addWidget(self.scroll, 1)
        lay.addWidget(self.box_multi, 1)
        self.btn_next = QPushButton()
        self.btn_next.clicked.connect(self.next_step.emit)
        lay.addWidget(self.btn_next, 0, Qt.AlignmentFlag.AlignRight)

    def retranslate(self) -> None:
        self.lbl_title.setText(tr("chars.title"))
        self.chk_multi.setText(tr("chars.multi"))
        self.lbl_multi_hint.setText(tr("chars.multi_hint"))
        self.lbl_single.setText(tr("chars.single_voice"))
        self.btn_single_listen.setText(tr("chars.listen"))
        self.btn_find.setText(tr("chars.find"))
        self.btn_merge.setText(tr("chars.merge"))
        self.btn_next.setText(tr("chars.next"))
        self.btn_catalog.setText(tr("chars.catalog"))
        self.lbl_actor.setText(tr("chars.actor_weight"))
        self.sld_actor.setToolTip(tr("chars.actor_weight_tip"))
        if self.project is not None:
            self.load(self.project)

    # ------------------------------------------------------------------ model -> view
    def count_lines(self, sid: str) -> int:
        return sum(1 for ln in self.project.lines if ln.speaker == sid) if self.project else 0

    def diarized(self) -> bool:
        st = (self.project.stages.get("diarization") or {}) if self.project else {}
        return bool(st.get("done")) and "skipped" not in str(st.get("summary", ""))

    def load(self, p: Project) -> None:
        self.project = p
        multi = bool(p.settings.get("multi_voice"))
        self.chk_multi.blockSignals(True)
        self.chk_multi.setChecked(multi)
        self.chk_multi.blockSignals(False)
        items = voice_items()
        fill_voice_combo(self.cmb_single, Voice.from_dict(p.settings.get("single_voice")), items)
        self.box_single.setVisible(not multi)
        self.box_multi.setVisible(multi)
        for c in self.cards.values():
            c.setParent(None)
            c.deleteLater()
        self.cards = {}
        self.sld_actor.setValue(int(round(100 * float(p.settings.get("actor_weight", actor_voice.DEFAULT_WEIGHT)))))
        card_items = [(Voice("auto", ""), tr("voice.auto"))] + items[:1] + [(Voice("actor", ""), tr("voice.actor"))] + items[1:]
        if multi:
            for sp in sorted(p.speakers, key=lambda s: -s.seconds):
                samples = [ln.text or ln.translation for ln in p.lines if ln.speaker == sp.id and (ln.text or ln.translation)][:3]
                cardw = SpeakerCard(self, sp, samples, card_items)
                self.cards[sp.id] = cardw
                self.cards_lay.insertWidget(self.cards_lay.count() - 1, cardw)
        need = multi and not self.diarized()
        self.btn_find.setVisible(need)
        self.btn_merge.setVisible(multi and len(p.speakers) > 1)
        self.lbl_multi_status.setText(tr("chars.need_find") if need else tr("chars.found", n=len(p.speakers)) if multi else "")

    # ------------------------------------------------------------------ edits
    def _on_multi(self, on: bool) -> None:
        if self.project is None:
            return
        self.project.settings["multi_voice"] = bool(on)
        self.project.save()
        self.load(self.project)
        self.changed.emit("multi")

    def _on_actor_weight(self) -> None:
        if self.project is not None:
            self.project.settings["actor_weight"] = round(self.sld_actor.value() / 100.0, 2)
            self.project.save()
            self.changed.emit("voice")

    def _on_single(self, _i: int) -> None:
        if self.project is None or self.cmb_single.currentData() is None:
            return
        v = voice_from_key(self.cmb_single.currentData())
        self.project.settings["single_voice"] = {"kind": v.kind, "id": v.id}
        self.project.save()
        self.changed.emit("voice")

    def set_voice(self, sid: str, data) -> None:
        sp = self.project.speaker(sid) if self.project else None
        if sp is None or data is None:
            return
        sp.voice = voice_from_key(data)
        self.project.save()
        self.changed.emit("voice")

    def set_key(self, sid: str, on: bool) -> None:
        sp = self.project.speaker(sid) if self.project else None
        if sp is not None and on != self.project.is_key(sp):
            sp.key = bool(on)
            self.project.save()
            self.changed.emit("voice")

    def rename(self, sid: str, name: str) -> None:
        sp = self.project.speaker(sid) if self.project else None
        if sp is not None and name.strip() and name.strip() != sp.name:
            sp.name = name.strip()
            self.project.save()
            self.changed.emit("name")

    def selected(self) -> List[str]:
        return [sid for sid, c in self.cards.items() if c.chk.isChecked()]

    def merge_selected(self) -> None:
        sel = self.selected()
        if self.project is None or len(sel) < 2:
            self.lbl_multi_status.setText(tr("chars.merge_hint"))
            return
        keep = max(sel, key=lambda s: self.project.speaker(s).seconds)
        self.project.merge_speakers(keep, sel)
        self.load(self.project)
        self.changed.emit("merge")


# ================================================================================================ 3. Script
COL_TIME, COL_SPEAKER, COL_ORIG, COL_TRANS, COL_FLAG, COL_KEEP = range(6)
SOFTENED_BG = "#4a3a12"            # amber tint: the profanity filter changed this line (text stays >= 4.5:1, see theme tests)


class ScriptPage(QWidget):
    changed = Signal()
    next_step = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.project: Optional[Project] = None
        self._loading = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.lbl_title = label("sectiontitle", False)
        self.chk_only_long = QCheckBox()
        self.chk_only_long.toggled.connect(self._filter)
        self.lbl_count = label("hint", False)
        row.addWidget(self.lbl_title)
        row.addStretch(1)
        row.addWidget(self.lbl_count)
        row.addWidget(self.chk_only_long)
        lay.addLayout(row)
        self.table = QTableWidget(0, 6)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setWordWrap(True)
        hh = self.table.horizontalHeader()
        for col, mode in ((COL_TIME, QHeaderView.ResizeMode.ResizeToContents), (COL_SPEAKER, QHeaderView.ResizeMode.ResizeToContents),
                          (COL_ORIG, QHeaderView.ResizeMode.Stretch), (COL_TRANS, QHeaderView.ResizeMode.Stretch),
                          (COL_FLAG, QHeaderView.ResizeMode.ResizeToContents), (COL_KEEP, QHeaderView.ResizeMode.ResizeToContents)):
            hh.setSectionResizeMode(col, mode)
        self.table.itemChanged.connect(self._on_item)
        lay.addWidget(self.table, 1)
        self.lbl_hint = label("hint")
        lay.addWidget(self.lbl_hint)
        self.btn_next = QPushButton()
        self.btn_next.clicked.connect(self.next_step.emit)
        lay.addWidget(self.btn_next, 0, Qt.AlignmentFlag.AlignRight)

    def retranslate(self) -> None:
        self.lbl_title.setText(tr("script.title"))
        self.chk_only_long.setText(tr("script.only_long"))
        self.table.setHorizontalHeaderLabels([tr("script.col_time"), tr("script.col_speaker"), tr("script.col_original"),
                                              tr("script.col_translation"), tr("script.col_fit"), tr("script.col_keep")])
        self.lbl_hint.setText(tr("script.hint"))
        self.btn_next.setText(tr("script.next"))
        if self.project is not None:
            self.load(self.project)

    def _flag(self, i: int) -> str:
        p = self.project
        ln = p.lines[i]
        if ln.keep_original:
            return tr("script.flag_kept")
        if ln.softened:
            return tr("script.flag_softened")
        nxt = p.lines[i + 1].start if i + 1 < len(p.lines) else None
        if ln.fit == "too_long" or script.too_long(ln, p.settings.get("target_lang", "ru"), next_start=nxt):
            return tr("script.flag_long")
        return {"stretched": tr("script.flag_stretched"), "shifted": tr("script.flag_shifted")}.get(ln.fit, "")

    def load(self, p: Project) -> None:
        self.project = p
        self._loading = True
        self.table.setRowCount(len(p.lines))
        names = {s.id: (s.name or s.id) for s in p.speakers}
        multi = bool(p.settings.get("multi_voice"))
        ro = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
        for i, ln in enumerate(p.lines):
            cells = [QTableWidgetItem(f"{fmt_time(ln.start, True)}\n{fmt_time(ln.end, True)}"),
                     QTableWidgetItem(names.get(ln.speaker, ln.speaker)), QTableWidgetItem(ln.text),
                     QTableWidgetItem(ln.translation), QTableWidgetItem(self._flag(i)), QTableWidgetItem()]
            cells[COL_TIME].setData(Qt.ItemDataRole.UserRole, ln.id)
            for col in (COL_TIME, COL_ORIG, COL_FLAG):
                cells[col].setFlags(ro)
            if not multi:
                cells[COL_SPEAKER].setFlags(ro)
            if ln.softened:
                cells[COL_TRANS].setBackground(QColor(SOFTENED_BG))
                cells[COL_TRANS].setToolTip(tr("script.softened_tip", text=ln.softened))
            cells[COL_KEEP].setFlags(ro | Qt.ItemFlag.ItemIsUserCheckable)
            cells[COL_KEEP].setCheckState(Qt.CheckState.Checked if ln.keep_original else Qt.CheckState.Unchecked)
            for col, it in enumerate(cells):
                self.table.setItem(i, col, it)
        self._loading = False
        self.table.resizeRowsToContents()
        self._filter()
        n_long = sum(1 for i in range(len(p.lines)) if self._flag(i) == tr("script.flag_long"))
        self.lbl_count.setText(tr("script.count", n=len(p.lines), long=n_long))

    def _filter(self, *_a) -> None:
        only = self.chk_only_long.isChecked()
        for r in range(self.table.rowCount()):
            it = self.table.item(r, COL_FLAG)
            self.table.setRowHidden(r, only and (it is None or it.text() != tr("script.flag_long")))

    def _on_item(self, it: QTableWidgetItem) -> None:
        if self._loading or self.project is None:
            return
        r = it.row()
        ln = self.project.line(int(self.table.item(r, COL_TIME).data(Qt.ItemDataRole.UserRole)))
        if ln is None:
            return
        col = it.column()
        if col == COL_TRANS and it.text() != ln.translation:
            ln.translation, ln.spoken, ln.edited = it.text().strip(), "", True
            ln.audio, ln.fit, ln.softened = "", "", ""
            self._loading = True
            it.setBackground(QColor(0, 0, 0, 0))
            it.setToolTip("")
            self._loading = False
        elif col == COL_KEEP:
            ln.keep_original = it.checkState() == Qt.CheckState.Checked
        elif col == COL_SPEAKER:
            name = it.text().strip()
            sid = next((s.id for s in self.project.speakers if name in (s.id, s.name)), None)
            if sid is None and name:
                sid = name
            if sid and sid != ln.speaker:
                self.project.reassign(ln.id, sid)
        else:
            return
        self.project.save()
        self._loading = True
        self.table.item(r, COL_FLAG).setText(self._flag(r))
        self._loading = False
        self.changed.emit()

    def set_editable(self, on: bool) -> None:
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked | QAbstractItemView.EditTrigger.EditKeyPressed
                                   if on else QAbstractItemView.EditTrigger.NoEditTriggers)


# ================================================================================================ 4. Dub
class DubPage(QWidget):
    dub = Signal()
    cancel = Signal()
    preview = Signal()
    watch = Signal()
    open_result = Signal()
    external = Signal()
    volume = Signal(float)

    def __init__(self, player: QWidget) -> None:
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        c = card(primary=True)
        cl = QVBoxLayout(c)
        cl.setContentsMargins(18, 14, 18, 14)
        row = QHBoxLayout()
        self.btn_dub = QPushButton()
        self.btn_dub.setObjectName("primary")
        self.btn_dub.clicked.connect(self.dub.emit)
        self.btn_cancel = QPushButton()
        self.btn_cancel.clicked.connect(self.cancel.emit)
        self.btn_cancel.hide()
        row.addWidget(self.btn_dub, 1)
        row.addWidget(self.btn_cancel)
        cl.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.lbl_stage = label("status")
        self.lbl_eta = label("hint", False)
        cl.addWidget(self.progress)
        r2 = QHBoxLayout()
        r2.addWidget(self.lbl_stage, 1)
        r2.addWidget(self.lbl_eta)
        cl.addLayout(r2)
        vol = QHBoxLayout()
        self.lbl_volume = label("fileLabel", False)
        self.sld_volume = QSlider(Qt.Orientation.Horizontal)
        self.sld_volume.setRange(0, 100)
        self.sld_volume.setValue(15)
        self.sld_volume.valueChanged.connect(self._volume_text)
        self.sld_volume.sliderReleased.connect(lambda: self.volume.emit(self.sld_volume.value() / 100.0))
        self.lbl_volume_value = label("hint", False)
        vol.addWidget(self.lbl_volume)
        vol.addWidget(self.sld_volume, 1)
        vol.addWidget(self.lbl_volume_value)
        cl.addLayout(vol)
        lay.addWidget(c)

        c = card()
        cl = QVBoxLayout(c)
        cl.setContentsMargins(18, 14, 18, 14)
        row = QHBoxLayout()
        self.btn_preview = QPushButton()
        self.btn_preview.clicked.connect(self.preview.emit)
        self.btn_watch = QPushButton()
        self.btn_watch.setEnabled(False)
        self.btn_watch.clicked.connect(self.watch.emit)
        self.btn_open = QPushButton()
        self.btn_open.clicked.connect(self.open_result.emit)
        self.btn_open.setEnabled(False)
        self.btn_external = QPushButton()
        self.btn_external.clicked.connect(self.external.emit)
        self.btn_external.setEnabled(False)
        for b in (self.btn_preview, self.btn_watch, self.btn_open, self.btn_external):
            row.addWidget(b)
        row.addStretch(1)
        self.lbl_watch = label("hint")
        cl.addLayout(row)
        cl.addWidget(self.lbl_watch)
        cl.addWidget(player, 1)
        lay.addWidget(c, 1)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(800)
        self.log.setFixedHeight(110)
        lay.addWidget(self.log)
        self._volume_text(15)

    def _volume_text(self, v: int) -> None:
        self.lbl_volume_value.setText(f"{v} %" if v else tr("dub.volume_off"))

    def retranslate(self) -> None:
        self.btn_dub.setText(tr("dub.start"))
        self.btn_cancel.setText(tr("ui.btn_cancel"))
        self.lbl_volume.setText(tr("dub.volume"))
        self.btn_preview.setText(tr("dub.preview"))
        self.btn_watch.setText(tr("dub.watch"))
        self.btn_open.setText(tr("dub.open"))
        self.btn_external.setText(tr("dub.external"))
        self._volume_text(self.sld_volume.value())

    def load(self, p: Project) -> None:
        self.sld_volume.blockSignals(True)
        self.sld_volume.setValue(int(round(100 * float(p.settings.get("original_volume", 0.15)))))
        self.sld_volume.blockSignals(False)
        self._volume_text(self.sld_volume.value())
        out = p.settings.get("output_file")
        self.btn_open.setEnabled(bool(out) and Path(out).exists())
        self.btn_external.setEnabled(bool(out) and Path(out).exists())

    def running(self, on: bool) -> None:
        self.btn_dub.setEnabled(not on)
        self.btn_cancel.setVisible(on)
        self.btn_preview.setEnabled(not on)
        self.sld_volume.setEnabled(not on)
