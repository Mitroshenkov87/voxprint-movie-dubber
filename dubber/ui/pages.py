"""The four steps of the window: Film, Characters, Lines, Dub.

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


class AutoLabel(QLabel):
    """A status label that takes no room while it has nothing to say."""

    def setText(self, text: str) -> None:  # noqa: N802
        super().setText(text)
        self.setVisible(bool(text))


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
    """Drop a film, press Dub: the whole pipeline runs with the best settings and the dubbed file appears next to the film.

    Everything that can be changed sits in the collapsed Options (preset, remembered for the next film); "Prepare and review"
    is the optional path for people who want to check the characters and lines before the dub."""
    choose_file = Signal()
    choose_subtitles = Signal()
    dub = Signal()
    prepare = Signal()                     # "Prepare and review"
    cancel = Signal()
    show_result = Signal()
    watch = Signal()
    option_changed = Signal(str)           # target | audio | subs | profanity | multi | voice | volume | format
    options_toggled = Signal(bool)

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

        # ---- the one big action
        c = card(primary=True)
        cl = QVBoxLayout(c)
        cl.setContentsMargins(18, 14, 18, 14)
        self.lbl_plan = label("status")
        row = QHBoxLayout()
        self.btn_dub = QPushButton()
        self.btn_dub.setObjectName("primary")
        self.btn_dub.setProperty("big", "true")
        self.btn_dub.setMinimumHeight(56)
        self.btn_dub.clicked.connect(self.dub.emit)
        self.btn_cancel = QPushButton()
        self.btn_cancel.clicked.connect(self.cancel.emit)
        self.btn_cancel.hide()
        row.addWidget(self.btn_dub, 1)
        row.addWidget(self.btn_cancel)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.hide()
        r2 = QHBoxLayout()
        self.lbl_prepare = AutoLabel()                 # what runs now / the result
        self.lbl_prepare.setObjectName("status")
        self.lbl_prepare.setWordWrap(True)
        self.lbl_prepare.hide()
        self.lbl_eta = AutoLabel()
        self.lbl_eta.setObjectName("hint")
        self.lbl_eta.hide()
        r2.addWidget(self.lbl_prepare, 1)
        r2.addWidget(self.lbl_eta)
        r3 = QHBoxLayout()
        self.btn_show = QPushButton()
        self.btn_show.clicked.connect(self.show_result.emit)
        self.btn_watch = QPushButton()
        self.btn_watch.clicked.connect(self.watch.emit)
        for bt in (self.btn_show, self.btn_watch):
            bt.hide()
            r3.addWidget(bt)
        r3.addStretch(1)
        self.lbl_attention = label("warn")
        self.lbl_attention.hide()
        r4 = QHBoxLayout()
        self.btn_prepare = QPushButton()
        self.btn_prepare.clicked.connect(self.prepare.emit)
        self.lbl_review_hint = label("hint")
        r4.addWidget(self.btn_prepare)
        r4.addWidget(self.lbl_review_hint, 1)
        cl.addWidget(self.lbl_plan)
        cl.addLayout(row)
        cl.addWidget(self.progress)
        cl.addLayout(r2)
        cl.addLayout(r3)
        cl.addWidget(self.lbl_attention)
        cl.addLayout(r4)
        lay.addWidget(c)

        # ---- Options (collapsed, preset)
        c = card()
        ol = QVBoxLayout(c)
        ol.setContentsMargins(18, 10, 18, 10)
        self.btn_options = QPushButton()
        self.btn_options.setObjectName("disclosure")
        self.btn_options.setCheckable(True)
        self.btn_options.toggled.connect(self._toggle_options)
        ol.addWidget(self.btn_options, 0, Qt.AlignmentFlag.AlignLeft)
        self.options_body = QWidget()
        self.options_body.setObjectName("content")
        g = QGridLayout(self.options_body)
        g.setContentsMargins(0, 6, 0, 4)
        g.setHorizontalSpacing(12)
        self.lbl_options_hint = label("hint")
        self.lbl_target = label("fileLabel", False)
        self.cmb_target = QComboBox()
        for code in TARGET_LANGS:
            self.cmb_target.addItem("", code)
        self.lbl_audio = label("fileLabel", False)
        self.cmb_audio = QComboBox()
        self.cmb_audio.addItem("", -1)
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
        self.chk_multi = QCheckBox()
        self.lbl_multi_hint = label("hint")
        self.lbl_voice = label("fileLabel", False)
        self.cmb_voice = QComboBox()
        self.lbl_volume = label("fileLabel", False)
        self.sld_volume = QSlider(Qt.Orientation.Horizontal)
        self.sld_volume.setRange(0, 100)
        self.sld_volume.setValue(15)
        self.lbl_volume_value = label("hint", False)
        self.sld_volume.valueChanged.connect(self._volume_text)
        self.lbl_format = label("fileLabel", False)
        self.cmb_format = QComboBox()
        self.cmb_format.addItem("", "mkv")
        self.cmb_format.addItem("", "mp4")
        rows = [(self.lbl_target, self.cmb_target, None), (self.lbl_audio, self.cmb_audio, None),
                (self.lbl_subs, self.cmb_subs, self.btn_subs_file)]
        g.addWidget(self.lbl_options_hint, 0, 0, 1, 3)
        r = 1
        for lb, w, extra in rows:
            g.addWidget(lb, r, 0)
            g.addWidget(w, r, 1, 1, 1 if extra else 2)
            if extra:
                g.addWidget(extra, r, 2)
            r += 1
        g.addWidget(self.lbl_subs_status, r, 0, 1, 3)
        r += 1
        g.addWidget(self.lbl_profanity, r, 0)
        g.addWidget(self.cmb_profanity, r, 1)
        r += 1
        g.addWidget(self.chk_multi, r, 0, 1, 3)
        r += 1
        g.addWidget(self.lbl_multi_hint, r, 0, 1, 3)
        r += 1
        g.addWidget(self.lbl_voice, r, 0)
        g.addWidget(self.cmb_voice, r, 1, 1, 2)
        r += 1
        vol = QHBoxLayout()
        vol.addWidget(self.sld_volume, 1)
        vol.addWidget(self.lbl_volume_value)
        g.addWidget(self.lbl_volume, r, 0)
        g.addLayout(vol, r, 1, 1, 2)
        r += 1
        g.addWidget(self.lbl_format, r, 0)
        g.addWidget(self.cmb_format, r, 1)
        g.setColumnStretch(2, 1)
        g.setColumnStretch(1, 1)
        self.options_body.hide()
        ol.addWidget(self.options_body)
        lay.addWidget(c)
        lay.addStretch(1)

        self.cmb_target.currentIndexChanged.connect(lambda _i: self.option_changed.emit("target"))
        self.cmb_audio.currentIndexChanged.connect(lambda _i: self.option_changed.emit("audio"))
        self.cmb_subs.currentIndexChanged.connect(lambda _i: self.option_changed.emit("subs"))
        self.cmb_profanity.currentIndexChanged.connect(lambda _i: self.option_changed.emit("profanity"))
        def _on_multi(_on: bool) -> None:
            self._multi_visibility()
            self.option_changed.emit("multi")

        self.chk_multi.toggled.connect(_on_multi)
        self.cmb_voice.currentIndexChanged.connect(lambda _i: self.option_changed.emit("voice"))
        self.sld_volume.sliderReleased.connect(lambda: self.option_changed.emit("volume"))
        self.cmb_format.currentIndexChanged.connect(lambda _i: self.option_changed.emit("format"))
        self.info: Optional[media.MediaInfo] = None
        self.subs_file: str = ""
        self._volume_text(15)

    # ------------------------------------------------------------------ texts
    def retranslate(self) -> None:
        self.lbl_drop.setText(tr("film.title"))
        self.btn_file.setText(tr("ui.choose_file"))
        if not self.info:
            self.lbl_file.setText(tr("film.drop_hint"))
        self.btn_dub.setText(tr("film.dub"))
        self.btn_cancel.setText(tr("ui.btn_cancel"))
        self.btn_show.setText(tr("film.show"))
        self.btn_watch.setText(tr("film.watch"))
        self.btn_prepare.setText(tr("film.review"))
        self.lbl_review_hint.setText(tr("film.review_hint"))
        self._options_title()
        self.lbl_options_hint.setText(tr("film.options_hint"))
        self.lbl_target.setText(tr("ui.lang_target"))
        for i, code in enumerate(TARGET_LANGS):
            self.cmb_target.setItemText(i, tr(f"lang.{code}"))
        self.lbl_audio.setText(tr("film.audio_track"))
        self._audio_auto_text()
        self.lbl_subs.setText(tr("film.subtitles"))
        self.cmb_subs.setItemText(0, tr("film.subs_auto"))
        self.cmb_subs.setItemText(1, tr("film.subs_none"))
        if self.cmb_subs.count() > 2:
            self.cmb_subs.setItemText(2, tr("film.subs_file", name=Path(self.subs_file).name))
        self.btn_subs_file.setText(tr("film.subs_choose"))
        self.lbl_profanity.setText(tr("film.profanity"))
        self.cmb_profanity.setItemText(0, tr("film.profanity_keep"))
        self.cmb_profanity.setItemText(1, tr("film.profanity_soften"))
        self.chk_multi.setText(tr("chars.multi"))
        self.lbl_multi_hint.setText(tr("chars.multi_hint"))
        self.lbl_voice.setText(tr("chars.single_voice"))
        self.lbl_volume.setText(tr("dub.volume"))
        self._volume_text(self.sld_volume.value())
        self.lbl_format.setText(tr("film.format"))
        self.cmb_format.setItemText(0, tr("film.format_mkv"))
        self.cmb_format.setItemText(1, tr("film.format_mp4"))

    def _options_title(self) -> None:
        self.btn_options.setText(("\u25be  " if self.btn_options.isChecked() else "\u25b8  ") + tr("film.options"))

    def _toggle_options(self, on: bool) -> None:
        self.options_body.setVisible(on)
        self._options_title()
        self.options_toggled.emit(on)

    def set_options_open(self, on: bool) -> None:
        self.btn_options.blockSignals(True)
        self.btn_options.setChecked(on)
        self.btn_options.blockSignals(False)
        self.options_body.setVisible(on)
        self._options_title()

    def _volume_text(self, v: int) -> None:
        self.lbl_volume_value.setText(f"{v} %" if v else tr("dub.volume_off"))

    def _audio_auto_text(self) -> None:
        if self.info is not None and self.info.audio:
            t = next((t for t in self.info.audio if t.index == self.auto_track), self.info.audio[0])
            self.cmb_audio.setItemText(0, tr("film.audio_auto", label=t.label()))
        else:
            self.cmb_audio.setItemText(0, tr("film.audio_auto_none"))

    def _multi_visibility(self) -> None:
        multi = self.chk_multi.isChecked()
        self.lbl_voice.setVisible(not multi)
        self.cmb_voice.setVisible(not multi)

    # ------------------------------------------------------------------ model <-> view
    @property
    def auto_track(self) -> int:
        return media.pick_original_track(self.info, self.target_lang())

    def target_lang(self) -> str:
        return self.cmb_target.currentData() or "ru"

    def set_media(self, path: Path, info: Optional[media.MediaInfo], error: str = "") -> None:
        self.info = info
        self.lbl_file.setText(str(path))
        self.cmb_audio.blockSignals(True)
        while self.cmb_audio.count() > 1:
            self.cmb_audio.removeItem(1)
        if info is not None:
            for t in info.audio:
                self.cmb_audio.addItem(t.label(), t.index)
        self._audio_auto_text()
        self.cmb_audio.blockSignals(False)
        if info is None:
            self.lbl_info.setText(tr("film.probe_failed", error=error))
            return
        subs_txt = ", ".join(t.label() for t in info.subtitles) or tr("film.none")
        self.lbl_info.setText(tr("film.info", duration=fmt_time(info.duration), video=info.video_codec or "?",
                                 audio=len(info.audio), subs=subs_txt))

    def set_subs_file(self, path: str) -> None:
        self.subs_file = path
        while self.cmb_subs.count() > 2:
            self.cmb_subs.removeItem(2)
        if path:
            self.cmb_subs.addItem(tr("film.subs_file", name=Path(path).name), path)
            self.cmb_subs.setCurrentIndex(2)

    def subtitle_choice(self) -> str:
        return self.cmb_subs.currentData() or "auto"

    def multi(self) -> bool:
        return self.chk_multi.isChecked()

    def _widgets(self):
        return (self.cmb_target, self.cmb_audio, self.cmb_subs, self.cmb_profanity, self.chk_multi, self.cmb_voice, self.sld_volume,
                self.cmb_format)

    def load(self, p: Project) -> None:
        for w in self._widgets():
            w.blockSignals(True)
        self.cmb_target.setCurrentIndex(max(0, self.cmb_target.findData(p.settings.get("target_lang"))))
        self.cmb_audio.setCurrentIndex(0 if p.settings.get("audio_track_auto", True)
                                       else max(0, self.cmb_audio.findData(int(p.settings.get("audio_track") or 0))))
        self._audio_auto_text()
        self.cmb_profanity.setCurrentIndex(max(0, self.cmb_profanity.findData(p.settings.get("profanity", "keep"))))
        ch = str(p.settings.get("subtitle_choice") or "auto")
        if ch not in ("auto", "none"):
            self.set_subs_file(ch)
        else:
            self.cmb_subs.setCurrentIndex(0 if ch == "auto" else 1)
        self.chk_multi.setChecked(bool(p.settings.get("multi_voice")))
        fill_voice_combo(self.cmb_voice, Voice.from_dict(p.settings.get("single_voice")))
        self.sld_volume.setValue(int(round(100 * float(p.settings.get("original_volume", 0.15)))))
        self._volume_text(self.sld_volume.value())
        fmt = str(p.settings.get("output_format") or "mkv")
        self.cmb_format.setCurrentIndex(1 if fmt == "mp4" else 0)
        for w in self._widgets():
            w.blockSignals(False)
        self._multi_visibility()
        self.show_subtitle_status(p)

    def store(self, p: Project) -> None:
        p.settings["target_lang"] = self.target_lang()
        auto = (self.cmb_audio.currentData() if self.cmb_audio.currentData() is not None else -1) == -1
        p.settings["audio_track_auto"] = auto
        p.settings["audio_track"] = self.auto_track if auto else int(self.cmb_audio.currentData() or 0)
        p.settings["subtitle_choice"] = self.subtitle_choice()
        p.settings["profanity"] = self.cmb_profanity.currentData() or "keep"
        p.settings["multi_voice"] = self.multi()
        if self.cmb_voice.currentData() is not None:
            v = voice_from_key(self.cmb_voice.currentData())
            p.settings["single_voice"] = {"kind": v.kind, "id": v.id}
        p.settings["original_volume"] = round(self.sld_volume.value() / 100.0, 2)
        p.settings["output_format"] = self.cmb_format.currentData() or "mkv"

    def set_plan(self, p: Optional[Project], out_name: str = "") -> None:
        """One line under the button: what pressing Dub will make."""
        if p is None:
            self.lbl_plan.setText(tr("film.plan_none"))
            return
        if p.settings.get("multi_voice"):
            voice = tr("film.voice_multi")
        else:
            v = Voice.from_dict(p.settings.get("single_voice"))
            lv = voices.get_library_voice(v.id) if v.kind == "library" else None
            voice = tr("film.voice_single", name=lv.name) if lv else tr("film.voice_clone")
        self.lbl_plan.setText(tr("film.plan", lang=tr(f"lang.{p.settings.get('target_lang', 'ru')}"), voice=voice, name=out_name))

    def running(self, on: bool) -> None:
        self.btn_dub.setEnabled(not on)
        self.btn_prepare.setEnabled(not on)
        self.btn_cancel.setVisible(on)
        self.btn_file.setEnabled(not on)
        self.options_body.setEnabled(not on)
        if on:
            self.btn_show.hide()
            self.lbl_eta.setText("")

    def set_attention(self, items: List[str]) -> None:
        self.lbl_attention.setText(tr("review.title", items="; ".join(items)) if items else "")
        self.lbl_attention.setVisible(bool(items))

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
    """Optional review of the characters: only in the multi-voice mode (switched on in the Film screen's Options)."""
    changed = Signal(str)                  # what changed: voice | merge | name
    listen = Signal(str)                   # speaker id
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
        self.lbl_disabled = label("hint")
        cl.addWidget(self.lbl_title)
        cl.addWidget(self.lbl_disabled)
        self.btn_catalog = QPushButton()
        self.btn_catalog.clicked.connect(self.open_catalog.emit)
        cl.addWidget(self.btn_catalog, 0, Qt.AlignmentFlag.AlignLeft)
        lay.addWidget(c)

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
        self.lbl_disabled.setText(tr("chars.disabled"))
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
        items = voice_items()
        self.lbl_disabled.setVisible(not multi)
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
    def _on_actor_weight(self) -> None:
        if self.project is not None:
            self.project.settings["actor_weight"] = round(self.sld_actor.value() / 100.0, 2)
            self.project.save()
            self.changed.emit("voice")

    def set_voice(self, sid: str, data) -> None:
        project = self.project
        sp = project.speaker(sid) if project else None
        if project is None or sp is None or data is None:
            return
        sp.voice = voice_from_key(data)
        project.save()
        self.changed.emit("voice")

    def set_key(self, sid: str, on: bool) -> None:
        project = self.project
        sp = project.speaker(sid) if project else None
        if project is not None and sp is not None and on != project.is_key(sp):
            sp.key = bool(on)
            project.save()
            self.changed.emit("voice")

    def rename(self, sid: str, name: str) -> None:
        project = self.project
        sp = project.speaker(sid) if project else None
        if project is not None and sp is not None and name.strip() and name.strip() != sp.name:
            sp.name = name.strip()
            project.save()
            self.changed.emit("name")

    def selected(self) -> List[str]:
        return [sid for sid, c in self.cards.items() if c.chk.isChecked()]

    def merge_selected(self) -> None:
        sel = self.selected()
        if self.project is None or len(sel) < 2:
            self.lbl_multi_status.setText(tr("chars.merge_hint"))
            return
        project = self.project
        if project is None:
            return

        def _seconds(sid: str) -> float:
            speaker = project.speaker(sid)
            return speaker.seconds if speaker is not None else 0.0

        keep = max(sel, key=_seconds)
        project.merge_speakers(keep, sel)
        self.load(project)
        self.changed.emit("merge")


# ================================================================================================ 3. Lines
COL_TIME, COL_SPEAKER, COL_ORIG, COL_TRANS, COL_FLAG, COL_KEEP = range(6)
SOFTENED_BG = "#4a3a12"            # amber tint: the profanity filter changed this line (text stays >= 4.5:1, see theme tests)


class LinesPage(QWidget):
    """Optional review of the lines (made automatically): edit a translation, keep a line in the original, find the long ones."""
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
        self.lbl_title.setText(tr("lines.title"))
        self.chk_only_long.setText(tr("lines.only_long"))
        self.table.setHorizontalHeaderLabels([tr("lines.col_time"), tr("lines.col_speaker"), tr("lines.col_original"),
                                              tr("lines.col_translation"), tr("lines.col_fit"), tr("lines.col_keep")])
        self.lbl_hint.setText(tr("lines.hint"))
        self.btn_next.setText(tr("lines.next"))
        if self.project is not None:
            self.load(self.project)

    def _flag(self, i: int) -> str:
        p = self.project
        if p is None:
            return ""
        ln = p.lines[i]
        if ln.keep_original:
            return tr("lines.flag_kept")
        if ln.softened:
            return tr("lines.flag_softened")
        nxt = p.lines[i + 1].start if i + 1 < len(p.lines) else None
        if ln.fit == "too_long" or script.too_long(ln, p.settings.get("target_lang", "ru"), next_start=nxt):
            return tr("lines.flag_long")
        return {"stretched": tr("lines.flag_stretched"), "shifted": tr("lines.flag_shifted")}.get(ln.fit, "")

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
                cells[COL_TRANS].setToolTip(tr("lines.softened_tip", text=ln.softened))
            cells[COL_KEEP].setFlags(ro | Qt.ItemFlag.ItemIsUserCheckable)
            cells[COL_KEEP].setCheckState(Qt.CheckState.Checked if ln.keep_original else Qt.CheckState.Unchecked)
            for col, it in enumerate(cells):
                self.table.setItem(i, col, it)
        self._loading = False
        self.table.resizeRowsToContents()
        self._filter()
        n_long = sum(1 for i in range(len(p.lines)) if self._flag(i) == tr("lines.flag_long"))
        self.lbl_count.setText(tr("lines.count", n=len(p.lines), long=n_long))

    def _filter(self, *_a) -> None:
        only = self.chk_only_long.isChecked()
        for r in range(self.table.rowCount()):
            it = self.table.item(r, COL_FLAG)
            self.table.setRowHidden(r, only and (it is None or it.text() != tr("lines.flag_long")))

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
        out_path = Path(str(out)) if isinstance(out, (str, Path)) else None
        exists = out_path is not None and out_path.exists()
        self.btn_open.setEnabled(exists)
        self.btn_external.setEnabled(exists)

    def running(self, on: bool) -> None:
        self.btn_dub.setEnabled(not on)
        self.btn_cancel.setVisible(on)
        self.btn_preview.setEnabled(not on)
        self.sld_volume.setEnabled(not on)
