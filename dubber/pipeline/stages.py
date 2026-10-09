"""Pipeline stages.  Each stage is ``fn(project, cfg, emit) -> summary`` and only reads/writes files of the project folder, so it
can run in the GUI process (light stages, tests) or in a worker process (models).  ``emit(kind, **payload)`` reports
``log`` (text), ``progress`` (0..1 of the stage) and ``until`` (seconds of film whose dub is final - Watch mode).

Order (research note 06): probe -> extract -> subtitles -> vad -> separation -> asr -> script -> diarization -> translation ->
voices -> tts (+ time fitting, block by block in film order) -> mix -> mux.  Models run one after another (one process each),
so the peak VRAM is that of the largest model; every stage keeps to 75 % of the free VRAM (``dubber.infra.resources``).
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from dubber.core import audio, media, mixing, script, subtitles, timefit, voices
from dubber.core.project import CHUNK_S, Line, Project, Speaker, hash_of, read_json, write_json

Emit = Callable[..., None]
ORDER = ["probe", "extract", "subtitles", "vad", "separation", "asr", "script", "diarization", "translation", "voices", "tts", "mix", "mux"]
HEAVY = {"vad", "separation", "asr", "diarization", "translation", "tts"}          # model stages: a worker process each
GPU = {"separation", "asr", "diarization", "translation", "tts"}                   # hold the shared Voxprint GPU lock
BLOCK_S = 60.0              # TTS + fitting work through the film in blocks of this length (Watch mode follows the blocks)
TAKES = 3                   # best-of-N for lines that do not fit

DEFAULT_CFG: Dict[str, Any] = {
    "device": "auto", "allow_download": True, "inprocess": False,
    "vad": "silero", "separation": "tiger", "asr": "whisper", "asr_repo": "deepdml/faster-whisper-large-v3-turbo-ct2",
    "diarization": "pyannote", "translation": "opus", "tts": "qwen", "tts_model": "tts_1_7b", "tts_backend": "auto",
    "roformer_model": "vocals_mel_band_roformer.ckpt",
    "subdl_key": "", "opensubtitles_key": "", "opensubtitles_user": "", "opensubtitles_password": "",
}
MOCK_CFG: Dict[str, Any] = {**DEFAULT_CFG, "inprocess": True, "device": "cpu", "allow_download": False, "vad": "energy",
                            "separation": "none", "asr": "none", "diarization": "cluster", "translation": "mock", "tts": "mock"}


def _device(cfg: Dict[str, Any]) -> str:
    from dubber.workers.common import torch_device

    return torch_device(cfg.get("device", "auto"))


def _rng(p: Project) -> Tuple[Optional[float], Optional[float]]:
    r = p.settings.get("range")
    return (float(r[0]), float(r[1])) if r else (None, None)


# ---------------------------------------------------------------------------------------------- cache keys
def inputs_for(key: str, p: Project, cfg: Dict[str, Any], prev: str) -> str:
    s, c = p.settings, cfg
    from dubber.core.project import file_fingerprint

    parts: Dict[str, Any] = {
        "probe": [file_fingerprint(p.source)],
        "extract": [s.get("audio_track"), s.get("range")],
        "subtitles": [s.get("source_lang"), s.get("target_lang"), s.get("subtitle_choice"), bool(c.get("subdl_key")), bool(c.get("opensubtitles_key"))],
        "vad": [c.get("vad")],
        "separation": [c.get("separation"), c.get("roformer_model") if c.get("separation") == "roformer" else ""],
        "asr": [c.get("asr"), c.get("asr_repo"), s.get("source_lang")],
        "script": [],
        "diarization": [s.get("multi_voice"), c.get("diarization")],
        "translation": [c.get("translation"), s.get("target_lang"), s.get("profanity")],
        "voices": [s.get("multi_voice"), s.get("single_voice"), [(sp.id, sp.voice.kind, sp.voice.id) for sp in p.speakers]],
        "tts": [c.get("tts"), c.get("tts_model"), s.get("actor_weight") if s.get("multi_voice") else None,
                [(sp.id, sp.voice.kind, sp.voice.id, p.is_key(sp)) for sp in p.speakers] if s.get("multi_voice") else None, [(ln.id, ln.translation, ln.speaker, ln.keep_original, ln.start, ln.end) for ln in p.lines]],
        "mix": [s.get("original_volume")],
        "mux": [s.get("output_format"), s.get("output")],
    }
    return hash_of(key, parts.get(key, []), prev)


# ---------------------------------------------------------------------------------------------- light stages
def st_probe(p: Project, cfg: Dict[str, Any], emit: Emit) -> str:
    info = media.probe(p.source)
    t0, t1 = _rng(p)
    dur = (min(t1, info.duration) - t0) if t0 is not None and t1 is not None else info.duration
    p.settings["duration"] = round(dur, 3)
    p.settings["media"] = {"duration": info.duration, "video": info.video_codec, "container": info.container,
                           "audio": [t.label() for t in info.audio], "subtitles": [t.label() for t in info.subtitles]}
    if not info.audio:
        raise RuntimeError("the file has no audio track")
    lang = info.audio[min(int(p.settings.get("audio_track") or 0), len(info.audio) - 1)].lang2
    if lang in ("en", "ru", "de", "fr", "es", "it") and not p.settings.get("source_lang_user"):
        p.settings["source_lang"] = lang
    return f"{info.duration / 60:.1f} min, {len(info.audio)} audio, {len(info.subtitles)} subtitle tracks"


def st_extract(p: Project, cfg: Dict[str, Any], emit: Emit) -> str:
    t0, t1 = _rng(p)
    length = (t1 - t0) if t0 is not None and t1 is not None else None
    tr = int(p.settings.get("audio_track") or 0)
    media.extract_audio(p.source, p.path("audio", "mix16.wav"), 16000, 1, tr, t0, length)
    media.extract_audio(p.source, p.path("audio", "mix44.wav"), 44100, 1, tr, t0, length)
    media.extract_audio(p.source, p.path("audio", "mix48s.wav"), 48000, 2, tr, t0, length)
    p.settings["duration"] = round(audio.duration(p.path("audio", "mix16.wav")), 3)
    return "soundtrack decoded (16 kHz analysis, 48 kHz stereo for the mix)"


def _crop(cues: List[subtitles.Cue], p: Project) -> List[subtitles.Cue]:
    t0, t1 = _rng(p)
    if t0 is None:
        return cues
    return [c for c in subtitles.shift(cues, -t0) if c.end > 0 and c.start < (t1 - t0)]


def find_subs(p: Project, lang: str, cfg: Dict[str, Any], emit: Emit, online: bool = True) -> Optional[subtitles.Found]:
    """Embedded text track -> file next to the film -> SubDL -> OpenSubtitles."""
    info = media.probe(p.source)
    tr = media.embedded_for(info, lang)
    if tr is not None:
        try:
            cues = media.extract_subtitle(p.source, tr, p.path("subs", f"embedded.{lang}.srt"))
            if cues:
                return subtitles.Found("embedded", cues, tr.label())
        except Exception as exc:  # noqa: BLE001
            emit("log", text=f"embedded subtitles not readable: {exc}")
    for path in subtitles.sidecar_candidates(p.source, lang):
        try:
            cues = subtitles.load_file(path)
        except OSError:
            continue
        if cues:
            return subtitles.Found("sidecar", cues, path.name)
    if online:
        return subtitles.find_online(p.source, lang, cfg, log=lambda m: emit("log", text=m))
    return None


def st_subtitles(p: Project, cfg: Dict[str, Any], emit: Emit) -> str:
    choice = p.settings.get("subtitle_choice", "auto")
    tgt, src = p.settings["target_lang"], p.settings["source_lang"]
    found: Dict[str, Any] = {}
    if choice == "none":
        pass
    elif choice not in ("auto", "") and Path(choice).is_file():
        found["target"] = subtitles.Found("file", subtitles.load_file(Path(choice)), Path(choice).name)
    else:
        f = find_subs(p, tgt, cfg, emit)
        if f:
            found["target"] = f
    if choice != "none":
        f = find_subs(p, src, cfg, emit, online=False)        # original-language text helps the voice references (no download)
        if f:
            found["source"] = f
    summary = []
    for role, f in found.items():
        cues = _crop(f.cues, p)
        subtitles.write_srt(cues, p.path("subs", f"{role}.srt"))
        summary.append(f"{role}: {f.source} ({len(cues)} cues)")
    for role in ("target", "source"):
        if role not in found:
            (p.folder / "subs" / f"{role}.srt").unlink(missing_ok=True)
    p.settings["subs_found"] = {r: {"source": f.source, "label": f.label} for r, f in found.items()}
    return "; ".join(summary) or "no subtitles: the speech will be recognised and translated offline"


def st_script(p: Project, cfg: Dict[str, Any], emit: Emit) -> str:
    windows = [tuple(w) for w in read_json(p.path("analysis", "windows.json"), [])]
    tgt = p.path("subs", "target.srt")
    src = p.path("subs", "source.srt")
    asr = read_json(p.path("analysis", "asr.json"), {}) or {}
    target_cues = subtitles.load_file(tgt) if tgt.exists() else []
    source_cues = subtitles.load_file(src) if src.exists() else []
    if target_cues:
        off = subtitles.estimate_offset(target_cues, windows)
        if off:
            emit("log", text=f"subtitles shifted by {off:+.2f} s to match the speech")
            target_cues = subtitles.shift(target_cues, off)
        lines = script.lines_from_cues(target_cues, target=True)
        if source_cues:
            script.attach_text(lines, subtitles.shift(source_cues, subtitles.estimate_offset(source_cues, windows)), "text")
        elif asr.get("segments"):
            script.attach_text(lines, [subtitles.Cue(s["start"], s["end"], s["text"]) for s in asr["segments"]], "text")
        how = "target-language subtitles"
    elif source_cues:
        lines = script.lines_from_cues(subtitles.shift(source_cues, subtitles.estimate_offset(source_cues, windows)), target=False)
        how = "original-language subtitles"
    elif asr.get("segments"):
        lines = script.lines_from_asr(asr["segments"])
        how = "speech recognition"
    else:
        raise RuntimeError("no subtitles and no speech recognition result: enable recognition (ASR) or add subtitles")
    script.snap_to_windows(lines, windows)
    p.lines = lines
    p.speakers = [Speaker("S1", "Speaker 1", seconds=sum(ln.duration for ln in lines))]
    music = sum(ln.keep_original for ln in lines)
    return f"{len(lines)} lines from {how}" + (f"; {music} songs/music kept original" if music else "")


def st_voices(p: Project, cfg: Dict[str, Any], emit: Emit) -> str:
    """Reference clips for cloned voices (from the separated speech, else the mix)."""
    speech = p.path("stems", "speech.wav")
    src = speech if speech.exists() else p.path("audio", "mix44.wav")
    made = []
    if p.settings.get("multi_voice"):
        for sp in p.speakers:
            if sp.voice.kind not in ("clone", "actor", "auto"):
                continue
            chosen = voices.pick_reference_lines(p.lines, sp.id)
            if chosen:
                secs, text = voices.build_reference(chosen, src, p.path("voices", f"{sp.id}.wav"))
                sp.ref_audio, sp.ref_text = p.rel(p.path("voices", f"{sp.id}.wav")), text
                made.append(f"{sp.id} {secs:.0f} s")
    else:
        v = p.settings.get("single_voice") or {}
        if (v.get("kind") or "clone") == "clone":
            main = max(p.speakers, key=lambda s: s.seconds).id if p.speakers else None
            chosen = voices.pick_reference_lines(p.lines, main) or voices.pick_reference_lines(p.lines, None)
            if chosen:
                secs, text = voices.build_reference(chosen, src, p.path("voices", "single.wav"))
                p.settings["single_ref"] = {"audio": p.rel(p.path("voices", "single.wav")), "text": text}
                made.append(f"one voice {secs:.0f} s")
    return "reference clips: " + (", ".join(made) or "none needed (library voices)")


# ---------------------------------------------------------------------------------------------- model stages
def st_vad(p: Project, cfg: Dict[str, Any], emit: Emit) -> str:
    x, sr = audio.read(p.path("audio", "mix16.wav"), 16000)
    segs = None
    if cfg.get("vad") == "silero":
        try:
            import torch
            from silero_vad import get_speech_timestamps, load_silero_vad

            ts = get_speech_timestamps(torch.from_numpy(x), load_silero_vad(), sampling_rate=16000, return_seconds=True,
                                       min_silence_duration_ms=300)
            segs = [(float(t["start"]), float(t["end"])) for t in ts]
        except ImportError:
            emit("log", text="Silero VAD is not installed; using the energy detector")
    if segs is None:
        segs = audio.energy_vad(x, sr)
    win = script.dialogue_windows(segs, total=len(x) / sr)
    write_json(p.path("analysis", "windows.json"), [list(w) for w in win])
    return f"{len(win)} dialogue windows, {sum(b - a for a, b in win) / 60:.1f} min of {len(x) / sr / 60:.1f}"


def st_separation(p: Project, cfg: Dict[str, Any], emit: Emit) -> str:
    kind = cfg.get("separation", "tiger")
    for n in ("speech.wav", "background.wav", "speech16.wav"):
        (p.folder / "stems" / n).unlink(missing_ok=True)
    if kind == "none":
        return "no separation: the original is ducked under the dub"
    windows = [tuple(w) for w in read_json(p.path("analysis", "windows.json"), [])]
    if kind == "roformer":
        from dubber.infra import shared_paths

        fn = __import__("dubber.engines.separation", fromlist=["x"]).roformer_model(
            _device(cfg), shared_paths.models_dir() / "audio-separator", cfg.get("roformer_model"), lambda m: emit("log", text=m))
    else:
        from dubber.engines.separation import tiger_model

        fn = tiger_model(_device(cfg), cfg.get("allow_download", True), lambda m: emit("log", text=m))
    from dubber.engines.separation import separate_windows

    separate_windows(p.path("audio", "mix44.wav"), windows, p.path("stems", "speech.wav"), p.path("stems", "background.wav"), fn,
                     lambda m: emit("log", text=m))
    x, sr = audio.read(p.path("stems", "speech.wav"), 16000)
    audio.write(p.path("stems", "speech16.wav"), x, sr)
    return f"{kind}: speech and background stems on {len(windows)} windows"


def st_asr(p: Project, cfg: Dict[str, Any], emit: Emit) -> str:
    out = p.path("analysis", "asr.json")
    kind = cfg.get("asr", "whisper")
    have_target = (p.folder / "subs" / "target.srt").exists()
    have_source = (p.folder / "subs" / "source.srt").exists()
    if kind == "none" or (have_source and kind != "force"):
        out.unlink(missing_ok=True)
        return "skipped (subtitles give the text)" if have_source or have_target else "skipped"
    wav = p.path("stems", "speech16.wav")
    if not wav.exists():
        wav = p.path("audio", "mix16.wav")
    from dubber.engines.asr import transcribe_faster_whisper

    res = transcribe_faster_whisper(str(wav), p.settings.get("source_lang"), cfg.get("asr_repo"), _device(cfg),
                                    cfg.get("allow_download", True), lambda m: emit("log", text=m))
    write_json(out, res)
    if res.get("language") and not p.settings.get("source_lang_user"):
        p.settings["source_lang"] = res["language"]
    return f"{len(res['segments'])} segments, language {res.get('language')}"


def st_diarization(p: Project, cfg: Dict[str, Any], emit: Emit) -> str:
    if not p.settings.get("multi_voice"):
        for ln in p.lines:
            ln.speaker = "S1"
        old = p.speaker("S1")
        p.speakers = [Speaker("S1", "Speaker 1", voice=old.voice if old else Speaker("S1").voice,
                              seconds=sum(ln.duration for ln in p.lines))]
        return "one voice for all lines (multi-voice is off): diarization skipped"
    wav = p.path("stems", "speech16.wav")
    if not wav.exists():
        wav = p.path("audio", "mix16.wav")
    how = cfg.get("diarization", "pyannote")
    mapping: Dict[int, str] = script.speakers_from_tags(p.lines)
    if mapping:                                    # the subtitles name the speakers: no voice analysis needed
        _renumber(p, mapping)
        names = {}
        for ln in p.lines:
            names.setdefault(ln.speaker, mapping.get(ln.id, ""))
        for sp in p.speakers:
            if names.get(sp.id) and sp.name.startswith("Speaker "):
                sp.name = names[sp.id]
        return f"{len(p.speakers)} speakers (subtitle speaker tags)"
    if how == "pyannote":
        try:
            from dubber.engines.diarization import pyannote_turns

            turns = pyannote_turns(str(wav), _device(cfg), cfg.get("allow_download", True), lambda m: emit("log", text=m))
            script.assign_speakers(p.lines, turns)
            mapping = {ln.id: ln.speaker for ln in p.lines}
        except Exception as exc:  # noqa: BLE001 - no token / not installed: fall back
            emit("log", text=f"pyannote not available ({type(exc).__name__}: {str(exc)[:120]}); using voice clustering")
            how = "cluster"
    if how == "cluster":
        from dubber.engines.diarization import cluster_lines

        mapping = cluster_lines(p.lines, str(wav))
    _renumber(p, mapping)
    return f"{len(p.speakers)} speakers ({how})"


def _renumber(p: Project, mapping: Dict[int, str]) -> None:
    """Speakers S1, S2 ... by amount of speech; voices chosen earlier are kept for speakers that keep their id."""
    totals: Dict[str, float] = {}
    for ln in p.lines:
        lab = mapping.get(ln.id, ln.speaker or "S1")
        totals[lab] = totals.get(lab, 0.0) + ln.duration
    order = sorted(totals, key=lambda k: -totals[k])
    new = {lab: f"S{i + 1}" for i, lab in enumerate(order)}
    old = {s.id: s for s in p.speakers}
    for ln in p.lines:
        ln.speaker = new.get(mapping.get(ln.id, ln.speaker or "S1"), "S1")
    p.speakers = [Speaker(new[lab], old[new[lab]].name if new[lab] in old else f"Speaker {i + 1}",
                          voice=old[new[lab]].voice if new[lab] in old else Speaker("x").voice, seconds=round(totals[lab], 1))
                  for i, lab in enumerate(order)]


def st_translation(p: Project, cfg: Dict[str, Any], emit: Emit) -> str:
    summary = _translate(p, cfg, emit)
    from dubber.core import profanity

    mode = p.settings.get("profanity", profanity.DEFAULT_MODE)
    n = profanity.apply_to_lines(p.lines, mode, p.settings["target_lang"])
    if mode == "soften":
        summary += f"; profanity softened in {n} lines" if profanity.supported(p.settings["target_lang"]) else \
            f"; no profanity filter for '{p.settings['target_lang']}' yet"
    return summary


_RU_ACRONYMS = [(re.compile(r"(?<![\w.])(?:A\.\s?I\.?|AI|Эй\.\s?И\.?|Эй-Ай)(?![\w])"), "ИИ")]


def tidy_translation(text: str, tgt: str) -> str:
    """Fix machine-translation leftovers the voice cannot read: Latin acronyms in a Russian line ("A.I." / "AI" / "Эй.И." -> "ИИ")."""
    if tgt == "ru":
        for rx, rep in _RU_ACRONYMS:
            def sub(m: "re.Match[str]", rep: str = rep) -> str:
                rest = m.string[m.end():].lstrip()
                ends_sentence = m.group(0).endswith(".") and (not rest or rest[:1].isupper())
                return rep + ("." if ends_sentence else "")
            text = rx.sub(sub, text)
    return text


def _translate(p: Project, cfg: Dict[str, Any], emit: Emit) -> str:
    todo = [ln for ln in p.lines if not ln.keep_original and not ln.translation.strip() and ln.text.strip()]
    if not todo:
        return "nothing to translate (subtitles in the dub language)"
    src, tgt = p.settings["source_lang"], p.settings["target_lang"]
    texts = [ln.text for ln in todo]
    if cfg.get("translation") == "mock":
        from dubber.engines.translation import mock_translate

        out = mock_translate(texts, src, tgt)
    else:
        from dubber.engines.translation import opus_translate

        out = opus_translate(texts, src, tgt, _device(cfg), cfg.get("allow_download", True), lambda m: emit("log", text=m))
    for ln, t in zip(todo, out):
        ln.translation = tidy_translation(t.strip(), tgt)
    return f"{len(todo)} lines translated {src}->{tgt} ({cfg.get('translation')})"


# ---------------------------------------------------------------------------------------------- TTS + time fitting
def voice_spec(p: Project, ln: Line):
    from dubber.engines.tts import VoiceSpec

    v = p.voice_for(ln)
    if v.kind == "library" and v.id:
        lv = voices.get_library_voice(v.id)
        if lv and lv.ref_audio:
            return VoiceSpec(f"lib:{lv.id}", "library", str(lv.ref_audio), lv.ref_text, str(lv.path), lv.adapter_scale)
    if p.settings.get("multi_voice"):
        sp = p.speaker(ln.speaker)
        if sp and v.kind == "actor":
            return actor_spec(p, sp)
        if sp and v.kind == "auto":                # key character: actor-like; the others: the closest library voice
            return actor_spec(p, sp, library_only=not p.is_key(sp))
        if sp and sp.ref_audio:
            return VoiceSpec(f"clone:{sp.id}", "clone", str(p.abs(sp.ref_audio)), sp.ref_text)
    ref = p.settings.get("single_ref") or {}
    if ref.get("audio"):
        return VoiceSpec("clone:single", "clone", str(p.abs(ref["audio"])), ref.get("text", ""))
    return VoiceSpec("default", "clone", "", "")


def actor_spec(p: Project, sp, library_only: bool = False):
    """Actor-like voice of one speaker: his clip + the library voices to compare with (the engine picks and blends)."""
    from dubber.core import actor_voice as av
    from dubber.engines.tts import VoiceSpec

    ref = str(p.abs(sp.ref_audio)) if sp.ref_audio and p.abs(sp.ref_audio).exists() else ""
    ok = False
    if ref and not library_only:
        x, sr = audio.read(Path(ref))
        ok = av.ref_quality(x, sr)[2]
    w = float(p.settings.get("actor_weight", av.DEFAULT_WEIGHT))
    return VoiceSpec(f"{'match' if library_only else 'actor'}:{sp.id}:{w:.2f}", "actor", ref, sp.ref_text, actor_weight=w, actor_ok=ok,
                     candidates=tuple(av.candidates_from_library(voices.list_library())),
                     record_dir=str(av.folder(p.folder, sp.id)))


def make_tts(p: Project, cfg: Dict[str, Any], need_adapters: bool, emit: Emit):
    from dubber.engines import tts as tts_mod

    if cfg.get("tts") == "mock":
        return tts_mod.MockTTS(p.settings["target_lang"])
    from dubber import models

    folder, _ = models.ensure(models.SPECS[cfg.get("tts_model", "tts_1_7b")].repo, cfg.get("allow_download", True),
                              log=lambda m: emit("log", text=m))
    return tts_mod.QwenTTS(folder, p.settings["target_lang"], _device(cfg), cfg.get("tts_backend", "auto"), need_adapters,
                           lambda m: emit("log", text=m))


def _blocks(lines: List[Line], block_s: float) -> List[Tuple[float, List[Line]]]:
    out: List[Tuple[float, List[Line]]] = []
    for ln in sorted(lines, key=lambda x: x.start):
        if not out or ln.start >= out[-1][0]:
            end = (int(ln.start // block_s) + 1) * block_s
            out.append((end, []))
        out[-1][1].append(ln)
    return out


def st_tts(p: Project, cfg: Dict[str, Any], emit: Emit) -> str:
    lines = sorted(p.dub_lines(), key=lambda ln: ln.start)
    total = float(p.settings.get("duration") or (lines[-1].end + 5 if lines else 0))
    lang = p.settings["target_lang"]
    specs = {ln.id: voice_spec(p, ln) for ln in lines}
    if any(not s.ref_audio for s in specs.values()) and cfg.get("tts") != "mock":
        raise RuntimeError("no voice reference: choose a library voice or let the program clone one from the film")
    engine = make_tts(p, cfg, any(s.kind in ("library", "actor") for s in specs.values()), emit)
    sr = engine.sample_rate
    tdir = p.path("tts", "x").parent
    fitdir = p.path("tts", "fit", "x").parent
    shutil.rmtree(p.folder / "tts" / "blocks", ignore_errors=True)
    est = {ln.id: script.estimate_seconds(ln.translation, lang) for ln in lines}
    secs: Dict[int, float] = {}
    wavs: Dict[int, np.ndarray] = {}
    stats = {"fits": 0, "shifted": 0, "stretched": 0, "too_long": 0, "retried": 0}

    def cache_path(text: str, spec, seed: Optional[int]) -> Path:
        return tdir / f"{hash_of(text, spec.tag(), getattr(engine, 'backend', ''), Path(str(getattr(engine, 'base_dir', ''))).name, seed)}.wav"

    def synth(items: List[Tuple[int, str]], seed: Optional[int] = None) -> Dict[int, np.ndarray]:
        got: Dict[int, np.ndarray] = {}
        todo: Dict[str, List[Tuple[int, str]]] = {}
        for i, t in items:
            cp = cache_path(t, specs[i], seed)
            if cp.exists():
                got[i] = audio.read(cp, sr)[0]
            else:
                todo.setdefault(specs[i].key, []).append((i, t))
        for key, group in todo.items():
            spec = specs[group[0][0]]

            def done(i: int, w: np.ndarray, _seed=seed) -> None:
                w = audio.trim_silence(w, engine.sample_rate)
                audio.write(cache_path(dict(group)[i], spec, _seed), w, engine.sample_rate)
                got[i] = w
            if seed is None:
                engine.run_queue(group, spec, done)
            else:
                for i, t in group:
                    done(i, engine.synthesize_batch([t], spec, seed=seed)[0])
        return got

    blocks = _blocks(lines, BLOCK_S)
    for bi, (bend, blines) in enumerate(blocks):
        for ln in blines:
            ln.spoken = ln.translation
        got = synth([(ln.id, ln.spoken) for ln in blines])
        wavs.update(got)
        secs.update({i: len(w) / sr for i, w in got.items()})
        # ---- time fitting: placement, then shorter wording / extra takes for the lines that still do not fit
        for attempt in range(2):
            plan = {pl.id: pl for pl in timefit.place(lines, {**est, **secs}, total)}
            bad = [ln for ln in blines if plan[ln.id].needs_retry]
            if not bad:
                break
            for ln in bad:
                slot = timefit.slot_for(ln, lines, total) * timefit.MAX_STRETCH
                cands: List[Tuple[str, np.ndarray]] = [(ln.spoken, wavs[ln.id])]
                for variant in script.shorten(ln.translation, lang)[:2] if attempt == 0 else []:
                    cands.append((variant, synth([(ln.id, variant)])[ln.id]))
                if attempt == 1:
                    for seed in range(1, TAKES):
                        cands.append((ln.spoken, synth([(ln.id, ln.spoken)], seed=seed)[ln.id]))
                k = timefit.best_take([len(w) / sr for _, w in cands], slot)
                ln.spoken, wavs[ln.id] = cands[k]
                secs[ln.id] = len(wavs[ln.id]) / sr
                stats["retried"] += 1
        plan = {pl.id: pl for pl in timefit.place(lines, {**est, **secs}, total)}
        records = []
        for ln in blines:
            pl = plan[ln.id]
            w = audio.stretch(wavs[ln.id], sr, pl.stretch) if pl.stretch != 1.0 else wavs[ln.id]
            fp = fitdir / f"line_{ln.id}.wav"
            audio.write(fp, w, sr)
            ln.audio, ln.audio_s, ln.place_start, ln.stretch, ln.fit = p.rel(fp), round(len(w) / sr, 3), round(pl.start, 3), pl.stretch, pl.verdict
            stats[pl.verdict] += 1
            records.append({"id": ln.id, "audio": ln.audio, "start": ln.place_start, "end": ln.end, "keep": False})
        until = total if bi == len(blocks) - 1 else min(total, blocks[bi + 1][1][0].start - 0.05)
        write_json(p.path("tts", "blocks", f"{bi:05d}.json"), {"until": until, "lines": records})
        emit("until", seconds=round(until, 2))
        emit("progress", value=(bi + 1) / len(blocks))
    engine.close()
    return (f"{len(lines)} lines ({getattr(engine, 'backend', '?')}): {stats['fits']} fit, {stats['shifted']} shifted into pauses, "
            f"{stats['stretched']} stretched <=1.15x, {stats['too_long']} too long; {stats['retried']} retries")


# ---------------------------------------------------------------------------------------------- mix / mux
def line_audio_map(p: Project, lines: List[Line]) -> Dict[int, Tuple[np.ndarray, int]]:
    out = {}
    for ln in lines:
        if ln.audio and p.abs(ln.audio).exists():
            out[ln.id] = audio.read(p.abs(ln.audio))
    return out


def mix_into(p: Project, t0: float, t1: float, lines: List[Line], cache: Dict[int, Tuple[np.ndarray, int]]) -> Tuple[np.ndarray, int]:
    windows = [tuple(w) for w in read_json(p.path("analysis", "windows.json"), [])]
    bg, sp = p.folder / "stems" / "background.wav", p.folder / "stems" / "speech.wav"
    near = [ln for ln in lines if ln.place_start < t1 and ln.place_start + ln.audio_s > t0 or ln.keep_original]
    need = {ln.id for ln in near} - set(cache)
    if need:
        cache.update(line_audio_map(p, [ln for ln in near if ln.id in need]))
    return mixing.mix_range(t0, t1, p.path("audio", "mix48s.wav"), windows, near, cache, bg if bg.exists() else None,
                            sp if sp.exists() else None, float(p.settings.get("original_volume", 0.15)))


def st_mix(p: Project, cfg: Dict[str, Any], emit: Emit) -> str:
    import soundfile as sf

    total = float(p.settings.get("duration") or audio.duration(p.path("audio", "mix48s.wav")))
    out = p.path("out", "dub_track.wav")
    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp.wav")
    cache: Dict[int, Tuple[np.ndarray, int]] = {}
    bounds = mixing.chunk_bounds(total, CHUNK_S)
    with sf.SoundFile(str(tmp), "w", 48000, 2, subtype="PCM_16") as fh:
        for i, (a, b) in enumerate(bounds):
            x, sr = mix_into(p, a, b, p.lines, cache)
            fh.write(x)
            for k in [k for k, _ in cache.items() if (p.line(k) and p.line(k).place_start + p.line(k).audio_s < a)]:
                cache.pop(k, None)
            emit("progress", value=(i + 1) / max(1, len(bounds)))
    os.replace(tmp, out)
    return f"dub track {total / 60:.1f} min, original voice at {int(100 * float(p.settings.get('original_volume', 0.15)))} %"


def output_path(p: Project) -> Path:
    if p.settings.get("output"):
        return Path(p.settings["output"])
    src = p.source
    fmt = p.settings.get("output_format", "same")
    ext = src.suffix.lower() if fmt == "same" and src.suffix.lower() in (".mkv", ".mp4", ".m4v", ".mov") else f".{fmt if fmt != 'same' else 'mkv'}"
    if ext in (".m4v", ".mov"):
        ext = ".mp4"
    return src.with_name(f"{src.stem}.dub-{p.settings['target_lang']}{ext}")


def st_mux(p: Project, cfg: Dict[str, Any], emit: Emit) -> str:
    if p.settings.get("range"):
        return "preview: no new file"
    out = output_path(p)
    tmp = out.with_name(f".{out.stem}.{os.getpid()}.partial{out.suffix}")
    idx = media.mux_dub(p.source, p.path("out", "dub_track.wav"), tmp, p.settings["target_lang"],
                        f"AI dub ({p.settings['target_lang']}) - Voxprint")
    os.replace(tmp, out)
    p.settings["output_file"] = str(out)
    return f"{out.name}: dub added as audio track #{idx + 1} (default), video and original tracks copied"


FUNCS: Dict[str, Callable[[Project, Dict[str, Any], Emit], str]] = {
    "probe": st_probe, "extract": st_extract, "subtitles": st_subtitles, "vad": st_vad, "separation": st_separation, "asr": st_asr,
    "script": st_script, "diarization": st_diarization, "translation": st_translation, "voices": st_voices, "tts": st_tts,
    "mix": st_mix, "mux": st_mux,
}
