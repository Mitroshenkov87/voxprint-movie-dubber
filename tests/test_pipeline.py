"""Pipeline logic on the CPU: subtitles, script building, time fitting, mixing, watch logic, voices, and an end-to-end run with
the stand-in engines (energy VAD, no separation, subtitles as text, clustering, mock translation, mock TTS)."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from dubber.core import audio, mixing, script, subtitles, timefit, voices, watch
from dubber.core.project import Line, Project, Voice, read_json_gz, write_json_gz
from dubber.engines import tts as tts_mod
from dubber.pipeline import runner as R
from dubber.pipeline import stages as S

SRT = """1
00:00:00,800 --> 00:00:04,300
Good evening. I was hoping you would come tonight.

2
00:00:04,750 --> 00:00:09,310
I almost stayed home, but the rain stopped, so here I am.

3
00:00:09,760 --> 00:00:13,730
Then let us not waste the evening. Tell me everything.

4
00:00:14,180 --> 00:00:17,460
It is a long story, and it begins with a letter.
"""


# ------------------------------------------------------------------ subtitles
def test_parse_srt_vtt_ass_and_cleanup():
    cues = subtitles.parse_srt(SRT)
    assert len(cues) == 4 and cues[1].start == pytest.approx(4.75)
    vtt = subtitles.parse_vtt("WEBVTT\n\n00:01.000 --> 00:02.500\n<i>Hi</i> there\n")
    assert vtt[0].end == pytest.approx(2.5)
    ass = subtitles.parse_ass("[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
                              "Dialogue: 0,0:00:01.50,0:00:03.00,Default,,0,0,0,,{\\i1}Hello,{\\i0} world\\Nagain\n")
    assert ass[0].start == pytest.approx(1.5) and ass[0].text == "Hello, world\nagain"
    assert subtitles.clean_text("JOHN: [door slams] <i>Hello!</i>\n- Bye.") == "Hello! Bye."
    assert subtitles.is_music("♪ la la la ♪") and subtitles.is_music("[MUSIC PLAYING]") and not subtitles.is_music("Music is life")
    assert subtitles.sdh_speaker("MARY: hi") == "MARY"


def test_write_srt_roundtrip(tmp_path):
    cues = subtitles.parse_srt(SRT)
    subtitles.write_srt(cues, tmp_path / "a.srt")
    assert [c.text for c in subtitles.load_file(tmp_path / "a.srt")] == [c.text for c in cues]


def test_cp1251_subtitles_decode():
    assert "Привет" in subtitles.decode_bytes("Привет".encode("cp1251"))


def test_offset_estimate_finds_shift():
    cues = [subtitles.Cue(s + 2.0, e + 2.0, "x") for s, e in [(1, 3), (5, 7), (10, 12), (15, 16)]]
    windows = [(1, 3), (5, 7), (10, 12), (15, 16)]
    assert subtitles.estimate_offset(cues, windows) == pytest.approx(-2.0, abs=0.11)


def test_sidecar_search_and_title_guess(tmp_path):
    v = tmp_path / "The.Movie.2019.1080p.mkv"
    v.write_bytes(b"")
    (tmp_path / "The.Movie.2019.1080p.ru.srt").write_text(SRT)
    (tmp_path / "The.Movie.2019.1080p.en.srt").write_text(SRT)
    assert subtitles.sidecar_candidates(v, "ru")[0].name.endswith(".ru.srt")
    assert subtitles.guess_title(v) == ("The Movie", 2019, None)
    assert subtitles.guess_title(Path("Show.Name.S02E05.mkv"))[2] == (2, 5)


def test_online_search_order_with_fake_http(tmp_path):
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("movie.ru.srt", SRT)
    calls = []

    def http(method, url, headers, body):
        calls.append(url)
        if "api.subdl.com" in url:
            return json.dumps({"status": True, "subtitles": [{"url": "/subtitle/1.zip"}]}).encode()
        if "dl.subdl.com" in url:
            return buf.getvalue()
        raise AssertionError(url)
    found = subtitles.find_online(Path("Movie.2001.mkv"), "ru", {"subdl_key": "k", "opensubtitles_key": "o"}, http)
    assert found.source == "subdl" and len(found.cues) == 4
    assert "api_key=k" in calls[0] and "film_name=Movie" in calls[0]


def test_opensubtitles_flow_with_fake_http():
    def http(method, url, headers, body):
        if url == "https://dl.example/x.srt":
            return SRT.encode()
        assert headers.get("Api-Key") == "o"
        if url.endswith("/login"):
            return b'{"token": "T"}'
        if "/subtitles?" in url:
            assert headers.get("Authorization") == "Bearer T"
            return json.dumps({"data": [{"attributes": {"files": [{"file_id": 7}]}}]}).encode()
        if url.endswith("/download"):
            assert json.loads(body)["file_id"] == 7
            return b'{"link": "https://dl.example/x.srt", "file_name": "x.srt"}'
        if url == "https://dl.example/x.srt":
            return SRT.encode()
        raise AssertionError(url)
    name, data = subtitles.opensubtitles_search("o", Path("Film.mkv"), "ru", http, "user", "pw")
    assert name == "x.srt" and b"Good evening" in data


def test_online_errors_only_log():
    def http(*a):
        raise OSError("offline")
    logs = []
    assert subtitles.find_online(Path("a.mkv"), "ru", {"subdl_key": "k"}, http, logs.append) is None
    assert "offline" in logs[0]


# ------------------------------------------------------------------ script
def test_dialogue_windows_merge_and_pad():
    w = script.dialogue_windows([(1.0, 2.0), (2.3, 3.0), (6.0, 6.1), (8.0, 9.0)], total=8.9)
    assert w[0] == (0.85, 3.15) and w[-1][1] == 8.9 and all(b - a >= 0.3 for a, b in w)


def test_lines_from_cues_marks_songs_and_attach_text():
    cues = subtitles.parse_srt(SRT) + [subtitles.Cue(18, 20, "♪ Happy birthday ♪")]
    lines = script.lines_from_cues(cues, target=True)
    assert lines[-1].keep_original and lines[-1].kind == "music" and lines[0].translation.startswith("Good")
    src = [subtitles.Cue(0.9, 4.2, "Добрый вечер.")]
    assert script.attach_text(lines, src, "text") == 1 and lines[0].text == "Добрый вечер."


def test_lines_from_asr_split_long_segments():
    words = [{"w": f" w{i}", "start": i * 0.5, "end": i * 0.5 + 0.4} for i in range(40)]
    lines = script.lines_from_asr([{"start": 0, "end": 20, "text": "x", "words": words}], max_len=8)
    assert len(lines) >= 2 and all(ln.duration <= 8.5 for ln in lines) and [ln.id for ln in lines] == list(range(1, len(lines) + 1))


def test_cluster_speakers_two_groups():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.05, (5, 26)) + np.r_[np.ones(13), np.zeros(13)]
    b = rng.normal(0, 0.05, (3, 26)) + np.r_[np.zeros(13), np.ones(13)]
    labels = script.cluster_speakers(np.vstack([a, b]))
    assert labels[:5] == [0] * 5 and labels[5:] == [1] * 3


def test_length_estimate_and_too_long_flag():
    assert script.estimate_seconds("Привет, как дела?", "ru") < script.estimate_seconds("Здравствуйте, как поживаете в этот прекрасный вечер?", "ru")
    ln = Line(1, 0.0, 1.0, translation="Это очень длинная фраза, которая точно не поместится в одну секунду времени.")
    assert script.too_long(ln, "ru") and not script.too_long(Line(2, 0, 3, translation="Да."), "ru")


def test_short_interjection_uses_the_pause_before_it_is_too_long():
    # clip 1: Whoa! is a 0.36 s slot. A reading longer than 1.25x of that slot is flagged until the next line's pause can take it.
    whoa = Line(1, 6.84, 7.20, "Whoa!", translation="Ого-го!")
    assert whoa.duration == pytest.approx(0.36)
    assert script.too_long(whoa, "ru")
    assert not script.too_long(whoa, "ru", next_start=7.58)
    eww = Line(2, 0.0, 0.36, "Ewwwww.", translation="Фу, нет!")
    assert script.too_long(eww, "ru") and not script.too_long(eww, "ru", next_start=1.4)
    tight = Line(3, 0.0, 0.36, "Yes!", translation="Ого, подождите, это никак не влезет в паузу.")
    assert script.too_long(tight, "ru", next_start=0.50)


def test_shorten_variants_are_shorter():
    text = "Ну, знаешь, я просто хотел сказать (тихо), что это было очень, очень важно, понимаешь"
    out = script.shorten(text, "ru")
    assert out and all(len(v) < len(text) for v in out) and "(тихо)" not in out[0]


# ------------------------------------------------------------------ time fitting
def _lines(*spans):
    return [Line(i + 1, a, b, translation="x") for i, (a, b) in enumerate(spans)]


def test_place_short_interjection_extends_into_the_pause():
    lines = [Line(1, 0.0, 0.36, "Whoa!"), Line(2, 1.4, 2.2, "Next words here")]
    plan = {p.id: p for p in timefit.place(lines, {1: 0.9, 2: 0.6}, total=4.0)}
    assert plan[1].verdict != "too_long" and plan[1].stretch == 1.0
    overlap = [Line(1, 0.0, 0.36, "Yes!"), Line(2, 0.45, 1.2, "Go now please")]
    plan2 = {p.id: p for p in timefit.place(overlap, {1: 2.0, 2: 0.5}, total=3.0)}
    assert plan2[1].verdict == "too_long"


def test_place_fits_shifts_stretches_and_flags():
    lines = _lines((0, 2), (3, 4), (4.5, 5), (5.2, 6))
    plan = {p.id: p for p in timefit.place(lines, {1: 1.5, 2: 1.3, 3: 0.85, 4: 3.0}, total=7.0)}
    assert plan[1].verdict == "fits"
    assert plan[2].verdict == "shifted"                       # runs into the pause before line 3
    assert plan[3].verdict in ("shifted", "stretched")
    assert plan[4].verdict == "too_long" and plan[4].stretch == timefit.HARD_STRETCH
    starts = sorted((p.start, p.start + {1: 1.5, 2: 1.3, 3: 0.85, 4: 3.0}[p.id] / p.stretch) for p in plan.values())
    assert all(b[0] >= a[1] for a, b in zip(starts, starts[1:]))   # never overlapping


def test_best_take():
    assert timefit.best_take([2.5, 1.8, 1.9], 2.0) == 2
    assert timefit.best_take([3.0, 2.6], 2.0) == 1


def test_stretch_signalsmith_length():
    x = np.sin(np.linspace(0, 200, 24000)).astype(np.float32)
    y = audio.stretch(x, 24000, 1.15)
    assert abs(len(y) - 24000 / 1.15) < 600


# ------------------------------------------------------------------ mixing
def test_mix_keeps_original_outside_windows_and_adds_dub(tmp_path):
    sr = 48000
    t = np.arange(sr * 4) / sr
    orig = np.stack([0.2 * np.sin(2 * np.pi * 200 * t)] * 2, axis=1).astype(np.float32)
    import soundfile as sf

    sf.write(tmp_path / "o.wav", orig, sr)
    ln = Line(1, 1.0, 2.0, translation="x")
    ln.place_start = 1.0
    dub = (0.3 * np.ones(sr // 2)).astype(np.float32)
    x, rate = mixing.mix_range(0, 4, tmp_path / "o.wav", [(1.0, 2.0)], [ln], {1: (dub, sr)}, original_volume=0.0)
    assert rate == sr and x.shape == (4 * sr, 2)
    assert np.allclose(x[: int(0.5 * sr)], orig[: int(0.5 * sr)], atol=1e-3)      # outside: untouched
    inside = x[int(1.1 * sr):int(1.4 * sr)]
    assert inside.mean() > 0.2                                                  # dub present, original ducked
    song = Line(2, 1.0, 2.0, keep_original=True)
    x2, _ = mixing.mix_range(0, 4, tmp_path / "o.wav", [(1.0, 2.0)], [song], {}, original_volume=0.0)
    assert np.allclose(x2, orig, atol=1e-3)                                     # songs keep the original


def test_chunk_bounds():
    assert mixing.chunk_bounds(65, 30) == [(0, 30), (30, 60), (60, 65)]


# ------------------------------------------------------------------ watch mode
def test_watch_ready_buffering_and_resume():
    w = watch.WatchState(total=3600)
    w.update(0, 120)
    assert not w.ready and not w.start()
    w.update(0, 300)
    assert w.ready and w.start()
    assert w.update(100, 320) == "none"
    assert w.update(319, 320) == "pause" and w.buffering
    assert w.update(319, 340) == "none"                     # not enough ahead yet
    assert w.update(319, 370) == "play"
    short = watch.WatchState(total=90)
    short.update(0, 90, finished=True)
    assert short.ready


def test_eta():
    e = watch.Eta()
    assert e.update(0, 0, 100) == -1
    assert e.update(10, 10, 100) == pytest.approx(90)


# ------------------------------------------------------------------ voices
def _fake_voice(root: Path, vid: str, name: str):
    d = root / vid
    d.mkdir(parents=True)
    (d / "adapter_model.safetensors").write_bytes(b"x")
    (d / "adapter_config.json").write_text("{}")
    audio.write(d / "ref_sample.wav", np.zeros(2400, np.float32), 24000)
    (d / "voice.json").write_text(json.dumps({"name": name, "language": "ru", "gender": "female", "base_model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base"}))
    (d / "training_meta.json").write_text(json.dumps({"ref_sample_text": "Привет"}))


def test_shared_voice_library_is_read_only(tmp_path):
    from dubber.infra import shared_paths

    root = shared_paths.voices_dir()
    assert voices.list_library() == []
    _fake_voice(root, "anna", "Anna")
    (root / "broken").mkdir()
    lib = voices.list_library()
    assert [v.id for v in lib] == ["anna"] and lib[0].ref_text == "Привет" and lib[0].gender == "female"
    assert voices.get_library_voice("anna").ref_audio.name == "ref_sample.wav"


def test_reference_pick_and_build(tmp_path):
    lines = [Line(1, 0, 3, "a", speaker="S1"), Line(2, 3, 9, "b", speaker="S1"), Line(3, 9, 10, "c", speaker="S1"),
             Line(4, 10, 14, "d", speaker="S2"), Line(5, 14, 20, "song", speaker="S1", keep_original=True)]
    chosen = voices.pick_reference_lines(lines, "S1")
    assert [ln.id for ln in chosen] == [1, 2]
    audio.write(tmp_path / "s.wav", 0.1 * np.ones(24000 * 20, np.float32), 24000)
    secs, text = voices.build_reference(chosen, tmp_path / "s.wav", tmp_path / "ref.wav")
    # line 2 is padded 0.12 s past its end (the lines touch, so the join itself is not padded) plus the 0.25 s gap
    assert secs == pytest.approx(9.37, abs=0.02) and text == "a b"
    faded, _ = audio.read(tmp_path / "ref.wav")
    assert abs(float(faded[0])) < 1e-4 and abs(float(faded[-1])) < 1e-4 and abs(float(faded[len(faded) // 2])) > 1e-3


# ------------------------------------------------------------------ project
def test_project_roundtrip_merge_reassign_and_voice(tmp_path):
    p = Project.create(tmp_path / "p", tmp_path / "film.mkv")
    p.lines = [Line(1, 0, 1, speaker="S1"), Line(2, 1, 2, speaker="S2"), Line(3, 2, 3, speaker="S3")]
    from dubber.core.project import Speaker

    p.speakers = [Speaker("S1"), Speaker("S2"), Speaker("S3")]
    p.save()
    q = Project(p.folder)
    q.merge_speakers("S1", ["S2"])
    assert {ln.speaker for ln in q.lines} == {"S1", "S3"} and [s.id for s in q.speakers] == ["S1", "S3"]
    q.reassign(3, "S1")
    assert q.line(3).speaker == "S1"
    q.settings["single_voice"] = {"kind": "library", "id": "anna"}
    assert q.voice_for(q.line(1)) == Voice("library", "anna")          # multi-voice off: one voice for all
    q.settings["multi_voice"] = True
    q.speaker("S1").voice = Voice("library", "bob")
    assert q.voice_for(q.line(1)).id == "bob"
    q.save()
    assert Project(p.folder).settings["multi_voice"] is True             # the toggle is stored per project


def test_gzip_is_reproducible(tmp_path):
    write_json_gz(tmp_path / "a.gz", {"x": 1})
    b1 = (tmp_path / "a.gz").read_bytes()
    write_json_gz(tmp_path / "a.gz", {"x": 1})
    assert b1 == (tmp_path / "a.gz").read_bytes() and read_json_gz(tmp_path / "a.gz") == {"x": 1}


# ------------------------------------------------------------------ TTS batching
def test_next_group_similar_lengths_and_order():
    texts = {i: "x" * L for i, L in enumerate([5, 50, 6, 48, 7, 49, 100, 4])}
    q = list(texts)
    g = tts_mod.next_group(q, texts, 3)
    assert 0 in g and len(g) == 3 and max(len(texts[i]) for i in g) <= 7
    assert tts_mod.max_tokens_for("a") == tts_mod.MIN_TOKENS and tts_mod.max_tokens_for("a" * 5000) == tts_mod.MAX_TOKENS


def test_run_queue_halves_on_oom():
    class Eng(tts_mod.MockTTS):
        calls = []

        def synthesize_batch(self, texts, voice, seed=None):
            self.calls.append(len(texts))
            if len(texts) > 4:
                raise RuntimeError("CUDA out of memory")
            return super().synthesize_batch(texts, voice, seed)
    e = Eng()
    got = {}
    e.run_queue([(i, f"line {i}") for i in range(20)], tts_mod.VoiceSpec("v", "clone"), lambda i, w: got.__setitem__(i, w))
    assert len(got) == 20 and max(e.calls[1:]) <= 6


def test_backend_order(monkeypatch):
    import importlib.util as iu

    monkeypatch.setattr(iu, "find_spec", lambda n: object() if n in ("flash_attn", "faster_qwen3_tts") else None)
    assert tts_mod.QwenTTS.candidates(True) == [("standard", "flash_attention_2"), ("graphs", "sdpa"), ("standard", "sdpa")]
    assert ("graphs", "sdpa") not in tts_mod.QwenTTS.candidates(True, need_adapters=True)
    assert tts_mod.QwenTTS.candidates(False) == [("standard", "sdpa")]


# ------------------------------------------------------------------ end to end (stand-in engines)
@pytest.fixture
def film(tmp_path, clip):
    d = tmp_path / "films"
    d.mkdir()
    shutil.copy(clip, d / "clip.mkv")
    (d / "clip.en.srt").write_text(SRT, encoding="utf-8")
    return d / "clip.mkv"


def test_end_to_end_mock_multi_voice(tmp_path, film):
    p = Project.create(tmp_path / "proj", film, target_lang="ru", source_lang="en", multi_voice=True)
    until = []
    res = R.Runner(p, S.MOCK_CFG, R.Callbacks(dubbed_until=until.append)).run()
    assert res.ok, res
    out = Path(res.message)
    assert out.name == "clip.dub-ru.mkv" and out.exists()
    from dubber.core import media

    info = media.probe(out)
    assert len(info.audio) == 2 and info.audio[1].lang == "rus" and info.video_codec == "h264"
    q = Project(p.folder)
    assert [ln.speaker for ln in q.lines] in (["S1", "S2", "S1", "S2"], ["S2", "S1", "S2", "S1"])
    assert all(ln.fit and ln.place_start >= 0 for ln in q.lines)
    assert until and until[-1] == pytest.approx(q.settings["duration"], abs=0.1)
    assert json.loads((p.folder / "watch" / "index.json").read_text())["chunks"] == [0]
    again = R.Runner(Project(p.folder), S.MOCK_CFG).run()
    assert set(again.stages.values()) == {"cached"}
    q.lines[0].translation = "Добрый вечер."                                 # an edit re-runs only TTS and later stages
    q.save()
    third = R.Runner(Project(p.folder), S.MOCK_CFG).run()
    assert third.stages["script"] == "cached" and third.stages["tts"] == "done" and third.stages["mux"] == "done"


def test_end_to_end_single_voice_skips_diarization_and_preview(tmp_path, film):
    p = Project.create(tmp_path / "proj", film, target_lang="ru", source_lang="en")
    res = R.Runner(p, S.MOCK_CFG).run(until_stage="voices")
    assert res.ok and "skipped" in Project(p.folder).stages["diarization"]["summary"]
    start = R.best_preview_start(Project(p.folder), length=10)
    sub = R.preview_project(Project(p.folder), start, 10)
    r2 = R.Runner(sub, S.MOCK_CFG).run(until_stage="mix")
    assert r2.ok and (sub.folder / "out" / "dub_track.wav").exists()
    assert audio.duration(sub.folder / "out" / "dub_track.wav") == pytest.approx(10, abs=0.1)


def test_failing_stage_reports_clearly(tmp_path, film):
    (film.parent / "clip.en.srt").unlink()
    p = Project.create(tmp_path / "proj", film, target_lang="ru", source_lang="en")
    res = R.Runner(p, S.MOCK_CFG).run()
    assert not res.ok and res.message.startswith("script:")


# ------------------------------------------------------------------ profanity filter
def test_profanity_soften_ru_and_restore():
    from dubber.core import profanity
    from dubber.core.project import Line
    f = profanity.soften_ru
    assert f("Блядь, какой пиздец!") == "Чёрт, какой капец!"
    assert f("Иди на хуй, понял?") == "Иди к чёрту, понял?"
    assert f("Ни хуя себе, он охуел.") == "Ни фига себе, он обалдел."
    assert f("ЭТО ПИЗДЕЦ") == "ЭТО КАПЕЦ"
    clean = "Небо голубое, хлеба нет, себе веб-сайт. Сука, мудак! Хулиган, ребята, бляха-муха, Ебола."
    assert f(clean) == clean                       # rude but not mat, and look-alike words stay
    lines = [Line(1, 0, 1, translation="Это пиздец."), Line(2, 1, 2, translation="Привет."),
             Line(3, 2, 3, translation="Охуеть.", edited=True)]
    assert profanity.apply_to_lines(lines, "soften", "ru") == 1
    assert lines[0].translation == "Это капец." and lines[0].softened == "Это пиздец." and lines[2].translation == "Охуеть."
    assert profanity.apply_to_lines(lines, "keep", "ru") == 0
    assert lines[0].translation == "Это пиздец." and not lines[0].softened
    assert profanity.soften("fuck", "de") == "fuck" and not profanity.supported("de")


def test_profanity_mode_runs_in_translation_stage(tmp_path, film):
    srt = SRT.replace("Good evening. I was hoping you would come tonight.", "Какой пиздец, опять дождь.")
    (film.parent / "clip.ru.srt").write_text(srt, encoding="utf-8")
    p = Project.create(tmp_path / "proj", film, target_lang="ru", source_lang="en", profanity="soften")
    res = R.Runner(p, S.MOCK_CFG).run(until_stage="translation")
    assert res.ok, res
    q = Project(p.folder)
    assert q.lines[0].translation == "Какой капец, опять дождь." and q.lines[0].softened
    assert "softened in 1" in q.stages["translation"]["summary"]


# ------------------------------------------------------------------ voice catalog (fake network)
def test_voice_catalog_lists_and_installs_into_shared_library(tmp_path):
    import hashlib
    import io
    import zipfile
    from dubber.core import voice_catalog as vc, voices
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n in ("adapter_model.safetensors", "adapter_config.json", "ref_sample.wav", "speaker_centroid.safetensors"):
            z.writestr(f"levi/{n}", b"x")
        z.writestr("levi/voice.json", json.dumps({"name": "Levi", "language": "ru", "adapter_scale": 0.5}))
        z.writestr("levi/training_meta.json", json.dumps({"ref_sample_text": "привет"}))
        z.writestr("../evil.txt", b"no")
    data = buf.getvalue()
    sha = hashlib.sha256(data).hexdigest()
    pages = {"https://huggingface.co/R/v/resolve/main/SHA256SUMS.txt": f"{sha}  levi.zip\n".encode(),
             "https://huggingface.co/api/models/R/v?blobs=true": json.dumps({"siblings": [{"rfilename": "levi.zip", "size": len(data)}]}).encode(),
             "https://huggingface.co/R/v/resolve/main/levi/voice.json": json.dumps({"name": "Levi", "gender": "male", "license": "CC0-1.0"}).encode()}
    found = vc.list_remote("R/v", fetch=lambda u: pages[u])
    assert [(v.id, v.name, v.gender, v.size_bytes) for v in found] == [("levi", "Levi", "male", len(data))]

    def dl(url, dest, size, progress, cancel):
        assert url.endswith("/levi.zip")
        dest.write_bytes(data)
    root = tmp_path / "voices"
    path = vc.install(found[0], root=root, downloader=dl)
    assert path == root / "levi" and not (root / "evil.txt").exists() and not (tmp_path / "evil.txt").exists()
    lib = voices.list_library(root)
    assert lib[0].id == "levi" and lib[0].repo_id == "levi" and lib[0].adapter_scale == 0.5 and lib[0].ref_text == "привет"
    assert vc.install(found[0], root=root, downloader=lambda *a: (_ for _ in ()).throw(AssertionError("again"))) == path
    bad = vc.RemoteVoice("noa", "Noa", "R/v", "0" * 64)
    with pytest.raises(vc.VoiceCatalogError):
        vc.install(bad, root=root, downloader=lambda u, d, *a: d.write_bytes(b"zz"))
    assert not (root / "noa").exists()


# ------------------------------------------------------------------ actor-like voice
def _lib_voice(root, vid, gender, emb, scale=0.5):
    from safetensors.numpy import save_file
    d = root / vid
    d.mkdir(parents=True)
    for n in ("adapter_model.safetensors", "adapter_config.json"):
        (d / n).write_bytes(b"x")
    audio.write(d / "ref_sample.wav", np.zeros(2400, np.float32), 24000)
    (d / "voice.json").write_text(json.dumps({"name": vid.title(), "gender": gender, "adapter_scale": scale}), encoding="utf-8")
    save_file({"speaker_embedding": np.asarray(emb, np.float32)}, str(d / "speaker_centroid.safetensors"))
    return d


def test_actor_voice_helpers(tmp_path):
    from dubber.core import actor_voice as av
    sr = 16000
    t = np.arange(sr * 6) / sr
    speech = (0.3 * np.sin(2 * np.pi * 150 * t) * (np.sin(2 * np.pi * 2 * t) > 0)).astype(np.float32)
    assert av.ref_quality(speech, sr)[2]
    assert not av.ref_quality(speech[: sr * 2], sr)[2]                       # too short
    noisy = speech + 0.2 * np.random.default_rng(0).standard_normal(len(speech)).astype(np.float32)
    assert not av.ref_quality(noisy, sr)[2]                                  # too noisy
    a, m, f = _lib_voice(tmp_path, "levi", "male", [1, 0, 0]), _lib_voice(tmp_path, "natan", "male", [0.6, 0.8, 0]), \
        _lib_voice(tmp_path, "noa", "female", [0.7, 0.7, 0.1])
    cands = [av.Candidate("levi", a, "male", 0.5), av.Candidate("natan", m, "male", 0.5), av.Candidate("noa", f, "female", 0.5)]
    best, score = av.pick_closest(np.array([0.5, 0.86, 0.0]), cands, "male")
    assert best.id == "natan" and score > 0.99
    assert av.pick_closest(np.array([0.5, 0.86, 0.0]), cands, "female")[0].id == "noa"
    assert av.pick_closest(None, cands, "female")[0].id == "noa"
    actor = np.array([3.0, 0.0, 0.0])
    b = av.blend(actor, np.array([0.0, 1.0, 0.0]), 0.7)
    assert np.linalg.norm(b) == pytest.approx(3.0) and b[0] > b[1] > 0
    assert av.adapter_scale(0.5, 0.7) == pytest.approx(0.15)


def test_actor_voice_resolves_blends_and_saves(tmp_path, monkeypatch):
    from dubber.core import actor_voice as av
    from dubber.engines import tts as T
    root = tmp_path / "lib"
    _lib_voice(root, "levi", "male", [1, 0, 0])
    _lib_voice(root, "noa", "female", [0, 1, 0])
    ref = tmp_path / "actor.wav"
    audio.write(ref, np.zeros(24000, np.float32), 24000)
    cands = tuple(av.candidates_from_library(voices.list_library(root)))
    rec_dir = tmp_path / "proj" / "voices" / "actor_S1"
    spec = T.VoiceSpec("actor:S1", "actor", str(ref), "hello", actor_weight=0.7, candidates=cands, record_dir=str(rec_dir))

    class Fake(T.MockTTS):
        def actor_embedding(self, voice):
            return np.array([0.1, 2.0, 0.0], np.float32)          # sounds like Noa
    eng = Fake()
    got = eng.resolve(spec)
    assert got.kind == "library" and Path(got.adapter_dir).name == "noa" and got.ref_audio == str(ref)
    assert got.adapter_scale == pytest.approx(0.15) and got.key in eng._blends
    rec = av.read_record(rec_dir)
    assert rec["mode"] == "blend" and rec["library_id"] == "noa" and (rec_dir / "blended_embedding.npy").is_file()
    assert eng.resolve(spec) is got                                # once per voice
    # bad actor clip -> the library voice as is
    poor = T.VoiceSpec("actor:S2", "actor", str(ref), "", actor_weight=0.7, actor_ok=False, candidates=cands)
    fb = Fake().resolve(poor)
    assert fb.ref_audio.endswith("ref_sample.wav") and fb.adapter_scale == 0.5
    # keep it: only with a consent note, in the library format
    with pytest.raises(ValueError):
        av.save_to_library(rec_dir, "Actor", "", root)
    saved = av.save_to_library(rec_dir, "Hero Actor", "user confirmed", root, "ru")
    info = json.loads((saved / "voice.json").read_text(encoding="utf-8"))
    assert saved.name == "hero-actor" and info["consent"]["confirmed"] and not info["commercial_use"]
    assert {v.id for v in voices.list_library(root)} == {"levi", "noa", "hero-actor"}
    assert T.load_centroid(saved) is not None


def test_end_to_end_mock_with_actor_voice(tmp_path, film):
    from dubber.infra import shared_paths
    _lib_voice(shared_paths.voices_dir(), "levi", "male", [1, 0, 0])
    p = Project.create(tmp_path / "proj", film, target_lang="ru", source_lang="en", multi_voice=True)
    assert R.Runner(p, S.MOCK_CFG).run(until_stage="translation").ok
    q = Project(p.folder)
    for sp in q.speakers:
        sp.voice = Voice("actor", "")
    q.save()
    res = R.Runner(Project(p.folder), S.MOCK_CFG).run()
    assert res.ok, res
    rec = json.loads((p.folder / "voices" / f"actor_{q.speakers[0].id}" / "actor_voice.json").read_text())
    assert rec["library_id"] == "levi"


def test_key_characters_and_subtitle_speaker_tags(tmp_path, film):
    from dubber.core.project import Speaker
    p = Project.create(tmp_path / "proj", film)
    p.speakers = [Speaker("S1", seconds=50), Speaker("S2", seconds=30), Speaker("S3", seconds=12), Speaker("S4", seconds=8)]
    assert [p.is_key(s) for s in p.speakers] == [True, True, False, False]
    p.speakers[2].key, p.speakers[0].key = True, False
    assert [p.is_key(s) for s in p.speakers] == [False, True, True, False]
    assert Speaker("x").voice.kind == "auto"
    tagged = SRT.replace("Good evening.", "ANNA: Good evening.").replace("I almost", "BOB: I almost") \
        .replace("Then let us", "ANNA: Then let us")
    (film.parent / "clip.en.srt").write_text(tagged, encoding="utf-8")
    q = Project.create(tmp_path / "proj2", film, target_lang="ru", source_lang="en", multi_voice=True)
    res = R.Runner(q, S.MOCK_CFG).run()
    assert res.ok, res
    q = Project(q.folder)
    assert "subtitle speaker tags" in q.stages["diarization"]["summary"]
    assert [ln.speaker for ln in q.lines] == ["S1", "S2", "S1", "S1"] and {s.name for s in q.speakers} == {"Anna", "Bob"}


def test_original_track_is_picked_automatically():
    from pathlib import Path
    from dubber.core.media import MediaInfo, Track, pick_original_track
    info = MediaInfo(Path("x.mkv"), 10.0, "h264", [Track(0, 1, "ac3", "rus", "Dub"), Track(1, 2, "eac3", "eng", "Commentary"),
                                                     Track(2, 3, "dts", "eng", "Original")])
    assert pick_original_track(info, "ru") == 2
    info.audio[2].title = ""
    assert pick_original_track(info, "ru") == 2                      # not the Russian dub, not the commentary
    assert pick_original_track(None, "ru") == 0


def test_mp4_export_remuxes_video_and_converts_only_unsupported_audio(tmp_path, clip, monkeypatch):
    from dubber import ffmpeg
    from dubber.core import media
    calls = []
    info = media.MediaInfo(clip, 18.0, "hevc", [media.Track(0, 1, "truehd", "eng", "", 8), media.Track(1, 2, "ac3", "eng")],
                           [media.Track(0, 3, "hdmv_pgs_subtitle", "eng"), media.Track(1, 4, "subrip", "rus")])
    monkeypatch.setattr(media, "probe", lambda p: info)
    monkeypatch.setattr(ffmpeg, "run", lambda args, timeout=0: calls.append([str(a) for a in args]))
    media.mux_dub(clip, tmp_path / "d.wav", tmp_path / "out.mp4", "ru", "AI dub")
    a = calls[-1]
    assert "-c:v" not in a and a[a.index("-c") + 1] == "copy"           # the video is copied, never re-encoded
    assert a[a.index("-c:a:0") + 1] == "aac" and "-c:a:1" not in a      # TrueHD converted, AC-3 kept
    assert "0:s:1" in a and "0:s:0" not in a                            # text subtitles kept, PGS left out
    assert a[a.index("-tag:v") + 1] == "hvc1"
    media.mux_dub(clip, tmp_path / "d.wav", tmp_path / "out.mkv", "ru", "AI dub")
    b = calls[-1]
    assert b[b.index("-map") + 1] == "0" and "-c:a:0" not in b          # MKV: every stream copied as it is


def test_tidy_translation_russian_acronyms():
    from dubber.pipeline.stages import tidy_translation as t

    assert t("Бивис, нам нужно достать кое-что из этого A.I.", "ru") == "Бивис, нам нужно достать кое-что из этого ИИ."
    assert t("Что такое AI?", "ru") == "Что такое ИИ?"
    assert t("если мы не будем осторожны, Эй.И. может уничтожить", "ru") == "если мы не будем осторожны, ИИ может уничтожить"
    assert t("AIR и SAID", "ru") == "AIR и SAID" and t("What is AI?", "en") == "What is AI?"
