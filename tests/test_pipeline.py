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


def test_shorten_variants_are_shorter():
    text = "Ну, знаешь, я просто хотел сказать (тихо), что это было очень, очень важно, понимаешь"
    out = script.shorten(text, "ru")
    assert out and all(len(v) < len(text) for v in out) and "(тихо)" not in out[0]


# ------------------------------------------------------------------ time fitting
def _lines(*spans):
    return [Line(i + 1, a, b, translation="x") for i, (a, b) in enumerate(spans)]


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
    assert secs == pytest.approx(9.25, abs=0.01) and text == "a b"


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
