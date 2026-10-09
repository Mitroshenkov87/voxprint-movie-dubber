"""Regression tests built from the two real user clips (runs of 2026-10-09 on the RTX 4090 laptop and on the box):
segmentation of recognised speech, translation completeness, shortening / best take, the MP4 remux cache bug, runtime pins
and the shared Voxprint settings."""
import json
import re
import shutil
from pathlib import Path

import pytest

from dubber.core import script, segment, timefit
from dubber.core.project import Project
from dubber.engines import translation as T
from dubber.pipeline import runner as R
from dubber.pipeline import stages as S

DATA = Path(__file__).resolve().parent / "data"


def _asr(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def _lines(name, **kw):
    d = _asr(name)
    return script.lines_from_asr(d["segments"], lang="en", **kw)


# ------------------------------------------------------------------ 1. segmentation
def test_clip2_lines_follow_sentences_not_asr_chunks():
    lines = _lines("asr-user-clip2.json")
    texts = [ln.text for ln in lines]
    # run 2 had "wow you cleaned yeah man i deep cleaned the entire apartment what's the occasion well i got a hot day" as ONE line
    assert "Wow, you cleaned." in texts and "What's the occasion?" in texts and "Good for you, man." in texts
    assert not any("occasion?" in t and "Well" in t for t in texts)
    assert all(ln.duration <= segment.MAX_S + 0.01 for ln in lines)
    assert all(re.search(r"[.!?…]$", t) for t in texts)                 # every line is whole sentence(s)
    assert len(lines) >= 40                                             # 16 chunks before


def test_lines_never_span_a_long_pause():
    d = _asr("asr-user-clip2.json")
    words = segment.words_of(d["segments"], "en")
    pauses = [(a["end"], b["start"]) for a, b in zip(words, words[1:]) if b["start"] - a["end"] > segment.HARD_PAUSE]
    assert pauses
    for ln in script.lines_from_asr(d["segments"], lang="en"):
        assert not any(ln.start < p0 and p1 < ln.end for p0, p1 in pauses), ln.text


def test_clip1_short_exclamations_and_acronyms():
    texts = [ln.text for ln in _lines("asr-user-clip1.json")]
    assert "Learning sucks!" in texts and "Destroy!" in texts and "Yeah, yeah, yeah, cool!" in texts
    assert any(t.startswith("It's come to my attention") and t.endswith("homework for you.") for t in texts)   # "A.I. to do" not cut
    assert "Beavis, we need to get us some of this A.I." in texts
    assert sum(t == "B.I." for t in texts) <= 1                       # the recogniser's loop over the laugh is collapsed


def test_junk_tokens_removed():
    ws = [{"w": " Wellсем,", "start": 0, "end": .3}, {"w": " no.", "start": .3, "end": .5}, {"w": " Choi443buz", "start": .5, "end": .7},
          {"w": " not=#", "start": .7, "end": .9}, {"w": " A", "start": .9, "end": 1.0}, {"w": ".I.", "start": 1.0, "end": 1.2},
          {"w": " очему.", "start": 1.2, "end": 1.4}]
    assert [w["w"] for w in segment.clean_words(ws, "en")] == [" Well,", " no.", " not", " A.I."]


def test_unpunctuated_stream_is_capped_and_split_at_speaker_turns():
    d = _asr("asr-user-clip2-unpunctuated.json")              # the PC run: lower case, no punctuation, chunks of 4-8 s
    assert not segment.punctuated(segment.words_of(d["segments"]))
    turns = [{"start": 0.0, "end": 0.96, "speaker": "A"}, {"start": 0.96, "end": 3.8, "speaker": "B"},
             {"start": 3.8, "end": 4.9, "speaker": "A"}, {"start": 4.9, "end": 8.4, "speaker": "B"},
             {"start": 8.4, "end": 30.0, "speaker": "A"}]
    lines = script.lines_from_asr(d["segments"], lang="en", turns=turns)
    for b in (0.96, 3.8, 4.9, 8.4):
        assert not any(ln.start < b - 0.05 and ln.end > b + 0.05 for ln in lines), b
    assert lines[0].text == "Wow you cleaned"
    plain = script.lines_from_asr(d["segments"], lang="en")
    assert all(ln.duration <= segment.MAX_S_UNPUNCTUATED + 0.01 for ln in plain)


def test_asr_decode_options_keep_punctuation():
    from dubber.engines.asr import decode_options
    o = decode_options("en")
    assert o["condition_on_previous_text"] and o["initial_prompt"].count(".") >= 2 and o["word_timestamps"]
    assert max(o["temperature"]) <= 0.4 and o["language"] == "en"
    assert "initial_prompt" not in decode_options("ja")


# ------------------------------------------------------------------ 2. translation completeness
def _marian_like(texts):
    """Stand-in that behaves like Opus-MT on the real lines: with several sentences it drops the last short one."""
    out = []
    for t in texts:
        sents = T.split_sentences(t)
        if len(sents) > 1 and len(sents[-1]) < 20:
            sents = sents[:-1]
        out.append(" ".join(f"<{s}>" for s in sents))
    return out


def test_sentence_split():
    assert T.split_sentences("Yeah, that sounds cool! Learning sucks!") == ["Yeah, that sounds cool!", "Learning sucks!"]
    assert T.split_sentences("Beavis, we need to get us some of this A.I.") == ["Beavis, we need to get us some of this A.I."]
    assert T.split_sentences("There's a concern that A.I. will take over.") == ["There's a concern that A.I. will take over."]
    assert len(T.split_sentences("Yeah, yeah, yeah, cool. Destroy it all. Destroy.")) == 3


def test_short_exclamations_survive_translation():
    src = ["Yeah, that sounds cool! Learning sucks!", "Yeah, yeah, yeah, cool. Destroy it all. Destroy."]
    whole = _marian_like(src)                                        # the old way: line by line
    assert "Learning" not in whole[0] and not whole[1].endswith("<Destroy.>")
    out, _ = T.complete_translate(src, _marian_like)
    assert "<Learning sucks!>" in out[0] and out[1].endswith("<Destroy.>")


def test_incomplete_sentence_is_retried():
    def drops(texts):
        return ["" if t == "Destroy!" else f"[{t}]" for t in texts]

    out, retried = T.complete_translate(["Destroy it all! Destroy!"], drops, retry_fn=lambda xs: [f"<{x}>" for x in xs])
    assert retried == 1 and out[0] == "[Destroy it all!] <Destroy!>"
    assert T.suspicious("Learning sucks!", "") and T.suspicious("We need to get us some of this", "Бивис.")
    assert T.suspicious("Learning sucks!", "Learning sucks!", "ru") and not T.suspicious("Learning sucks!", "Учиться - отстой!", "ru")


# ------------------------------------------------------------------ 3. shortening and best take
def test_shorten_never_reduces_to_one_word():
    line = "Бивис, нам нужно достать кое-что из этого ИИ."              # run 1, line 11: was cut to "Бивис."
    out = script.shorten(line, "ru")
    assert "Бивис." not in out
    for t in ("Ну, знаешь, я просто хотел сказать (тихо), что это было очень, очень важно, понимаешь", line,
              "В самом деле, грубо, так что я был в своем уме, я могу представить, как заставить этих взрослых убраться после"):
        for v in script.shorten(t, "ru"):
            assert script.kept_share(v, t) >= script.MIN_KEEP and len(v.split()) >= 2


def test_best_take_prefers_complete_meaning():
    # full take slightly too long (needs ~1.2x), a cut variant that fits but kept 12 %: the full line wins
    assert timefit.best_take([4.0, 0.61], 3.83, [1.0, 0.12]) == 0
    # among fitting takes the more complete wording wins over the longer one
    assert timefit.best_take([3.0, 3.5], 3.83, [1.0, 0.8]) == 0
    # nothing fits within the hard limit for the full text, a 90 % variant does: take it
    assert timefit.best_take([6.0, 4.1], 3.83, [1.0, 0.9]) == 1
    assert timefit.best_take([2.5, 1.8, 1.9], 2.0) == 2               # old behaviour without completeness info


# ------------------------------------------------------------------ 4. MP4 remux re-runs only the mux
@pytest.fixture
def film_no_subs(tmp_path, clip):
    d = tmp_path / "films"
    d.mkdir()
    shutil.copy(clip, d / "clip.mkv")
    return d / "clip.mkv"


def test_mp4_remux_after_auto_language_reruns_only_mux(tmp_path, film_no_subs, monkeypatch):
    segs = [{"start": 0.8, "end": 3.9, "text": "Good evening. I was hoping you would come.",
             "words": [{"w": f" {w}", "start": 0.8 + i * 0.4, "end": 1.1 + i * 0.4} for i, w in
                       enumerate("Good evening. I was hoping you would come.".split())]}]

    def fake_asr(p, cfg, emit):
        S.write_json(p.path("analysis", "asr.json"), {"language": "en", "segments": segs})
        p.settings["source_lang_detected"] = "en"                      # 'auto' resolved by recognition
        return "1 segment"

    monkeypatch.setitem(S.FUNCS, "asr", fake_asr)
    p = Project.create(tmp_path / "proj", film_no_subs, target_lang="ru", output_format="mkv")
    assert p.settings["source_lang"] == "auto"
    res = R.Runner(p, S.MOCK_CFG).run()
    assert res.ok, res
    q = Project(p.folder)
    assert q.settings["source_lang"] == "auto" and S.source_lang(q) == "en"
    q.settings["output_format"] = "mp4"
    q.save()
    res2 = R.Runner(Project(p.folder), S.MOCK_CFG).run(from_stage="mux")
    assert res2.ok, res2
    assert res2.stages["mux"] == "done"
    assert all(v == "cached" for k, v in res2.stages.items() if k != "mux"), res2.stages
    assert Path(Project(p.folder).settings["output_file"]).suffix == ".mp4"


# ------------------------------------------------------------------ 5. runtime pins and users
def test_runtime_lock_is_the_audiobook_builder_pin():
    from dubber.infra import runtime
    lock = runtime.load_lock()
    assert lock["torch_version"] == "2.11.0" and lock["python"] == "3.11" and "cu128" in lock["flavors"]
    assert any(w["dist"] == "torch" and w["version"] == "2.11.0+cu128" and "cp311" in w["file"] for w in lock["wheels"])
    assert runtime.choose_flavor(lock, (12, 8)) == "cu128" and runtime.choose_flavor(lock, (13, 0)) == "cu128"
    assert runtime.choose_flavor(lock, (12, 6)) == "cu126" and runtime.choose_flavor(lock, None) == "cpu"


def test_runtime_key_is_exact_and_matches_the_installer_script():
    from dubber.infra import runtime
    req = (Path(__file__).resolve().parent.parent / "requirements.txt").read_text(encoding="utf-8")
    con = (Path(__file__).resolve().parent.parent / "installer" / "runtime-constraints.txt").read_text(encoding="utf-8")
    k = runtime.runtime_key("cu128", req, con)
    assert re.fullmatch(r"[0-9a-f]{12}", k)
    assert runtime.runtime_key("cu128", req + "\n# a comment\n\n", con) == k            # comments / blank lines do not count
    assert runtime.runtime_key("cu126", req, con) != k and runtime.runtime_key("cu128", req + "\nnumpy<3", con) != k
    ps = (Path(__file__).resolve().parent.parent / "installer" / "install-runtime.ps1").read_text(encoding="utf-8")
    assert 'python=$($lockObj.python);platform=$($lockObj.platform);torch=$($lockObj.torch_version)+$flavor;' in ps
    assert "runtime-$key" in ps and ".users.json" in ps and "New-Item -ItemType Junction" in ps
    assert "never upgraded" in ps


def test_runtime_users_and_cli(tmp_path, monkeypatch):
    import main as app_main
    from dubber.infra import runtime, shared_paths
    rt = shared_paths.voxprint_home() / "runtime-0123456789ab"
    (rt / "env").mkdir(parents=True)
    (rt / "runtime-key.json").write_text("{}", encoding="utf-8")
    (rt / ".users.json").write_text('{"movie-dubber": true, "future-app": true}', encoding="utf-8")
    monkeypatch.setattr(runtime, "current_runtime_dir", lambda: rt)
    out = tmp_path / "o.txt"
    assert app_main.main(["--unregister-runtime-user", "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8").splitlines() == ["1", str(rt)]
    assert json.loads((rt / ".users.json").read_text()) == {"future-app": True}
    assert runtime.is_runtime_dir(rt) and not runtime.is_runtime_dir(shared_paths.voxprint_home() / "runtime")
    assert app_main.main(["--register-runtime-user"]) == 0 and json.loads((rt / ".users.json").read_text())["movie-dubber"]


def test_cuda_dll_dirs_are_found_without_importing_torch():
    from dubber.infra import cuda_dlls
    dirs = cuda_dlls.candidate_dirs()
    assert all(d.is_dir() for d in dirs)
    assert cuda_dlls.expose() == [] or isinstance(cuda_dlls.expose(), list)


# ------------------------------------------------------------------ 6. suite.json
def test_suite_json_schema_atomic_and_unknown_keys_kept():
    from dubber.infra import suite
    f = suite.path()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text('{"schema": 1, "future_key": {"x": 1}, "theme": "neon"}', encoding="utf-8")
    assert suite.theme() == "glass-dark" and suite.gpu() == "auto" and suite.models_dir() is None
    suite.set_value("gpu", "cuda")
    suite.set_value("ui_language", "ru-RU")
    data = json.loads(f.read_text(encoding="utf-8"))
    assert data == {"schema": 1, "future_key": {"x": 1}, "theme": "neon", "gpu": "cuda:0", "ui_language": "ru"}
    assert not list(f.parent.glob("*.tmp"))
    with pytest.raises(ValueError):
        suite.set_value("gpu", "rocm")
    with pytest.raises(ValueError):
        suite.set_value("output_format", "mp4")                     # app-specific settings never go to suite.json


def test_models_dir_from_suite_then_legacy_txt(tmp_path, monkeypatch):
    from dubber.infra import shared_paths, suite
    legacy = tmp_path / "legacy-models"
    (shared_paths.shared_state_dir() / "models_dir.txt").write_text(str(legacy), encoding="utf-8")
    assert shared_paths.configured_models_dir() == legacy               # no suite.json yet: the older file counts
    suite.set_value("models_dir", None)
    assert shared_paths.configured_models_dir() is None                # explicit null = the default folder
    chosen = tmp_path / "chosen"
    shared_paths.set_models_dir(chosen)
    assert json.loads(suite.path().read_text())["models_dir"] == str(chosen) and shared_paths.models_dir() == chosen
    monkeypatch.setenv("VOXPRINT_MODELS_DIR", str(tmp_path / "env"))
    assert shared_paths.configured_models_dir() == tmp_path / "env"    # the environment variable always wins


def test_device_and_language_are_shared(monkeypatch):
    from dubber import i18n, settings
    from dubber.infra import suite
    assert settings.device() == "auto"
    settings.set_device("cpu")
    assert suite.gpu() == "cpu" and settings.engine_cfg()["device"] == "cpu"
    suite.set_value("ui_language", "ru")
    i18n.reset()
    monkeypatch.delenv("VOXPRINT_LANG", raising=False)
    assert i18n.detect_language() == "ru"
    i18n.set_language("en")
    assert suite.ui_language() == "en"


def test_sync_suite_settings_cli(tmp_path):
    import main as app_main
    from dubber.infra import shared_paths, suite
    (shared_paths.shared_state_dir() / "models_dir.txt").write_text(str(tmp_path / "m"), encoding="utf-8")
    assert app_main.main(["--sync-suite-settings"]) == 0
    d = json.loads(suite.path().read_text())
    assert d["schema"] == 1 and d["models_dir"] == str(tmp_path / "m") and d["ui_language"] in ("en", "ru")


def test_installer_suite_integration():
    iss = (Path(__file__).resolve().parent.parent / "installer" / "VoxprintMovieDubber.iss").read_text(encoding="utf-8")
    assert "DefaultGroupName=Voxprint" in iss and "UsePreviousGroup=no" in iss
    assert "{6F1D2B7A-3C54-4E0B-9A41-7B5E0C9D2F18}_is1" in iss and '"audiobook-builder": true' in iss    # sibling detection
    assert "--unregister-runtime-user" in iss and "--unregister-models-user" in iss and "--sync-suite-settings" in iss
    assert "MB_DEFBUTTON2, IDNO) = IDYES" in iss                       # models deleted only on an explicit Yes (default No)
    assert "LicenseFile=..\\LICENSE" in iss
    lic = (Path(__file__).resolve().parent.parent / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in lic and "Version 2.0" in lic
    assert "OFFLINE" not in iss.upper().split("ONLINE", 1)[0]


# ------------------------------------------------------------------ box re-run of 2026-10-09 (CPU, no separation)
def test_hallucinated_burst_lines_are_dropped():
    # end of clip 2: "I'll give you some space." three more times within 0.2 s (93.26-93.48) after the real line at 89.46
    words = [{"w": " I'll", "start": 89.46, "end": 89.8}, {"w": " give", "start": 89.8, "end": 90.0},
             {"w": " you", "start": 90.0, "end": 90.2}, {"w": " some", "start": 90.2, "end": 90.4},
             {"w": " space.", "start": 90.4, "end": 90.68}, {"w": " No.", "start": 92.82, "end": 93.26}]
    for k, (a, b) in enumerate([(93.26, 93.28), (93.28, 93.34), (93.34, 93.48)]):
        step = (b - a) / 5
        words += [{"w": w, "start": a + i * step, "end": a + (i + 1) * step} for i, w in
                  enumerate([" I'll", " give", " you", " some", " space."])]
    texts = [t for _, _, t in segment.segment(words)]
    assert texts == ["I'll give you some space.", "No."]
    assert not segment.impossible_rate(67.76, 67.98, "Wait.")             # a short real exclamation stays


def test_stray_marks_are_tidied():
    t = segment.tidy_punctuation
    assert t("Up at your place and wants to meet up at your place.).,") == "Up at your place and wants to meet up at your place."
    assert t("Wait... what?") == "Wait... what?" and t("Hello , world.") == "Hello, world." and t("Yes, (really) fine.") == "Yes, (really) fine."


def test_cyrillic_acronym_only_when_the_source_says_ai():
    t = S.tidy_translation
    assert t("А.И. надрал задницу.", "ru", "A.I. kicks ass.") == "ИИ надрал задницу."
    assert t("Есть опасения, что А.И. возьмет на себя всю нашу работу", "ru", "There's a concern that A.I. will take over") == \
        "Есть опасения, что ИИ возьмет на себя всю нашу работу"
    assert t("А.И. Пушкин написал это.", "ru", "A. I. Pushkin wrote it.") == "А.И. Пушкин написал это."
    assert t("если мы не будем осторожны, Ай-Ий может уничтожить само человечество.", "ru", "A.I. could destroy humanity itself.") == \
        "если мы не будем осторожны, ИИ может уничтожить само человечество."


def test_uninstaller_finds_the_runtime_from_the_installer_note(tmp_path, monkeypatch):
    from dubber.infra import runtime

    rt = tmp_path / "Voxprint" / "runtime-0123456789ab"
    (rt / "env").mkdir(parents=True)
    (rt / runtime.KEY_FILE).write_text("{}", encoding="utf-8")
    app = tmp_path / "app"
    app.mkdir()
    assert runtime.current_runtime_dir(app) is None or runtime.current_runtime_dir(app) != rt
    (app / runtime.DIR_FILE).write_text(str(rt), encoding="utf-8")
    assert runtime.current_runtime_dir(app) == rt
    ps1 = (Path(__file__).resolve().parents[1] / "installer" / "install-runtime.ps1").read_text(encoding="utf-8")
    iss = (Path(__file__).resolve().parents[1] / "installer" / "VoxprintMovieDubber.iss").read_text(encoding="utf-8")
    assert runtime.DIR_FILE in ps1 and runtime.DIR_FILE in iss


def test_asr_hole_is_found_and_filled_from_the_second_pass():
    from dubber.engines import asr

    d = _asr("asr-user-clip2.json")
    words = [w for s in d["segments"] for w in s["words"]]
    # the box run without separation: the conditioned pass jumped from 9.3 s to 24.2 s
    first = [{"start": 0, "end": 9.34, "text": "", "words": [w for w in words if w["end"] <= 9.34]},
             {"start": 24.16, "end": 93.3, "text": "", "words": [w for w in words if w["start"] >= 24.16]}]
    holes = asr.find_holes([(0.0, 93.5)], first)
    assert len(holes) == 1 and 9.3 < holes[0][0] < 10.5 and 23.0 < holes[0][1] < 24.2
    merged, added = asr.fill_holes(first, d["segments"], holes)
    got = [w for s in merged for w in s["words"]]
    assert added > 20 and len(got) == len(words) and [w["start"] for w in got] == sorted(w["start"] for w in got)
    assert asr.find_holes([(0.0, 93.5)], d["segments"]) == []          # the complete transcript has none
    assert asr.find_holes([], first) == []
