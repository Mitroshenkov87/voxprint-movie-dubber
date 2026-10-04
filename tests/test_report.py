import re
import time

from dubber.diag import report as R
from dubber.diag.report import CheckResult, Report, Status


def make_report():
    rep = Report(app_line="test 1.0", options={"quick": True})
    rep.add(CheckResult("system.os", "OS", Status.INFO, "Windows 11"))
    rep.add(CheckResult("gpu.torch", "Torch", Status.OK, "CUDA ok", seconds=1.5, details=["a : b"]))
    rep.add(CheckResult("tts.graphs_sdpa", "TTS", Status.FAIL, "RuntimeError: boom", traceback="Traceback...\nRuntimeError: boom"))
    rep.add(CheckResult("stage.vad", "VAD", Status.SKIP, "no clip"))
    return rep


def test_render_has_all_sections_and_counts():
    rep = make_report()
    rep.finish(True)
    text = rep.render()
    for part in ("SUMMARY", "KEY NUMBERS", "DETAILS", "ERRORS AND TRACEBACKS", "END OF REPORT"):
        assert part in text
    assert "State         : COMPLETE" in text
    assert "RuntimeError: boom" in text and "Totals:" in text
    assert rep.counts()["FAIL"] == 1


def test_incomplete_report_says_what_was_running():
    rep = make_report()
    rep.current_step = "tts.graphs_fa2"
    text = rep.render()
    assert "INCOMPLETE" in text and "tts.graphs_fa2" in text


def test_same_id_replaces():
    rep = make_report()
    rep.add(CheckResult("stage.vad", "VAD", Status.OK, "now fine"))
    assert len([r for r in rep.results if r.id == "stage.vad"]) == 1
    assert rep.get("stage.vad").status == Status.OK


def test_redaction_of_tokens_and_user_names():
    tok = "hf_" + "A1b2C3d4E5f6G7h8"
    assert tok not in R.redact(f"token={tok}")
    assert "alex" not in R.redact(r"C:\Users\alex\Desktop\x.txt")
    assert "alex" not in R.redact("/home/alex/models")
    assert "/tmp/wk/home/models" in R.redact("/tmp/wk/home/models")      # a folder called "home" is not a user profile
    assert "SECRETVALUE123" not in R.redact("x SECRETVALUE123 y", extra_secrets=["SECRETVALUE123"])


def test_report_redacts_secret_given_by_user():
    rep = Report(secrets=["my-private-token-xyz"])
    rep.add(CheckResult("a", "A", Status.FAIL, "failed with my-private-token-xyz", traceback="my-private-token-xyz"))
    assert "my-private-token-xyz" not in rep.render()


def test_save_atomic_and_fallback(tmp_path):
    rep = make_report()
    rep.finish(True)
    p = R.save_report(rep, tmp_path / "d" / "r.txt")
    assert p.read_text(encoding="utf-8").startswith("=")
    assert not list((tmp_path / "d").glob("*.tmp"))
    blocker = tmp_path / "file"
    blocker.write_text("x")                                  # a FILE where a folder is needed -> primary write fails
    alt = R.save_report(rep, blocker / "r.txt", fallback_dir=tmp_path / "fb")
    assert alt.parent == tmp_path / "fb" and alt.exists()


def test_filename_pattern():
    assert re.fullmatch(r"Voxprint-MovieDubber-Diagnostics-\d{8}-\d{6}\.txt", R.report_filename(time.time()))


def test_table_alignment():
    lines = R.table(["a", "bb"], [["1", "2"], ["333", "4"]])
    assert len({len(l.rstrip()) for l in lines[:1]}) == 1 and len(lines) == 4


def test_key_numbers_with_tts_rows():
    rep = make_report()
    ok = CheckResult("tts.standard_sdpa", "TTS", Status.OK, "ok", metrics={"mode": "standard_sdpa", "rtf": 0.8, "load_s": 5.0, "warmup_s": 2.0})
    rep.add(ok)
    text = "\n".join(R.key_numbers(rep.results))
    assert "0.8" in text and "standard" in text.lower()
