import sys
import textwrap

from dubber.diag import procs
from dubber.diag.procs import decode_throttle, describe_exit_code, run_worker
from dubber.diag.report import CheckResult, Status
from dubber.diag.runner import DiagnosticRunner, DiagOptions, guarded


def fake(tmp_path, body):
    f = tmp_path / "fake.py"
    f.write_text("import sys, json\nMARK='@@VX@@ '\n" + textwrap.dedent(body), encoding="utf-8")
    return [sys.executable, str(f)]


def test_decode_throttle():
    assert decode_throttle(0) in ("", "none", "no throttling")
    assert "power" in decode_throttle(0x4).lower() or "sw" in decode_throttle(0x4).lower()


def test_exit_codes():
    assert "access violation" in describe_exit_code(0xC0000005)
    assert "access violation" in describe_exit_code(-1073741819)       # same value as signed 32-bit
    assert "exit code 3" in describe_exit_code(3)


def test_worker_ok(tmp_path):
    cmd = fake(tmp_path, "print('stray library noise')\nprint(MARK + json.dumps({'t':'log','msg':'hi'}))\n"
                         "print(MARK + json.dumps({'t':'result','status':'OK','summary':'fine','details':['x : y']}))\n")
    logs = []
    out = run_worker("fake", {}, timeout=30, command=cmd, sample_gpu=False, on_log=logs.append)
    assert out.result["status"] == "OK" and logs == ["hi"] and out.returncode == 0
    r = DiagnosticRunner.result_from_outcome("c", "C", out)
    assert r.status == Status.OK and r.summary == "fine"


def test_worker_crash_is_a_result_not_an_exception(tmp_path):
    cmd = fake(tmp_path, "sys.stderr.write('native boom\\n')\nsys.exit(-1073741819 & 0xFFFFFFFF if False else 3)\n")
    out = run_worker("fake", {}, timeout=30, command=cmd, sample_gpu=False)
    r = DiagnosticRunner.result_from_outcome("c", "C", out)
    assert r.status == Status.FAIL and "CRASHED" in r.summary and "native boom" in r.traceback


def test_worker_timeout_kills_it(tmp_path):
    cmd = fake(tmp_path, "import time\ntime.sleep(60)\n")
    out = run_worker("fake", {}, timeout=1.5, command=cmd, sample_gpu=False)
    assert out.timed_out
    r = DiagnosticRunner.result_from_outcome("c", "C", out)
    assert r.status == Status.FAIL and "TIMEOUT" in r.summary


def test_garbage_stdout_is_ignored(tmp_path):
    cmd = fake(tmp_path, "print(MARK + '{not json')\nprint(MARK + json.dumps({'t':'result','status':'WARN','summary':'s'}))\n")
    out = run_worker("fake", {}, timeout=30, command=cmd, sample_gpu=False)
    assert out.result["status"] == "WARN"


def test_cannot_start(tmp_path):
    out = run_worker("fake", {}, timeout=5, command=[str(tmp_path / "does-not-exist.exe")], sample_gpu=False)
    assert out.start_error
    assert DiagnosticRunner.result_from_outcome("c", "C", out).status == Status.FAIL


def test_guarded_turns_exceptions_into_fail():
    def boom():
        raise ValueError("bad thing")
    r = guarded("x.y", "X", boom)
    assert r.status == Status.FAIL and "ValueError" in r.summary and "Traceback" in r.traceback and r.seconds is not None


def test_guarded_passes_results_through():
    r = guarded("x.y", "X", lambda: CheckResult("x.y", "X", Status.OK, "fine"))
    assert r.status == Status.OK


def test_runner_survives_a_broken_step_and_writes_report(tmp_path, monkeypatch):
    """One check raising must not stop the report; the file lands on the Desktop."""
    opt = DiagOptions(allow_download=False, quick=True, skip=("models", "tts", "stages", "gpu", "network"))
    run = DiagnosticRunner(opt)
    from dubber.diag import checks_system
    def broken():
        raise RuntimeError("os probe exploded")
    monkeypatch.setattr(checks_system, "check_os", broken)
    path = run.run()
    text = path.read_text(encoding="utf-8")
    assert path.parent == __import__("dubber.paths", fromlist=["x"]).desktop_dir()
    assert "os probe exploded" in text and "Traceback" in text          # the failure is in the report ...
    assert "system.hw" in text and "system.ffmpeg" in text             # ... and the later checks still ran
    assert "END OF REPORT" in text and "State         : COMPLETE" in text
