"""One worker process for every model stage of a dub.

The process stays up across stages and across films started from this program, so the resident model cache
(``dubber.infra.resident``) can keep weights loaded.  The parent holds the shared GPU lock for the whole run and
releases it when the run ends.  Diagnostics still use one process per check.
"""
from __future__ import annotations

import atexit
import json
import os
import queue
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dubber.diag.procs import MARK, worker_command
from dubber.infra.thermal import telemetry_from_summary

_session: Optional["WorkerSession"] = None


def _plain_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in cfg.items() if k != "hf_token" and not callable(v)}


class WorkerSession:
    """A living ``session`` worker.  :meth:`run_stage` sends one stage and waits for its result line."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = _plain_cfg(cfg)
        self.proc: Optional[subprocess.Popen] = None
        self._q: queue.Queue = queue.Queue()
        self._err: List[str] = []
        self._args: Optional[Path] = None
        self._start()

    def alive(self) -> bool:
        """True while the session worker process is still running."""
        return self.proc is not None and self.proc.poll() is None

    def _start(self) -> None:
        fd, tmp = tempfile.mkstemp(prefix="vmd-session-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"cfg": self.cfg, "session": True}, fh)
        self._args = Path(tmp)
        env = dict(os.environ)
        env.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
        kw: Dict[str, Any] = {}
        if os.name == "nt":
            kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            kw["start_new_session"] = True
        self.proc = subprocess.Popen(worker_command("session", self._args), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, env=env, text=True, encoding="utf-8", errors="replace", bufsize=1, **kw)
        threading.Thread(target=self._read_out, daemon=True).start()
        threading.Thread(target=self._read_err, daemon=True).start()

    def _read_out(self) -> None:
        proc = self.proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            text = line.strip()
            if not text.startswith(MARK):
                continue
            try:
                self._q.put(json.loads(text[len(MARK):]))
            except ValueError:
                continue

    def _read_err(self) -> None:
        proc = self.proc
        if proc is None or proc.stderr is None:
            return
        for line in proc.stderr:
            if len(self._err) < 80:
                self._err.append(line.rstrip("\n"))

    def run_stage(self, folder: Path, key: str, cfg: Dict[str, Any], on_log: Callable[[str], None],
                  cancel: Optional[Callable[[], bool]] = None, timeout: float = 0.0) -> str:
        """Run one stage.  ``timeout`` 0 waits until the stage finishes (the runner has its own limit)."""
        if not self.alive() or self.proc is None or self.proc.stdin is None:
            raise RuntimeError("the model session is not running")
        payload = {"stage": key, "project": str(folder), "cfg": _plain_cfg(cfg)}
        self.proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        deadline = time.monotonic() + timeout if timeout else None
        while True:
            if cancel is not None and cancel():
                self.close()
                from dubber.pipeline.runner import Cancelled

                raise Cancelled()
            try:
                msg = self._q.get(timeout=0.25)
            except queue.Empty:
                if not self.alive():
                    tail = "\n".join(self._err)[-400:]
                    raise RuntimeError(f"session ended while running {key}" + (f": {tail}" if tail else ""))
                if deadline is not None and time.monotonic() > deadline:
                    self.close()
                    raise RuntimeError(f"{key} timed out in the model session")
                continue
            if msg.get("t") == "log":
                on_log(str(msg.get("msg", "")))
                continue
            if msg.get("t") == "result" and msg.get("stage") == key:
                line = telemetry_from_summary(key, msg.get("gpu") or {})
                if line:
                    on_log(line)
                if msg.get("status") == "OK":
                    return str(msg.get("summary", ""))
                # runner imports this module, so the helper stays inside the method.
                from dubber.pipeline.runner import stage_error

                raise stage_error(str(msg.get("summary") or f"{key} failed"), str(msg.get("traceback") or ""))
            if msg.get("t") == "result" and msg.get("status") == "FAIL" and not msg.get("stage"):
                raise RuntimeError(str(msg.get("summary") or "session failed"))

    def close(self) -> None:
        """Stop the session worker, waiting up to 15 seconds before killing it."""
        proc = self.proc
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin is not None:
                proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                proc.stdin.flush()
        except Exception:  # noqa: BLE001 - the process may already be gone
            pass
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        if self._args is not None:
            self._args.unlink(missing_ok=True)
            self._args = None
        self.proc = None


def get_session(cfg: Dict[str, Any]) -> WorkerSession:
    """The process shared by every dub started in this program."""
    global _session
    if _session is None or not _session.alive():
        _session = WorkerSession(cfg)
    return _session


def close_session() -> None:
    """Stop the shared session worker for this program, if one is running."""
    global _session
    if _session is not None:
        _session.close()
        _session = None


atexit.register(close_session)
