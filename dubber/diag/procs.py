"""Run a heavy step in a separate process ("worker") and watch it.

Why a process per model: a crash (access violation in a CUDA DLL, driver reset, out-of-memory kill) or a hang kills only the
worker; the parent turns that into a FAIL entry with the exit code and the tail of stderr, and the report goes on.  A process
also gives clean VRAM numbers (everything is freed when it exits) and avoids the transformers-version conflicts between libraries.
It is also the architecture decided for the product (research note 06, section 3).

Protocol: the worker prints lines ``@@VX@@ {json}`` on stdout (anything else on stdout is ignored, stderr is kept as a log):
``{"t":"log","msg":...}`` progress text, ``{"t":"result", ...}`` the final result (status/summary/details/metrics/traceback).
"""
from __future__ import annotations

import collections
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

from dubber import paths
from dubber.appinfo import resource_dir

MARK = "@@VX@@ "
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def worker_command(name: str, args_file: Path) -> List[str]:
    """Command line that starts worker ``name`` (frozen exe: ``exe --worker ...``; source: ``python main.py --worker ...``)."""
    if getattr(sys, "frozen", False):
        return [sys.executable, "--worker", name, str(args_file)]
    return [sys.executable, str(resource_dir() / "main.py"), "--worker", name, str(args_file)]


# ---------------------------------------------------------------------------------------------- nvidia-smi sampling
def find_nvidia_smi() -> Optional[str]:
    """``nvidia-smi`` from PATH or its usual Windows locations."""
    exe = shutil.which("nvidia-smi")
    if exe:
        return exe
    for p in (r"C:\Windows\System32\nvidia-smi.exe", r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"):
        if os.path.isfile(p):
            return p
    return None


THROTTLE_BITS = {0x1: "idle", 0x2: "app-clocks-setting", 0x4: "SW power cap", 0x8: "HW slowdown", 0x10: "sync boost",
                 0x20: "SW thermal slowdown", 0x40: "HW thermal slowdown", 0x80: "HW power brake", 0x100: "display clock setting"}


def decode_throttle(mask: int) -> str:
    """Human text for the nvidia-smi throttle-reason bit mask (``0x0`` -> ``none``)."""
    if mask == 0:
        return "none"
    names = [n for bit, n in THROTTLE_BITS.items() if mask & bit and bit not in (0x1,)]
    return ", ".join(names) if names else "idle only"


def _num(s: str) -> Optional[float]:
    try:
        return float(s.strip().split()[0])
    except (ValueError, IndexError):
        return None


class GpuSampler(threading.Thread):
    """Polls nvidia-smi once a second while a worker runs; :meth:`summary` gives max/min values.  Silent if nvidia-smi is absent."""
    FIELDS = ["utilization.gpu", "memory.used", "power.draw", "temperature.gpu", "clocks.sm", "clocks.mem"]

    def __init__(self, interval: float = 1.0) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.smi = find_nvidia_smi()
        self._halt = threading.Event()
        self.samples: List[Dict[str, Optional[float]]] = []
        self.throttle_seen = 0
        self.throttle_field = ""

    def _query(self, fields: List[str]) -> Optional[List[str]]:
        smi = self.smi
        if not smi:
            return None
        try:
            p = subprocess.run([smi, f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"], capture_output=True,
                               text=True, timeout=10, creationflags=_NO_WINDOW, encoding="utf-8", errors="replace")
            if p.returncode == 0 and p.stdout.strip():
                return [c.strip() for c in p.stdout.strip().splitlines()[0].split(",")]
        except (OSError, subprocess.SubprocessError):
            pass
        return None

    def run(self) -> None:
        """Poll nvidia-smi about once a second until :meth:`stop` is called."""
        if not self.smi:
            return
        thr = None
        for cand in ("clocks_throttle_reasons.active", "clocks_event_reasons.active"):
            if self._query([cand]):
                thr = cand
                self.throttle_field = cand
                break
        fields = self.FIELDS + ([thr] if thr else [])
        while not self._halt.is_set():
            row = self._query(fields)
            if row and len(row) == len(fields):
                s = {f: _num(v) for f, v in zip(self.FIELDS, row)}
                self.samples.append(s)
                if thr:
                    try:
                        self.throttle_seen |= int(row[-1], 16)
                    except ValueError:
                        pass
            self._halt.wait(self.interval)

    def stop(self) -> None:
        """Stop sampling and wait up to 12 seconds for this thread to finish."""
        self._halt.set()
        if self.is_alive():
            self.join(timeout=12)

    def summary(self) -> Dict[str, Any]:
        """``{"n": samples, "util_max", "mem_used_max_gb", "power_max_w", "temp_max_c", "sm_clock_min/max", "throttle": text}``."""
        def col(name: str) -> List[float]:
            return [s[name] for s in self.samples if s.get(name) is not None]       # type: ignore[misc]
        out: Dict[str, Any] = {"n": len(self.samples)}
        if not self.samples:
            return out
        for key, name, scale in (("util_max", "utilization.gpu", 1), ("mem_used_max_gb", "memory.used", 1 / 1024),
                                 ("power_max_w", "power.draw", 1), ("temp_max_c", "temperature.gpu", 1)):
            c = col(name)
            if c:
                out[key] = round(max(c) * scale, 1)
        sm = col("clocks.sm")
        if sm:
            out["sm_clock_min"], out["sm_clock_max"] = int(min(sm)), int(max(sm))
        out["throttle"] = decode_throttle(self.throttle_seen)
        return out

    @staticmethod
    def describe(s: Dict[str, Any]) -> str:
        """One line for the report, e.g. ``GPU during step: util max 92%, VRAM max 7.1 GB, ...`` ('' when nothing was sampled)."""
        if not s.get("n"):
            return ""
        parts = []
        if "util_max" in s:
            parts.append(f"util max {s['util_max']:.0f}%")
        if "mem_used_max_gb" in s:
            parts.append(f"VRAM used max {s['mem_used_max_gb']:.1f} GB")
        if "power_max_w" in s:
            parts.append(f"power max {s['power_max_w']:.0f} W")
        if "temp_max_c" in s:
            parts.append(f"temp max {s['temp_max_c']:.0f} C")
        if "sm_clock_max" in s:
            parts.append(f"SM clock {s['sm_clock_min']}-{s['sm_clock_max']} MHz")
        parts.append(f"throttle: {s.get('throttle', '?')}")
        return "GPU during step : " + ", ".join(parts)


# ---------------------------------------------------------------------------------------------- the worker run
@dataclass
class WorkerOutcome:
    """What came back from one worker process."""
    name: str
    result: Optional[Dict[str, Any]] = None       # the worker's own result dict (None if it never produced one)
    returncode: Optional[int] = None
    timed_out: bool = False
    cancelled: bool = False
    seconds: float = 0.0
    stderr_tail: str = ""
    gpu: Dict[str, Any] = field(default_factory=dict)
    start_error: str = ""


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, timeout=15, creationflags=_NO_WINDOW)
        else:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


def describe_exit_code(rc: Optional[int]) -> str:
    """Readable meaning of a worker exit code (Windows NTSTATUS values appear as negative/large numbers)."""
    if rc is None:
        return "not finished"
    u = rc & 0xFFFFFFFF
    known = {0xC0000005: "access violation (native crash, usually a CUDA/driver/DLL problem)",
             0xC0000409: "stack buffer overrun / fast-fail abort (native crash)",
             0xC000001D: "illegal instruction (CPU feature missing for a native library)",
             0xC0000135: "a required DLL was not found",
             0xC0000142: "DLL initialisation failed",
             0xC00000FD: "stack overflow", 0xC0000374: "heap corruption",
             0x40010004: "terminated (Ctrl+C / close)"}
    if u in known:
        return f"0x{u:08X} {known[u]}"
    if rc < 0 and os.name != "nt":
        try:
            return f"killed by signal {-rc} ({signal.Signals(-rc).name})"
        except ValueError:
            return f"killed by signal {-rc}"
    return f"exit code {rc}" + (f" (0x{u:08X})" if u > 255 else "")


def run_worker(name: str, args: Dict[str, Any], timeout: float = 600.0, env: Optional[Dict[str, str]] = None,
               on_log: Optional[Callable[[str], None]] = None, cancel: Optional[threading.Event] = None,
               sample_gpu: bool = True, command: Optional[List[str]] = None) -> WorkerOutcome:
    """Start worker ``name`` with ``args`` and wait for it (at most ``timeout`` seconds).  Never raises."""
    out = WorkerOutcome(name=name)
    t0 = time.time()
    args_file = None
    sampler: Optional[GpuSampler] = None
    try:
        fd, tmp = tempfile.mkstemp(prefix=f"vmd-{name}-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(args, fh, ensure_ascii=False)
        args_file = Path(tmp)
        full_env = dict(os.environ)
        full_env.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
                         "HF_HUB_DISABLE_SYMLINKS_WARNING": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "TRANSFORMERS_VERBOSITY": "error"})
        full_env.update(env or {})
        cmd = command or worker_command(name, args_file)
        if command is not None:
            cmd = [*command, str(args_file)] if "{args}" not in " ".join(command) else [c.replace("{args}", str(args_file)) for c in command]
        kw: Dict[str, Any] = {}
        if os.name == "nt":
            kw["creationflags"] = _NO_WINDOW | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            kw["start_new_session"] = True
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, env=full_env,
                                text=True, encoding="utf-8", errors="replace", bufsize=1, **kw)
    except Exception as exc:  # noqa: BLE001
        out.start_error = f"{type(exc).__name__}: {exc}"
        out.seconds = time.time() - t0
        if args_file:
            args_file.unlink(missing_ok=True)
        return out

    stderr_tail: Deque[str] = collections.deque(maxlen=120)

    def drain_err() -> None:
        try:
            log_file = paths.logs_dir() / f"worker-{name}.log"
            with open(log_file, "w", encoding="utf-8", errors="replace") as lf:
                for line in proc.stderr:                       # type: ignore[union-attr]
                    stderr_tail.append(line.rstrip("\n"))
                    lf.write(line)
        except Exception:  # noqa: BLE001
            for _ in proc.stderr:                              # type: ignore[union-attr]
                pass

    def read_out() -> None:
        for line in proc.stdout:                               # type: ignore[union-attr]
            if line.startswith(MARK):
                try:
                    msg = json.loads(line[len(MARK):])
                except ValueError:
                    continue
                if msg.get("t") == "result":
                    out.result = msg
                elif msg.get("t") == "log" and on_log:
                    try:
                        on_log(str(msg.get("msg", "")))
                    except Exception:  # noqa: BLE001
                        pass

    threads = [threading.Thread(target=drain_err, daemon=True), threading.Thread(target=read_out, daemon=True)]
    for t in threads:
        t.start()
    if sample_gpu:
        sampler = GpuSampler()
        sampler.start()
    deadline = t0 + timeout
    while True:
        try:
            proc.wait(timeout=0.25)
            break
        except subprocess.TimeoutExpired:
            pass
        if cancel is not None and cancel.is_set():
            out.cancelled = True
            _kill_tree(proc)
            break
        if time.time() > deadline:
            out.timed_out = True
            _kill_tree(proc)
            break
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
    for t in threads:
        t.join(timeout=5)
    out.returncode = proc.returncode
    out.seconds = time.time() - t0
    out.stderr_tail = "\n".join(stderr_tail)
    if sampler:
        sampler.stop()
        out.gpu = sampler.summary()
    if args_file:
        args_file.unlink(missing_ok=True)
    return out
