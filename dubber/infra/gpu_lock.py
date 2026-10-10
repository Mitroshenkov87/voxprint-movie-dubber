"""GPU lock shared by the Voxprint programs (spec agreed with Voxprint AI Audiobook Builder).

Before a GPU job a program creates ``%TEMP%\\voxprint-gpu.lock`` (``$TMPDIR`` elsewhere) with
``{"owner": "movie-dubber", "job": "...", "started": ISO8601, "eta": ISO8601}`` and deletes it when the job ends.
If the file exists and belongs to another program, we wait.  It counts as stale only when its ``eta`` passed more than two hours
ago AND nvidia-smi shows no python / Voxprint process using GPU memory (or the file is unreadable garbage older than two hours).
A lock with our own owner name left by a crashed run of this program is taken over.

Used by the diagnostics and by the dubbing pipeline (:func:`gpu_job`).
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

OWNER = "movie-dubber"
LOCK_NAME = "voxprint-gpu.lock"
STALE_AFTER = timedelta(hours=2)
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def lock_file() -> Path:
    """Return the shared GPU lock path: ``VOXPRINT_GPU_LOCK``, or ``voxprint-gpu.lock`` in the temp directory."""
    override = os.environ.get("VOXPRINT_GPU_LOCK")
    return Path(override) if override else Path(tempfile.gettempdir()) / LOCK_NAME


def _now() -> datetime:
    return datetime.now(timezone.utc).astimezone()


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _parse(s: Any) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def read_lock(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Contents of the lock file; ``{}`` for an unreadable one; None when there is no lock."""
    p = path or lock_file()
    try:
        text = p.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return None
    except OSError:
        return {}
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except ValueError:
        return {}


def gpu_processes() -> List[str]:
    """Names of processes holding GPU memory (nvidia-smi); [] when nvidia-smi is absent."""
    from dubber.diag.procs import find_nvidia_smi

    smi = find_nvidia_smi()
    if not smi:
        return []
    try:
        p = subprocess.run([smi, "--query-compute-apps=process_name,used_memory", "--format=csv,noheader"], capture_output=True,
                           text=True, timeout=15, creationflags=_NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return []
    return [ln.split(",")[0].strip() for ln in (p.stdout or "").splitlines() if ln.strip()]


def voxprint_gpu_busy(procs: Optional[List[str]] = None) -> bool:
    """Return whether a Python or Voxprint process is holding GPU memory."""
    procs = gpu_processes() if procs is None else procs
    return any(("python" in n.lower() or "voxprint" in n.lower()) for n in procs)


def is_stale(data: Dict[str, Any], path: Path, now: Optional[datetime] = None, busy: Optional[Callable[[], bool]] = None) -> bool:
    """Return whether the lock is older than two hours and no Voxprint process is using the GPU."""
    now = now or _now()
    eta = _parse(data.get("eta")) or _parse(data.get("started"))
    if eta is None:                                           # garbage: use the file age
        try:
            eta = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        except OSError:
            return True
    if now - eta <= STALE_AFTER:
        return False
    return not (busy or voxprint_gpu_busy)()


class GpuLockTimeout(TimeoutError):
    """The wait for the shared GPU lock was cancelled or exceeded its timeout."""
    pass


def try_acquire(job: str, eta_s: float, path: Optional[Path] = None, owner: str = OWNER,
                busy: Optional[Callable[[], bool]] = None) -> Optional[Dict[str, Any]]:
    """Create the lock; returns None on success, else the holder's data."""
    p = path or lock_file()
    current = read_lock(p)
    if current is not None:
        pid = current.get("pid")
        mine = current.get("owner") == owner and (pid == os.getpid() or not _pid_alive(pid))
        if not mine and not is_stale(current, p, busy=busy):
            return current
        try:
            p.unlink()
        except FileNotFoundError:
            pass
        except PermissionError:
            return current
    now = _now()
    data = {"owner": owner, "job": job, "started": _iso(now), "eta": _iso(now + timedelta(seconds=max(60.0, eta_s))), "pid": os.getpid()}
    tmp = p.with_name(f"{p.name}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    try:
        if os.name == "nt":
            os.rename(tmp, p)                                  # fails if someone created it in between
        else:
            os.link(tmp, p)                                    # atomic "create if absent"
            tmp.unlink()
    except (FileExistsError, PermissionError):
        tmp.unlink(missing_ok=True)
        return read_lock(p) or {"owner": "?"}
    return None


def _pid_alive(pid: Any) -> bool:
    try:
        import psutil

        return bool(pid) and psutil.pid_exists(int(pid))      # no pid recorded -> treated as not alive
    except Exception:  # noqa: BLE001
        return True


def release(path: Optional[Path] = None, owner: str = OWNER) -> None:
    """Delete the lock file when this process owns it."""
    p = path or lock_file()
    cur = read_lock(p)
    if cur is not None and cur.get("owner") == owner and cur.get("pid") in (None, os.getpid()):
        try:
            p.unlink()
        except OSError:
            pass


@contextmanager
def gpu_job(job: str, eta_s: float = 600.0, on_wait: Callable[[Dict[str, Any], float], None] = lambda d, s: None,
            timeout: Optional[float] = None, poll: float = 5.0, path: Optional[Path] = None,
            cancel: Optional[Callable[[], bool]] = None) -> Iterator[None]:
    """Hold the shared GPU lock for the duration of the block (waits while another Voxprint program holds it).

    Re-entrant: when this process already holds the lock (an outer ``gpu_job`` around a whole run), the inner block neither
    re-creates nor releases it, so the GPU is not left unlocked between the stages of one run."""
    cur = read_lock(path or lock_file())
    if cur and cur.get("owner") == OWNER and cur.get("pid") == os.getpid():
        yield
        return
    t0 = time.monotonic()
    while True:
        holder = try_acquire(job, eta_s, path)
        if holder is None:
            break
        waited = time.monotonic() - t0
        if cancel and cancel():
            raise GpuLockTimeout("cancelled while waiting for the GPU")
        if timeout is not None and waited > timeout:
            raise GpuLockTimeout(f"the GPU is busy: {holder.get('owner', '?')} / {holder.get('job', '?')}")
        on_wait(holder, waited)
        time.sleep(poll)
    try:
        yield
    finally:
        release(path)
