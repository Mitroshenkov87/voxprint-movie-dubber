"""Shared model store: one plain folder per model, a per-model OS lock while downloading, SHA-256 manifests, a users file.

Spec agreed with Voxprint AI Audiobook Builder (its ``infra/model_downloader.py`` ``ModelLock`` / ``sweep_stale_locks`` are copied
here); change locations only in :mod:`dubber.infra.shared_paths`.

* Layout: ``<models>/<owner>--<name>/`` - plain files, not the Hugging Face cache.  A download goes into ``<name>.partial`` (kept
  between runs, so the next run resumes) and is renamed after the check; a half-downloaded model is never used.
* Lock: empty file ``<models>/.<owner>--<name>.lock``, ``msvcrt.locking(LK_NBLCK, 1 byte)`` / ``fcntl.flock(LOCK_EX|LOCK_NB)``,
  held for the whole download and removed after it.  :func:`sweep_stale_locks` deletes unheld leftovers.
* Manifest: ``model_manifest.json`` next to this file: pinned revision + size + SHA-256 per file.  A model without a manifest entry
  is checked structurally (weights, and a config file unless the model is checkpoint-only).
* Users: ``<models>/.users.json`` ``{"audiobook-builder": true, "movie-dubber": true}`` - the installer adds our key, the uninstaller
  removes it and deletes models only when no other key remains AND the user agrees.
* Watchdog: while a download runs, the size of the ``.partial`` folder is reported every few seconds; no progress for
  ``STALL_SECONDS`` gives up the attempt (the data stays and the next attempt resumes).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from dubber.infra import shared_paths

log = logging.getLogger("dubber.models")

USER_KEY = "movie-dubber"
USERS_FILE = ".users.json"
MANIFEST_PATH = Path(__file__).with_name("model_manifest.json")
STALL_SECONDS = 180.0
WATCH_INTERVAL = 5.0
DEFAULT_WEIGHTS = ("*.safetensors", "*.bin")
CONFIG_NAMES = ("config.json", "config.yaml", "params.json")


class ModelUnavailable(RuntimeError):
    """The model is missing and cannot be fetched (downloads off, gated without token, network error, stalled)."""


def models_root() -> Path:
    return shared_paths.models_dir()


def folder_name(repo: str) -> str:
    return repo.replace("/", "--")


def local_dir_for(repo: str, root: Optional[Path] = None, weights: Sequence[str] = DEFAULT_WEIGHTS,
                  require_config: bool = True) -> Path:
    """``Qwen/X`` -> ``<models>/Qwen--X``; with a user-chosen folder a model complete only in the default folder is used there."""
    name = folder_name(repo)
    if root is not None:
        return Path(root) / name
    target = models_root() / name
    if not verify_structure(target, weights, require_config):
        default = shared_paths.default_models_dir() / name
        if default != target and verify_structure(default, weights, require_config):
            return default
    return target


def verify_structure(path: Path, weights: Sequence[str] = DEFAULT_WEIGHTS, require_config: bool = True) -> bool:
    """At least one weights file, and a config file unless the model is a checkpoint with no config."""
    if not path.is_dir():
        return False
    if not any(any(path.glob(w)) for w in weights):
        return False
    if not require_config:
        return True
    return any((path / n).exists() for n in CONFIG_NAMES)


# ---------------------------------------------------------------------------------------------- manifest (size + SHA-256)
def load_manifest(path: Optional[Path] = None) -> Dict[str, Any]:
    try:
        data = json.loads(Path(path or MANIFEST_PATH).read_text(encoding="utf-8"))
        return data.get("models", {}) if isinstance(data, dict) and data.get("schema") == 1 else {}
    except (OSError, ValueError):
        return {}


def pinned_revision(repo: str) -> Optional[str]:
    return (load_manifest().get(repo) or {}).get("revision")


def sha256_file(path: Path, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(bufsize), b""):
            h.update(block)
    return h.hexdigest()


def _checked(name: str) -> bool:
    return not name.rsplit("/", 1)[-1].startswith(".") and not name.lower().endswith((".md",))


def manifest_bad_files(folder: Path, repo: str, patterns: Optional[Iterable[str]] = None, manifest: Optional[Dict[str, Any]] = None,
                       full_hash: bool = True) -> Optional[List[str]]:
    """Missing / wrong files of ``folder`` against the manifest (size first, then SHA-256); None = no manifest entry."""
    import fnmatch

    entry = (manifest if manifest is not None else load_manifest()).get(repo)
    if not entry:
        return None
    pats = list(patterns or [])
    bad = []
    for name, meta in entry.get("files", {}).items():
        if not _checked(name) or (pats and not any(fnmatch.fnmatch(name, p) for p in pats)):
            continue
        p = Path(folder) / Path(*name.split("/"))
        try:
            if not p.is_file() or p.stat().st_size != int(meta["size"]):
                bad.append(name)
            elif full_hash and sha256_file(p) != meta["sha256"]:
                bad.append(name)
        except OSError:
            bad.append(name)
    return bad


def write_verified_marker(folder: Path, repo: str) -> None:
    """``.verified`` = the hashes were checked once (later starts only compare sizes, hashing 4 GB on every start is slow)."""
    try:
        (Path(folder) / ".verified").write_text(json.dumps({"repo": repo, "revision": pinned_revision(repo), "at": time.time()}),
                                                encoding="utf-8")
    except OSError:
        pass


def is_ready(folder: Path, repo: str, weights: Sequence[str] = DEFAULT_WEIGHTS, patterns: Optional[Iterable[str]] = None,
             require_config: bool = True) -> bool:
    if not verify_structure(folder, weights, require_config):
        return False
    bad = manifest_bad_files(folder, repo, patterns, full_hash=not (folder / ".verified").exists())
    return not bad


# ---------------------------------------------------------------------------------------------- one download per model (copied)
class ModelLock:
    """Cross-process lock for one model folder (an OS file lock: it disappears with a crashed process, so it is never stale)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._fh: Any = None

    def try_acquire(self) -> bool:
        for _ in range(5):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                fh = open(self.path, "a+b")
            except OSError:
                # Windows: a lock file held by another thread of this process raises PermissionError on open - it is taken.
                continue
            try:
                if sys.platform == "win32":
                    import msvcrt

                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    st = os.fstat(fh.fileno())        # the previous holder may have unlinked the file right after release
                    try:
                        cur = os.stat(self.path)
                        same = (cur.st_ino, cur.st_dev) == (st.st_ino, st.st_dev)
                    except OSError:
                        same = False
                    if not same:
                        fh.close()
                        continue
            except OSError:
                fh.close()
                return False
            self._fh = fh
            return True
        return False

    @property
    def held(self) -> bool:
        return self._fh is not None

    def release(self, remove: bool = False) -> None:
        fh, self._fh = self._fh, None
        if fh is None:
            return
        if remove and os.name != "nt":
            try:
                self.path.unlink()
            except OSError:
                pass
        try:
            if sys.platform == "win32":
                import msvcrt

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            fh.close()
        if remove and os.name == "nt":
            try:
                self.path.unlink()
            except OSError:                    # PermissionError: another process opened it meanwhile (it waits for the model)
                pass

    def acquire(self, on_wait: Callable[[float], None] = lambda s: None, poll: float = 0.5, timeout: Optional[float] = None) -> None:
        t0 = time.monotonic()
        while not self.try_acquire():
            waited = time.monotonic() - t0
            if timeout is not None and waited > timeout:
                raise TimeoutError(f"{self.path.name} is held by another process")
            on_wait(waited)
            time.sleep(poll)


def lock_path(repo: str, root: Optional[Path] = None) -> Path:
    return (Path(root) if root else models_root()) / f".{folder_name(repo)}.lock"


def sweep_stale_locks(root: Optional[Path] = None) -> int:
    """Delete the empty ``.<model>.lock`` files no process holds (a crash leftover).  Held locks and non-empty files are kept."""
    folder = Path(root) if root is not None else models_root()
    try:
        candidates = [p for p in folder.glob(".*.lock") if p.is_file()]
    except OSError:
        return 0
    removed = 0
    for p in candidates:
        try:
            if p.stat().st_size:
                continue
        except OSError:
            continue
        lock = ModelLock(p)
        try:
            if not lock.try_acquire():
                continue
        except PermissionError:
            continue
        lock.release(remove=True)
        if not p.exists():
            removed += 1
    return removed


# ---------------------------------------------------------------------------------------------- users (reference count)
def _users_path(root: Optional[Path] = None) -> Path:
    return (Path(root) if root else models_root()) / USERS_FILE


def read_users(root: Optional[Path] = None) -> Dict[str, bool]:
    try:
        data = json.loads(_users_path(root).read_text(encoding="utf-8-sig"))
        return {str(k): bool(v) for k, v in data.items()} if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_users(users: Dict[str, bool], root: Optional[Path] = None) -> None:
    path = _users_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps(users, indent=1, sort_keys=True), encoding="utf-8")
    for i in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:               # Windows: the other program is reading it right now
            time.sleep(0.1 * (i + 1))
    tmp.unlink(missing_ok=True)


def register_user(key: str = USER_KEY, root: Optional[Path] = None) -> Dict[str, bool]:
    users = read_users(root)
    users[key] = True
    _write_users(users, root)
    return users


def unregister_user(key: str = USER_KEY, root: Optional[Path] = None) -> List[str]:
    """Remove ``key``; returns the other programs still using the models folder (empty = models may be deleted, if the user agrees)."""
    users = read_users(root)
    users.pop(key, None)
    _write_users(users, root)
    return sorted(k for k, v in users.items() if v)


# ---------------------------------------------------------------------------------------------- download
def _dir_size(p: Path) -> int:
    total = 0
    for base, _dirs, files in os.walk(p):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(base, f))
            except OSError:
                pass
    return total


@dataclass
class EnsureInfo:
    path: Path
    source: str                 # "present" | "downloaded"
    download_s: float = 0.0


Downloader = Callable[[str, Path, Optional[str], Optional[List[str]], Optional[str]], None]


def hf_download(repo: str, dest: Path, revision: Optional[str], patterns: Optional[List[str]], token: Optional[str]) -> None:
    """Plain files into ``dest`` (``huggingface_hub`` resumes partial files that are already there)."""
    from huggingface_hub import snapshot_download

    kw: Dict[str, Any] = dict(repo_id=repo, local_dir=str(dest), token=token)
    if revision:
        kw["revision"] = revision
    if patterns:
        kw["allow_patterns"] = list(patterns)
    snapshot_download(**kw)


def _watched(func: Callable[[], None], part: Path, size_gb: float, log_fn: Callable[[str], None], repo: str,
             stall_s: float) -> None:
    """Run ``func`` in a thread; report progress; give up when the folder does not grow for ``stall_s`` seconds."""
    err: List[BaseException] = []
    done = threading.Event()

    def target() -> None:
        try:
            func()
        except BaseException as exc:  # noqa: BLE001
            err.append(exc)
        finally:
            done.set()

    threading.Thread(target=target, daemon=True, name=f"download-{repo}").start()
    last_size, last_change = -1, time.monotonic()
    while not done.wait(WATCH_INTERVAL):
        size = _dir_size(part)
        if size != last_size:
            last_size, last_change = size, time.monotonic()
        log_fn(f"downloading {repo}: {size / 1024 ** 3:.2f}" + (f" / ~{size_gb:.1f}" if size_gb else "") + " GB")
        if time.monotonic() - last_change > stall_s:
            raise ModelUnavailable(f"download of {repo} stalled (no data for {stall_s:.0f} s); run it again to resume")
    if err:
        raise err[0]


def ensure_model(repo: str, *, allow_download: bool = True, token: Optional[str] = None, patterns: Optional[Sequence[str]] = None,
                 weights: Sequence[str] = DEFAULT_WEIGHTS, gated: bool = False, size_gb: float = 0.0,
                 log_fn: Callable[[str], None] = lambda m: None, downloader: Optional[Downloader] = None,
                 root: Optional[Path] = None, stall_s: float = STALL_SECONDS, require_config: bool = True) -> EnsureInfo:
    """Return the folder of a complete model, downloading it (under the per-model lock) when allowed."""
    target = local_dir_for(repo, root, weights, require_config)
    if is_ready(target, repo, weights, patterns, require_config):
        return EnsureInfo(target, "present")
    if not allow_download:
        raise ModelUnavailable(f"{repo} is not on this computer and downloading is switched off")
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or None
    if gated and not token:
        raise ModelUnavailable(f"{repo} is a gated model: accept its terms on huggingface.co and enter a Hugging Face token in Settings")
    target = (Path(root) if root else models_root()) / folder_name(repo)
    lock = ModelLock(lock_path(repo, target.parent))
    lock.acquire(lambda s: log_fn(f"waiting for another program downloading {repo} ({s:.0f} s)") if int(s) % 10 == 0 else None)
    done = False
    try:
        if is_ready(target, repo, weights, patterns, require_config):            # the other process finished it meanwhile
            done = True
            return EnsureInfo(target, "present")
        part = target.with_name(target.name + ".partial")
        part.mkdir(parents=True, exist_ok=True)
        dl = downloader or hf_download
        t0 = time.time()
        try:
            _watched(lambda: dl(repo, part, pinned_revision(repo), list(patterns) if patterns else None, token), part, size_gb,
                     log_fn, repo, stall_s)
        except ModelUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ModelUnavailable(f"download of {repo} failed: {type(exc).__name__}: {' '.join(str(exc).split())[:200]}") from exc
        log_fn(f"verifying {repo} (checksums)")
        if not verify_structure(part, weights, require_config):
            raise ModelUnavailable(f"{repo}: the download finished but the folder looks incomplete ({part})")
        bad = manifest_bad_files(part, repo, patterns)
        if bad:
            for name in bad:                                     # broken files go; the next run downloads them again
                (part / name).unlink(missing_ok=True)
            raise ModelUnavailable(f"{repo}: checksum mismatch in {', '.join(bad[:4])}; run again to re-download them")
        if bad is not None:
            write_verified_marker(part, repo)
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        for i in range(10):
            try:
                os.replace(part, target)
                break
            except PermissionError:
                time.sleep(0.5 * (i + 1))
        else:
            raise ModelUnavailable(f"{repo}: could not rename {part.name} (a file is open in another program)")
        done = True
        return EnsureInfo(target, "downloaded", round(time.time() - t0, 1))
    finally:
        lock.release(remove=True)
        if not done:
            log.info("model %s not finished; the .partial folder is kept for resuming", repo)
