"""Refcounted manifest of suite-wide shared resources (schema 1).

One file, ``<Voxprint home>/shared/manifest.json``, is the list of runtimes, ffmpeg builds, model stores and other
heavy pieces that every Voxprint program may use. Each entry names the resource, its version, the directory it lives
in, a checksum and size when those apply, and the apps that hold a reference. Installing adds this app's reference.
Uninstalling removes it and deletes the directory only when the list is empty, and only when that directory sits
inside ``shared/``. Keys this program does not know are copied through unchanged.

The file is replaced atomically. Writers take ``shared/manifest.lock`` (an OS file lock) and, in this process, a
thread lock, so overlapping updates cannot drop a reference. Other suite apps implement this file the same way;
the field list is in ``docs/SHARED-RESOURCES.md``. The module is stdlib-only so an installer can run the file
before the package is importable.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

SCHEMA = 1
MANIFEST_NAME = "manifest.json"
LOCK_NAME = "manifest.lock"
APP_ID = "movie-dubber"
_KNOWN_TOP = {"schema", "resources"}
_KNOWN_RESOURCE = {"id", "version", "path", "sha256", "size", "apps"}
_THREAD_LOCKS: Dict[str, threading.Lock] = {}
_THREAD_GUARD = threading.Lock()


@dataclass(frozen=True)
class AddResult:
    """What :func:`add_ref` did: whether an identical resource was already on disk, and the path it uses."""

    reused: bool
    path: str


@dataclass(frozen=True)
class ReleaseResult:
    """Directories removed, and directories that still need removing after this process exits."""

    deleted: tuple[str, ...]
    pending: tuple[str, ...]


def shared_dir(home: Path) -> Path:
    """Return ``<home>/shared``, creating it when that is possible."""
    path = Path(home) / "shared"
    path.mkdir(parents=True, exist_ok=True)
    return path


def manifest_path(home: Path) -> Path:
    """Return the path of ``shared/manifest.json`` (the file need not exist yet)."""
    return shared_dir(home) / MANIFEST_NAME


def _lock_path(home: Path) -> Path:
    return shared_dir(home) / LOCK_NAME


def _thread_lock(path: Path) -> threading.Lock:
    key = os.path.normcase(str(path))
    with _THREAD_GUARD:
        lock = _THREAD_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _THREAD_LOCKS[key] = lock
        return lock


class _FileLock:
    """Exclusive OS lock. A crashed process releases it. Callers also hold :func:`_thread_lock`."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh: Any = None

    def __enter__(self) -> "_FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+b")
        try:
            if fh.seek(0, os.SEEK_END) == 0:
                fh.write(b"\0")
                fh.flush()
            if sys.platform == "win32":
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
            else:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        except BaseException:
            fh.close()
            raise
        self._fh = fh
        return self

    def __exit__(self, *exc: object) -> None:
        fh, self._fh = self._fh, None
        if fh is None:
            return
        try:
            if sys.platform == "win32":
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            fh.close()


def _locked(home: Path, fn: Callable[[], Any]) -> Any:
    path = _lock_path(home)
    with _thread_lock(path):
        with _FileLock(path):
            return fn()


def _empty() -> Dict[str, Any]:
    return {"schema": SCHEMA, "resources": []}


def read_manifest(home: Path) -> Dict[str, Any]:
    """The manifest object. A missing or unreadable file is an empty schema-1 document."""
    try:
        data = json.loads(manifest_path(home).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return _empty()
    if not isinstance(data, dict):
        return _empty()
    if not isinstance(data.get("resources"), list):
        data = dict(data)
        data["resources"] = []
    return data


def _write(home: Path, data: Dict[str, Any]) -> None:
    path = manifest_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    text = json.dumps(data, indent=1, ensure_ascii=True) + "\n"
    tmp.write_text(text, encoding="utf-8")
    try:
        for i in range(10):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                time.sleep(0.05 * (i + 1))
        raise PermissionError(f"cannot replace {path}")
    finally:
        if tmp.exists():
            tmp.unlink()


def _apps(value: object) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        if isinstance(item, str) and item and item not in out:
            out.append(item)
    return out


def _match(entry: Dict[str, Any], resource_id: str, version: str) -> bool:
    return entry.get("id") == resource_id and entry.get("version") == version


def _inside_shared(home: Path, folder: Path) -> bool:
    try:
        folder.resolve().relative_to(shared_dir(home).resolve())
    except (OSError, ValueError):
        return False
    return True


def _running_from(folder: Path) -> bool:
    try:
        prefix = Path(sys.prefix).resolve()
        root = folder.resolve()
    except OSError:
        return False
    return prefix == root or root in prefix.parents


def add_ref(home: Path, *, resource_id: str, version: str, path: Path, app: str = APP_ID,
            sha256: Optional[str] = None, size: Optional[int] = None) -> AddResult:
    """Record that ``app`` uses this resource. An entry with the same id, version and checksum is reused."""
    home = Path(home)
    folder = Path(path)

    def op() -> AddResult:
        data = read_manifest(home)
        resources = [item for item in data.get("resources", []) if isinstance(item, dict)]
        reused = False
        found: Optional[Dict[str, Any]] = None
        for entry in resources:
            if _match(entry, resource_id, version):
                found = entry
                break
        if found is None:
            found = {"id": resource_id, "version": version, "path": str(folder), "sha256": sha256, "size": size, "apps": []}
            resources.append(found)
        else:
            existing_sha = found.get("sha256")
            same_sha = not sha256 or not existing_sha or existing_sha == sha256
            reused = bool(same_sha and folder.exists())
            if sha256:
                found["sha256"] = sha256
            if size is not None:
                found["size"] = size
            if not reused:
                found["path"] = str(folder)
        apps = _apps(found.get("apps"))
        if app not in apps:
            apps.append(app)
        found["apps"] = apps
        data["schema"] = SCHEMA
        data["resources"] = resources
        _write(home, data)
        return AddResult(reused, str(found.get("path") or folder))

    return _locked(home, op)


def _drop_app(entry: Dict[str, Any], app: str) -> Dict[str, Any]:
    entry = dict(entry)
    entry["apps"] = [name for name in _apps(entry.get("apps")) if name != app]
    return entry


def _delete_tree(folder: Path) -> bool:
    if not folder.exists():
        return True
    try:
        if folder.is_dir() and not folder.is_symlink():
            shutil.rmtree(folder)
        else:
            folder.unlink()
        return not folder.exists()
    except OSError:
        return False


def release_app(home: Path, app: str = APP_ID, *, resource_id: Optional[str] = None) -> ReleaseResult:
    """Remove ``app`` from every matching resource. Delete an unreferenced directory inside ``shared/``."""
    home = Path(home)

    def op() -> ReleaseResult:
        data = read_manifest(home)
        deleted: List[str] = []
        pending: List[str] = []
        kept: List[Dict[str, Any]] = []
        for raw in data.get("resources", []):
            if not isinstance(raw, dict):
                kept.append(raw)
                continue
            entry = _drop_app(raw, app) if resource_id in (None, raw.get("id")) else dict(raw)
            apps = _apps(entry.get("apps"))
            if apps:
                entry["apps"] = apps
                kept.append(entry)
                continue
            folder = Path(str(entry.get("path") or ""))
            if not str(entry.get("path") or ""):
                continue
            if not _inside_shared(home, folder):
                continue
            if _running_from(folder):
                pending.append(str(folder))
                continue
            if _delete_tree(folder):
                deleted.append(str(folder))
            else:
                pending.append(str(folder))
        data["schema"] = SCHEMA
        data["resources"] = kept
        if kept:
            _write(home, data)
        else:
            manifest = manifest_path(home)
            manifest.unlink(missing_ok=True)
        return ReleaseResult(tuple(deleted), tuple(pending))

    return _locked(home, op)


def resource_apps(home: Path, resource_id: str, version: str) -> List[str]:
    """Apps that currently reference ``resource_id`` at ``version`` (empty when there is no entry)."""
    for entry in read_manifest(home).get("resources", []):
        if isinstance(entry, dict) and _match(entry, resource_id, version):
            return _apps(entry.get("apps"))
    return []


def iter_resources(home: Path) -> Iterable[Dict[str, Any]]:
    """Resource entries that are JSON objects. Unknown top-level keys are not yielded."""
    for entry in read_manifest(home).get("resources", []):
        if isinstance(entry, dict):
            yield entry


def known_resource_keys() -> frozenset[str]:
    """Field names this schema writes on a resource. Anything else is preserved as an unknown key."""
    return frozenset(_KNOWN_RESOURCE)


def known_top_keys() -> frozenset[str]:
    """Field names this schema writes on the document. Anything else is preserved as an unknown key."""
    return frozenset(_KNOWN_TOP)


def _cli(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="shared_manifest", description="Update shared/manifest.json.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    add = sub.add_parser("add", help="Add an app reference to one resource.")
    add.add_argument("--home", required=True)
    add.add_argument("--id", required=True)
    add.add_argument("--version", required=True)
    add.add_argument("--path", required=True)
    add.add_argument("--app", default=APP_ID)
    add.add_argument("--sha256", default="")
    add.add_argument("--size", type=int, default=-1)
    release = sub.add_parser("release", help="Drop an app reference and delete unreferenced shared directories.")
    release.add_argument("--home", required=True)
    release.add_argument("--app", default=APP_ID)
    release.add_argument("--id", default="")
    args = parser.parse_args(argv)
    home = Path(args.home)
    if args.cmd == "add":
        added = add_ref(home, resource_id=args.id, version=args.version, path=Path(args.path), app=args.app,
                        sha256=args.sha256 or None, size=None if args.size < 0 else args.size)
        print(json.dumps({"reused": added.reused, "path": added.path}), flush=True)
        return 0
    if args.cmd == "release":
        released = release_app(home, args.app, resource_id=args.id or None)
        print(json.dumps({"deleted": list(released.deleted), "pending": list(released.pending)}), flush=True)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(_cli())
