"""Shared model store: locate a model that VOXPrint MDL has already fetched.

Downloads belong to ``voxprint-mdl``. This module does not fetch, verify-by-redownload,
or call ensure_ready. That helper downloads inside the process, so the app never calls it.

* Layout: ``<models>/<owner>--<name>/`` holds the files an engine opens. MDL stores
  the bytes at ``<models>/blobs/<sha256>``. :func:`link_blobs` hardlinks (then
  symlinks, then copies) those blobs into the named folder.
* Manifest: ``mdl.json`` at the repository root. :func:`load_manifest` turns it
  back into ``{repo: {revision, license, files}}``.
* Users: ``<models>/.users.json``. The installer adds ``movie-dubber``. Unknown keys and
  other programs' entries stay. A folder is deleted only when this app recorded it and
  no other program still uses the store. The models directory itself is not a manifest ref.
* :class:`ModelLock` is the voice-library lock. It is not a model download.
"""
from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None  # type: ignore[assignment]
try:
    import msvcrt
except ImportError:  # Linux and macOS
    msvcrt = None  # type: ignore[assignment]

from dubber import paths
from dubber.infra import model_guard
from voxprint_common.hashing import sha256_file as file_digest

USER_KEY = "movie-dubber"
USERS_FILE = ".users.json"
MANIFEST_PATH = Path(__file__).resolve().parents[2] / "mdl.json"
DEFAULT_WEIGHTS = ("*.safetensors", "*.bin")
CONFIG_NAMES = ("config.json", "config.yaml", "params.json")


class ModelUnavailable(RuntimeError):
    """The model is not on disk. VOXPrint MDL is what fetches it."""


def models_root() -> Path:
    """Return the shared models folder, creating the configured directory when that is possible."""
    return paths.models_dir()


def folder_name(repo: str) -> str:
    """Turn ``owner/name`` into the on-disk folder name ``owner--name``."""
    return repo.replace("/", "--")


def local_dir_for(repo: str, root: Optional[Path] = None, weights: Sequence[str] = DEFAULT_WEIGHTS,
                  require_config: bool = True) -> Path:
    """``Qwen/X`` -> ``<models>/Qwen--X``; a model complete only in the default folder is used from there."""
    name = folder_name(repo)
    if root is not None:
        return Path(root) / name
    target = models_root() / name
    if not verify_structure(target, weights, require_config):
        default = paths.default_models_dir() / name
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


def load_manifest(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load pinned models as ``{repo: {revision, license, files}}``.

    ``mdl.json`` (schema 1) is the file on disk. A document whose ``models`` value is
    already that dict is returned unchanged. An unreadable file or another schema
    yields an empty dict.
    """
    try:
        data = json.loads(Path(path or MANIFEST_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or data.get("schema") != 1:
        return {}
    models = data.get("models", {})
    if isinstance(models, dict):
        return models
    if not isinstance(models, list):
        return {}
    out: Dict[str, Any] = {}
    for row in models:
        if not isinstance(row, dict):
            continue
        repo = row.get("repo")
        if not isinstance(repo, str) or not repo:
            continue
        files: Dict[str, Dict[str, Any]] = {}
        for item in row.get("files") or []:
            if not isinstance(item, dict):
                continue
            name = item.get("path")
            if not isinstance(name, str) or not name:
                continue
            files[name] = {"size": int(item["size"]), "sha256": str(item["sha256"])}
        out[repo] = {"revision": row.get("version") or row.get("revision"), "license": row.get("license"), "files": files}
    return out


def pinned_revision(repo: str) -> Optional[str]:
    """Return the pinned revision for ``repo``, or None when the manifest has no entry."""
    revision = (load_manifest().get(repo) or {}).get("revision")
    return str(revision) if revision else None


def sha256_file(path: Path, bufsize: int = 1 << 20) -> str:
    """Return the hex SHA-256 of ``path``. ``bufsize`` is accepted and ignored; the shared hasher streams the file."""
    del bufsize
    return file_digest(path)


def _checked(name: str) -> bool:
    return not name.rsplit("/", 1)[-1].startswith(".") and not name.lower().endswith((".md",))


def manifest_bad_files(folder: Path, repo: str, patterns: Optional[Iterable[str]] = None, manifest: Optional[Dict[str, Any]] = None,
                       full_hash: bool = True) -> Optional[List[str]]:
    """Missing or wrong files of ``folder`` against the manifest (size first, then SHA-256). None means no entry."""
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
    """``.verified`` records that the hashes were checked, so later starts only compare sizes."""
    try:
        (Path(folder) / ".verified").write_text(json.dumps({"repo": repo, "revision": pinned_revision(repo), "at": time.time()}),
                                                encoding="utf-8")
    except OSError:
        pass


def is_ready(folder: Path, repo: str, weights: Sequence[str] = DEFAULT_WEIGHTS, patterns: Optional[Iterable[str]] = None,
             require_config: bool = True) -> bool:
    """Return whether the folder has the expected weights and matches the manifest.

    Files are hashed only when ``.verified`` is absent. A blob MDL stored under its
    SHA-256 name is treated as already checked once :func:`link_blobs` writes the marker.
    """
    if not verify_structure(folder, weights, require_config):
        return False
    bad = manifest_bad_files(folder, repo, patterns, full_hash=not (folder / ".verified").exists())
    return not bad


def folder_bytes(path: Path) -> int:
    """Return the total size in bytes of the files under ``path``."""
    total = 0
    for base, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(base, name))
            except OSError:
                pass
    return total


def _blob_roots(root: Path) -> List[Path]:
    found: List[Path] = []
    for base in (root, paths.default_models_dir()):
        blob = base / "blobs"
        if blob not in found:
            found.append(blob)
    return found


def _place(src: Path, dest: Path) -> None:
    """Put ``src`` at ``dest``: hardlink, then symlink, then copy."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dest)
        return
    except OSError:
        pass
    try:
        os.symlink(src, dest)
        return
    except OSError:
        pass
    shutil.copy2(src, dest)


def link_blobs(repo: str, weights: Sequence[str] = DEFAULT_WEIGHTS, patterns: Optional[Iterable[str]] = None,
               require_config: bool = True, root: Optional[Path] = None) -> Optional[Path]:
    """Link MDL blobs for ``repo`` into ``<models>/owner--name`` when every file is present.

    A size match against the manifest is enough. The blob file name is the SHA-256,
    which MDL checked when it stored the file. Nothing is downloaded.
    """
    entry = load_manifest().get(repo)
    if not entry:
        return None
    target_root = Path(root) if root is not None else models_root()
    target = target_root / folder_name(repo)
    pats = list(patterns or [])
    needed: List[Tuple[str, int, str]] = []
    for name, meta in entry.get("files", {}).items():
        if not _checked(name) or (pats and not any(fnmatch.fnmatch(name, p) for p in pats)):
            continue
        digest = str(meta.get("sha256") or "")
        if ".." in name.split("/") or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
            return None
        needed.append((name, int(meta["size"]), digest))
    if not needed:
        return None
    sources: List[Tuple[str, Path]] = []
    for name, size, digest in needed:
        found: Optional[Path] = None
        for blob_root in _blob_roots(target_root):
            candidate = blob_root / digest
            try:
                if candidate.is_file() and candidate.stat().st_size == size:
                    found = candidate
                    break
            except OSError:
                continue
        if found is None:
            return None
        sources.append((name, found))
    for name, src in sources:
        dest = target / Path(*name.split("/"))
        try:
            if dest.is_file() and dest.stat().st_size == src.stat().st_size:
                continue
            if dest.exists():
                return None
        except OSError:
            return None
        _place(src, dest)
    write_verified_marker(target, repo)
    if is_ready(target, repo, weights, patterns, require_config):
        return target
    return None


# ---------------------------------------------------------------------------------------------- voice-library lock
class ModelLock:
    """Cross-process lock (an OS file lock). The voice library uses it; model downloads do not."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._fh: Any = None

    def try_acquire(self) -> bool:
        """Take the OS file lock without waiting, or return False when another holder still has it."""
        for _ in range(5):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                fh = open(self.path, "a+b")
            except OSError:
                continue
            try:
                if sys.platform == "win32":
                    if msvcrt is None:
                        fh.close()
                        return False
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    if fcntl is None:
                        fh.close()
                        return False
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    st = os.fstat(fh.fileno())
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
        """Return whether this instance currently holds the file lock."""
        return self._fh is not None

    def release(self, remove: bool = False) -> None:
        """Unlock the file, and delete it when ``remove`` is true."""
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
                if msvcrt is not None:
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            elif fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            fh.close()
        if remove and os.name == "nt":
            try:
                self.path.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------------------------- users
def _users_path(root: Optional[Path] = None) -> Path:
    return (Path(root) if root else models_root()) / USERS_FILE


def read_users(root: Optional[Path] = None) -> Dict[str, bool]:
    """Read ``.users.json`` as a map of program name to in-use flag, or return an empty dict."""
    try:
        return model_guard.user_flags(model_guard.load_users_document(_users_path(root)))
    except OSError:
        return {}


def register_user(key: str = USER_KEY, root: Optional[Path] = None) -> Dict[str, bool]:
    """Mark ``key`` as using the models folder and return the updated program map.

    The write keeps unknown keys and other programs. A file that cannot be parsed is left unchanged.

    Raises:
        OSError: ``.users.json`` exists and is not valid JSON.
    """
    base = Path(root) if root is not None else models_root()
    document = model_guard.update_users(base, lambda current: model_guard.with_user(current, key))
    return model_guard.user_flags(document)


def unregister_user(key: str = USER_KEY, root: Optional[Path] = None) -> List[str]:
    """Remove ``key`` and return the other programs still using the models folder.

    Raises:
        OSError: ``.users.json`` exists and is not valid JSON.
    """
    base = Path(root) if root is not None else models_root()
    document = model_guard.update_users(base, lambda current: model_guard.without_user(current, key))
    return [name for name in model_guard.active_users(document) if name != key]
