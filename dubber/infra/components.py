"""Movie Dubber's references in the shared runtime manifest.

The file format, the lock (``manifest.lock``, 30 seconds), the atomic publish
and the Canvas / Enlarge migration live in ``voxprint_common.runtime_manifest``.
This module only chooses component ids, turns this app's older list-shaped
manifest into the object that library reads, and deletes a directory when no
component still points at it. The shared models directory is not removed while
another program's weights are inside it. The app id stored in ``refs`` is ``movie-dubber``.

A path must stay inside ``<home>/shared``, except a ``model`` component, which
may be an absolute path under ``voxprint_common.paths.models_dir``
(``<home>/models`` by default) or ``VOXPRINT_MODEL_CACHE``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

# Linux runs ``python <app>/dubber/infra/components.py``. The app root has to be
# importable before ``dubber`` loads. The Windows setup copies this file alone
# into a temp directory and only calls ``add``; ``release`` goes through main.py.
_script = Path(__file__).resolve()
if len(_script.parents) >= 3:
    _candidate = _script.parents[2]
    if (_candidate / "dubber" / "__init__.py").is_file():
        _root_text = str(_candidate)
        if _root_text not in sys.path:
            sys.path.insert(0, _root_text)

try:
    from dubber.infra import model_guard as _model_guard
except ImportError:
    model_guard: ModuleType | None = None
else:
    model_guard = _model_guard

from voxprint_common.atomic import LockTimeout, exclusive_lock, write_json
from voxprint_common.hashing import sha256_bytes, sha256_file
from voxprint_common.paths import models_dir
from voxprint_common.runtime_manifest import (
    ManifestError,
    add_ref,
    cpu_onnxruntime_conflicts,
    load,
    lock_path,
    remove_ref,
)

APP_ID = "movie-dubber"
CPU_ONNX_MESSAGE = (
    "The CPU package onnxruntime is installed. The shared runtime must not contain it, "
    "because it shadows onnxruntime-gpu. Remove that package and install onnxruntime-gpu instead."
)

_LEGACY_KIND = {
    "runtime": "runtime",
    "torch": "wheels",
    "ctranslate2": "wheels",
    "ffmpeg": "tool",
    "models": "model",
}


class OutsideShared(ValueError):
    """The directory is not an allowed component path.

    Runtime, wheel and tool paths stay inside ``<home>/shared``. A model path
    stays under the shared models directory or ``VOXPRINT_MODEL_CACHE``.
    """


@dataclass(frozen=True)
class ReleaseResult:
    """Directories removed, and directories the uninstaller still has to remove."""

    deleted: tuple[str, ...]
    pending: tuple[str, ...]


def manifest_file(home: Path) -> Path:
    """Return ``<home>/shared/manifest.json``. The file need not exist yet."""
    return Path(home) / "shared" / "manifest.json"


def shared_root(home: Path) -> Path:
    """Return ``<home>/shared``."""
    return Path(home) / "shared"


def component_id_for(piece_id: str, version: str) -> tuple[str, str]:
    """Return ``(component id, kind)`` for a self-test piece id.

    ``python-3.14`` is the user CUDA runtime. The CI CPU runtime uses
    ``python-3.14-cpu`` so the two copies sit side by side.
    """
    text = version.strip()
    if piece_id == "runtime":
        if text.endswith("cpu"):
            return "python-3.14-cpu", "runtime"
        return "python-3.14", "runtime"
    if piece_id == "torch":
        flavor = "cpu" if text.endswith("+cpu") or text.endswith("-cpu") else "cu130"
        return f"torch-2.11.0-{flavor}", "wheels"
    if piece_id == "ctranslate2":
        return "ctranslate2-cu12", "wheels"
    if piece_id == "ffmpeg":
        return "ffmpeg-8", "tool"
    if piece_id == "models":
        return "models", "model"
    return piece_id, _LEGACY_KIND.get(piece_id, "tool")


def digest_for(path: Path, fallback: str) -> str:
    """Hash a file, or ``runtime-key.json`` inside a directory, or the fallback text."""
    if path.is_file():
        return sha256_file(path)
    lock = path / "runtime-key.json"
    if lock.is_file():
        return sha256_file(lock)
    marker = path / "ffmpeg.exe" if sys.platform == "win32" else path / "ffmpeg"
    if marker.is_file():
        return sha256_file(marker)
    return sha256_bytes(fallback.encode("utf-8"))


def relative_path(home: Path, path: Path) -> str:
    """Return ``path`` relative to ``<home>/shared``.

    Raises:
        OutsideShared: ``path`` is not inside that directory.
    """
    shared = shared_root(home).resolve()
    try:
        target = Path(path).resolve()
        rel = target.relative_to(shared)
    except (OSError, ValueError) as exc:
        raise OutsideShared(f"{path} is outside {shared}") from exc
    if ".." in rel.parts:
        raise OutsideShared(f"{path} is outside {shared}")
    return rel.as_posix()


def _model_roots() -> tuple[Path, ...]:
    """Directories a model component may live in."""
    roots = [models_dir()]
    cache = os.environ.get("VOXPRINT_MODEL_CACHE", "").strip()
    if cache:
        roots.append(Path(os.path.expandvars(os.path.expanduser(cache))))
    return tuple(roots)


def _inside(folder: Path, root: Path) -> bool:
    try:
        base = root.resolve()
        target = folder.resolve()
    except OSError:
        return False
    return target == base or base in target.parents


def _absolute_model_path(path: Path) -> str:
    """Return an absolute model path that stays under a models root.

    Raises:
        OutsideShared: ``path`` is not absolute, or it is outside every models root.
    """
    target = Path(path)
    if not target.is_absolute():
        raise OutsideShared(f"{path} is outside the shared models directory")
    try:
        folder = target.resolve()
    except OSError as exc:
        raise OutsideShared(f"{path} is outside the shared models directory") from exc
    if not any(_inside(folder, root) for root in _model_roots()):
        raise OutsideShared(f"{path} is outside the shared models directory")
    return folder.as_posix()


def component_path(home: Path, path: Path, kind: str) -> str:
    """Return the path stored for one component.

    A model is an absolute path under the shared models directory. Every other
    kind is relative to ``<home>/shared``.

    Raises:
        OutsideShared: ``path`` is not in the directory that kind allows.
    """
    if kind == "model":
        return _absolute_model_path(path)
    return relative_path(home, path)


def running_from(path: Path) -> bool:
    """True when this process's prefix is ``path`` or a directory inside it."""
    try:
        here = Path(sys.prefix).resolve()
        folder = Path(path).resolve()
    except OSError:
        return False
    return here == folder or folder in here.parents


def guard_cpu_onnxruntime() -> str:
    """Return the refusal text when the CPU ``onnxruntime`` distribution is installed."""
    if cpu_onnxruntime_conflicts():
        return CPU_ONNX_MESSAGE
    return ""


def prepare(home: Path) -> None:
    """Rewrite a list-shaped manifest (this app's previous file) into the object shape.

    ``voxprint_common`` migrates a ``resources`` object and ``holders``. The file
    written by the shared-deps branch was a ``resources`` array with ``apps``.
    That array is converted here, once, before the library reads the file.
    """
    target = manifest_file(home)
    if not target.is_file():
        return
    with exclusive_lock(lock_path(target), timeout=30):
        try:
            data = json.loads(target.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, ValueError):
            return
        if not isinstance(data, dict) or isinstance(data.get("components"), dict):
            return
        resources = data.get("resources")
        if not isinstance(resources, list):
            return
        shared = shared_root(home)
        mapped: dict[str, dict[str, object]] = {}
        for row in resources:
            if not isinstance(row, dict):
                continue
            raw_id = str(row.get("id") or "").strip()
            if not raw_id:
                continue
            version = str(row.get("version") or "")
            component_id, kind = component_id_for(raw_id, version)
            try:
                rel = component_path(home, Path(str(row.get("path") or "")), str(row.get("kind") or kind))
            except OutsideShared:
                continue
            digest = str(row.get("sha256") or "").strip().lower()
            if len(digest) != 64:
                digest = digest_for(shared / rel, f"{component_id}\n{version}\n")
            names = row.get("refs") if "refs" in row else row.get("apps")
            body: dict[str, object] = {
                "kind": str(row.get("kind") or kind),
                "version": version,
                "path": rel,
                "sha256": digest,
                "installed_at": str(row.get("installed_at") or ""),
                "refs": names if isinstance(names, list) else [],
            }
            for key, value in row.items():
                if key not in body and key not in {"id", "apps"}:
                    body[key] = value
            mapped[component_id] = body
        data.pop("resources", None)
        data.pop("version", None)
        data["schema"] = 1
        data["components"] = mapped
        write_json(target, data)


def add_component(
    home: Path,
    component_id: str,
    *,
    kind: str,
    version: str,
    path: Path,
    app: str = APP_ID,
    sha256: str = "",
) -> object:
    """Add ``app`` to one component. The same id, version and digest is reused.

    Raises:
        OutsideShared: ``path`` is outside ``<home>/shared``, or a model path is
            outside the shared models directory.
        ManifestError: The id, kind, or digest is not valid.
    """
    prepare(home)
    rel = component_path(home, path, kind)
    digest = sha256.strip().lower()
    if len(digest) != 64:
        digest = digest_for(Path(path), f"{component_id}\n{version}\n")
    return add_ref(
        component_id,
        app,
        kind=kind,
        version=version,
        path=rel,
        sha256=digest,
        manifest=manifest_file(home),
    )


def release_component(home: Path, component_id: str, app: str = APP_ID) -> ReleaseResult:
    """Drop ``app`` from one component and delete its directory when nothing still uses it.

    Other components that share the directory (the runtime and its wheels) keep
    the directory. A directory this process is running from is left in ``pending``.
    """
    prepare(home)
    target = manifest_file(home)
    if not target.is_file():
        return ReleaseResult((), ())
    document = load(target)
    current = document.components.get(component_id)
    rel = current.path if current is not None and app in current.refs else ""
    if rel and remove_ref(component_id, app, manifest=target):
        return _delete_unused(home, [rel])
    return ReleaseResult((), ())


def component_refs(home: Path, component_id: str) -> list[str]:
    """Return the apps that reference ``component_id``, or an empty list."""
    prepare(home)
    target = manifest_file(home)
    if not target.is_file():
        return []
    document = load(target)
    current = document.components.get(component_id)
    if current is None:
        return []
    return list(current.refs)


def release_app(home: Path, app: str = APP_ID) -> ReleaseResult:
    """Drop ``app`` from every component and delete directories nobody still uses.

    A directory this process is running from is left on disk and named in
    ``pending``. A component record is removed only when its ``refs`` list is
    empty. Two components may share one directory (the runtime and its wheels);
    the directory stays while any component still points at it.
    """
    prepare(home)
    target = manifest_file(home)
    if not target.is_file():
        return ReleaseResult((), ())
    document = load(target)
    owned = [key for key, value in document.components.items() if app in value.refs]
    removed: list[str] = []
    for key in owned:
        rel = document.components[key].path
        if remove_ref(key, app, manifest=target):
            removed.append(rel)
    return _delete_unused(home, removed)


def _delete_unused(home: Path, removed: list[str]) -> ReleaseResult:
    """Delete directories in ``removed`` that no remaining component still names."""
    target = manifest_file(home)
    if not target.is_file():
        return ReleaseResult((), ())
    still = load(target)
    kept = {value.path for value in still.components.values()}
    deleted: list[str] = []
    pending: list[str] = []
    shared = shared_root(home).resolve()
    seen: set[str] = set()
    for rel in removed:
        if rel in kept or not rel or rel in seen:
            continue
        seen.add(rel)
        folder = _deletable(rel, shared)
        if folder is None:
            continue
        guard = model_guard
        if guard is not None and guard.is_models_root(folder):
            deleted.extend(guard.delete_owned_models(folder))
            continue
        if _covers_kept(folder, kept, shared):
            continue
        under_models = any(_inside(folder, root) for root in _model_roots())
        if guard is None:
            if under_models:
                continue
        elif guard.is_under_models(folder) and not guard.may_delete_model(folder):
            continue
        if running_from(folder):
            pending.append(str(folder))
            continue
        if folder.is_dir():
            shutil.rmtree(folder, ignore_errors=True)
        if folder.exists():
            pending.append(str(folder))
        else:
            deleted.append(str(folder))
            if guard is not None and guard.is_under_models(folder):
                guard.forget_installed(folder.parent, folder.name)
    if not still.components and target.is_file():
        target.unlink()
    return ReleaseResult(tuple(deleted), tuple(pending))


def _covers_kept(folder: Path, kept: set[str], shared: Path) -> bool:
    """True when a remaining component path is this directory or something inside it."""
    for rel in kept:
        raw = Path(rel)
        try:
            other = raw.resolve() if raw.is_absolute() else (shared / rel).resolve()
        except OSError:
            continue
        if other == folder or folder in other.parents:
            return True
    return False


def _deletable(rel: str, shared: Path) -> Path | None:
    """Return a directory this app may remove, or None when the path is not one of ours.

    Relative paths stay inside ``shared``. Absolute model paths stay under a
    models root. The shared directory itself is never removed.
    """
    raw = Path(rel)
    try:
        folder = raw.resolve() if raw.is_absolute() else (shared / rel).resolve()
    except OSError:
        return None
    if folder == shared:
        return None
    if shared in folder.parents:
        return folder
    if raw.is_absolute() and any(_inside(folder, root) for root in _model_roots()):
        return folder
    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="components", description="Update the shared runtime manifest.")
    sub = parser.add_subparsers(dest="command", required=True)
    add = sub.add_parser("add")
    add.add_argument("--home", required=True)
    add.add_argument("--id", required=True)
    add.add_argument("--kind", required=True)
    add.add_argument("--version", required=True)
    add.add_argument("--path", required=True)
    add.add_argument("--app", default=APP_ID)
    add.add_argument("--sha256", default="")
    release = sub.add_parser("release")
    release.add_argument("--home", required=True)
    release.add_argument("--app", default=APP_ID)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Installer entry: ``add`` one component, or ``release`` an app's references."""
    args = _build_parser().parse_args(argv)
    home = Path(args.home)
    try:
        if args.command == "add":
            add_component(
                home,
                args.id,
                kind=args.kind,
                version=args.version,
                path=Path(args.path),
                app=args.app,
                sha256=args.sha256,
            )
            return 0
        if args.command == "release":
            result = release_app(home, args.app)
            print("deleted: " + (", ".join(result.deleted) or "none"))
            print("pending: " + (", ".join(result.pending) or "none"))
            return 0
    except (ManifestError, OutsideShared, LockTimeout, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"unknown command {args.command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
