"""Refuse to delete shared model folders this app did not install or does not solely own.

``.users.json`` and ``.installed.json`` are updated under a file lock. Unknown keys and
other programs' entries stay. A publish is a temporary file plus ``os.replace``.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from dubber import paths
from voxprint_common.atomic import exclusive_lock, write_json

APP_ID = "movie-dubber"
_SKIP_USER_KEYS = frozenset({"schema", "note", "apps", "users", "by_app", "legacy_users", "extra"})
_BOOKKEEPING = frozenset({".users.json", ".installed.json", ".users.lock", ".installed.lock"})


def users_path(root: Path) -> Path:
    """Return ``<models>/.users.json``."""
    return Path(root) / ".users.json"


def ledger_path(root: Path) -> Path:
    """Return ``<models>/.installed.json``."""
    return Path(root) / ".installed.json"


def is_models_root(folder: Path) -> bool:
    """True when ``folder`` is the shared models directory itself."""
    try:
        return Path(folder).resolve() == paths.models_dir().resolve()
    except OSError:
        return False


def is_under_models(folder: Path) -> bool:
    """True when ``folder`` is inside the shared models directory and is not the directory itself."""
    try:
        target = Path(folder).resolve()
        root = paths.models_dir().resolve()
    except OSError:
        return False
    return root in target.parents


def load_users_document(path: Path) -> dict[str, Any]:
    """Read a users document. A missing file is ``{}``. A list of names becomes flags plus ``legacy_users``.

    Raises:
        OSError: The file exists and is not a JSON object or list. The caller must not replace it.
    """
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise OSError(f"cannot read {path}") from exc
    if isinstance(data, list):
        names = [str(item) for item in data if isinstance(item, str) and str(item)]
        body: dict[str, Any] = {"legacy_users": list(names)}
        for name in names:
            body[name] = True
        return body
    if not isinstance(data, dict):
        raise OSError(f"{path.name} is not a JSON object")
    return dict(data)


def user_flags(document: dict[str, Any]) -> dict[str, bool]:
    """Return program name to in-use flag. Bookkeeping keys are not programs."""
    flags: dict[str, bool] = {}
    for key, value in document.items():
        if key in _SKIP_USER_KEYS or not isinstance(value, bool):
            continue
        flags[str(key)] = value
    for key in ("apps", "users", "legacy_users"):
        value = document.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item and item not in flags:
                    flags[item] = True
        elif isinstance(value, dict):
            for name, flag in value.items():
                if isinstance(flag, bool):
                    flags[str(name)] = flag
    return flags


def active_users(document: dict[str, Any]) -> list[str]:
    """Sorted names whose flag is true."""
    return sorted(name for name, flag in user_flags(document).items() if flag)


def other_users(document: dict[str, Any], app: str = APP_ID) -> bool:
    """True when some other program still uses the shared models directory."""
    return any(name != app and flag for name, flag in user_flags(document).items())


def with_user(document: dict[str, Any], app: str) -> dict[str, Any]:
    """Copy ``document``, mark ``app`` in use, and append it to ``apps`` when it is absent."""
    updated = dict(document)
    updated[app] = True
    apps = updated.get("apps")
    names = [item for item in apps if isinstance(item, str)] if isinstance(apps, list) else []
    if app not in names:
        names.append(app)
    updated["apps"] = names
    return updated


def without_user(document: dict[str, Any], app: str) -> dict[str, Any]:
    """Copy ``document`` without ``app``. An empty name list is omitted so a last unregister leaves ``{}``."""
    updated = dict(document)
    updated.pop(app, None)
    for key in ("apps", "users", "legacy_users"):
        value = updated.get(key)
        if isinstance(value, list):
            names = [item for item in value if item != app]
            if names:
                updated[key] = names
            else:
                updated.pop(key, None)
        elif isinstance(value, dict):
            nested = {name: flag for name, flag in value.items() if name != app}
            if nested:
                updated[key] = nested
            else:
                updated.pop(key, None)
    return updated


def update_users(root: Path, change: Callable[[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
    """Read-modify-write ``.users.json`` under ``.users.lock``. Unknown keys are the caller's to keep.

    Raises:
        OSError: The current file cannot be parsed. It is left unchanged.
    """
    path = users_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(path.with_name(".users.lock")):
        document = load_users_document(path)
        updated = change(document)
        write_json(path, updated)
        return updated


def load_ledger(path: Path) -> dict[str, Any]:
    """Read ``.installed.json``. A missing file is an empty ledger. A bad file raises ``OSError``."""
    if not path.is_file():
        return {"schema": 1, "by_app": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise OSError(f"cannot read {path}") from exc
    if not isinstance(data, dict):
        raise OSError(f"{path.name} is not a JSON object")
    return dict(data)


def safe_rel(relative: str) -> str:
    """Return a relative model path that cannot escape the models directory.

    Raises:
        ValueError: The path is absolute or contains ``..``.
    """
    text = relative.replace("\\", "/").strip("/")
    if not text or Path(text).is_absolute() or ".." in Path(text).parts:
        raise ValueError(f"refusing model path {relative!r}")
    return text


def _rels(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    found: list[str] = []
    for item in value:
        if isinstance(item, str):
            try:
                found.append(safe_rel(item))
            except ValueError:
                continue
    return found


def _mutate_ledger(root: Path, change: Callable[[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
    path = ledger_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(path.with_name(".installed.lock")):
        document = load_ledger(path)
        updated = change(document)
        write_json(path, updated)
        return updated


def record_installed(root: Path, relative: str, app: str = APP_ID) -> None:
    """Remember that ``app`` installed ``relative`` under ``root``. Other apps' entries stay."""
    rel = safe_rel(relative)

    def change(document: dict[str, Any]) -> dict[str, Any]:
        updated = dict(document)
        raw = updated.get("by_app")
        by_app = dict(raw) if isinstance(raw, dict) else {}
        current = _rels(by_app.get(app))
        if rel not in current:
            current.append(rel)
        by_app[app] = current
        updated["by_app"] = by_app
        updated.setdefault("schema", 1)
        return updated

    _mutate_ledger(root, change)


def forget_installed(root: Path, relative: str, app: str = APP_ID) -> None:
    """Drop one installed path for ``app``. A missing ledger is left missing."""
    try:
        rel = safe_rel(relative)
    except ValueError:
        return
    path = ledger_path(root)
    if not path.is_file():
        return

    def change(document: dict[str, Any]) -> dict[str, Any]:
        updated = dict(document)
        raw = updated.get("by_app")
        if not isinstance(raw, dict):
            return updated
        by_app = dict(raw)
        current = [item for item in _rels(by_app.get(app)) if item != rel]
        if current:
            by_app[app] = current
        else:
            by_app.pop(app, None)
        updated["by_app"] = by_app
        return updated

    _mutate_ledger(root, change)


def may_delete_model(folder: Path, root: Path | None = None) -> bool:
    """True only for a folder this app recorded and solely owns, and only when no other program uses the store."""
    base = Path(root) if root is not None else paths.models_dir()
    try:
        target = Path(folder).resolve()
        models = Path(base).resolve()
    except OSError:
        return False
    if target == models or models not in target.parents:
        return False
    rel = target.relative_to(models).as_posix()
    if ".." in Path(rel).parts:
        return False
    try:
        users = load_users_document(users_path(models))
        ledger = load_ledger(ledger_path(models))
    except OSError:
        return False
    if other_users(users):
        return False
    raw = ledger.get("by_app")
    by_app = raw if isinstance(raw, dict) else {}
    if rel not in _rels(by_app.get(APP_ID)):
        return False
    for name, paths_for_app in by_app.items():
        if str(name) != APP_ID and rel in _rels(paths_for_app):
            return False
    return True


def _inside(folder: Path, root: Path) -> bool:
    """True when ``folder`` resolves to ``root`` or a directory inside it. A symlink that escapes does not."""
    try:
        if folder.is_symlink():
            return False
        target = folder.resolve()
        base = root.resolve()
    except OSError:
        return False
    return target == base or base in target.parents


def _remove_dir(folder: Path, root: Path) -> None:
    """Remove ``folder`` only when it is ``root`` or a real directory inside ``root``.

    Raises:
        OSError: The path is a symlink or resolves outside ``root``.
    """
    if not _inside(folder, root):
        raise OSError("refusing to delete outside the models directory")
    shutil.rmtree(folder)


def _only_bookkeeping(root: Path) -> bool:
    try:
        children = list(root.iterdir())
    except OSError:
        return False
    for child in children:
        if child.is_dir():
            return False
        if child.name in _BOOKKEEPING or (child.name.startswith(".") and child.name.endswith(".lock")):
            continue
        return False
    return True


def delete_owned_models(root: Path) -> list[str]:
    """Delete model folders this app recorded and solely owns. The models directory itself stays while anything else is inside it.

    An empty store that only this app uses is removed, because that directory is the models component this app recorded.
    """
    try:
        base = Path(root).resolve()
    except OSError:
        return []
    if not base.is_dir() or not is_models_root(base):
        return []
    deleted: list[str] = []
    try:
        ledger = load_ledger(ledger_path(base))
    except OSError:
        return []
    raw = ledger.get("by_app")
    by_app = raw if isinstance(raw, dict) else {}
    for rel in list(_rels(by_app.get(APP_ID))):
        folder = base / rel
        if not may_delete_model(folder, base):
            continue
        if folder.is_symlink():
            continue
        if folder.is_dir():
            _remove_dir(folder, base)
        elif folder.is_file():
            folder.unlink()
        if not folder.exists():
            forget_installed(base, rel)
            deleted.append(str(folder))
    try:
        users = load_users_document(users_path(base))
    except OSError:
        return deleted
    if not other_users(users) and base.is_dir() and not base.is_symlink() and _only_bookkeeping(base):
        shutil.rmtree(base)  # the models directory this app recorded, and it holds no other weights
        if not base.exists():
            deleted.append(str(base))
    return deleted


def replace_downloaded(part: Path, target: Path, wholesale: bool) -> None:
    """Move a finished ``.partial`` folder onto ``target``.

    A wholesale replace removes ``target`` first. A shared folder is updated file by file and the partial
    directory is the only tree removed. The shared models directory is never the target.

    Raises:
        OSError: ``target`` is the models directory, or the rename did not succeed.
    """
    if is_models_root(target) or target.is_symlink() or (target.exists() and not _inside(target, target.parent)):
        raise OSError("refusing to replace the shared models directory")
    if wholesale or not target.exists():
        if target.is_dir():
            _remove_dir(target, target.parent)
        elif target.exists():
            target.unlink()
        last: OSError | None = None
        for attempt in range(10):
            try:
                os.replace(part, target)
                return
            except PermissionError as exc:
                last = exc
                time.sleep(0.5 * (attempt + 1))
        raise OSError(f"could not rename {part.name}") from last
    for src in part.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(part)
        if ".." in rel.parts:
            continue
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    if part.is_symlink() or not _inside(part, part.parent):
        raise OSError("refusing to delete outside the models directory")
    shutil.rmtree(part)
