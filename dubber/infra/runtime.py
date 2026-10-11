"""The Python runtime: pinned versions shared with Voxprint AI Audiobook Builder, side-by-side folders, a users file.

Agreed rules (Audiobook Builder, project-notes/voxprint/DECISIONS.md):

* Pins: ``runtime_lock.json`` next to this file keeps the Audiobook Builder lock schema (so the two files can be aligned
  later). The sibling lock on main is still Python 3.11 / torch 2.11 cu128, which this app does not copy. This lock is
  Python 3.14, win_amd64, torch 2.11.0. The only GPU flavor is cu130. The cu130 index and the PyTorch
  previous-versions install commands have no torchaudio for 2.12, 2.13, or 2.14, so the newest consistent set is
  torch 2.11.0+cu130 with torchaudio 2.11.0+cu130. torchcodec 0.17.0+cu130 is the newest codec whose table allows
  torch >= 2.11. The cpu flavor remains only so CI can pass ``-Backend cpu``.
  CTranslate2 still needs CUDA 12, so the lock also pins ``nvidia-cublas-cu12`` and ``nvidia-cudnn-cu12``.
  ``installer/runtime-constraints.txt`` records that those torch pins are applied by the installer.
* Key: :func:`runtime_key` = SHA-256 over Python version, platform, torch version + flavor, our requirements and constraints
  (comments and blank lines ignored), 12 hex digits.  ``installer/install-runtime.ps1`` computes the same key.
* The installed copy lives at ``<Voxprint home>/shared/runtimes/py3.14-torch2.11-<flavor>/`` (``cu130`` for a user install,
  ``cpu`` only when CI asks for it). A folder that already matches the pin is reused. It is never upgraded in place to a
  different Python or torch line: that line has its own directory name. ``<home>/runtime`` belongs to the Audiobook Builder
  and is never touched. An older ``runtime-<key>`` directory is moved into this layout by :func:`dubber.infra.shared_deps.migrate`.
* Who uses it is recorded in ``shared/manifest.json`` (schema 1), not in a private users file. The uninstaller drops this
  app's reference and deletes the directory only when no app remains.

Layout of one runtime: ``python`` (the uv-managed CPython), ``env`` (the virtual environment), ``runtime-key.json``,
``.install-complete``. The program folder has a junction ``<app>\\runtime`` -> ``env``, so shortcuts keep using
``<app>\\runtime\\Scripts\\python.exe``.
"""
from __future__ import annotations

import hashlib
import json
import re
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dubber.infra import model_store, shared_paths

LOCK_PATH = Path(__file__).with_name("runtime_lock.json")
PREFIX = "py3.14-torch2.11-"
KEY_FILE = "runtime-key.json"
DONE_MARK = ".install-complete"
USER_KEY = model_store.USER_KEY


def load_lock(path: Optional[Path] = None) -> Dict[str, Any]:
    """Return the parsed runtime lock (``runtime_lock.json`` unless ``path`` is given)."""
    return json.loads(Path(path or LOCK_PATH).read_text(encoding="utf-8-sig"))


def normalize_requirements(text: str) -> str:
    """Requirement lines without comments / blanks / surrounding spaces, in file order, joined by LF."""
    out = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return "\n".join(out)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def key_material(flavor: str, requirements: str, constraints: str, lock: Optional[Dict[str, Any]] = None) -> str:
    """Return the canonical string hashed into a runtime key: Python, platform, torch flavor, and requirement hashes."""
    lock = lock or load_lock()
    return (f"python={lock['python']};platform={lock['platform']};torch={lock['torch_version']}+{flavor};"
            f"req={_sha(normalize_requirements(requirements))};constraints={_sha(normalize_requirements(constraints))}")


def runtime_key(flavor: str, requirements: str, constraints: str, lock: Optional[Dict[str, Any]] = None) -> str:
    """Return the first 12 hex digits of the SHA-256 of :func:`key_material`."""
    return _sha(key_material(flavor, requirements, constraints, lock))[:12]


def choose_flavor(lock: Dict[str, Any], driver_cuda: Optional[tuple]) -> Optional[str]:
    """Newest CUDA flavor of the lock that the driver supports (``driver_cuda`` = (13, 0) from nvidia-smi).

    cu130 needs a driver that can run CUDA 13.0 (driver branch 580 or newer). The hardware gate already
    requires branch 600, which is above that line.

    None when the driver cannot run one. The CPU flavor is never chosen here: a user install does not
    offer it, and CI passes ``-Backend cpu`` to the installer on its own.
    """
    best: Optional[str] = None
    best_cu = (0, 0)
    for f in lock.get("flavors", []):
        m = re.fullmatch(r"cu(\d+)(\d)", f)
        if not m or not driver_cuda:
            continue
        cu = (int(m.group(1)), int(m.group(2)))
        if cu <= tuple(driver_cuda) and cu > best_cu:
            best, best_cu = f, cu
    return best


def runtime_dir(flavor: str) -> Path:
    """Return ``<Voxprint home>/shared/runtimes/py3.14-torch2.11-<flavor>``."""
    return shared_paths.shared_root() / "runtimes" / f"{PREFIX}{flavor}"


def is_runtime_dir(path: Path) -> bool:
    """A versioned runtime folder (never the Audiobook Builder's ``runtime``)."""
    p = Path(path)
    return p.name.startswith(PREFIX) and (p / KEY_FILE).is_file()


#: written by install-runtime.ps1 into the program folder: the real shared runtime folder behind the ``runtime`` junction
DIR_FILE = "runtime-dir.txt"


def current_runtime_dir(app_dir: Optional[Path] = None) -> Optional[Path]:
    """The versioned runtime folder of the running interpreter, or None. First the installer's note
    ``<app>\\runtime-dir.txt``, then the interpreter prefix with the ``<app>\\runtime`` junction resolved."""
    app = Path(app_dir) if app_dir else Path(__file__).resolve().parents[2]
    try:
        note = (app / DIR_FILE).read_text(encoding="utf-8-sig").strip()
        if note and is_runtime_dir(Path(note)):
            return Path(note)
    except OSError:
        pass
    cands = []
    for raw in (sys.prefix, os.path.realpath(sys.prefix)):
        try:
            pr = Path(raw).resolve()
        except OSError:
            continue
        cands += [pr.parent, pr]
    for cand in cands:
        if is_runtime_dir(cand):
            return cand
    return None


def installed_runtimes() -> List[Path]:
    """Return versioned runtime folders that contain ``runtime-key.json``, sorted by path."""
    root = shared_paths.shared_root() / "runtimes"
    if not root.is_dir():
        return []
    return sorted(p for p in root.glob(f"{PREFIX}*") if p.is_dir() and is_runtime_dir(p))


def _manifest_version(root: Path) -> str:
    if root.name.startswith(PREFIX):
        return root.name
    try:
        data = json.loads((root / KEY_FILE).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        data = {}
    flavor = str(data.get("flavor") or "cu130")
    return f"{PREFIX}{flavor}"


def register_user(root: Optional[Path] = None) -> Optional[Dict[str, bool]]:
    """Add this program's reference to the runtime entry in ``shared/manifest.json``."""
    from dubber.infra.shared_manifest import add_ref

    root = root or current_runtime_dir()
    if root is None:
        return None
    add_ref(shared_paths.voxprint_home(), resource_id="runtime", version=_manifest_version(root), path=root, app=USER_KEY)
    return {USER_KEY: True}


def unregister_user(root: Optional[Path] = None) -> List[str]:
    """Drop this program's runtime reference. Returns the other apps still using that runtime."""
    from dubber.infra.shared_manifest import release_app, resource_apps

    root = root or current_runtime_dir()
    if root is None:
        return []
    version = _manifest_version(root)
    others = [name for name in resource_apps(shared_paths.voxprint_home(), "runtime", version) if name != USER_KEY]
    release_app(shared_paths.voxprint_home(), USER_KEY, resource_id="runtime")
    return sorted(others)


def unregister_cli(out: Optional[str], root: Optional[Path] = None) -> int:
    """``--unregister-runtime-user [--out FILE]``: FILE gets ``<number of other apps>\\n<runtime folder>\\n``. Exit code 0 unless writing failed."""
    try:
        root = root or current_runtime_dir()
        others = unregister_user(root) if root else []
        if out:
            Path(out).write_text(f"{len(others)}\n{root or ''}\n", encoding="utf-8")
        print(f"other programs using the runtime {root or '(not shared)'}: {', '.join(others) or 'none'}", flush=True)
        return 0
    except OSError as exc:
        print(f"runtime users file: {exc}", flush=True)
        return 1
