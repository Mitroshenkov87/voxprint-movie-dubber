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
* Reuse only on an exact key match: ``<Voxprint home>\\runtime-<key>`` (``%LOCALAPPDATA%\\Voxprint\\runtime-<key>``) with a
  ``.install-complete`` marker is used as it is; otherwise a new side-by-side folder is installed.  A shared runtime is never
  upgraded in place (a different set of versions is a different key).  ``<home>\\runtime`` itself belongs to the Audiobook
  Builder and is never touched.
* Users: ``runtime-<key>\\.users.json`` in the format of ``models\\.users.json`` (``{"movie-dubber": true}``).  The uninstaller
  removes our key (``--unregister-runtime-user --out FILE``) and deletes the folder only when no other program is listed.

Layout of one runtime: ``runtime-<key>\\python`` (the uv-managed CPython), ``runtime-<key>\\env`` (the virtual environment),
``runtime-<key>\\runtime-key.json`` (what the key was made of), ``.users.json``, ``.install-complete``.  The program folder has a
junction ``<app>\\runtime`` -> ``runtime-<key>\\env``, so shortcuts and scripts keep using ``<app>\\runtime\\Scripts\\python.exe``.
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
PREFIX = "runtime-"
KEY_FILE = "runtime-key.json"
DONE_MARK = ".install-complete"
USER_KEY = model_store.USER_KEY
_KEY_RE = re.compile(r"^runtime-[0-9a-f]{12}$")


def load_lock(path: Optional[Path] = None) -> Dict[str, Any]:
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
    lock = lock or load_lock()
    return (f"python={lock['python']};platform={lock['platform']};torch={lock['torch_version']}+{flavor};"
            f"req={_sha(normalize_requirements(requirements))};constraints={_sha(normalize_requirements(constraints))}")


def runtime_key(flavor: str, requirements: str, constraints: str, lock: Optional[Dict[str, Any]] = None) -> str:
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


def runtime_dir(key: str) -> Path:
    return shared_paths.voxprint_home() / f"{PREFIX}{key}"


def is_runtime_dir(path: Path) -> bool:
    """A folder made by this logic (never the Audiobook Builder's ``runtime``)."""
    p = Path(path)
    return bool(_KEY_RE.match(p.name)) and (p / KEY_FILE).is_file()


#: written by install-runtime.ps1 into the program folder: the real ``runtime-<key>`` folder behind the ``runtime`` junction
DIR_FILE = "runtime-dir.txt"


def current_runtime_dir(app_dir: Optional[Path] = None) -> Optional[Path]:
    """The ``runtime-<key>`` folder of the running interpreter, or None.  First the installer's note
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
    home = shared_paths.voxprint_home()
    return sorted(p for p in home.glob(f"{PREFIX}*") if p.is_dir() and is_runtime_dir(p))


def register_user(root: Optional[Path] = None) -> Optional[Dict[str, bool]]:
    root = root or current_runtime_dir()
    return model_store.register_user(USER_KEY, root) if root else None


def unregister_user(root: Optional[Path] = None) -> List[str]:
    """Remove our key from ``runtime-<key>\\.users.json``; returns the other programs still using that runtime."""
    root = root or current_runtime_dir()
    if root is None:
        return []
    return model_store.unregister_user(USER_KEY, root)


def unregister_cli(out: Optional[str], root: Optional[Path] = None) -> int:
    """``--unregister-runtime-user [--out FILE]``: FILE gets ``<number of other users>\\n<runtime folder>\\n`` (folder empty when this
    interpreter is not a shared runtime).  Exit code 0 unless writing failed."""
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
