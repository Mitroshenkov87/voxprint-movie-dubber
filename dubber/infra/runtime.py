"""The Python runtime: pinned versions shared with Voxprint AI Audiobook Builder, side-by-side folders, a users file.

Agreed rules (Audiobook Builder, project-notes/voxprint/DECISIONS.md):

* Pins: ``runtime_lock.json`` next to this file is a VERBATIM copy of ``infra/runtime_lock.json`` of
  Mitroshenkov87/voxprint-audiobook-builder (Source: voxprint-audiobook-builder@578646515234, build 704): Python 3.11, win_amd64,
  torch 2.11.0 in the flavors cu128 / cu126 / cpu.  The installer takes the torch version and the flavors from it; cu128 is the
  build whose ``torch\\lib`` ships ``cublas64_12.dll`` + cuDNN 9, which CTranslate2 (faster-whisper) needs for the GPU, and it also
  covers RTX 50 (Blackwell).  ``installer/runtime-constraints.txt`` adds the few pins that follow from torch (torchcodec).
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


def choose_flavor(lock: Dict[str, Any], driver_cuda: Optional[tuple]) -> str:
    """Newest CUDA flavor of the lock that the driver supports (``driver_cuda`` = (12, 8) from nvidia-smi), else cpu."""
    best, best_cu = "cpu", (0, 0)
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


def current_runtime_dir() -> Optional[Path]:
    """The ``runtime-<key>`` folder of the running interpreter (follows the ``<app>\\runtime`` junction), or None."""
    try:
        prefix = Path(sys.prefix).resolve()
    except OSError:
        return None
    for cand in (prefix.parent, prefix):
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
