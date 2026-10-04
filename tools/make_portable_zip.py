"""Build the portable bundle: ``dist/Voxprint-MovieDubber-<version>-portable.zip`` (program source + bat files + test clip).

The user unpacks it anywhere and runs ``setup.bat`` once (creates .venv with Python 3.11 + CUDA PyTorch), then ``run.bat`` or
``diagnose.bat``.  Works on any OS (the zip is plain files); the content list is a whitelist so no .venv/.git/caches get in.
Usage:  python tools/make_portable_zip.py [--out dist]
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

INCLUDE_DIRS = ["dubber", "assets"]
INCLUDE_FILES = ["main.py", "requirements.txt", "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md",
                 "setup.bat", "run.bat", "diagnose.bat", "diagnose-console.bat"]
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".venv", ".git"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".partial", ".tmp"}


def collect(root: Path = ROOT):
    files = []
    for name in INCLUDE_FILES:
        p = root / name
        if p.is_file():
            files.append(p)
    for d in INCLUDE_DIRS:
        for p in sorted((root / d).rglob("*")):
            if p.is_file() and not (set(p.relative_to(root).parts) & SKIP_PARTS) and p.suffix not in SKIP_SUFFIXES:
                files.append(p)
    return files


def build(out_dir: Path, root: Path = ROOT) -> Path:
    from dubber import appinfo
    out_dir.mkdir(parents=True, exist_ok=True)
    zpath = out_dir / f"Voxprint-MovieDubber-{appinfo.APP_VERSION}-portable.zip"
    top = f"Voxprint-MovieDubber-{appinfo.APP_VERSION}"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in collect(root):
            z.write(p, f"{top}/{p.relative_to(root).as_posix()}")
    return zpath


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "dist"))
    a = ap.parse_args()
    z = build(Path(a.out))
    print(f"{z}  ({z.stat().st_size / 1e6:.1f} MB)")
