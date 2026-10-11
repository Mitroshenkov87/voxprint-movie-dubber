"""Download the voxprint-common 0.2.1 wheel and check its release checksum.

``requirements.txt`` pins that wheel by URL and sha256. The offline installer
vendors the same file. The script prints the directory that contains the wheel.
"""
from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

WHEEL_URL = (
    "https://github.com/Mitroshenkov87/voxprint-common/releases/download/"
    "v0.2.1/voxprint_common-0.2.1-py3-none-any.whl"
)
WHEEL_SHA256 = "7a5a41284a74fc8eb2ad724647c05575d16c42881d30dba461fddf15b96a3d2f"
WHEEL_NAME = "voxprint_common-0.2.1-py3-none-any.whl"


def fetch(dest: Path) -> Path:
    """Save the release wheel into ``dest`` and return that directory.

    A file that is already there is kept when its checksum matches.

    Raises:
        SystemExit: The download is missing or its SHA-256 does not match.
    """
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / WHEEL_NAME
    if not target.is_file() or _sha256(target) != WHEEL_SHA256:
        try:
            urllib.request.urlretrieve(WHEEL_URL, target)  # noqa: S310 - the release URL is a constant
        except OSError as exc:
            raise SystemExit(f"could not download {WHEEL_URL}: {exc}") from exc
    digest = _sha256(target)
    if digest != WHEEL_SHA256:
        target.unlink(missing_ok=True)
        raise SystemExit(f"voxprint-common wheel is corrupt (SHA-256 {digest})")
    return dest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    """Download the wheel into ``--dest`` and print that directory."""
    parser = argparse.ArgumentParser(description="Download the voxprint-common 0.2.1 wheel.")
    parser.add_argument("--dest", required=True, help="directory that will hold the wheel")
    args = parser.parse_args(argv)
    print(fetch(Path(args.dest)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
