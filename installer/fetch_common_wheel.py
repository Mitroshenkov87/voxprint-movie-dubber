"""Download the voxprint-common 0.2.5 wheel and check its release checksum.

``requirements.txt`` pins that wheel by URL and sha256. The offline installer
vendors the same file. The script prints the directory that contains the wheel.
"""
from __future__ import annotations

import argparse
import hashlib
import time
import urllib.request
from pathlib import Path
from urllib.error import URLError

WHEEL_URL = (
    "https://github.com/Mitroshenkov87/voxprint-common/releases/download/"
    "v0.2.5/voxprint_common-0.2.5-py3-none-any.whl"
)
WHEEL_SHA256 = "4980c1ab5a75b22dec1a16b16474ccb350110c8d3d056f25b5ed74a623fe6a9b"
WHEEL_NAME = "voxprint_common-0.2.5-py3-none-any.whl"


def fetch(dest: Path) -> Path:
    """Save the release wheel into ``dest`` and return that directory.

    A file that is already there is kept when its checksum matches. A partial file is resumed.
    The download is tried five times with exponential backoff and a per-read timeout.

    Raises:
        SystemExit: The download is missing or its SHA-256 does not match.
    """
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / WHEEL_NAME
    partial = dest / f"{WHEEL_NAME}.partial"
    if target.is_file() and _sha256(target) == WHEEL_SHA256:
        return dest
    last = "download failed"
    for attempt in range(1, 6):
        try:
            _resume(WHEEL_URL, partial)
            partial.replace(target)
            break
        except (OSError, URLError) as exc:
            last = str(exc)
            if attempt >= 5:
                raise SystemExit(f"could not download {WHEEL_URL}: {exc}") from exc
            time.sleep(2 ** (attempt - 1))
    digest = _sha256(target) if target.is_file() else ""
    if digest != WHEEL_SHA256:
        target.unlink(missing_ok=True)
        raise SystemExit(f"voxprint-common wheel is corrupt (SHA-256 {digest or last})")
    return dest


def _resume(url: str, partial: Path) -> None:
    have = partial.stat().st_size if partial.is_file() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "VoxprintMovieDubber"})
    if have:
        request.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - the release URL is a constant
        status = getattr(response, "status", 200)
        mode = "ab" if have and status == 206 else "wb"
        with partial.open(mode) as handle:
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                handle.write(block)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    """Download the wheel into ``--dest`` and print that directory."""
    parser = argparse.ArgumentParser(description="Download the voxprint-common 0.2.5 wheel.")
    parser.add_argument("--dest", required=True, help="directory that will hold the wheel")
    args = parser.parse_args(argv)
    print(fetch(Path(args.dest)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
