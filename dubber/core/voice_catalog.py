"""Open voices published by Voxprint on Hugging Face (e.g. ``Mitroshenkov87/voxprint-voices-ru``: Levi, Natan, Shimon, Miriam,
Rivka, Noa - CC0) - listed next to the local library and installed into the SHARED voice library, in the same format the
Audiobook Builder uses (``<voxprint home>/voices/<id>/``: LoRA adapter, ``ref_sample.wav``, ``training_meta.json``,
``speaker_centroid.safetensors``, ``voice.json`` with ``repo_id``).  Both programs then see the same voice once.

Repository layout: ``<id>.zip`` per voice, ``SHA256SUMS.txt`` (``<sha256>  <id>.zip``) and ``<id>/voice.json`` for the catalog.
Download: lock file ``voices/.<id>.lock`` (same OS lock as the model store), ``.part`` resumed with HTTP Range, SHA-256 must
match before anything is extracted, only known file names are extracted, the folder appears atomically (rename).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Dict, List, Optional

from dubber.infra import shared_paths
from dubber.infra.model_store import ModelLock

DEFAULT_REPOS = ("Mitroshenkov87/voxprint-voices-ru",)
ENV_REPOS = "VOXPRINT_VOICE_REPOS"                     # comma-separated override
HF = "https://huggingface.co"
ALLOWED_FILES = ("adapter_model.safetensors", "adapter_config.json", "ref_sample.wav", "training_meta.json", "preview.wav",
                 "voice.json", "speaker_centroid.safetensors")
MAX_ZIP_BYTES = 1 << 30

Fetch = Callable[[str], bytes]                       # url -> body


class VoiceCatalogError(RuntimeError):
    pass


@dataclass
class RemoteVoice:
    id: str
    name: str
    repo: str
    sha256: str
    language: str = ""
    gender: str = ""
    license: str = ""
    description: str = ""
    size_bytes: int = 0

    @property
    def url(self) -> str:
        return f"{HF}/{self.repo}/resolve/main/{self.id}.zip"


def repos() -> List[str]:
    env = os.environ.get(ENV_REPOS, "").strip()
    return [r.strip() for r in env.split(",") if r.strip()] if env else list(DEFAULT_REPOS)


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Voxprint-Movie-Dubber"})
    with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310 - https to huggingface.co only
        return r.read()


def parse_sums(text: str) -> Dict[str, str]:
    out = {}
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) == 2 and len(parts[0]) == 64 and parts[1].lstrip("*").endswith(".zip"):
            out[parts[1].lstrip("*")[:-4]] = parts[0].lower()
    return out


def list_remote(repo: str, fetch: Fetch = _fetch) -> List[RemoteVoice]:
    """Voices of one repository (network errors raise :class:`VoiceCatalogError`)."""
    try:
        sums = parse_sums(fetch(f"{HF}/{repo}/resolve/main/SHA256SUMS.txt").decode("utf-8", "replace"))
        try:
            sizes = {s["rfilename"][:-4]: int(s.get("size") or 0) for s in json.loads(fetch(f"{HF}/api/models/{repo}?blobs=true"))
                     .get("siblings", []) if s.get("rfilename", "").endswith(".zip")}
        except (ValueError, OSError):
            sizes = {}
        out = []
        for vid, sha in sorted(sums.items()):
            try:
                info = json.loads(fetch(f"{HF}/{repo}/resolve/main/{vid}/voice.json").decode("utf-8-sig"))
            except (ValueError, OSError):
                info = {}
            out.append(RemoteVoice(vid, str(info.get("name") or vid.title()), repo, sha, str(info.get("language") or ""),
                                   str(info.get("gender") or info.get("voice_type") or ""), str(info.get("license") or ""),
                                   str(info.get("description") or ""), sizes.get(vid, 0)))
        return out
    except OSError as exc:
        raise VoiceCatalogError(f"{repo}: {exc}") from exc


def installed_ids(root: Optional[Path] = None) -> Dict[str, Path]:
    """``repo_id`` (else folder name) -> folder, for voices already in the shared library."""
    root = Path(root) if root is not None else shared_paths.voices_dir()
    out: Dict[str, Path] = {}
    try:
        for d in root.iterdir():
            if d.is_dir() and not d.name.startswith(".") and (d / "adapter_model.safetensors").is_file():
                try:
                    rid = str(json.loads((d / "voice.json").read_text(encoding="utf-8-sig")).get("repo_id") or "")
                except (OSError, ValueError):
                    rid = ""
                out.setdefault(rid or d.name, d)
    except OSError:
        pass
    return out


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, dest: Path, expected: int, progress: Callable[[float], None], cancel: Callable[[], bool]) -> None:
    have = dest.stat().st_size if dest.exists() else 0
    headers = {"User-Agent": "Voxprint-Movie-Dubber"}
    if have:
        headers["Range"] = f"bytes={have}-"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:  # noqa: S310
        if have and getattr(r, "status", 200) != 206:
            have = 0
        got = have
        with open(dest, "ab" if have else "wb") as out:
            while True:
                if cancel():
                    raise VoiceCatalogError("cancelled")
                block = r.read(1 << 20)
                if not block:
                    break
                got += len(block)
                if got > MAX_ZIP_BYTES:
                    raise VoiceCatalogError("download too large")
                out.write(block)
                if expected:
                    progress(min(1.0, got / expected))


def extract_voice(zip_path: Path, dest: Path) -> None:
    """Only the known files, flat (a single wrapping folder is accepted); no paths outside ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for m in zf.infolist():
            name = PurePosixPath(m.filename.replace("\\", "/"))
            if m.is_dir() or name.name not in ALLOWED_FILES or ".." in name.parts or name.is_absolute() or len(name.parts) > 2:
                continue
            if m.file_size > MAX_ZIP_BYTES:
                raise VoiceCatalogError(f"{name.name} is too large")
            with zf.open(m) as src, open(dest / name.name, "wb") as out:
                shutil.copyfileobj(src, out)
    if not all((dest / n).is_file() for n in ("adapter_model.safetensors", "adapter_config.json")):
        raise VoiceCatalogError("the archive has no voice adapter")


def install(voice: RemoteVoice, root: Optional[Path] = None, progress: Callable[[float], None] = lambda f: None,
            cancel: Callable[[], bool] = lambda: False,
            downloader: Optional[Callable[[str, Path, int, Callable[[float], None], Callable[[], bool]], None]] = None) -> Path:
    """Download, verify and add one voice to the shared library; returns its folder (an installed voice is not downloaded again)."""
    root = Path(root) if root is not None else shared_paths.voices_dir()
    have = installed_ids(root)
    if voice.id in have:
        return have[voice.id]
    root.mkdir(parents=True, exist_ok=True)
    lock = ModelLock(root / f".{voice.id}.lock")
    if not lock.try_acquire():
        raise VoiceCatalogError(f"{voice.name} is being downloaded by another Voxprint program")
    stage = root / f".{voice.id}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        have = installed_ids(root)                       # the other program may have finished meanwhile
        if voice.id in have:
            return have[voice.id]
        parts = root / ".downloads"
        parts.mkdir(exist_ok=True)
        part = parts / f"{voice.sha256}.part"
        (downloader or _download)(voice.url, part, voice.size_bytes, progress, cancel)
        digest = _sha256(part)
        if digest != voice.sha256:
            part.unlink(missing_ok=True)
            raise VoiceCatalogError(f"{voice.name}: checksum mismatch")
        extract_voice(part, stage)
        info_path = stage / "voice.json"
        try:
            info = json.loads(info_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            info = {}
        info.update({"id": voice.id, "repo_id": voice.id, "name": info.get("name") or voice.name,
                     "language": info.get("language") or voice.language, "license": voice.license or info.get("license", "")})
        info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        target = root / voice.id
        n = 2
        while target.exists():
            target, n = root / f"{voice.id}-{n}", n + 1
        os.replace(stage, target)
        part.unlink(missing_ok=True)
        return target
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        lock.release(remove=True)
