"""Shared suite dependencies: layout, migration, check and repair.

The installer puts the Python 3.14 runtime, the torch 2.11.0 / torchaudio / torchcodec cu130 wheels, the
CTranslate2 CUDA 12 libraries, ffmpeg and the default-pipeline models into ``<Voxprint home>/shared`` before
the first launch. Every launch and ``--selftest`` call :func:`live_selftest`, which checks those pieces and
repairs a missing or broken one. An old per-app runtime or ffmpeg directory is moved once by :func:`migrate`
and is not consulted again.

Progress events are dictionaries ``{"progress": true, "resource", "status", "message"}``. The window shows
them in a dialog. ``--selftest`` prints each one as a JSON line on stderr.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
from urllib.error import URLError

from dubber import models as model_registry
from dubber import paths
from dubber.appinfo import resource_dir
from dubber.i18n import tr
from dubber.infra import shared_manifest, shared_paths
from dubber.infra.gpu_policy import (
    NVIDIA_DRIVER_URL,
    driver_branch,
    gate_skipped,
    launch_block,
    probe_torch,
    startup_detail_key,
    with_driver_link,
)
from dubber.infra.shared_manifest import APP_ID

Progress = Callable[[Dict[str, Any]], None]
RUNTIME_STEM = "py3.14-torch2.11"
FFMPEG_VERSION = "n8.1"
TORCH_VERSION = "2.11.0"
TORCHAUDIO_VERSION = "2.11.0"
TORCHCODEC_VERSION = "0.17.0"
CUBLAS_VERSION = "12.9.2.10"
CUDNN_VERSION = "9.27.0.42"
CT2_VERSION = "cu12"
MODELS_VERSION = "store"
FFMPEG_RELEASE = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest"
_LEGACY_RUNTIME = re.compile(r"^runtime-[0-9a-f]{12}$")
_SKIP_MODELS = "VOXPRINT_SKIP_MODEL_FETCH"
_FLAVOR_ENV = "VOXPRINT_TORCH_FLAVOR"


class RepairError(RuntimeError):
    """A shared piece could not be restored. ``resource_id`` selects the process exit code."""

    def __init__(self, resource_id: str, message: str) -> None:
        super().__init__(message)
        self.resource_id = resource_id


@dataclass(frozen=True)
class GpuStatus:
    """Result of the launch-time graphics check."""

    ok: bool
    code: str
    message: str


@dataclass(frozen=True)
class Piece:
    """One shared component the self-test reports."""

    id: str
    version: str
    path: str
    ok: bool
    detail: str = ""
    sha256: Optional[str] = None
    size: Optional[int] = None


@dataclass
class Report:
    """Self-test result. ``outcome`` is ``ok``, ``repair``, ``gpu``, ``models`` or ``failed``."""

    outcome: str
    exit_code: int
    message: str
    repaired: List[str] = field(default_factory=list)
    resources: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        """Fields merged into the command's JSON object."""
        body: Dict[str, Any] = {
            "outcome": self.outcome,
            "repaired": list(self.repaired),
            "resources": list(self.resources),
        }
        if self.message:
            body["error"] = self.message
        return body


def runtime_version(flavor: str) -> str:
    """Directory name ``py3.14-torch2.11-<flavor>`` for one pin set."""
    return f"{RUNTIME_STEM}-{flavor}"


def runtime_path(home: Path, flavor: str) -> Path:
    """``shared/runtimes/py3.14-torch2.11-<flavor>``."""
    return Path(home) / "shared" / "runtimes" / runtime_version(flavor)


def ffmpeg_path(home: Path, version: str = FFMPEG_VERSION) -> Path:
    """``shared/ffmpeg/<version>``."""
    return Path(home) / "shared" / "ffmpeg" / version


def ffmpeg_executable(home: Optional[Path] = None, version: str = FFMPEG_VERSION) -> Path:
    """Path of the shared ffmpeg binary for this platform (the file need not exist yet)."""
    root = Path(home) if home is not None else shared_paths.voxprint_home()
    name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    return ffmpeg_path(root, version) / name


def model_fetch_skipped() -> bool:
    """True when ``VOXPRINT_SKIP_MODEL_FETCH`` is set. CI sets it; a user install does not."""
    return os.environ.get(_SKIP_MODELS, "").strip().lower() in {"1", "true", "yes", "on"}


def active_flavor() -> str:
    """``cpu`` or ``cu130``. ``VOXPRINT_TORCH_FLAVOR`` wins, otherwise the installed torch build, otherwise cu130."""
    forced = os.environ.get(_FLAVOR_ENV, "").strip()
    if forced in {"cpu", "cu130"}:
        return forced
    try:
        version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return "cu130"
    flavor = version.split("+", 1)[1] if "+" in version else ""
    return flavor if flavor in {"cpu", "cu130"} else "cu130"


def _exe(name: str) -> str:
    return f"{name}.exe" if os.name == "nt" else name


def _python_exe(root: Path) -> Path:
    scripts = "Scripts" if os.name == "nt" else "bin"
    return root / "env" / scripts / _exe("python")


def _event(resource: str, status: str, message: str) -> Dict[str, Any]:
    return {"progress": True, "resource": resource, "status": status, "message": message}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_text(cmd: Sequence[str], timeout: float = 30) -> str:
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
    except (OSError, subprocess.SubprocessError):
        return ""
    if completed.returncode != 0:
        return ""
    return (completed.stdout or completed.stderr or "").strip()


def _binary_ok(path: Path) -> str:
    """First line of ``path -version``, or '' when it does not run."""
    if not path.is_file():
        return ""
    text = _run_text([str(path), "-version"], 20)
    return text.splitlines()[0].strip() if text else ""


def _package_line(python: Path) -> str:
    code = (
        "import importlib.metadata as m\n"
        "def v(n):\n"
        "    try:\n"
        "        return m.version(n)\n"
        "    except m.PackageNotFoundError:\n"
        "        return ''\n"
        "names = ['torch','torchaudio','torchcodec','nvidia-cublas-cu12','nvidia-cudnn-cu12','ctranslate2']\n"
        "print('|'.join(v(n) for n in names))\n"
    )
    return _run_text([str(python), "-c", code], 90)


def _pins(flavor: str, line: str) -> tuple[bool, bool]:
    """Return ``(wheels_ok, cu12_ok)`` from the pipe-separated version line."""
    parts = (line or "").split("|")
    while len(parts) < 6:
        parts.append("")
    torch, audio, codec, cublas, cudnn, ct2 = parts[:6]
    flavor_ok = (flavor == "cpu" and torch.endswith("+cpu")) or (flavor != "cpu" and torch.endswith(f"+{flavor}"))
    wheels = torch.startswith(TORCH_VERSION) and audio.startswith(TORCHAUDIO_VERSION) and codec.startswith(TORCHCODEC_VERSION) and flavor_ok
    cu12 = bool(ct2) and cublas == CUBLAS_VERSION and cudnn == CUDNN_VERSION
    if flavor == "cpu":
        cu12 = bool(ct2)
    return wheels, cu12


def live_gpu() -> GpuStatus:
    """Graphics check used by launch and ``--selftest``. A skipped gate (CI) is a pass."""
    if gate_skipped():
        return GpuStatus(True, "", "")
    available, capability, name = probe_torch()
    block = launch_block(available, capability, name, driver_branch())
    if block is None:
        return GpuStatus(True, "", "")
    key, params = startup_detail_key(block)
    message = with_driver_link(tr("gpu.gate_body", detail=tr(key, **params)))
    return GpuStatus(False, block.code, message)


def live_pieces(home: Path) -> List[Piece]:
    """Runtime, wheels, CTranslate2, ffmpeg and models as they are on disk right now."""
    home = Path(home)
    flavor = active_flavor()
    version = runtime_version(flavor)
    root = runtime_path(home, flavor)
    python = _python_exe(root)
    py_version = _run_text([str(python), "-c", "import sys; print(sys.version.split()[0])"], 30) if python.is_file() else ""
    packages = _package_line(python) if python.is_file() else ""
    wheels_ok, cu12_ok = _pins(flavor, packages)
    runtime_ok = py_version.startswith("3.14.") and (root / ".install-complete").is_file() and python.is_file()
    ffmpeg = ffmpeg_executable(home)
    ffmpeg_line = _binary_ok(ffmpeg)
    models = shared_paths.models_dir()
    if model_fetch_skipped():
        models_ok = True
        models_detail = "model download is skipped for this run"
    else:
        missing = [key for key in model_registry.INSTALL_MODELS if model_registry.locate(model_registry.SPECS[key].repo) is None]
        models_ok = not missing
        models_detail = "" if models_ok else "missing " + ", ".join(missing)
    ffmpeg_sha = _sha256_file(ffmpeg) if ffmpeg.is_file() else None
    ffmpeg_size = ffmpeg.stat().st_size if ffmpeg.is_file() else None
    return [
        Piece("runtime", version, str(root), runtime_ok, "" if runtime_ok else "Python 3.14 runtime is missing or incomplete"),
        Piece("torch", f"{TORCH_VERSION}+{flavor}", str(root), wheels_ok,
              "" if wheels_ok else "torch, torchaudio or torchcodec does not match the pinned cu130/cpu build"),
        Piece("ctranslate2", CT2_VERSION, str(root), cu12_ok,
              "" if cu12_ok else "CTranslate2 CUDA 12 libraries are missing"),
        Piece("ffmpeg", FFMPEG_VERSION, str(ffmpeg_path(home)), bool(ffmpeg_line),
              "" if ffmpeg_line else "shared ffmpeg is missing or does not run", ffmpeg_sha, ffmpeg_size),
        Piece("models", MODELS_VERSION, str(models), models_ok, models_detail),
    ]


def _find_uv() -> Optional[Path]:
    name = _exe("uv")
    candidates = [resource_dir() / "tools" / "uv" / name]
    found = shutil.which("uv")
    if found:
        candidates.append(Path(found))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _requirements() -> Path:
    return resource_dir() / "requirements.txt"


def install_shared_runtime(home: Path, progress: Progress, flavor: Optional[str] = None) -> Path:
    """Install or repair the versioned runtime with uv. Raises :class:`RepairError` when uv is not there."""
    flavor = flavor or active_flavor()
    uv = _find_uv()
    if uv is None:
        raise RepairError("runtime", "uv is not available, so the shared Python runtime cannot be repaired. Run the installer again.")
    requirements = _requirements()
    if not requirements.is_file():
        raise RepairError("runtime", f"requirements.txt was not found at {requirements}")
    root = runtime_path(home, flavor)
    env = root / "env"
    python = _python_exe(root)
    root.mkdir(parents=True, exist_ok=True)
    env_vars = os.environ.copy()
    env_vars["UV_PYTHON_INSTALL_DIR"] = str(root / "python")
    env_vars["UV_PYTHON_PREFERENCE"] = "only-managed"
    env_vars["PYTHONUTF8"] = "1"
    progress(_event("runtime", "repairing", f"Installing Python 3.14 into {root}"))
    _check([str(uv), "venv", str(env), "--python", "3.14", "--allow-existing"], env_vars, "runtime")
    torch_spec = f"{TORCH_VERSION}+{flavor}"
    audio_spec = f"{TORCHAUDIO_VERSION}+{flavor}"
    codec_spec = f"{TORCHCODEC_VERSION}+{flavor}"
    progress(_event("torch", "repairing", f"Installing torch {torch_spec}, torchaudio and torchcodec"))
    _check([str(uv), "pip", "install", "--python", str(python), f"torch=={TORCH_VERSION}",
            f"torchaudio=={TORCHAUDIO_VERSION}", f"torchcodec=={TORCHCODEC_VERSION}", f"--torch-backend={flavor}"],
           env_vars, "torch")
    pins = root / "constraints.txt"
    pins.write_text(f"torch=={torch_spec}\ntorchaudio=={audio_spec}\ntorchcodec=={codec_spec}\n", encoding="utf-8")
    progress(_event("ctranslate2", "repairing", "Installing the other packages and the CUDA 12 libraries"))
    _check([str(uv), "pip", "install", "--python", str(python), "-r", str(requirements), "-c", str(pins)], env_vars, "ctranslate2")
    if flavor != "cpu":
        _check([str(uv), "pip", "install", "--python", str(python), f"nvidia-cublas-cu12=={CUBLAS_VERSION}",
                f"nvidia-cudnn-cu12=={CUDNN_VERSION}"], env_vars, "ctranslate2")
    (root / ".install-complete").write_text("ok\n", encoding="utf-8")
    (root / "runtime-key.json").write_text(json.dumps({
        "python": "3.14", "torch": torch_spec, "flavor": flavor, "created_by": APP_ID,
    }, indent=1) + "\n", encoding="utf-8")
    return root


def _check(cmd: Sequence[str], env: Dict[str, str], resource_id: str) -> None:
    try:
        completed = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    except OSError as exc:
        raise RepairError(resource_id, f"cannot start {cmd[0]}: {exc}") from exc
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip()[-600:]
        raise RepairError(resource_id, f"{cmd[0]} failed ({completed.returncode}): {tail}")


def download_resumable(url: str, dest: Path, progress: Progress, resource: str, opener: Any = urllib.request.urlopen) -> None:
    """Download ``url`` to ``dest``. A ``dest.partial`` file from an interrupted run is continued."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    have = partial.stat().st_size if partial.is_file() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "VoxprintMovieDubber"})
    if have:
        request.add_header("Range", f"bytes={have}-")
        progress(_event(resource, "repairing", f"Resuming download ({have} bytes already saved)"))
    else:
        progress(_event(resource, "repairing", f"Downloading {dest.name}"))
    try:
        response = opener(request, timeout=120)
    except URLError as exc:
        raise RepairError(resource, f"download failed: {exc}") from exc
    status = getattr(response, "status", 200)
    mode = "ab" if have and status == 206 else "wb"
    try:
        with partial.open(mode) as handle:
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                handle.write(block)
    finally:
        close = getattr(response, "close", None)
        if close:
            close()
    partial.replace(dest)


def _checksum(name: str, text: str) -> str:
    for line in text.splitlines():
        if name in line.split():
            token = line.split()[0].strip().lower()
            if len(token) == 64:
                return token
    return ""


def install_shared_ffmpeg(home: Path, progress: Progress, opener: Any = urllib.request.urlopen) -> Path:
    """Install the LGPL ffmpeg build into ``shared/ffmpeg/n8.1``. An existing working binary is kept."""
    dest_dir = ffmpeg_path(home)
    binary = ffmpeg_executable(home)
    if _binary_ok(binary):
        progress(_event("ffmpeg", "ready", "Reusing the shared ffmpeg"))
        return dest_dir
    archive_name = ("ffmpeg-n8.1-latest-win64-lgpl-8.1.zip" if os.name == "nt"
                    else "ffmpeg-n8.1-latest-linux64-lgpl-8.1.tar.xz")
    archive = dest_dir / archive_name
    download_resumable(f"{FFMPEG_RELEASE}/{archive_name}", archive, progress, "ffmpeg", opener)
    try:
        sums = opener(urllib.request.Request(f"{FFMPEG_RELEASE}/checksums.sha256", headers={"User-Agent": "VoxprintMovieDubber"}), timeout=60)
        text = sums.read().decode("ascii", "replace")
    except (OSError, URLError) as exc:
        raise RepairError("ffmpeg", f"could not read the ffmpeg checksum list: {exc}") from exc
    want = _checksum(archive_name, text)
    have = _sha256_file(archive)
    if want and have != want:
        archive.unlink(missing_ok=True)
        raise RepairError("ffmpeg", "ffmpeg download is corrupt (SHA-256 mismatch)")
    dest_dir.mkdir(parents=True, exist_ok=True)
    _extract_ffmpeg(archive, dest_dir)
    archive.unlink(missing_ok=True)
    if not _binary_ok(binary):
        raise RepairError("ffmpeg", "ffmpeg was extracted but does not run")
    return dest_dir


def _extract_ffmpeg(archive: Path, dest: Path) -> None:
    names = {_exe("ffmpeg"), _exe("ffprobe")}
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            for info in bundle.infolist():
                base = info.filename.replace("\\", "/").rsplit("/", 1)[-1]
                if base in names or base == "LICENSE.txt":
                    target_name = "FFMPEG-LICENSE.txt" if base == "LICENSE.txt" else base
                    target = dest / target_name
                    with bundle.open(info) as src, target.open("wb") as out:
                        shutil.copyfileobj(src, out)
                    if base in names:
                        target.chmod(0o755)
        return
    with tarfile.open(archive, "r:*") as bundle:
        for member in bundle.getmembers():
            base = member.name.replace("\\", "/").rsplit("/", 1)[-1]
            if not member.isfile() or base not in names | {"LICENSE.txt"}:
                continue
            extracted = bundle.extractfile(member)
            if extracted is None:
                continue
            target_name = "FFMPEG-LICENSE.txt" if base == "LICENSE.txt" else base
            target = dest / target_name
            with extracted, target.open("wb") as out:
                shutil.copyfileobj(extracted, out)
            if base in names:
                target.chmod(0o755)


def install_shared_models(progress: Progress) -> Path:
    """Download the models the default pipeline needs. Partial downloads are kept and resumed."""
    if model_fetch_skipped():
        path = shared_paths.models_dir()
        path.mkdir(parents=True, exist_ok=True)
        return path
    folder = shared_paths.models_dir()
    for key in model_registry.INSTALL_MODELS:
        spec = model_registry.SPECS[key]
        progress(_event("models", "repairing", f"Checking model {key}"))
        try:
            model_registry.ensure(spec.repo, log=lambda message: progress(_event("models", "repairing", message)))
        except Exception as exc:  # noqa: BLE001 - one message for the self-test, the store keeps the partial files
            raise RepairError("models", f"{key}: {exc}") from exc
    return folder


def live_fix(home: Path, resource_id: str, progress: Progress) -> None:
    """Repair one piece. Runtime, torch and CTranslate2 are restored together, because they share one environment."""
    if resource_id in {"runtime", "torch", "ctranslate2"}:
        install_shared_runtime(home, progress)
        return
    if resource_id == "ffmpeg":
        install_shared_ffmpeg(home, progress)
        return
    if resource_id == "models":
        install_shared_models(progress)
        return
    raise RepairError(resource_id, f"unknown shared resource {resource_id}")


def _flavor_of(key_file: Path) -> str:
    try:
        data = json.loads(key_file.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return "cu130"
    flavor = str(data.get("flavor") or "")
    if flavor in {"cpu", "cu130"}:
        return flavor
    torch = str(data.get("torch") or "")
    if torch.endswith("+cpu"):
        return "cpu"
    return "cu130"


def _relink(app_dir: Path, env_dir: Path) -> None:
    link = app_dir / "runtime"
    if not env_dir.is_dir():
        return
    try:
        if link.is_symlink() or (os.name == "nt" and link.exists() and os.path.islink(link)):
            link.unlink()
        elif link.exists():
            if env_dir.exists() and link.resolve() == env_dir.resolve():
                return
            if link.is_dir():
                shutil.rmtree(link)
            else:
                link.unlink()
        if os.name == "nt":
            subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(env_dir)], check=False, capture_output=True)
        else:
            os.symlink(env_dir, link, target_is_directory=True)
    except OSError:
        return


def _empty_dir(path: Path) -> None:
    try:
        path.rmdir()
    except OSError:
        return


def migrate(home: Optional[Path] = None, app_dir: Optional[Path] = None) -> List[str]:
    """Move this app's old runtime and ffmpeg into ``shared/`` and remove the old directories.

    Models that still sit in ``<home>/models`` (the previous default) move to ``shared/models`` when that
    folder has not been created yet. After this returns, discovery does not read the old paths.
    """
    home = Path(home) if home is not None else shared_paths.voxprint_home()
    actions: List[str] = []
    actions.extend(_migrate_runtimes(home, app_dir))
    actions.extend(_migrate_ffmpeg(home, app_dir))
    actions.extend(_migrate_models(home))
    return actions


def _migrate_runtimes(home: Path, app_dir: Optional[Path]) -> List[str]:
    actions: List[str] = []
    try:
        children = list(home.iterdir())
    except OSError:
        return actions
    for old in children:
        if not old.is_dir() or not _LEGACY_RUNTIME.match(old.name) or not (old / "runtime-key.json").is_file():
            continue
        dest = runtime_path(home, _flavor_of(old / "runtime-key.json"))
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not (dest / "runtime-key.json").is_file():
            if dest.exists():
                shutil.rmtree(dest, ignore_errors=True)
            shutil.move(str(old), str(dest))
            actions.append(f"moved {old.name} to {dest}")
        else:
            shutil.rmtree(old, ignore_errors=True)
            actions.append(f"removed {old.name}")
        if app_dir is not None:
            _relink(app_dir, dest / "env")
    if app_dir is not None:
        note = app_dir / "runtime-dir.txt"
        try:
            target = Path(note.read_text(encoding="utf-8-sig").strip()) if note.is_file() else None
        except OSError:
            target = None
        if target is not None and _LEGACY_RUNTIME.match(target.name):
            note.unlink(missing_ok=True)
    return actions


def _take_ffmpeg(folder: Path, dest: Path) -> bool:
    moved = False
    dest.mkdir(parents=True, exist_ok=True)
    for name in (_exe("ffmpeg"), _exe("ffprobe"), "FFMPEG-LICENSE.txt"):
        src = folder / name
        if not src.is_file():
            continue
        target = dest / name
        if target.exists():
            src.unlink()
        else:
            shutil.move(str(src), str(target))
        moved = True
    if moved:
        _empty_dir(folder)
    return moved


def _migrate_ffmpeg(home: Path, app_dir: Optional[Path]) -> List[str]:
    dest = ffmpeg_path(home)
    actions: List[str] = []
    folders = []
    if app_dir is not None:
        folders.append(app_dir / "bin")
    folders.append(paths.app_home() / "tools")
    for folder in folders:
        if folder.is_dir() and _take_ffmpeg(folder, dest):
            actions.append(f"moved ffmpeg from {folder}")
    return actions


def _migrate_models(home: Path) -> List[str]:
    old = home / "models"
    new = home / "shared" / "models"
    if not old.is_dir():
        return []
    if new.exists():
        return []
    new.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(old), str(new))
    return [f"moved models to {new}"]


def _register(home: Path, piece: Piece) -> None:
    if not piece.ok:
        return
    shared_manifest.add_ref(home, resource_id=piece.id, version=piece.version, path=Path(piece.path), app=APP_ID,
                            sha256=piece.sha256, size=piece.size)


def _exit_for(resource_id: str) -> tuple[str, int]:
    # Imported here: dubber.cli imports this module from the selftest command.
    from dubber.cli import EXIT_JOB, EXIT_MODELS

    if resource_id == "models":
        return "models", EXIT_MODELS
    return "failed", EXIT_JOB


def run_selftest(progress: Progress, *, gpu: Callable[[], GpuStatus], inspect: Callable[[], List[Piece]],
                 fix: Callable[[str, Progress], None], register: Optional[Callable[[Piece], None]] = None,
                 repair: bool = True) -> Report:
    """Check every piece. Repair the ones that fail when ``repair`` is true, then check those again.

    A graphics failure returns immediately with exit code 3 and does not download anything. ``outcome`` is
    ``repair`` when at least one piece was restored and the second check passed.
    """
    # Imported here: dubber.cli imports this module from the selftest command.
    from dubber.cli import EXIT_GPU, EXIT_OK

    status = gpu()
    if not status.ok:
        message = status.message or f"This PC cannot run the dubber. NVIDIA driver downloads: {NVIDIA_DRIVER_URL}"
        if NVIDIA_DRIVER_URL not in message:
            message = with_driver_link(message)
        progress(_event("gpu", "failed", message))
        return Report("gpu", EXIT_GPU, message)
    repaired: List[str] = []
    reported: List[Dict[str, Any]] = []
    seen = list(inspect())
    for piece in seen:
        progress(_event(piece.id, "checking", f"Checking {piece.id} {piece.version}"))
        current = piece
        if not current.ok:
            if not repair:
                outcome, code = _exit_for(piece.id)
                message = piece.detail or f"{piece.id} is missing"
                progress(_event(piece.id, "failed", message))
                return Report(outcome, code, message, repaired, reported)
            progress(_event(piece.id, "repairing", f"Repairing {piece.id}"))
            try:
                fix(piece.id, progress)
            except RepairError as exc:
                outcome, code = _exit_for(exc.resource_id or piece.id)
                progress(_event(piece.id, "failed", str(exc)))
                return Report(outcome, code, str(exc), repaired, reported)
            repaired.append(piece.id)
            again = {item.id: item for item in inspect()}
            current = again.get(piece.id, current)
            if not current.ok:
                outcome, code = _exit_for(piece.id)
                message = current.detail or f"{piece.id} is still missing after repair"
                progress(_event(piece.id, "failed", message))
                return Report(outcome, code, message, repaired, reported)
        progress(_event(piece.id, "ready", f"{piece.id} is ready"))
        reported.append({"id": current.id, "version": current.version, "path": current.path, "ok": current.ok})
        if register is not None:
            register(current)
    outcome = "repair" if repaired else "ok"
    return Report(outcome, EXIT_OK, "", repaired, reported)


def live_selftest(progress: Progress, *, repair: bool = True, check_gpu: bool = True) -> Report:
    """Migrate old folders, then check and repair the shared pieces on this machine."""
    home = shared_paths.voxprint_home()
    migrate(home, _app_dir())

    def gpu() -> GpuStatus:
        if not check_gpu:
            return GpuStatus(True, "", "")
        return live_gpu()

    def inspect() -> List[Piece]:
        return live_pieces(home)

    def fix(resource_id: str, prog: Progress) -> None:
        live_fix(home, resource_id, prog)

    def register(piece: Piece) -> None:
        _register(home, piece)

    return run_selftest(progress, gpu=gpu, inspect=inspect, fix=fix, register=register, repair=repair)


def _app_dir() -> Path:
    return resource_dir()
