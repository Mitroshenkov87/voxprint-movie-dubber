"""The ``.vxdub`` project file: one dub in a ZIP, without the video.

The container follows the ODF/EPUB rule: the first entry is an uncompressed ``mimetype``
file whose bytes are exactly :data:`MIME`.  The other members are JSON documents plus
optional voice-reference clips under ``assets/``.  See ``docs/formats/VXDUB.md``.

A save writes a temporary file in the destination directory and publishes it with
``os.replace``, so a crash never leaves a half-written project under the real name.
Opening a file again keeps JSON keys and zip members this version does not know.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from dubber.appinfo import APP_BUILD, APP_VERSION
from dubber.core.project import DEFAULT_SETTINGS, Line, Project, Speaker

SCHEMA = 1
MIME = "application/vnd.voxprint.dub+zip"
APP_ID = "movie-dubber"
EXTENSION = ".vxdub"
FINGERPRINT_CHUNK = 16 * 1024 * 1024

_MIME_NAME = "mimetype"
_JSON_NAMES = ("manifest.json", "transcript.json", "translation.json", "voices.json", "job.json")
_JSON_ORDER = _JSON_NAMES
_TRANSLATION_FIELDS = ("translation", "spoken", "softened", "review", "edited")
_RUNTIME_FIELDS = ("audio", "audio_s", "place_start", "stretch", "fit")

Locate = Callable[[str], Optional[Path]]


class VxdubError(Exception):
    """A project file cannot be read, linked to its video, or written."""


class SchemaTooNew(VxdubError):
    """The file was written by a newer format than this program opens."""

    def __init__(self, found: int) -> None:
        self.found = found
        super().__init__(
            f"This project uses .vxdub format version {found}. "
            f"This version of Voxprint AI Movie Dubber opens version {SCHEMA}. "
            "Update the program and open the file again."
        )


class SourceMissing(VxdubError):
    """The video is not at a saved path and was not located."""


class FingerprintMismatch(VxdubError):
    """The chosen file is not the video this project was saved with."""

    def __init__(self) -> None:
        super().__init__("The selected file is not the video this project was saved with.")


@dataclass
class Document:
    """One project file in memory.  ``extras`` and unknown JSON keys are kept for the next save."""

    manifest: Dict[str, Any]
    transcript: Dict[str, Any]
    translation: Dict[str, Any]
    voices: Dict[str, Any]
    job: Dict[str, Any]
    assets: Dict[str, bytes] = field(default_factory=dict)
    extras: Dict[str, bytes] = field(default_factory=dict)


def utc_now() -> str:
    """Current time as ISO 8601 UTC with a ``Z`` suffix and whole seconds."""
    return _utc(datetime.now(timezone.utc))


def _utc(moment: datetime) -> str:
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    else:
        moment = moment.astimezone(timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def fingerprint(path: Path) -> str:
    """SHA-256 of the first 16 MiB, the last 16 MiB, and the size.

    A file of at most 16 MiB is hashed twice (the whole file is both the head and the tail).
    A longer file seeks to ``size - 16 MiB`` for the tail, so a file shorter than 32 MiB
    overlaps the head.  The size is an unsigned 64-bit big-endian integer after those bytes.
    """
    path = Path(path)
    size = path.stat().st_size
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        head = handle.read(min(FINGERPRINT_CHUNK, size))
        digest.update(head)
        if size > FINGERPRINT_CHUNK:
            handle.seek(size - FINGERPRINT_CHUNK)
            digest.update(handle.read(FINGERPRINT_CHUNK))
        else:
            digest.update(head)
    digest.update(size.to_bytes(8, "big"))
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    """SHA-256 of every byte.  Optional in the manifest; a movie is not hashed unless asked."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_hint(video: Path, package: Path) -> str:
    """Path of the video relative to the project file's directory, or ``""`` when there is none."""
    try:
        rel = os.path.relpath(Path(video).resolve(), Path(package).resolve().parent)
    except ValueError:
        return ""
    return Path(rel).as_posix()


def read(path: Path) -> Document:
    """Read a project file.  Raises :class:`SchemaTooNew` when ``schema`` is newer than :data:`SCHEMA`."""
    path = Path(path)
    try:
        with zipfile.ZipFile(path) as archive:
            return _read_archive(archive)
    except SchemaTooNew:
        raise
    except VxdubError:
        raise
    except (OSError, zipfile.BadZipFile, ValueError) as exc:
        raise VxdubError(f"This project file could not be opened: {exc}") from exc


def resolve_source(source: Mapping[str, Any], package: Path, locate: Optional[Locate] = None) -> Path:
    """Find the video by absolute path, then by the relative hint, checking the fingerprint.

    When neither candidate matches, ``locate`` is called with the saved absolute path and the
    file it returns is accepted only when the fingerprint matches.
    """
    if not isinstance(source, Mapping):
        raise VxdubError("This project file has no video reference.")
    expected = str(source.get("fingerprint") or "")
    if not expected:
        raise VxdubError("This project file has no video fingerprint.")
    try:
        size = int(source["size"])
    except (KeyError, TypeError, ValueError):
        size = -1
    saved = str(source.get("path") or "")
    relative = str(source.get("relative") or "")
    candidates: List[Path] = []
    if saved:
        candidates.append(Path(saved))
    if relative:
        candidates.append(Path(package).parent / relative)
    for candidate in candidates:
        if _matches(candidate, expected, size):
            return candidate.resolve()
    if locate is None:
        raise SourceMissing("The video for this project could not be found.")
    chosen = locate(saved)
    if chosen is None:
        raise SourceMissing("The video for this project could not be found.")
    if not _matches(Path(chosen), expected, size):
        raise FingerprintMismatch()
    return Path(chosen).resolve()


def open_project(path: Path, root: Path, locate: Optional[Locate] = None) -> tuple[Project, Document]:
    """Open a project file, relink its video, and load it into the local project folder."""
    path = Path(path)
    doc = read(path)
    video = resolve_source(doc.manifest.get("source") or {}, path, locate)
    return materialise(doc, video, root), doc


def materialise(doc: Document, video: Path, root: Path) -> Project:
    """Write the document into the project folder for ``video``.

    The folder is the same one a dub of this path already uses, so a save on this computer
    resumes the cached stages.  A new folder (the video was relinked to a different path)
    keeps the script but clears ``done``: the stage outputs are not inside the ``.vxdub``.
    """
    video = Path(video).resolve()
    folder = Project.folder_for(video, root)
    existed = (folder / "project.json").is_file()
    project = Project(folder) if existed else Project.create(folder, video)
    settings = {**DEFAULT_SETTINGS, **_copy(doc.manifest.get("settings") or {})}
    settings["source"] = str(video)
    if doc.manifest.get("source_lang"):
        settings["source_lang"] = doc.manifest["source_lang"]
    if doc.manifest.get("target_lang"):
        settings["target_lang"] = doc.manifest["target_lang"]
    if isinstance(doc.voices.get("single_voice"), Mapping) and doc.voices.get("single_voice"):
        settings["single_voice"] = dict(doc.voices["single_voice"])
    if doc.voices.get("actor_weight") is not None:
        settings["actor_weight"] = doc.voices["actor_weight"]
    single = doc.voices.get("single_ref")
    if isinstance(single, Mapping) and single.get("audio"):
        settings["single_ref"] = {"audio": str(single.get("audio") or ""), "text": str(single.get("text") or "")}
    project.settings = settings
    project.lines = _lines_from(doc)
    project.speakers = _speakers_from(doc)
    stages = _copy(doc.job.get("stages") or {})
    if not isinstance(stages, dict):
        stages = {}
    if not existed:
        for state in stages.values():
            if isinstance(state, dict):
                state["done"] = False
    project.stages = stages
    _extract_assets(doc, project)
    project.save()
    return project


def write_project(project: Project, dest: Path, *, previous: Optional[Document] = None,
                  include_sha256: bool = False) -> Document:
    """Write ``project`` to ``dest``.  An existing ``.vxdub`` keeps unknown keys and members.

    ``previous`` is the document already in memory (for example the file the user opened).
    When it is omitted and ``dest`` is already a project file, that file is read first.
    A newer schema is never overwritten.
    """
    dest = Path(dest)
    if previous is None and dest.is_file():
        try:
            previous = read(dest)
        except SchemaTooNew:
            raise
        except VxdubError:
            previous = None
    if previous is not None:
        found = _schema_number(previous.manifest)
        if found > SCHEMA:
            raise SchemaTooNew(found)
    document = _build(project, dest, previous, include_sha256)
    _atomic_write(dest, lambda tmp: _write_zip(tmp, document))
    return document


def _read_archive(archive: zipfile.ZipFile) -> Document:
    infos = archive.infolist()
    if not infos or infos[0].filename != _MIME_NAME or infos[0].compress_type != zipfile.ZIP_STORED:
        raise VxdubError("This is not a .vxdub project: the mimetype entry must be first and stored uncompressed.")
    if archive.read(_MIME_NAME) != MIME.encode("ascii"):
        raise VxdubError("This is not a .vxdub project.")
    if "manifest.json" not in archive.namelist():
        raise VxdubError("This project file has no manifest.json.")
    manifest = _json_member(archive, "manifest.json")
    _check_schema(manifest)
    assets: Dict[str, bytes] = {}
    extras: Dict[str, bytes] = {}
    for info in infos:
        name = info.filename
        if name in (_MIME_NAME, *_JSON_NAMES) or name.endswith("/"):
            continue
        payload = archive.read(name)
        if name.startswith("assets/"):
            assets[name] = payload
        else:
            extras[name] = payload
    return Document(
        manifest=manifest,
        transcript=_json_member(archive, "transcript.json"),
        translation=_json_member(archive, "translation.json"),
        voices=_json_member(archive, "voices.json"),
        job=_json_member(archive, "job.json"),
        assets=assets,
        extras=extras,
    )


def _json_member(archive: zipfile.ZipFile, name: str) -> Dict[str, Any]:
    try:
        raw = archive.read(name)
    except KeyError:
        return {}
    if not raw:
        return {}
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise VxdubError(f"{name} is not valid JSON.") from exc
    if not isinstance(data, dict):
        raise VxdubError(f"{name} must be a JSON object.")
    return data


def _check_schema(manifest: Mapping[str, Any]) -> None:
    found = _schema_number(manifest)
    if found > SCHEMA:
        raise SchemaTooNew(found)
    if found != SCHEMA:
        raise VxdubError(f"This project uses .vxdub format version {found}, which this program does not open.")


def _schema_number(manifest: Mapping[str, Any]) -> int:
    raw = manifest.get("schema") if isinstance(manifest, Mapping) else None
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise VxdubError("This project file has no usable format version.")
    return raw


def _matches(path: Path, expected: str, size: int) -> bool:
    try:
        if not path.is_file():
            return False
        if size >= 0 and path.stat().st_size != size:
            return False
        return fingerprint(path) == expected
    except OSError:
        return False


def _build(project: Project, dest: Path, previous: Optional[Document], include_sha256: bool) -> Document:
    video = project.source
    if not video.is_file():
        raise VxdubError(f"The video is not available to save: {video}")
    base = previous or Document({}, {}, {}, {}, {})
    now = utc_now()
    created = base.manifest.get("created")
    if not isinstance(created, str) or not created:
        created = now
    source = _source_record(video, dest, base.manifest.get("source") or {}, include_sha256)
    app = dict(base.manifest.get("app") or {}) if isinstance(base.manifest.get("app"), Mapping) else {}
    app.update({"id": APP_ID, "version": APP_VERSION, "build": int(APP_BUILD)})
    settings = dict(base.manifest.get("settings") or {}) if isinstance(base.manifest.get("settings"), Mapping) else {}
    settings.update(project.settings)
    manifest = dict(base.manifest)
    manifest.update({
        "schema": SCHEMA,
        "app": app,
        "created": created,
        "modified": now,
        "source": source,
        "source_lang": project.settings.get("source_lang", "auto"),
        "target_lang": project.settings.get("target_lang", "ru"),
        "settings": settings,
    })
    transcript_rows, translation_rows, runtime_rows = _split_lines(project)
    transcript = dict(base.transcript)
    transcript["lines"] = _merge_rows(base.transcript.get("lines"), transcript_rows)
    translation = dict(base.translation)
    translation["target_lang"] = project.settings.get("target_lang", "ru")
    translation["lines"] = _merge_rows(base.translation.get("lines"), translation_rows)
    voices = _voices(project, base)
    job = dict(base.job)
    job["stages"] = _merge_stages(base.job.get("stages"), project.stages)
    job["lines"] = _merge_rows(base.job.get("lines"), runtime_rows)
    assets = dict(base.assets)
    assets.update(_reference_assets(project))
    assets = {name: blob for name, blob in assets.items() if not _is_video_member(name, video)}
    extras = {name: blob for name, blob in base.extras.items() if not _is_video_member(name, video)}
    return Document(manifest, transcript, translation, voices, job, assets, extras)


def _source_record(video: Path, dest: Path, previous: Any, include_sha256: bool) -> Dict[str, Any]:
    stat = video.stat()
    digest = fingerprint(video)
    record: Dict[str, Any] = {
        "path": str(video.resolve()),
        "relative": relative_hint(video, dest),
        "size": stat.st_size,
        "mtime": _utc(datetime.fromtimestamp(stat.st_mtime, timezone.utc)),
        "fingerprint": digest,
    }
    source = dict(previous) if isinstance(previous, Mapping) else {}
    old_fingerprint = source.get("fingerprint")
    if include_sha256:
        record["sha256"] = sha256_file(video)
    elif old_fingerprint == digest and isinstance(source.get("sha256"), str) and source.get("sha256"):
        record["sha256"] = source["sha256"]
    source.update(record)
    if "sha256" not in record:
        source.pop("sha256", None)
    return source


def _split_lines(project: Project) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    transcript, translation, runtime = [], [], []
    for line in project.lines:
        raw = asdict(line)
        spoken = {"id": line.id}
        placed = {"id": line.id}
        heard = {"id": line.id}
        for key, value in raw.items():
            if key == "id":
                continue
            if key in _TRANSLATION_FIELDS:
                spoken[key] = value
            elif key in _RUNTIME_FIELDS:
                placed[key] = value
            else:
                heard[key] = value
        transcript.append(heard)
        translation.append(spoken)
        runtime.append(placed)
    return transcript, translation, runtime


def _voices(project: Project, previous: Document) -> Dict[str, Any]:
    voices = dict(previous.voices)
    voices["actor_weight"] = project.settings.get("actor_weight")
    single = dict(voices.get("single_voice") or {}) if isinstance(voices.get("single_voice"), Mapping) else {}
    single.update(dict(project.settings.get("single_voice") or {}))
    voices["single_voice"] = single
    fresh: List[Dict[str, Any]] = []
    assets = _reference_assets(project)
    by_audio = {row.get("ref_audio"): name for name, row in _character_assets(project)}
    for speaker in project.speakers:
        row: Dict[str, Any] = {
            "id": speaker.id,
            "name": speaker.name,
            "voice": {"kind": speaker.voice.kind, "id": speaker.voice.id},
            "actor_weight": speaker.actor_weight,
            "key": speaker.key,
            "seconds": speaker.seconds,
            "ref_text": speaker.ref_text,
            "ref_audio": speaker.ref_audio,
        }
        asset = by_audio.get(speaker.ref_audio)
        if asset:
            row["reference"] = asset
        fresh.append(row)
    voices["characters"] = _merge_characters(voices.get("characters"), fresh)
    ref = project.settings.get("single_ref") or {}
    if isinstance(ref, Mapping) and ref.get("audio"):
        single_ref = {"audio": str(ref.get("audio") or ""), "text": str(ref.get("text") or "")}
        for name in assets:
            if name.endswith("/" + Path(str(ref.get("audio"))).as_posix()):
                single_ref["reference"] = name
        voices["single_ref"] = single_ref
    return voices


def _character_assets(project: Project) -> List[tuple[str, Dict[str, Any]]]:
    rows: List[tuple[str, Dict[str, Any]]] = []
    for speaker in project.speakers:
        name = _asset_name(speaker.ref_audio) if speaker.ref_audio else None
        if name is not None:
            rows.append((name, {"ref_audio": speaker.ref_audio}))
    return rows


def _reference_assets(project: Project) -> Dict[str, bytes]:
    """Voice-reference clips that live inside the project folder.  The video is never copied."""
    found: Dict[str, bytes] = {}
    rels = [speaker.ref_audio for speaker in project.speakers if speaker.ref_audio]
    single = project.settings.get("single_ref") or {}
    if isinstance(single, Mapping) and single.get("audio"):
        rels.append(str(single["audio"]))
    video = project.source.resolve() if project.source.exists() else None
    root = project.folder.resolve()
    for rel in rels:
        path = _inside_project(root, rel)
        if path is None or (video is not None and path == video):
            continue
        name = _asset_name(rel)
        if name is None or _is_video_member(name, project.source):
            continue
        try:
            found[name] = path.read_bytes()
        except OSError:
            continue
    return found


def _asset_name(rel: str) -> Optional[str]:
    target = _safe_relative(rel)
    if target is None:
        return None
    return "assets/" + target.as_posix()


def _inside_project(root: Path, rel: str) -> Optional[Path]:
    target = _safe_relative(rel)
    if target is None:
        return None
    path = (root / target).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return None
    return path if path.is_file() else None


def _safe_relative(rel: str) -> Optional[Path]:
    if not rel:
        return None
    path = Path(rel)
    if path.is_absolute() or ".." in path.parts:
        return None
    return path


def _is_video_member(name: str, video: Path) -> bool:
    return Path(name).name == Path(video).name and Path(name).suffix.lower() == Path(video).suffix.lower()


def _merge_rows(previous: Any, fresh: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prior: Dict[Any, Dict[str, Any]] = {}
    if isinstance(previous, list):
        for row in previous:
            if isinstance(row, dict) and "id" in row:
                prior[row["id"]] = dict(row)
    merged = []
    for row in fresh:
        base = prior.get(row.get("id"), {})
        base.update(row)
        merged.append(base)
    return merged


def _merge_characters(previous: Any, fresh: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prior: Dict[Any, Dict[str, Any]] = {}
    if isinstance(previous, list):
        for row in previous:
            if isinstance(row, dict) and row.get("id"):
                prior[row["id"]] = dict(row)
    merged = []
    for row in fresh:
        base = prior.get(row["id"], {})
        voice = dict(base.get("voice") or {}) if isinstance(base.get("voice"), Mapping) else {}
        voice.update(row.get("voice") or {})
        base.update(row)
        base["voice"] = voice
        merged.append(base)
    return merged


def _merge_stages(previous: Any, current: Mapping[str, Any]) -> Dict[str, Any]:
    """The project's stage map, with unknown fields inside each stage kept."""
    prior = dict(previous) if isinstance(previous, Mapping) else {}
    merged: Dict[str, Any] = {}
    for key, value in current.items():
        base = dict(prior.get(key) or {}) if isinstance(prior.get(key), Mapping) else {}
        if isinstance(value, Mapping):
            base.update(value)
            merged[key] = base
        else:
            merged[key] = value
    return merged


def _lines_from(doc: Document) -> List[Line]:
    translation = _by_id(doc.translation.get("lines"))
    runtime = _by_id(doc.job.get("lines"))
    lines = []
    seen = set()
    for row in doc.transcript.get("lines") or []:
        if not isinstance(row, dict) or "id" not in row:
            continue
        data = dict(row)
        data.update(translation.get(row["id"], {}))
        data.update(runtime.get(row["id"], {}))
        lines.append(_line_from(data))
        seen.add(row["id"])
    for line_id, row in translation.items():
        if line_id in seen:
            continue
        data = dict(row)
        data.update(runtime.get(line_id, {}))
        lines.append(_line_from(data))
    return lines


def _line_from(data: Mapping[str, Any]) -> Line:
    try:
        return Line.from_dict(dict(data))
    except TypeError as exc:
        raise VxdubError(f"A transcript line is incomplete: {exc}") from exc


def _speakers_from(doc: Document) -> List[Speaker]:
    speakers = []
    for row in doc.voices.get("characters") or []:
        if isinstance(row, dict) and row.get("id"):
            speakers.append(Speaker.from_dict(row))
    return speakers


def _by_id(rows: Any) -> Dict[Any, Dict[str, Any]]:
    found: Dict[Any, Dict[str, Any]] = {}
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and "id" in row:
                found[row["id"]] = row
    return found


def _extract_assets(doc: Document, project: Project) -> None:
    root = project.folder.resolve()
    video = project.source.resolve() if project.source.exists() else None
    for name, blob in doc.assets.items():
        if not name.startswith("assets/"):
            continue
        rel = _safe_relative(name[len("assets/"):])
        if rel is None:
            continue
        dest = (root / rel).resolve()
        try:
            dest.relative_to(root)
        except ValueError:
            continue
        if video is not None and dest == video:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".{dest.name}.{uuid.uuid4().hex[:8]}.tmp")
        tmp.write_bytes(blob)
        os.replace(tmp, dest)


def _write_zip(path: Path, document: Document) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        info = zipfile.ZipInfo(_MIME_NAME, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = 0o644 << 16
        archive.writestr(info, MIME.encode("ascii"))
        for name in _JSON_ORDER:
            payload = json.dumps(getattr(document, name.split(".")[0]), ensure_ascii=False, indent=2) + "\n"
            archive.writestr(name, payload.encode("utf-8"))
        for name in sorted(document.assets):
            archive.writestr(name, document.assets[name])
        for name, blob in document.extras.items():
            if name in (_MIME_NAME, *_JSON_NAMES) or name in document.assets or name.endswith("/"):
                continue
            archive.writestr(name, blob)


def _atomic_write(dest: Path, write: Callable[[Path], None]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        write(tmp)
        _replace(tmp, dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _replace(tmp: Path, dest: Path) -> None:
    for attempt in range(10):
        try:
            os.replace(tmp, dest)
            return
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))
    tmp.unlink(missing_ok=True)
    raise PermissionError(f"cannot replace {dest}")


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value))
