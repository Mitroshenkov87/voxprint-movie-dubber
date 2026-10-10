"""The .vxdub project file: container, round trip, relink, and the file menu."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from dubber.appinfo import APP_BUILD, APP_VERSION
from dubber.core.project import Line, Project, Speaker, Voice
from dubber.core import vxdub

ROOT = Path(__file__).resolve().parents[1]
MIME = b"application/vnd.voxprint.dub+zip"
SENTINEL = b"VXVIDEO-SENTINEL-DO-NOT-EMBED-7f3a"


def _film(tmp_path: Path, payload: bytes = SENTINEL + b"-body") -> tuple[Project, Path]:
    video = tmp_path / "films" / "clip.mkv"
    video.parent.mkdir(parents=True)
    video.write_bytes(payload)
    root = tmp_path / "projects"
    project = Project.create(Project.folder_for(video, root), video, target_lang="de", source_lang="en")
    project.lines = [Line(1, 0.2, 1.4, "Hello", "Hallo", speaker="S1", spoken="Hallo!", fit="fits",
                          audio="tts/line_1.wav", audio_s=1.1, place_start=0.2, stretch=1.0)]
    project.speakers = [Speaker("S1", "Ann", voice=Voice("actor", "lib-1"), actor_weight=0.35, ref_text="Hello",
                                seconds=1.2, key=True)]
    clip = project.path("voices", "S1.wav")
    clip.write_bytes(b"RIFFfake")
    project.speakers[0].ref_audio = project.rel(clip)
    project.stages = {"asr": {"done": True, "inputs": "cache-key", "seconds": 0.4}}
    project.settings["single_voice"] = {"kind": "library", "id": "anna", "actor_weight": 0.5}
    project.settings["actor_weight"] = 0.5
    project.save()
    return project, video


def _manifest(schema: int, fingerprint: str = "abc", size: int = 1) -> dict:
    return {
        "schema": schema,
        "app": {"id": "movie-dubber", "version": "1", "build": 1},
        "created": "2020-01-02T03:04:05Z",
        "modified": "2020-01-02T03:04:05Z",
        "source": {"path": "C:/missing.mkv", "relative": "", "size": size, "mtime": "2020-01-02T03:04:05Z",
                   "fingerprint": fingerprint},
        "source_lang": "en",
        "target_lang": "ru",
        "settings": {},
    }


def _write_raw(path: Path, schema: int = 1, *, first: str = "mimetype", stored: bool = True, mime: bytes = MIME) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        members = [("manifest.json", json.dumps(_manifest(schema)).encode("utf-8"))]
        mime_info = zipfile.ZipInfo("mimetype", date_time=(1980, 1, 1, 0, 0, 0))
        mime_info.compress_type = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
        ordered = [("mimetype", None)] + members if first == "mimetype" else members + [("mimetype", None)]
        for name, payload in ordered:
            if name == "mimetype":
                archive.writestr(mime_info, mime)
            else:
                archive.writestr(name, payload)


def test_fingerprint_is_head_tail_and_size(tmp_path):
    small = tmp_path / "small.bin"
    data = b"abcdefghij"
    small.write_bytes(data)
    digest = hashlib.sha256()
    digest.update(data)
    digest.update(data)
    digest.update(len(data).to_bytes(8, "big"))
    assert vxdub.fingerprint(small) == digest.hexdigest()

    chunk = vxdub.FINGERPRINT_CHUNK
    large = tmp_path / "large.bin"
    blob = bytearray(chunk + 5)
    blob[0] = 1
    blob[-1] = 2
    large.write_bytes(blob)
    digest = hashlib.sha256()
    digest.update(bytes(blob[:chunk]))
    digest.update(bytes(blob[-chunk:]))
    digest.update(len(blob).to_bytes(8, "big"))
    assert vxdub.fingerprint(large) == digest.hexdigest()


def test_mimetype_is_first_and_stored(tmp_path):
    project, _video = _film(tmp_path)
    dest = tmp_path / "film.vxdub"
    vxdub.write_project(project, dest)
    raw = dest.read_bytes()
    assert raw[:4] == b"PK\x03\x04"
    assert raw[8:10] == b"\x00\x00"
    name_len = int.from_bytes(raw[26:28], "little")
    extra_len = int.from_bytes(raw[28:30], "little")
    assert extra_len == 0 and raw[30:30 + name_len] == b"mimetype"
    assert raw[30 + name_len:30 + name_len + len(MIME)] == MIME
    with zipfile.ZipFile(dest) as archive:
        info = archive.infolist()[0]
        assert info.filename == "mimetype" and info.compress_type == zipfile.ZIP_STORED
        assert archive.read("mimetype") == MIME
        assert "clip.mkv" not in archive.namelist()


def test_round_trip_without_the_video(tmp_path):
    project, video = _film(tmp_path)
    dest = tmp_path / "out" / "film.vxdub"
    vxdub.write_project(project, dest)
    with zipfile.ZipFile(dest) as archive:
        blobs = [archive.read(name) for name in archive.namelist()]
    assert all(SENTINEL not in blob for blob in blobs)
    assert any(b"RIFFfake" in blob for blob in blobs)

    root = tmp_path / "projects"
    again, _doc = vxdub.open_project(dest, root)
    assert again.source.resolve() == video.resolve()
    assert again.settings["target_lang"] == "de" and again.settings["source_lang"] == "en"
    assert again.settings["single_voice"]["id"] == "anna"
    line = again.lines[0]
    assert (line.text, line.translation, line.spoken, line.speaker, line.fit, line.audio) == (
        "Hello", "Hallo", "Hallo!", "S1", "fits", "tts/line_1.wav")
    speaker = again.speaker("S1")
    assert speaker is not None and speaker.name == "Ann" and speaker.actor_weight == 0.35
    assert speaker.voice.kind == "actor" and speaker.voice.id == "lib-1" and speaker.key is True
    assert (again.folder / "voices" / "S1.wav").read_bytes() == b"RIFFfake"
    assert again.stages["asr"]["inputs"] == "cache-key" and again.stages["asr"]["done"] is True

    fresh, _doc = vxdub.open_project(dest, tmp_path / "other-root")
    assert fresh.stages["asr"]["done"] is False
    assert vxdub.read(dest).job["stages"]["asr"]["done"] is True

    manifest = vxdub.read(dest).manifest
    assert manifest["schema"] == 1
    assert manifest["app"] == {"id": "movie-dubber", "version": APP_VERSION, "build": APP_BUILD}
    assert manifest["created"].endswith("Z") and manifest["modified"].endswith("Z")
    source = manifest["source"]
    assert source["path"] == str(video.resolve())
    assert source["relative"] == "../films/clip.mkv"
    assert source["size"] == video.stat().st_size
    assert source["fingerprint"] == vxdub.fingerprint(video)
    assert "sha256" not in source
    vxdub.write_project(project, dest, include_sha256=True)
    assert vxdub.read(dest).manifest["source"]["sha256"] == vxdub.sha256_file(video)


def test_unknown_keys_and_files_survive_a_resave(tmp_path):
    project, _video = _film(tmp_path)
    dest = tmp_path / "film.vxdub"
    vxdub.write_project(project, dest)
    doc = vxdub.read(dest)
    doc.manifest["editor"] = "hand"
    doc.manifest["created"] = "2020-01-02T03:04:05Z"
    doc.manifest["app"]["channel"] = "studio"
    doc.transcript["vendor"] = {"keep": True}
    doc.transcript["lines"][0]["dialect"] = "us"
    doc.extras["notes/readme.txt"] = b"keep me"
    doc.assets["assets/extra.bin"] = b"\x01\x02"
    vxdub.write_project(project, dest, previous=doc)
    project.lines[0].text = "Hello!"
    vxdub.write_project(project, dest)
    again = vxdub.read(dest)
    assert again.manifest["editor"] == "hand"
    assert again.manifest["created"] == "2020-01-02T03:04:05Z"
    assert again.manifest["app"]["channel"] == "studio"
    assert again.manifest["app"]["id"] == "movie-dubber"
    assert again.transcript["vendor"] == {"keep": True}
    assert again.transcript["lines"][0]["dialect"] == "us"
    assert again.transcript["lines"][0]["text"] == "Hello!"
    assert again.extras["notes/readme.txt"] == b"keep me"
    assert again.assets["assets/extra.bin"] == b"\x01\x02"
    outside = tmp_path / "outside.txt"
    slipped = vxdub.read(dest)
    slipped.assets["assets/../../outside.txt"] = b"nope"
    vxdub.write_project(project, dest, previous=slipped)
    vxdub.open_project(dest, tmp_path / "projects")
    assert not outside.exists()


def test_newer_schema_is_rejected_and_not_overwritten(tmp_path):
    project, _video = _film(tmp_path)
    dest = tmp_path / "new.vxdub"
    _write_raw(dest, schema=2)
    before = dest.read_bytes()
    with pytest.raises(vxdub.SchemaTooNew) as raised:
        vxdub.read(dest)
    message = str(raised.value)
    assert "version 2" in message and "version 1" in message and "Update the program" in message
    with pytest.raises(vxdub.SchemaTooNew):
        vxdub.write_project(project, dest)
    assert dest.read_bytes() == before
    _write_raw(dest, schema=0)
    with pytest.raises(vxdub.VxdubError) as old:
        vxdub.read(dest)
    assert not isinstance(old.value, vxdub.SchemaTooNew)


def test_bad_container_is_rejected(tmp_path):
    compressed = tmp_path / "compressed.vxdub"
    _write_raw(compressed, stored=False)
    with pytest.raises(vxdub.VxdubError, match="uncompressed"):
        vxdub.read(compressed)
    misplaced = tmp_path / "misplaced.vxdub"
    _write_raw(misplaced, first="manifest")
    with pytest.raises(vxdub.VxdubError, match="first"):
        vxdub.read(misplaced)
    wrong = tmp_path / "wrong.vxdub"
    _write_raw(wrong, mime=b"application/epub+zip")
    with pytest.raises(vxdub.VxdubError):
        vxdub.read(wrong)


def test_relink_by_path_and_fingerprint(tmp_path):
    project, video = _film(tmp_path)
    dest = tmp_path / "pack" / "film.vxdub"
    vxdub.write_project(project, dest)
    opened, _doc = vxdub.open_project(dest, tmp_path / "projects")
    assert opened.source.resolve() == video.resolve()

    sibling_dir = tmp_path / "sibling"
    sibling_dir.mkdir()
    sibling = sibling_dir / "clip.mkv"
    sibling.write_bytes(video.read_bytes())
    decoy_dir = tmp_path / "decoy"
    decoy_dir.mkdir()
    decoy = decoy_dir / "clip.mkv"
    decoy.write_bytes(b"a-different-file-at-the-old-path")
    doc = vxdub.read(dest)
    doc.manifest["source"]["path"] = str(decoy)
    doc.manifest["source"]["relative"] = "clip.mkv"
    doc.manifest["source"]["fingerprint"] = vxdub.fingerprint(sibling)
    doc.manifest["source"]["size"] = sibling.stat().st_size
    near = sibling_dir / "film.vxdub"
    _dump(near, doc)
    found, _doc = vxdub.open_project(near, tmp_path / "relinked")
    assert found.source.resolve() == sibling.resolve()

    sibling.write_bytes(b"replaced-after-the-save")
    asked = []

    def locate(saved_path: str) -> Path:
        asked.append(saved_path)
        copy = tmp_path / "located.mkv"
        copy.write_bytes(video.read_bytes())
        return copy

    located, _doc = vxdub.open_project(near, tmp_path / "asked", locate)
    assert asked and Path(asked[0]) == decoy
    assert located.source.resolve() == (tmp_path / "located.mkv").resolve()

    def wrong(_saved: str) -> Path:
        return sibling

    with pytest.raises(vxdub.FingerprintMismatch):
        vxdub.open_project(near, tmp_path / "bad", wrong)
    with pytest.raises(vxdub.SourceMissing):
        vxdub.resolve_source(vxdub.read(near).manifest["source"], near)


def _dump(path: Path, document: vxdub.Document) -> None:
    """Rewrite a document without going through the project (tests patch source fields)."""
    vxdub._atomic_write(path, lambda tmp: vxdub._write_zip(tmp, document))


def test_atomic_write_keeps_the_previous_file(tmp_path, monkeypatch):
    project, _video = _film(tmp_path)
    dest = tmp_path / "film.vxdub"
    vxdub.write_project(project, dest)
    good = dest.read_bytes()
    seen = []

    def fail(src, dst):
        seen.append((Path(src), Path(dst)))
        raise OSError("disk full")

    monkeypatch.setattr(vxdub.os, "replace", fail)
    project.lines[0].text = "changed"
    with pytest.raises(OSError, match="disk full"):
        vxdub.write_project(project, dest)
    assert dest.read_bytes() == good
    assert seen and seen[0][1] == dest and seen[0][0].parent == dest.parent
    assert list(dest.parent.glob(".film.vxdub.*.tmp")) == []


def test_installer_registers_vxdub_and_removes_it():
    iss = (ROOT / "installer" / "VoxprintMovieDubber.iss").read_text(encoding="utf-8")
    icon = ROOT / "installer" / "vxdub.ico"
    assert icon.is_file() and icon.read_bytes()[:4] == b"\x00\x00\x01\x00"
    assert 'ValueData: "Voxprint.MovieDubber.Project"' in iss
    assert 'ValueData: "Voxprint dubbing project"' in iss
    assert "Software\\Classes\\.vxdub" in iss
    assert "uninsdeletekey" in iss
    assert 'Source: "vxdub.ico"' in iss
    assert '"{app}\\main.py" "%1"' in iss or '{app}\\main.py' in iss and "%1" in iss


def test_cli_accepts_a_vxdub_as_the_first_argument():
    from main import project_file_from_argv

    assert project_file_from_argv(["movie.vxdub"]) == "movie.vxdub"
    assert project_file_from_argv(["Movie.VXDUB"]) == "Movie.VXDUB"
    assert project_file_from_argv(["--diagnose"]) is None
    assert project_file_from_argv(["clip.mkv"]) is None
    assert project_file_from_argv(["--run-project", "movie.vxdub"]) is None
    text = (ROOT / "main.py").read_text(encoding="utf-8")
    assert "project_file_from_argv(argv)" in text and "open_vxdub" in text


def test_file_menu_saves_and_opens(tmp_path, clip):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from dubber.pipeline import stages as stages
    from dubber.ui.window import MainWindow

    QApplication.instance() or QApplication([])
    window = MainWindow(cfg=stages.MOCK_CFG)
    assert window.btn_file.text() == "File"
    assert window.act_open.text() == "Open…"
    assert window.act_save.text() == "Save"
    assert window.act_save_as.text() == "Save As…"
    assert not window.act_save.isEnabled()
    window.set_source(clip)
    dest = tmp_path / "clip.vxdub"
    assert window.save_vxdub(dest)
    target = window.project.settings["target_lang"]
    window.close()

    again = MainWindow(cfg=stages.MOCK_CFG)
    notes = []
    assert again.open_vxdub(dest, notify=lambda title, text: notes.append((title, text)))
    assert notes == []
    assert again.project is not None and again.project.source.resolve() == clip.resolve()
    assert again.project.settings["target_lang"] == target
    assert again.vxdub_path == dest
    again.close()
