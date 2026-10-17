"""The film queue: one running dub, then the next, saved on disk."""
from __future__ import annotations

from pathlib import Path

import pytest

from dubber.ui.dub_queue import DubQueue


def _films(tmp_path: Path) -> tuple[Path, Path, Path]:
    paths = []
    for name in ("one.mkv", "two.mkv", "three.mkv"):
        path = tmp_path / name
        path.write_bytes(b"x")
        paths.append(path)
    return paths[0], paths[1], paths[2]


def test_one_film_runs_and_the_next_starts_after_it(tmp_path):
    first, second, third = _films(tmp_path)
    queue = DubQueue(tmp_path / "state" / "dub-queue.json")
    assert queue.add(first)
    assert queue.add(second)
    assert queue.add(first) is False
    assert queue.begin_next() == str(first.resolve())
    assert queue.begin_next() is None
    assert queue.running() == str(first.resolve())
    assert queue.waiting() == 1
    assert queue.complete("done") == str(second.resolve())
    assert queue.running() == str(second.resolve())
    assert queue.complete("failed") is None
    assert queue.add(third)
    assert queue.begin_next() == str(third.resolve())
    assert queue.complete("cancelled") is None
    assert queue.running() is None
    assert queue.waiting() == 0


def test_remove_drops_a_waiting_film_and_not_the_running_one(tmp_path):
    first, second, _third = _films(tmp_path)
    queue = DubQueue(tmp_path / "dub-queue.json")
    queue.add(first)
    queue.add(second)
    queue.begin_next()
    assert queue.remove(str(first.resolve())) is False
    assert queue.remove(str(second.resolve())) is True
    assert queue.waiting() == 0
    assert queue.running() == str(first.resolve())


def test_the_queue_is_reloaded_from_the_app_state(tmp_path):
    first, second, _third = _films(tmp_path)
    path = tmp_path / "movie-dubber" / "dub-queue.json"
    queue = DubQueue(path)
    queue.add(first)
    queue.add(second)
    queue.activate(first)
    again = DubQueue(path)
    assert again.running() == str(first.resolve())
    assert again.waiting() == 1
    assert [item.status for item in again.items()] == ["running", "queued"]


def test_a_broken_queue_file_starts_empty(tmp_path):
    path = tmp_path / "dub-queue.json"
    path.write_text("{", encoding="utf-8")
    queue = DubQueue(path)
    assert queue.items() == ()


def test_complete_rejects_an_unknown_status(tmp_path):
    queue = DubQueue(tmp_path / "dub-queue.json")
    with pytest.raises(ValueError):
        queue.complete("paused")
