"""Films waiting to be dubbed, one after another.

One film is running. The others wait. The list is a JSON file in the app
state directory, so it is still there after the window closes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_STATUSES = ("queued", "running", "done", "failed", "cancelled")
_ADVANCE = ("done", "failed")


@dataclass(frozen=True)
class QueueItem:
    """One film and where it is in the queue."""

    path: str
    status: str


class DubQueue:
    """A saved list of films. Only one of them is running."""

    def __init__(self, path: Path) -> None:
        """Load the list at ``path``, or start empty when the file is missing.

        Args:
            path: JSON file under the app state directory.
        """
        self._path = path
        self._items: list[QueueItem] = []
        self.load()

    def load(self) -> None:
        """Read the list from disk. A missing or broken file starts empty."""
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self._items = []
            return
        rows = raw.get("items") if isinstance(raw, dict) else None
        parsed: list[QueueItem] = []
        if isinstance(rows, list):
            running = False
            for row in rows:
                if not isinstance(row, dict):
                    continue
                film = row.get("path")
                status = row.get("status")
                if not isinstance(film, str) or not film or status not in _STATUSES:
                    continue
                if status == "running" and running:
                    status = "queued"
                running = running or status == "running"
                parsed.append(QueueItem(film, status))
        self._items = parsed

    def save(self) -> None:
        """Write the list to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"items": [{"path": item.path, "status": item.status} for item in self._items]}
        text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        self._path.write_text(text, encoding="utf-8")

    def items(self) -> tuple[QueueItem, ...]:
        """Return the films in list order."""
        return tuple(self._items)

    def waiting(self) -> int:
        """Return how many films are still waiting."""
        return sum(1 for item in self._items if item.status == "queued")

    def running(self) -> str | None:
        """Return the film that is dubbing, or None."""
        for item in self._items:
            if item.status == "running":
                return item.path
        return None

    def add(self, film: Path) -> bool:
        """Append ``film`` when it is not already waiting or running.

        Args:
            film: A video file.

        Returns:
            True when the film was added.
        """
        resolved = str(film.resolve())
        for item in self._items:
            if item.path == resolved and item.status in {"queued", "running"}:
                return False
        self._items.append(QueueItem(resolved, "queued"))
        self.save()
        return True

    def remove(self, film: str) -> bool:
        """Drop a film that is not the one currently running.

        Args:
            film: The stored path.

        Returns:
            True when a row was removed.
        """
        kept: list[QueueItem] = []
        removed = False
        for item in self._items:
            if item.path == film and item.status != "running":
                removed = True
                continue
            kept.append(item)
        if removed:
            self._items = kept
            self.save()
        return removed

    def activate(self, film: Path) -> bool:
        """Make ``film`` the running job.

        Args:
            film: The video the Dub button is starting.

        Returns:
            False when a different film is already running.
        """
        resolved = str(film.resolve())
        current = self.running()
        if current is not None and current != resolved:
            return False
        updated: list[QueueItem] = []
        found = False
        for item in self._items:
            if item.path == resolved and item.status in {"queued", "running"}:
                updated.append(QueueItem(resolved, "running"))
                found = True
            else:
                updated.append(item)
        if not found:
            updated.append(QueueItem(resolved, "running"))
        self._items = updated
        self.save()
        return True

    def begin_next(self) -> str | None:
        """Start the next waiting film.

        Returns:
            The film that is now running, or None when one is already running
            or nothing is waiting.
        """
        if self.running() is not None:
            return None
        for index, item in enumerate(self._items):
            if item.status != "queued":
                continue
            self._items[index] = QueueItem(item.path, "running")
            self.save()
            return item.path
        return None

    def finish(self, status: str) -> None:
        """Mark the running film ``status`` and leave the rest waiting.

        Args:
            status: ``done``, ``failed``, or ``cancelled``.

        Raises:
            ValueError: ``status`` is not one of those three.
        """
        if status not in {"done", "failed", "cancelled"}:
            raise ValueError(f"unknown queue status {status}")
        self._items = [
            QueueItem(item.path, status) if item.status == "running" else item
            for item in self._items
        ]
        self.save()

    def complete(self, status: str) -> str | None:
        """Finish the running film and start the next one when it should continue.

        A cancelled film stops the queue. A finished or failed film starts the
        next waiting film, still one at a time.

        Args:
            status: ``done``, ``failed``, or ``cancelled``.

        Returns:
            The film that is now running, or None when the queue stays idle.
        """
        if self.running() is None:
            if status not in {"done", "failed", "cancelled"}:
                raise ValueError(f"unknown queue status {status}")
            return None
        self.finish(status)
        if status not in _ADVANCE:
            return None
        return self.begin_next()
