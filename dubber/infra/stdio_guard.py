"""Stdout / stderr that cannot raise (copied from Voxprint AI Audiobook Builder ``infra/stdio_guard.py``).

The installed program starts with ``pythonw.exe`` (GUI subsystem): ``sys.stdout`` may be None or a handle whose ``write`` fails
with ``OSError`` errno 22.  Such an error must never escape (it would end the program with a traceback window).
"""
from __future__ import annotations

import sys
from typing import Optional, TextIO


class GuardedStream:
    def __init__(self, inner: Optional[TextIO]) -> None:
        self._inner = inner
        self._dead = inner is None

    def write(self, text: str) -> int:
        if self._dead or self._inner is None:
            return 0
        try:
            return self._inner.write(text)
        except UnicodeEncodeError:              # a cp1251/cp866 console or redirected file: replace the character, keep the stream alive
            try:
                enc = getattr(self._inner, "encoding", None) or "ascii"
                return self._inner.write(text.encode(enc, errors="replace").decode(enc, errors="replace"))
            except Exception:  # noqa: BLE001
                return 0
        except Exception:  # noqa: BLE001 - errno 22 and a missing console must not escape
            self._dead = True
            return 0

    def flush(self) -> None:
        if self._dead or self._inner is None:
            return
        try:
            self._inner.flush()
        except Exception:  # noqa: BLE001
            self._dead = True

    def isatty(self) -> bool:
        try:
            return bool(self._inner and self._inner.isatty())
        except Exception:  # noqa: BLE001
            return False

    def __getattr__(self, name: str):
        if self._inner is None:
            raise AttributeError(name)
        return getattr(self._inner, name)


def guard_stdio() -> None:
    if not isinstance(sys.stdout, GuardedStream):
        sys.stdout = GuardedStream(sys.stdout)  # type: ignore[assignment]
    if not isinstance(sys.stderr, GuardedStream):
        sys.stderr = GuardedStream(sys.stderr)  # type: ignore[assignment]
