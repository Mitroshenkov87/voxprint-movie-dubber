"""Windows specifics (guarded by ``sys.platform``): minimum build check and the Acrylic/dark window backdrop.

Same approach as Voxprint Audiobook Builder: only native DWM through ctypes (no GPL frameless-window libraries).  Every function is a
safe no-op on other platforms so the Linux port keeps working.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional

IS_WINDOWS = sys.platform == "win32"
MIN_BUILD = 26100            # Windows 11 24H2
BACKDROP_MIN_BUILD = 22621   # DWMWA_SYSTEMBACKDROP_TYPE appeared in 22H2
DWMWA_USE_IMMERSIVE_DARK_MODE, DWMWA_WINDOW_CORNER_PREFERENCE, DWMWA_SYSTEMBACKDROP_TYPE = 20, 33, 38
DWMSBT_TRANSIENTWINDOW = 3     # Acrylic
DWMWCP_ROUND = 2


@dataclass(frozen=True)
class OsCheck:
    ok: bool
    build: Optional[int]
    message: str


def windows_build() -> Optional[int]:
    if not IS_WINDOWS:
        return None
    try:
        return int(sys.getwindowsversion().build)          # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return None


def check_os(build: Optional[int] = None, is_windows: Optional[bool] = None) -> OsCheck:
    """Soft check, never raises."""
    win = IS_WINDOWS if is_windows is None else is_windows
    if not win:
        return OsCheck(False, None, "not Windows")
    b = build if build is not None else windows_build()
    if b is None:
        return OsCheck(True, None, "")
    if b < MIN_BUILD:
        return OsCheck(False, b, f"Windows build {b} is older than {MIN_BUILD}")
    return OsCheck(True, b, "")


def apply_backdrop(hwnd: int, dark: bool = True) -> str:
    """Enable Acrylic through ``DwmSetWindowAttribute``; returns ``'acrylic'`` or ``'plain'`` (older Windows / any error)."""
    if not IS_WINDOWS or (windows_build() or 0) < BACKDROP_MIN_BUILD:
        return "plain"
    try:
        import ctypes
        from ctypes import wintypes

        dwm = ctypes.windll.dwmapi                              # type: ignore[attr-defined]
        h = wintypes.HWND(hwnd)

        def _set(attr: int, value: int) -> int:
            v = ctypes.c_int(value)
            return dwm.DwmSetWindowAttribute(h, ctypes.c_uint(attr), ctypes.byref(v), ctypes.sizeof(v))

        _set(DWMWA_USE_IMMERSIVE_DARK_MODE, 1 if dark else 0)
        _set(DWMWA_WINDOW_CORNER_PREFERENCE, DWMWCP_ROUND)

        class MARGINS(ctypes.Structure):
            _fields_ = [("l", ctypes.c_int), ("r", ctypes.c_int), ("t", ctypes.c_int), ("b", ctypes.c_int)]

        dwm.DwmExtendFrameIntoClientArea(h, ctypes.byref(MARGINS(-1, -1, -1, -1)))     # glass over the whole client area
        return "acrylic" if _set(DWMWA_SYSTEMBACKDROP_TYPE, DWMSBT_TRANSIENTWINDOW) == 0 else "plain"
    except Exception:  # noqa: BLE001
        return "plain"
