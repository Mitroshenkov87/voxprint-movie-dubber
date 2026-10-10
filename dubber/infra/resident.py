"""Keep stage models loaded while they fit in the VRAM budget.

The budget is the memory that is free right now minus ``max(2 GB, 8 % of the card)``
(``dubber.infra.resources.vram_allowance_gb``).  A model stays resident across stages and clips in this process.
Another model is loaded beside it when the sum still fits.  Under pressure the largest other model is unloaded
first, and only then.  With no NVIDIA GPU there is no VRAM pressure, so models stay until :func:`reset`.
"""
from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

_enabled = False
_slots: Dict[str, Any] = {}
_sizes: Dict[str, float] = {}
_loaders: Dict[str, Callable[[], Any]] = {}
_noted: List[str] = []
_lock = threading.Lock()


def enable() -> None:
    """Turn the cache on for this process (the dubbing session worker)."""
    global _enabled
    _enabled = True


def enabled() -> bool:
    """Return whether the resident-model cache is on in this process."""
    return _enabled


def disable() -> None:
    """Turn the cache off.  Tests use this so a later test does not keep a model."""
    global _enabled
    _enabled = False


def reset() -> None:
    """Unload everything held here.  Used at the end of a session and by tests."""
    with _lock:
        for name in list(_slots):
            _discard(name)
        _noted.clear()


def loaded() -> List[str]:
    """Return the names of models currently held in this process."""
    return list(_slots)


def noted() -> List[str]:
    """Models a stage asked to prefetch, whether or not a loader ran."""
    return list(_noted)


def register_loader(name: str, factory: Callable[[], Any]) -> None:
    """Remember ``factory`` as the way to build ``name`` for a later :func:`prefetch`."""
    _loaders[name] = factory


def vram_now() -> Tuple[float, float]:
    """``(free GB, total GB)`` of the first GPU, or ``(0, 0)`` when there is none."""
    from dubber.infra.resources import gpu_from_smi, gpu_from_torch

    g = gpu_from_torch() or gpu_from_smi()
    if not g:
        return 0.0, 0.0
    return float(g[2]), float(g[1])


def unload_plan(loaded_sizes: Sequence[Tuple[str, float]], incoming: str, need_gb: float,
                free_gb: float, total_gb: float) -> List[str]:
    """Names to drop so ``incoming`` fits.  Empty when it already fits, or when there is no GPU.

    Largest residents go first.  The incoming name is never listed.  If it still does not fit after every
    other model is listed, the list is every other model (the new one still has to run)."""
    if total_gb <= 0:
        return []
    from dubber.infra.resources import vram_allowance_gb

    room = vram_allowance_gb(free_gb, total_gb)
    if need_gb <= room + 1e-6:
        return []
    others = sorted(((n, s) for n, s in loaded_sizes if n != incoming), key=lambda t: (-t[1], t[0]))
    drop: List[str] = []
    freed = 0.0
    for name, size in others:
        drop.append(name)
        freed += size
        if need_gb <= room + freed + 1e-6:
            break
    return drop


def _close(obj: Any) -> None:
    if obj is None:
        return
    try:
        obj._kept = False  # noqa: SLF001 - the holder asked for a real unload
    except Exception:  # noqa: BLE001 - not every cached object allows attributes
        pass
    close = getattr(obj, "close", None)
    if close is None:
        return
    try:
        close()
    except Exception:  # noqa: BLE001 - unloading must not hide the stage error that follows
        pass


def _discard(name: str) -> None:
    _close(_slots.pop(name, None))
    _sizes.pop(name, None)


def _mark_kept(obj: Any) -> None:
    try:
        obj._kept = True  # noqa: SLF001
    except Exception:  # noqa: BLE001
        pass


def _take(name: str, need_gb: float, build: Callable[[], Any], free_gb: float, total_gb: float, allow_unload: bool) -> Any:
    held = _slots.get(name)
    if held is not None:
        return held
    plan = unload_plan(list(_sizes.items()), name, need_gb, free_gb, total_gb)
    if plan and not allow_unload:
        return None
    for dropped in plan:
        _discard(dropped)
    obj = build()
    _mark_kept(obj)
    _slots[name] = obj
    _sizes[name] = float(need_gb)
    return obj


def slot(name: str, need_gb: float, build: Callable[[], Any], free_gb: Optional[float] = None,
         total_gb: Optional[float] = None) -> Any:
    """Return the resident object named ``name``, building it once.  A disabled cache just calls ``build``.

    Under pressure the largest other model is unloaded first.  The build runs while the cache lock is held so two
    stages cannot construct the same name twice."""
    if not _enabled:
        return build()
    if free_gb is None or total_gb is None:
        free_gb, total_gb = vram_now()
    with _lock:
        return _take(name, need_gb, build, free_gb, total_gb, True)


def note(name: str) -> None:
    """Remember a non-empty ``name`` as a prefetch request without loading it."""
    if name and name not in _noted:
        _noted.append(name)


def prefetch(name: str, need_gb: float, free_gb: Optional[float] = None, total_gb: Optional[float] = None) -> Any:
    """Load ``name`` if a loader was registered and it fits beside what is already loaded.

    Always records the request.  Unlike :func:`slot`, this never unloads a model: the stage that is running still
    needs its weights.  A miss returns None."""
    note(name)
    factory = _loaders.get(name)
    if factory is None or not _enabled:
        return None
    try:
        if free_gb is None or total_gb is None:
            free_gb, total_gb = vram_now()
        with _lock:
            return _take(name, need_gb, factory, free_gb, total_gb, False)
    except Exception:  # noqa: BLE001 - a prefetch miss must not fail the stage that is already running
        return None
