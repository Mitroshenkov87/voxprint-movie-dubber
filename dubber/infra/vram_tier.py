"""VRAM-aware model tiers (suite rule, project-notes suite/COLLABORATION.md section 3).

The installer and the program detect the card's VRAM (at install and at every start) and pick the matching tier on their own:

* ``8gb``: below 15.5 GiB;
* ``16gb``: from 15.5 GiB up to but not including 23.5 GiB (RTX 4060 Ti 16 GB, 4080, RTX 4090 Laptop ...);
* ``24gb``: 23.5 GiB and above (RTX 4090, RTX 5090 ...).

The cut points come from ``voxprint_common.gpu_probe``. Batch caps stay in this module.

A tier is a set of caps, not a different pipeline: how many lines one TTS generate call may take (the measured VRAM
budget still decides below that cap) and which models are offered at all.  A model is offered only when it fits the
card with the suite headroom (``max(2 GB, 8 %)``, :mod:`dubber.infra.resources`) left free, so options the card cannot
run are disabled, never offered.  Nothing here is a user mode: Settings has an optional advanced override that can only
pick a tier the card supports.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from voxprint_common.gpu_probe import tier_for_gib

AUTO = "auto"
TIER_8 = "8gb"
TIER_16 = "16gb"
TIER_24 = "24gb"


@dataclass(frozen=True)
class Tier:
    """One VRAM tier and the maximum number of TTS lines allowed in a single generate call."""
    key: str
    title: str
    tts_batch_cap: int             # lines per TTS generate call at most; the VRAM budget decides below that


TIERS: Dict[str, Tier] = {t.key: t for t in (
    Tier(TIER_8, "8 GB", 2),
    Tier(TIER_16, "16 GB", 6),
    Tier(TIER_24, "24 GB or more", 12),
)}
#: preferred TTS models, best first
TTS_ORDER: Tuple[str, ...] = ("tts_1_7b", "tts_0_6b")


def detect(total_gb: float) -> str:
    """The tier for a card with ``total_gb`` of VRAM. Unknown, zero, or NaN is ``unknown``."""
    size = float(total_gb or 0.0)
    return tier_for_gib(None if size <= 0 else size).value


def choices(total_gb: float) -> List[str]:
    """Tiers the advanced override may pick: the detected one and the ones below it.

    An unknown card may pick any tier.
    """
    keys = [TIER_8, TIER_16, TIER_24]
    found = detect(total_gb)
    if found not in keys:
        return keys
    return keys[: keys.index(found) + 1]


def resolve(total_gb: float, override: Optional[str] = AUTO) -> str:
    """The tier in use: the override when the card supports it, otherwise the detected tier."""
    if override and override != AUTO and override in choices(total_gb):
        return override
    return detect(total_gb)


def batch_cap(tier: Optional[str]) -> int:
    """Return the TTS line cap for ``tier``, using the 16 GB cap when the tier is unknown."""
    t = TIERS.get(str(tier or ""))
    return t.tts_batch_cap if t else TIERS[TIER_16].tts_batch_cap


def _usable_gb(total_gb: float) -> float:
    from dubber.infra.resources import vram_headroom_gb

    return max(0.0, float(total_gb) - vram_headroom_gb(float(total_gb)))


def tts_need_gb(model: str) -> float:
    """VRAM one TTS model needs to speak at least one line: weights + workspace + one line."""
    from dubber.infra.resources import MODEL_VRAM_GB, TTS_ACT_RESERVE_GB, TTS_ITEM_VRAM_GB

    return MODEL_VRAM_GB.get(model, 5.0) + TTS_ACT_RESERVE_GB + TTS_ITEM_VRAM_GB


def tts_models(total_gb: float) -> List[str]:
    """TTS models this card can run with the headroom left free (all of them when the VRAM is unknown)."""
    if not total_gb:
        return list(TTS_ORDER)
    room = _usable_gb(total_gb)
    return [m for m in TTS_ORDER if tts_need_gb(m) <= room]


def separation_choices(total_gb: float) -> List[str]:
    """Separation models this card can run (``none`` is always possible)."""
    from dubber.infra.resources import MODEL_VRAM_GB

    room = _usable_gb(total_gb) if total_gb else float("inf")
    out = [k for k in ("tiger", "roformer") if MODEL_VRAM_GB.get("separation", 1.6) <= room]
    return out + ["none"]


def tts_model_for(total_gb: float, preferred: str) -> str:
    """``preferred`` when the card can run it, else the best model it can run (the lightest one as the last resort)."""
    fits = tts_models(total_gb)
    if preferred in fits:
        return preferred
    return fits[0] if fits else TTS_ORDER[-1]


def install_models(total_gb: float, keys: Tuple[str, ...]) -> List[str]:
    """The installer's model list with the TTS model swapped for one the card can run."""
    out: List[str] = []
    for k in keys:
        m = tts_model_for(total_gb, k) if k in TTS_ORDER else k
        if m not in out:
            out.append(m)
    return out


def describe(total_gb: float, override: Optional[str] = AUTO) -> str:
    """Return one line naming the tier in use, the card size in gigabytes, the TTS batch cap, and the TTS models that fit."""
    tier = resolve(total_gb, override)
    how = "override" if override and override != AUTO and tier == override else "auto"
    size = f"{total_gb:.1f} GB" if total_gb else "unknown"
    return (f"VRAM tier {tier} ({how}; card {size}): TTS batch up to {batch_cap(tier)}, "
            f"TTS models {', '.join(tts_models(total_gb)) or 'none'}")
