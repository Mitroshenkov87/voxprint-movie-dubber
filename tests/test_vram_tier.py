"""VRAM-aware model tiers (suite rule, COLLABORATION.md section 3)."""
from __future__ import annotations

from dubber import models
from dubber.infra import vram_tier as vt


def test_tier_is_picked_from_the_card():
    assert vt.detect(16.0) == vt.TIER_16          # RTX 4090 Laptop, 4080
    assert vt.detect(23.99) == vt.TIER_24         # RTX 4090 reports a little under 24 GiB
    assert vt.detect(31.5) == vt.TIER_24          # RTX 5090
    assert vt.detect(0.0) == vt.TIER_16           # unknown size: the smaller tier


def test_override_can_only_pick_a_tier_the_card_supports():
    assert vt.choices(16.0) == [vt.TIER_16]
    assert vt.choices(24.0) == [vt.TIER_16, vt.TIER_24]
    assert vt.resolve(16.0, vt.TIER_24) == vt.TIER_16
    assert vt.resolve(24.0, vt.TIER_16) == vt.TIER_16
    assert vt.resolve(24.0, "auto") == vt.TIER_24
    assert vt.resolve(24.0, "nonsense") == vt.TIER_24


def test_batch_cap_follows_the_tier():
    assert vt.batch_cap(vt.TIER_16) < vt.batch_cap(vt.TIER_24) == 12
    assert vt.batch_cap("") == vt.batch_cap(vt.TIER_16)


def test_models_the_card_cannot_run_are_not_offered():
    assert vt.tts_models(16.0) == ["tts_1_7b", "tts_0_6b"]
    assert vt.tts_models(8.0) == ["tts_0_6b"]                 # RTX 4060 8 GB: 1.7B does not fit with 2 GB headroom
    assert vt.tts_model_for(8.0, "tts_1_7b") == "tts_0_6b"
    assert vt.tts_model_for(24.0, "tts_0_6b") == "tts_0_6b"
    assert vt.tts_models(0.0) == list(vt.TTS_ORDER)           # unknown VRAM: nothing is hidden
    assert "none" in vt.separation_choices(8.0) and "tiger" in vt.separation_choices(16.0)


def test_installer_fetches_a_tts_model_the_card_can_run():
    small = vt.install_models(8.0, models.INSTALL_MODELS)
    assert "tts_0_6b" in small and "tts_1_7b" not in small
    big = vt.install_models(24.0, models.INSTALL_MODELS)
    assert big == list(models.INSTALL_MODELS)
    assert "VRAM tier 24gb (auto" in vt.describe(24.0)
