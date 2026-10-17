"""Shared settings state for the simple screen: presets, custom, reset, and Expert mode."""
from __future__ import annotations

from pathlib import Path

from dubber.ui.shell_state import (
    apply_engine,
    apply_project,
    bind_model_forge,
    build_layout,
    clear_model_forge,
    dub_settings,
    registry_ids,
    resolve_role,
    stop_stage,
)
from voxprint_common.ui.expert import ExpertMode
from voxprint_common.ui.presets import PresetMatch
from voxprint_common.ui.variants import InputKind, ResultKind, VariantId, state_names, surface_for


def _layout(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    clear_model_forge()
    return build_layout(tmp_path / "state")


def test_layout_uses_the_library_audio_surface(tmp_path, monkeypatch):
    surface = surface_for("movie-dubber")
    assert surface.variant is VariantId.AUDIO
    assert surface.input_kind is InputKind.MEDIA_VOICE
    assert surface.result_kind is ResultKind.PLAYER
    settings_name, history_name = state_names(surface)
    assert (settings_name, history_name) == ("audio-settings.json", "audio-history.json")
    layout = _layout(tmp_path, monkeypatch)
    layout.choose("language", "en")
    state = tmp_path / "state"
    assert (state / "audio-settings.json").is_file()
    assert not (state / "video-settings.json").exists()
    source = (Path(__file__).resolve().parents[1] / "dubber" / "ui" / "shell_state.py").read_text(encoding="utf-8")
    window = (Path(__file__).resolve().parents[1] / "dubber" / "ui" / "shell_window.py").read_text(encoding="utf-8")
    assert "AppSurface(" not in source
    assert "AppSurface(" not in window


def test_balanced_is_the_default_and_fast_writes_concrete_values(tmp_path, monkeypatch):
    layout = _layout(tmp_path, monkeypatch)
    status = layout.preset_status("quality")
    assert status is not None and status.kind is PresetMatch.EXACT and status.option_id == "balanced"
    assert layout.get("dub.actor_weight") == 50
    assert layout.get("dub.voice_style") == "actor"
    before = {key: layout.get(key) for key in ("dub.tts_size", "dub.separation", "dub.tts_batch", "dub.mt_context")}
    layout.choose("quality", "fast")
    assert layout.get("dub.tts_size") == "small"
    assert layout.get("dub.separation") == "off"
    assert layout.get("dub.tts_batch") == 8
    assert layout.get("dub.mt_context") == 0
    assert layout.get("dub.tts_size") != before["dub.tts_size"]
    assert layout.preset_status("quality").kind is PresetMatch.EXACT
    assert layout.get("dub.actor_weight") == 50


def test_close_edit_marks_changes_and_reset_restores_the_preset(tmp_path, monkeypatch):
    layout = _layout(tmp_path, monkeypatch)
    original = layout.get("dub.tts_batch")
    layout.set_value("dub.tts_batch", 3)
    status = layout.preset_status("quality")
    assert status is not None
    assert status.kind is PresetMatch.MODIFIED
    assert status.option_id == "balanced"
    layout.reset_group("quality")
    assert layout.get("dub.tts_batch") == original
    assert layout.preset_status("quality").kind is PresetMatch.EXACT
    assert layout.get("dub.actor_weight") == 50


def test_far_edit_marks_custom_and_reset_restores_the_stored_preset(tmp_path, monkeypatch):
    layout = _layout(tmp_path, monkeypatch)
    layout.set_value("dub.tts_batch", 7)
    layout.set_value("dub.mt_context", 7)
    layout.set_value("dub.mix_original", 0.33)
    layout.set_value("dub.tts_size", "small")
    layout.set_value("dub.diarization", "embed")
    status = layout.preset_status("quality")
    assert status is not None and status.kind is PresetMatch.CUSTOM
    assert status.option_id == "balanced"
    layout.reset_group("quality")
    assert layout.preset_status("quality").kind is PresetMatch.EXACT
    assert layout.get("dub.tts_batch") == 4
    assert layout.get("dub.tts_size") == "large"
    assert layout.get("dub.mix_original") == 0.15


def test_expert_mode_switch_keeps_values(tmp_path, monkeypatch):
    layout = _layout(tmp_path, monkeypatch)
    layout.choose("quality", "best")
    layout.choose("voices", "neutral")
    layout.set_value("dub.actor_weight", 80)
    snap = layout.snapshot()
    mode = ExpertMode()
    assert mode.enabled() is False
    mode.set_enabled(True)
    mode.set_enabled(False)
    assert mode.enabled() is False
    assert layout.snapshot() == snap
    assert layout.get("dub.actor_weight") == 80
    assert layout.preset_status("voices").kind is PresetMatch.MODIFIED


def test_model_choices_come_from_the_mdl_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    clear_model_forge()
    ids = registry_ids()
    assert "qwen3-tts" in ids and "sep" in ids and "roformer" in ids
    choices = {item.value for item in dub_settings().get("dub.tts_model").choices}
    assert set(ids) <= choices
    assert all("/" not in value and "\\" not in value for value in choices)
    assert resolve_role("tts") == "qwen3-tts"
    bind_model_forge({"tts": "not-published-yet"})
    assert resolve_role("tts") == "qwen3-tts"
    bind_model_forge({"separation.best": "roformer"})
    assert resolve_role("separation.best") == "roformer"
    clear_model_forge()


def test_engine_mapping_uses_resolved_roles(tmp_path, monkeypatch):
    layout = _layout(tmp_path, monkeypatch)
    layout.choose("quality", "fast")
    layout.choose("voices", "neutral")
    engine = apply_engine(layout)
    project = apply_project(layout)
    assert engine["separation"] == "none"
    assert engine["tts_model"] == "tts_0_6b"
    assert engine["tts_batch"] == 8
    assert project["actor_weight"] == 0.0
    assert project["multi_voice"] is False
    assert project["original_volume"] == 0.15
    layout.choose("quality", "best")
    layout.choose("voices", "keep")
    assert apply_engine(layout)["separation"] == "roformer"
    assert apply_engine(layout)["tts_model"] == "tts_1_7b"
    assert apply_project(layout)["actor_weight"] == 0.5
    layout.choose("voices", "pick")
    assert layout.get("dub.multi_voice") is True
    assert apply_project(layout)["multi_voice"] is True
    assert apply_project(layout)["actor_weight"] == 0.5
    labels = [item.label_key for item in dub_settings().get("dub.tts_model").choices]
    assert "Qwen3-TTS 1.7B Base (voice cloning)" in labels
    assert all("placeholder" not in label.lower() and "заглушка" not in label.lower() for label in labels)


def test_source_language_and_stop_stage_stay_in_the_shared_state(tmp_path, monkeypatch):
    layout = _layout(tmp_path, monkeypatch)
    specs = dub_settings()
    source = specs.get("dub.source_lang")
    until = specs.get("dub.until")
    assert source is not None and source.expert is True and source.default == "auto"
    assert until is not None and until.expert is True and until.default == "mux"
    assert layout.get("dub.source_lang") == "auto"
    assert stop_stage(layout) == "mux"
    assert layout.preset_status("quality").kind is PresetMatch.EXACT
    layout.set_value("dub.source_lang", "en")
    layout.set_value("dub.until", "translation")
    assert apply_project(layout)["source_lang"] == "en"
    assert stop_stage(layout) == "translation"
    assert layout.preset_status("quality").kind is PresetMatch.EXACT
    restored = build_layout(tmp_path / "state")
    assert restored.get("dub.source_lang") == "en"
    assert restored.get("dub.until") == "translation"
    assert stop_stage(restored) == "translation"
