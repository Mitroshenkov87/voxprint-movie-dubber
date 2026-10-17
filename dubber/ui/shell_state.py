"""One settings state for the simple screen and the expert panel.

Presets write concrete values. A close hand edit reads as the preset plus
changes. A far edit reads as Custom. Reset restores the stored preset.
Expert mode is a suite flag and does not touch this state.

Model ids come from ``mdl.json``. The table below is a placeholder until
Model Forge publishes role bindings. Call :func:`bind_model_forge` with that
map. Widgets never name a model id or a model path.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from voxprint_common.mdl.catalog import load_manifest
from voxprint_common.ui.presets import PresetFile, PresetGroup, PresetOption
from voxprint_common.ui.settings import Choice, SettingSpec, Settings, SettingType
from voxprint_common.ui.simple_image import SimpleLayout
from voxprint_common.ui.variants import state_names, surface_for

from dubber.appinfo import resource_dir

# Placeholder role -> current mdl.json id. Model Forge replaces this map.
PLACEHOLDER_ROLES: dict[str, str] = {
    "tts": "qwen3-tts",
    "asr": "asr",
    "separation": "sep",
    "separation.best": "roformer",
    "diarization": "diar",
    "embedding": "embed",
}

_FORGE: dict[str, str] = {}

_QUALITY_ORDER = (
    "dub.tts_model",
    "dub.tts_size",
    "dub.separation",
    "dub.diarization",
    "dub.tts_backend",
    "dub.tts_batch",
    "dub.mt_context",
    "dub.mix_original",
)
_VOICE_ORDER = ("dub.voice_style", "dub.actor_weight", "dub.multi_voice")
_BACKENDS = {"auto": "auto", "standard": "standard/sdpa", "graphs": "graphs/sdpa"}


class DubReview:
    """Let a dub start. This app does not review pictures, and the gate stays on."""

    def review_prompt(self, prompt: str) -> str | None:
        """Allow ``prompt``. A video path is not a picture request.

        Args:
            prompt: The chosen video path.

        Returns:
            None. The dub may start.
        """
        del prompt
        return None

    def review_result(self, prompt: str, item_id: str) -> str | None:
        """Allow the finished dub into the history strip.

        Args:
            prompt: The video that was dubbed.
            item_id: History id of the result.

        Returns:
            None. The result may be kept.
        """
        del prompt, item_id
        return None


def manifest_path() -> Path:
    """Return this app's ``mdl.json``."""
    return resource_dir() / "mdl.json"


def bind_model_forge(outputs: Mapping[str, str]) -> None:
    """Install Model Forge role bindings.

    An id that is not in the MDL registry is ignored. An empty map restores
    the placeholders.

    Args:
        outputs: Role name to model id.
    """
    _FORGE.clear()
    _FORGE.update(outputs)


def clear_model_forge() -> None:
    """Drop Model Forge bindings so roles use the placeholder ids again."""
    _FORGE.clear()


def model_forge_outputs() -> dict[str, str]:
    """Return the Model Forge map. It is empty until Forge is plugged in."""
    return dict(_FORGE)


def registry_ids() -> tuple[str, ...]:
    """Return model ids from ``mdl.json``, in file order, without duplicates."""
    return tuple(model_id for model_id, _title in registry_entries())


def registry_entries() -> tuple[tuple[str, str], ...]:
    """Return ``(id, display name)`` from ``mdl.json``, first tier only."""
    manifest = load_manifest(manifest_path())
    found: dict[str, str] = {}
    for model in manifest.models:
        if model.id not in found:
            found[model.id] = model.display_name or model.id
    return tuple(found.items())


def resolve_role(role: str) -> str:
    """Return the model id for ``role``.

    A Forge id wins when it is in the registry. Otherwise the placeholder id
    is used. The placeholder must be in the registry.

    Args:
        role: A name from :data:`PLACEHOLDER_ROLES`.

    Raises:
        KeyError: ``role`` has no placeholder.
        ValueError: The placeholder id is not in ``mdl.json``.
    """
    ids = set(registry_ids())
    forged = _FORGE.get(role, "")
    if forged and forged in ids:
        return forged
    placeholder = PLACEHOLDER_ROLES[role]
    if placeholder not in ids:
        raise ValueError(f"placeholder {placeholder} for {role} is not in the MDL registry")
    return placeholder


def model_label_key(model_id: str) -> str:
    """Return the catalog key for a registry id. ``off`` is not a model."""
    if model_id == "off":
        return "model.off"
    return "model." + model_id.replace("-", "_")


def dub_settings() -> Settings:
    """Return the descriptors the chips and the expert panel share."""
    tts = resolve_role("tts")
    separation = resolve_role("separation")
    diarization = resolve_role("diarization")
    models = _model_choices()
    models_or_off = _model_choices("off")
    return Settings(
        (
            SettingSpec(
                "dub.target_lang",
                "shell.language",
                "shell.language.help",
                SettingType.ENUM,
                "ru",
                choices=(
                    Choice("ru", "shell.language.ru"),
                    Choice("en", "shell.language.en"),
                    Choice("de", "shell.language.de"),
                ),
            ),
            SettingSpec(
                "dub.voice_style",
                "shell.voice",
                "shell.voice.help",
                SettingType.ENUM,
                "actor",
                choices=(
                    Choice("actor", "shell.voice.keep"),
                    Choice("neutral", "shell.voice.neutral"),
                ),
            ),
            SettingSpec(
                "dub.actor_weight",
                "setting.dub.actor_weight",
                "setting.dub.actor_weight.help",
                SettingType.FLOAT,
                0.5,
                expert=True,
                group="voices",
                minimum=0.0,
                maximum=1.0,
            ),
            SettingSpec(
                "dub.multi_voice",
                "setting.dub.multi",
                "setting.dub.multi.help",
                SettingType.BOOL,
                False,
                expert=True,
                group="voices",
            ),
            SettingSpec(
                "dub.tts_model",
                "setting.dub.tts_model",
                "setting.dub.tts_model.help",
                SettingType.ENUM,
                tts,
                expert=True,
                group="speech",
                choices=models,
            ),
            SettingSpec(
                "dub.tts_size",
                "setting.dub.tts_size",
                "setting.dub.tts_size.help",
                SettingType.ENUM,
                "large",
                expert=True,
                group="speech",
                choices=(
                    Choice("small", "setting.dub.tts_size.small"),
                    Choice("large", "setting.dub.tts_size.large"),
                ),
            ),
            SettingSpec(
                "dub.tts_backend",
                "setting.dub.tts_backend",
                "setting.dub.tts_backend.help",
                SettingType.ENUM,
                "auto",
                expert=True,
                group="speech",
                choices=(
                    Choice("auto", "setting.dub.tts_backend.auto"),
                    Choice("standard", "setting.dub.tts_backend.standard"),
                    Choice("graphs", "setting.dub.tts_backend.graphs"),
                ),
            ),
            SettingSpec(
                "dub.tts_batch",
                "setting.dub.tts_batch",
                "setting.dub.tts_batch.help",
                SettingType.INT,
                4,
                expert=True,
                group="speech",
                minimum=1,
                maximum=16,
            ),
            SettingSpec(
                "dub.separation",
                "setting.dub.separation",
                "setting.dub.separation.help",
                SettingType.ENUM,
                separation,
                expert=True,
                group="sound",
                choices=models_or_off,
            ),
            SettingSpec(
                "dub.diarization",
                "setting.dub.diarization",
                "setting.dub.diarization.help",
                SettingType.ENUM,
                diarization,
                expert=True,
                group="sound",
                choices=models_or_off,
            ),
            SettingSpec(
                "dub.mix_original",
                "setting.dub.mix",
                "setting.dub.mix.help",
                SettingType.FLOAT,
                0.15,
                expert=True,
                group="sound",
                minimum=0.0,
                maximum=1.0,
            ),
            SettingSpec(
                "dub.mt_context",
                "setting.dub.mt_context",
                "setting.dub.mt_context.help",
                SettingType.INT,
                2,
                expert=True,
                group="words",
                minimum=0,
                maximum=8,
            ),
            SettingSpec(
                "dub.subtitles",
                "setting.dub.subtitles",
                "setting.dub.subtitles.help",
                SettingType.ENUM,
                "auto",
                expert=True,
                group="words",
                choices=(
                    Choice("auto", "setting.dub.subtitles.auto"),
                    Choice("none", "setting.dub.subtitles.none"),
                ),
            ),
            SettingSpec(
                "dub.profanity",
                "setting.dub.profanity",
                "setting.dub.profanity.help",
                SettingType.ENUM,
                "keep",
                expert=True,
                group="words",
                choices=(
                    Choice("keep", "setting.dub.profanity.keep"),
                    Choice("soften", "setting.dub.profanity.soften"),
                ),
            ),
            SettingSpec(
                "dub.output_format",
                "setting.dub.output",
                "setting.dub.output.help",
                SettingType.ENUM,
                "mkv",
                expert=True,
                group="output",
                choices=(
                    Choice("mkv", "setting.dub.output.mkv"),
                    Choice("mp4", "setting.dub.output.mp4"),
                ),
            ),
            SettingSpec(
                "dub.device",
                "setting.dub.device",
                "setting.dub.device.help",
                SettingType.ENUM,
                "auto",
                expert=True,
                group="output",
                choices=(
                    Choice("auto", "setting.dub.device.auto"),
                    Choice("cuda", "setting.dub.device.cuda"),
                ),
            ),
            SettingSpec(
                "dub.vram_tier",
                "setting.dub.vram",
                "setting.dub.vram.help",
                SettingType.ENUM,
                "auto",
                expert=True,
                group="output",
                choices=(
                    Choice("auto", "setting.dub.vram.auto"),
                    Choice("8gb", "setting.dub.vram.8gb"),
                    Choice("16gb", "setting.dub.vram.16gb"),
                    Choice("24gb", "setting.dub.vram.24gb"),
                ),
            ),
            SettingSpec(
                "dub.subdl_key",
                "setting.dub.subdl",
                "setting.dub.subdl.help",
                SettingType.TEXT,
                "",
                expert=True,
                group="accounts",
            ),
            SettingSpec(
                "dub.opensubtitles_key",
                "setting.dub.opensubtitles",
                "setting.dub.opensubtitles.help",
                SettingType.TEXT,
                "",
                expert=True,
                group="accounts",
            ),
        )
    )


def dub_presets(settings: Settings | None = None) -> PresetFile:
    """Return Quality, Voices, and Language. Each option writes concrete values.

    Args:
        settings: Unused. Present so callers can pass the descriptor list.
    """
    del settings
    tts = resolve_role("tts")
    separate = resolve_role("separation")
    separate_best = resolve_role("separation.best")
    diar = resolve_role("diarization")
    quality = PresetGroup(
        "quality",
        "shell.quality",
        (
            _option(
                "fast",
                "shell.quality.fast",
                _quality(tts, "small", "off", "off", "standard", 8, 0, 0.15),
            ),
            _option(
                "balanced",
                "shell.quality.balanced",
                _quality(tts, "large", separate, diar, "auto", 4, 2, 0.15),
            ),
            _option(
                "best",
                "shell.quality.best",
                _quality(tts, "large", separate_best, diar, "graphs", 1, 4, 0.15),
            ),
        ),
        "balanced",
    )
    voices = PresetGroup(
        "voices",
        "shell.voice",
        (
            _option("keep", "shell.voice.keep", _voice("actor", 0.5, False)),
            _option("neutral", "shell.voice.neutral", _voice("neutral", 0.0, False)),
        ),
        "keep",
    )
    language = PresetGroup(
        "language",
        "shell.language",
        (
            _option("ru", "shell.language.ru", (("dub.target_lang", "ru"),)),
            _option("en", "shell.language.en", (("dub.target_lang", "en"),)),
            _option("de", "shell.language.de", (("dub.target_lang", "de"),)),
        ),
        "ru",
    )
    return PresetFile((quality, voices, language))


def build_layout(state_dir: Path | None) -> SimpleLayout:
    """Build the dub screen state. ``state_dir`` None keeps it in memory.

    The widget uses :func:`surface_for` ``movie-dubber``. That surface is the
    audio variant, so the files are ``audio-settings.json`` and
    ``audio-history.json``.

    Args:
        state_dir: App state directory, or None for a memory-only state.
    """
    settings = dub_settings()
    settings_file, history_file = state_names(surface_for("movie-dubber"))
    return SimpleLayout(
        presets=dub_presets(settings),
        settings=settings,
        state_dir=state_dir,
        gate=DubReview(),
        state_name=settings_file,
        history_name=history_file,
    )


def _as_float(value: object) -> float:
    """Return ``value`` as a float.

    Args:
        value: A settings number. Booleans are rejected.

    Raises:
        TypeError: ``value`` is not a number.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("setting value is not a number")
    return float(value)


def _as_int(value: object) -> int:
    """Return ``value`` as a whole number.

    Args:
        value: A settings number. Booleans and fractions are rejected.

    Raises:
        TypeError: ``value`` is not a whole number.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("setting value is not a whole number")
    return value


def apply_project(layout: SimpleLayout) -> dict[str, object]:
    """Return project settings written from the live state.

    ``mt_context`` is stored on the project. The translation engine does not
    read it yet.

    Args:
        layout: Screen state after presets and any hand edits.
    """
    style = str(layout.get("dub.voice_style"))
    weight = _as_float(layout.get("dub.actor_weight"))
    return {
        "target_lang": str(layout.get("dub.target_lang")),
        "multi_voice": bool(layout.get("dub.multi_voice")),
        "actor_weight": weight,
        "original_volume": _as_float(layout.get("dub.mix_original")),
        "profanity": str(layout.get("dub.profanity")),
        "subtitle_choice": str(layout.get("dub.subtitles")),
        "output_format": str(layout.get("dub.output_format")),
        "mt_context": _as_int(layout.get("dub.mt_context")),
        "audio_track_auto": True,
        "single_voice": {
            "kind": "clone" if style == "actor" else "library",
            "id": "",
            "actor_weight": weight,
        },
    }


def apply_engine(layout: SimpleLayout) -> dict[str, object]:
    """Return engine config keys written from the live state.

    ``tts_batch`` is included for a later engine. The current pipeline still
    chooses the speech batch from the VRAM tier.

    Args:
        layout: Screen state after presets and any hand edits.
    """
    separation_id = str(layout.get("dub.separation"))
    diarization_id = str(layout.get("dub.diarization"))
    if separation_id == "off":
        separation = "none"
    elif separation_id == resolve_role("separation.best"):
        separation = "roformer"
    else:
        separation = "tiger"
    diarization = "pyannote" if diarization_id == resolve_role("diarization") else "cluster"
    backend = _BACKENDS[str(layout.get("dub.tts_backend"))]
    size = str(layout.get("dub.tts_size"))
    return {
        "separation": separation,
        "diarization": diarization,
        "tts_model": "tts_0_6b" if size == "small" else "tts_1_7b",
        "tts_backend": backend,
        "tts_batch": _as_int(layout.get("dub.tts_batch")),
        "device": str(layout.get("dub.device")),
        "vram_tier": str(layout.get("dub.vram_tier")),
    }


def _model_choices(*extra: str) -> tuple[Choice, ...]:
    rows: list[Choice] = []
    seen: set[str] = set()
    for model_id in extra:
        if model_id in seen:
            continue
        rows.append(Choice(model_id, model_label_key(model_id)))
        seen.add(model_id)
    for model_id, _title in registry_entries():
        if model_id in seen:
            continue
        rows.append(Choice(model_id, model_label_key(model_id)))
        seen.add(model_id)
    return tuple(rows)


def _quality(
    tts: str,
    size: str,
    separation: str,
    diarization: str,
    backend: str,
    batch: int,
    context: int,
    mix: float,
) -> tuple[tuple[str, str | int | float], ...]:
    values: dict[str, str | int | float] = {
        "dub.tts_model": tts,
        "dub.tts_size": size,
        "dub.separation": separation,
        "dub.diarization": diarization,
        "dub.tts_backend": backend,
        "dub.tts_batch": batch,
        "dub.mt_context": context,
        "dub.mix_original": mix,
    }
    return tuple((key, values[key]) for key in _QUALITY_ORDER)


def _voice(style: str, weight: float, multi: bool) -> tuple[tuple[str, str | float | bool], ...]:
    values: dict[str, str | float | bool] = {
        "dub.voice_style": style,
        "dub.actor_weight": weight,
        "dub.multi_voice": multi,
    }
    return tuple((key, values[key]) for key in _VOICE_ORDER)


def _option(
    option_id: str,
    label_key: str,
    values: tuple[tuple[str, str | int | float | bool], ...],
) -> PresetOption:
    return PresetOption(option_id, label_key, values)
