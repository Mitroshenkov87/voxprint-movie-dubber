"""Application settings (``<app home>/state/settings.json``): service keys, Hugging Face token, engine choices, folders.

Secrets (subtitle keys, the HF token, the OpenSubtitles password) stay in the user's profile folder only; they are passed to
worker processes through the environment / argument file and never written to logs or diagnostic reports.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from dubber import paths
from dubber.core.project import read_json, write_json

SECRET_KEYS = ("subdl_key", "opensubtitles_key", "opensubtitles_password", "hf_token")
DEFAULTS: Dict[str, Any] = {
    "subdl_key": "", "opensubtitles_key": "", "opensubtitles_user": "", "opensubtitles_password": "", "hf_token": "",
    "separation": "tiger", "asr": "whisper", "diarization": "pyannote", "translation": "opus", "tts": "qwen",
    "tts_model": "tts_1_7b", "tts_backend": "auto", "device": "auto", "allow_download": True,
    # VRAM tier (dubber.infra.vram_tier): "auto" picks it from the card; the advanced override can only pick a tier the card supports
    "vram_tier": "auto",
    "output_format": "mkv", "projects_dir": "",
    # the Film screen's Options, remembered for the next film (a film's own project keeps what was used for it)
    "dub_target_lang": "", "dub_subtitles": "auto", "dub_profanity": "keep", "dub_multi_voice": False, "dub_original_volume": 0.15,
    "options_open": False,
}
#: keys of the Film screen's Options -> project setting they preset
DUB_DEFAULTS = {"dub_target_lang": "target_lang", "dub_subtitles": "subtitle_choice", "dub_profanity": "profanity",
                "dub_multi_voice": "multi_voice", "dub_original_volume": "original_volume", "output_format": "output_format"}
ENGINE_KEYS = ("separation", "asr", "diarization", "translation", "tts", "tts_model", "tts_backend", "device", "allow_download", "vram_tier",
               "subdl_key", "opensubtitles_key", "opensubtitles_user", "opensubtitles_password", "hf_token")


def _file():
    return paths.state_dir() / "settings.json"


def load() -> Dict[str, Any]:
    """Load known settings from ``state/settings.json``, filling gaps from :data:`DEFAULTS`."""
    data = read_json(_file(), {}) or {}
    return {**DEFAULTS, **{k: v for k, v in data.items() if k in DEFAULTS}}


def save(values: Dict[str, Any]) -> None:
    """Merge known keys from ``values`` into ``state/settings.json``."""
    cur = load()
    cur.update({k: v for k, v in values.items() if k in DEFAULTS})
    write_json(_file(), cur)


def output_format() -> str:
    """mkv (default) | mp4; the old "same as the film" choice reads as MKV (the video is only ever remuxed)."""
    v = str(load().get("output_format") or "mkv")
    return v if v in ("mkv", "mp4") else "mkv"


def engine_cfg() -> Dict[str, Any]:
    """Return engine, device, and secret settings for a worker, with ``device`` normalized."""
    s = load()
    cfg = {k: s[k] for k in ENGINE_KEYS}
    cfg["device"] = device()
    return cfg


def device() -> str:
    """auto | cuda. A stored ``cpu`` (an older choice or the sibling app) is read as auto."""
    from voxprint_common.suite_settings import normalize_gpu, read

    raw = read()
    if "gpu" in raw:
        normalized = normalize_gpu(raw.get("gpu"))
        return "cuda" if normalized and normalized.startswith("cuda") else "auto"
    value = str(load().get("device") or "auto")
    if value == "cpu":
        return "auto"
    return value if value in ("auto", "cuda") else "auto"


def set_device(value: str) -> None:
    """Remember auto or cuda in the shared suite file. ``cpu`` is not a choice this program offers."""
    from voxprint_common.suite_settings import set_gpu

    choice = "cuda" if value == "cuda" else "auto"
    try:
        set_gpu(choice)
    except ValueError:
        return


def sync_suite_settings() -> int:
    """Write the models folder and UI language into suite.json when those keys are still absent.

    Exit code 0 when done, 1 on any error. The installer ignores the error.
    """
    import json

    from voxprint_common.paths import state_dir
    from voxprint_common.suite_settings import normalize_language, read, set_models_dir, set_ui_language, values

    try:
        if "models_dir" not in read():
            legacy = _legacy_models_choice(state_dir() / "models_dir.txt")
            set_models_dir(legacy)
        if "ui_language" not in read():
            from dubber import i18n

            code = i18n.saved_language() or i18n.system_language() or "en"
            set_ui_language(normalize_language(code) or "en")
        print(json.dumps(values()), flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001 - the installer must keep going
        print(f"suite settings: {exc}", flush=True)
        return 1


def _legacy_models_choice(path: Path) -> Path | None:
    """Absolute path from the older one-line models file, or None."""
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return None
    text = next((line.strip().strip('"') for line in lines if line.strip()), "")
    if not text:
        return None
    folder = Path(text)
    return folder if folder.is_absolute() else None


def redacted(values: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``values`` with secret keys replaced by ``"set"`` when present and ``""`` when empty."""
    return {k: ("set" if v else "") if k in SECRET_KEYS else v for k, v in values.items()}
