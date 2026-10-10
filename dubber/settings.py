"""Application settings (``<app home>/state/settings.json``): service keys, Hugging Face token, engine choices, folders.

Secrets (subtitle keys, the HF token, the OpenSubtitles password) stay in the user's profile folder only; they are passed to
worker processes through the environment / argument file and never written to logs or diagnostic reports.
"""
from __future__ import annotations

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
    data = read_json(_file(), {}) or {}
    return {**DEFAULTS, **{k: v for k, v in data.items() if k in DEFAULTS}}


def save(values: Dict[str, Any]) -> None:
    cur = load()
    cur.update({k: v for k, v in values.items() if k in DEFAULTS})
    write_json(_file(), cur)


def output_format() -> str:
    """mkv (default) | mp4; the old "same as the film" choice reads as MKV (the video is only ever remuxed)."""
    v = str(load().get("output_format") or "mkv")
    return v if v in ("mkv", "mp4") else "mkv"


def engine_cfg() -> Dict[str, Any]:
    s = load()
    cfg = {k: s[k] for k in ENGINE_KEYS}
    cfg["device"] = device()
    return cfg


def device() -> str:
    """auto | cuda. Shared suite.json may still say ``cpu`` (an older choice or the sibling app); this program reads that as auto."""
    from dubber.infra import suite

    if suite.has("gpu"):
        v = suite.device_setting()
    else:
        v = str(load().get("device") or "auto")
    if v == "cpu":
        return "auto"
    return v if v in ("auto", "cuda") else "auto"


def set_device(value: str) -> None:
    """Remember auto or cuda in the shared suite file. ``cpu`` is not a choice this program offers."""
    from dubber.infra import suite

    if value not in ("auto", "cuda"):
        value = "auto"
    suite.set_quietly("gpu", {"cuda": "cuda:0"}.get(value, value))


def redacted(values: Dict[str, Any]) -> Dict[str, Any]:
    return {k: ("set" if v else "") if k in SECRET_KEYS else v for k, v in values.items()}
