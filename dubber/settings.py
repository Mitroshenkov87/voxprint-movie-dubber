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
    "output_format": "same", "projects_dir": "",
}
ENGINE_KEYS = ("separation", "asr", "diarization", "translation", "tts", "tts_model", "tts_backend", "device", "allow_download",
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


def engine_cfg() -> Dict[str, Any]:
    s = load()
    return {k: s[k] for k in ENGINE_KEYS}


def redacted(values: Dict[str, Any]) -> Dict[str, Any]:
    return {k: ("set" if v else "") if k in SECRET_KEYS else v for k, v in values.items()}
