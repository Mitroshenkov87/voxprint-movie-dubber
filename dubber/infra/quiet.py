"""Mute known harmless noise of the ML libraries (lesson from Voxprint AI Audiobook Builder): the "SoX could not be found"
message of qwen-tts, triton "not found" warnings on Windows, and transformers' "Setting pad_token_id to eos_token_id" line printed
for every generate call.  Call :func:`mute_library_noise` in each worker process before importing the libraries."""
from __future__ import annotations

import logging
import os
import warnings

_PATTERNS = (".*[Ss]o[Xx].*", ".*triton.*", ".*pad_token_id.*", ".*TypedStorage is deprecated.*", ".*torch.jit.load.*",
             ".*`torch_dtype` is deprecated.*")


class _DropNoise(logging.Filter):
    WORDS = ("sox", "triton", "pad_token_id", "flash attention", "flash_attn is not installed")

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage().lower()
        except Exception:  # noqa: BLE001
            return True
        return not any(w in msg for w in self.WORDS)


def mute_library_noise() -> None:
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    for pat in _PATTERNS:
        warnings.filterwarnings("ignore", message=pat)
    flt = _DropNoise()
    for name in ("", "transformers", "transformers.generation.utils", "qwen_tts", "faster_qwen3_tts", "torch", "sox", "triton"):
        logging.getLogger(name).addFilter(flt)
    for name in ("sox", "triton", "numba", "urllib3", "httpx", "filelock"):
        logging.getLogger(name).setLevel(logging.ERROR)
