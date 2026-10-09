"""Qwen3-TTS synthesis (logic taken from Voxprint AI Audiobook Builder ``core/tts_engine.py`` and ``core/narration.py``).

Backend order on an NVIDIA GPU for dialogue (no LoRA adapters): CUDA Graphs via ``faster-qwen3-tts`` (MIT) first, then
FlashAttention-2 if that package exists, then standard SDPA.  ``auto`` tries Graphs first; the batched standard backend is used when
``tts_backend`` is ``standard/sdpa`` (or ``standard/flash_attention_2``) and stays the fallback when Graphs cannot load.  Clip 3
(16 short lines) measured graphs/sdpa at 67.5 s against batched standard/sdpa at 208.9 s.  On the CPU: SDPA.  Every step falls back
to the next one if loading fails.
Batching: up to ``MAX_BATCH`` (12) lines per generate call, sized by the VRAM budget in ``dubber.infra.resources``; lines of similar
length are batched together (sorted inside a small look-ahead window so the output stays close to film order for Watch mode);
out-of-memory halves the batch.  Library voices (LoRA adapters) stay loaded side by side while the budget allows, else they are
swapped one after another.
The CUDA-Graphs backend synthesises one line at a time (static shapes).

Voices: ``clone`` = reference clip + its text (ICL; without a text the x-vector-only mode is used); ``library`` = a LoRA adapter
from the shared Voxprint voice library, applied to the talker with peft (several adapters can be loaded and switched).
"""
from __future__ import annotations

import hashlib
import importlib.util
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("dubber.tts")

FRAMES_PER_SECOND = 12.5
MAX_SECONDS_PER_CHAR = 0.19
MIN_TOKENS, MAX_TOKENS = 48, 2048
MAX_BATCH = 12
SORT_WINDOW_BATCHES = 3
QWEN_LANG = {"ru": "Russian", "en": "English", "de": "German", "fr": "French", "es": "Spanish", "it": "Italian", "pt": "Portuguese",
             "ja": "Japanese", "ko": "Korean", "zh": "Chinese"}


def max_tokens_for(text: str) -> int:
    """Upper bound of codec frames (without it a missing end-of-speech token babbles for minutes)."""
    seconds = 3.0 + MAX_SECONDS_PER_CHAR * len(text.strip())
    return max(MIN_TOKENS, min(MAX_TOKENS, int(round(seconds * FRAMES_PER_SECOND))))


@dataclass
class VoiceSpec:
    key: str                       # stable id (cache key part)
    kind: str                      # clone | library
    ref_audio: str = ""
    ref_text: str = ""
    adapter_dir: str = ""          # library voice folder
    adapter_scale: float = 1.0     # LoRA strength from the voice's voice.json (Audiobook Builder: new voices 0.5, old ones 1.0)
    # kind "actor" (dubber.core.actor_voice): the actor's clip blended with the closest library voice
    actor_weight: float = 0.0
    actor_ok: bool = True          # the actor clip is long / clean enough; else the library voice is used as is
    gender: str = ""
    candidates: tuple = ()         # actor_voice.Candidate items
    record_dir: str = ""

    def tag(self) -> str:
        fp = ""
        for p in (self.ref_audio, str(Path(self.adapter_dir) / "adapter_model.safetensors") if self.adapter_dir else ""):
            try:
                st = Path(p).stat() if p else None
                fp += f"{st.st_size}:{int(st.st_mtime)};" if st else ""
            except OSError:
                pass
        return hashlib.sha1(f"{self.kind}|{self.key}|{self.ref_text}|{fp}|{self.adapter_scale}".encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------------------------- batching (from the audiobook narrator)
def next_group(queue: List[int], texts: Dict[int, str], limit: int) -> List[int]:
    """Take the next batch off ``queue`` (in place): ``limit`` texts of similar length from a look-ahead window, always containing
    the earliest pending item, so the order stays close to the film order."""
    if limit <= 1:
        return [queue.pop(0)]
    window = sorted(queue[:limit * SORT_WINDOW_BATCHES], key=lambda i: len(texts[i]))
    if len(window) <= limit:
        group = window
    else:
        first = window.index(queue[0])
        lo = max(0, min(first - limit // 2, len(window) - limit))
        group = window[lo:lo + limit]
    for i in group:
        queue.remove(i)
    return group


def is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError"


class BaseTTS:
    sample_rate = 24000
    backend = "base"
    def resolve(self, voice: VoiceSpec) -> VoiceSpec:
        """An "actor" voice becomes a concrete one (library adapter + blended embedding, or a fallback) - once per voice."""
        if voice.kind != "actor":
            return voice
        if not hasattr(self, "_resolved"):
            self._resolved, self._blends = {}, {}
        if voice.key not in self._resolved:
            self._resolved[voice.key] = self._resolve_actor(voice)
        return self._resolved[voice.key]

    def actor_embedding(self, voice: VoiceSpec) -> Optional[np.ndarray]:
        return None

    def _resolve_actor(self, voice: VoiceSpec) -> VoiceSpec:
        from dubber.core import actor_voice as av

        emb = self.actor_embedding(voice) if voice.ref_audio else None
        cand, score = av.pick_closest(emb, list(voice.candidates), voice.gender)
        rec: Dict[str, Any] = {"actor_ok": bool(voice.actor_ok), "weight": float(voice.actor_weight), "ref_audio": voice.ref_audio,
                               "ref_text": voice.ref_text, "similarity": round(score, 4)}
        if cand is None:                                       # empty library: a plain clone of the actor
            out = VoiceSpec(voice.key, "clone", voice.ref_audio, voice.ref_text)
            rec.update(mode="clone (no library voice)")
        elif not voice.actor_ok or emb is None:
            lib = _library_spec(cand)
            out = VoiceSpec(f"{voice.key}>{cand.id}", "library", lib[0], lib[1], str(cand.path), cand.adapter_scale)
            rec.update(mode="library fallback", library_id=cand.id, library_path=str(cand.path), adapter_scale=cand.adapter_scale)
        else:
            centroid = av.load_centroid(cand.path)
            blended = av.blend(emb, centroid, voice.actor_weight)
            scale = av.adapter_scale(cand.adapter_scale, voice.actor_weight)
            out = VoiceSpec(f"{voice.key}>{cand.id}", "library", voice.ref_audio, voice.ref_text, str(cand.path), scale)
            self._blends[out.key] = blended
            rec.update(mode="blend", library_id=cand.id, library_path=str(cand.path), adapter_scale=scale)
            if voice.record_dir:
                av.write_record(Path(voice.record_dir), rec, emb, blended)
                return out
        if voice.record_dir:
            av.write_record(Path(voice.record_dir), rec, emb)
        return out

    def synthesize_batch(self, texts: Sequence[str], voice: VoiceSpec, seed: Optional[int] = None) -> List[np.ndarray]:
        raise NotImplementedError

    def max_batch(self) -> int:
        return 1

    def close(self) -> None:
        pass

    def run_queue(self, items: Sequence[Tuple[int, str]], voice: VoiceSpec, on_done: Callable[[int, np.ndarray], None]) -> None:
        """Synthesize ``items`` (id, text) for one voice in batches; OOM halves the batch size; a failing batch goes one by one."""
        texts = {i: t for i, t in items}
        queue = [i for i, _ in items]
        limit = max(1, self.max_batch())
        while queue:
            group = next_group(queue, texts, limit)
            try:
                wavs = self.synthesize_batch([texts[i] for i in group], voice)
            except Exception as exc:  # noqa: BLE001
                if len(group) == 1:
                    raise
                if is_oom(exc):
                    limit = max(1, limit // 2)
                    self._free()
                log.warning("batch of %d failed (%s); retrying one by one", len(group), type(exc).__name__)
                wavs = [self.synthesize_batch([texts[i]], voice)[0] for i in group]
            for i, w in zip(group, wavs):
                on_done(i, w)

    def _free(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


class MockTTS(BaseTTS):
    """Speech-like tone with the length a speaker would need (deterministic per text): lets the whole pipeline run on a CPU."""
    backend = "mock"

    def max_batch(self) -> int:
        return MAX_BATCH

    def __init__(self, lang: str = "ru", rate: float = 1.0) -> None:
        self.lang, self.rate = lang, rate
        self._resolved, self._blends = {}, {}

    def synthesize_batch(self, texts: Sequence[str], voice: VoiceSpec, seed: Optional[int] = None) -> List[np.ndarray]:
        from dubber.core.script import estimate_seconds

        voice = self.resolve(voice)

        out = []
        for t in texts:
            h = int(hashlib.md5(f"{t}|{voice.key}|{seed}".encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.default_rng(h)
            dur = max(0.3, estimate_seconds(t, self.lang) * rng.uniform(0.92, 1.12) / self.rate)
            n = int(dur * self.sample_rate)
            tt = np.arange(n) / self.sample_rate
            f0 = 110 + (h % 120)
            sig = sum(np.sin(2 * np.pi * f0 * k * tt) / k for k in range(1, 6))
            env = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * tt) ** 2
            fade = np.minimum(1.0, np.minimum(tt, dur - tt) / 0.03)
            out.append((0.15 * sig * env * fade).astype(np.float32))
        return out


class QwenTTS(BaseTTS):
    """Qwen3-TTS Base (voice cloning). ``auto`` tries CUDA Graphs, then FA2, then SDPA."""

    def __init__(self, base_dir: Path, language: str, device: str = "auto", prefer: str = "auto", need_adapters: bool = False,
                 log_fn: Callable[[str], None] = lambda m: None) -> None:
        import torch

        from dubber.workers.common import apply_qwen_tts_compat

        self.log = log_fn
        self.language = QWEN_LANG.get(language, "Auto")
        self.use_cuda = device == "cuda" or (device == "auto" and torch.cuda.is_available())
        self.dtype = torch.bfloat16 if self.use_cuda else torch.float32
        self.base_dir = Path(base_dir)
        self._prompts: Dict[str, Any] = {}
        self._resolved, self._blends = {}, {}
        self._adapters: Dict[str, str] = {}
        self._peft = None
        self.model = None
        from dubber.infra import resources

        self.budget = resources.VramBudget() if self.use_cuda else None      # measured before our model takes any memory
        errors = []
        for kind, attn in self.candidates(self.use_cuda, prefer, need_adapters):
            try:
                if kind == "graphs":
                    from faster_qwen3_tts import FasterQwen3TTS

                    apply_qwen_tts_compat()
                    self.model = FasterQwen3TTS.from_pretrained(str(base_dir), device="cuda", dtype=self.dtype, attn_implementation=attn)
                    self.model.warmup()
                else:
                    from qwen_tts import Qwen3TTSModel

                    apply_qwen_tts_compat()
                    self.model = Qwen3TTSModel.from_pretrained(str(base_dir), device_map="cuda:0" if self.use_cuda else None,
                                                               dtype=self.dtype, attn_implementation=attn)
                self.backend = f"{kind}/{attn}"
                break
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{kind}/{attn}: {type(exc).__name__}: {str(exc)[:160]}")
                self.model = None
        if self.model is None:
            raise RuntimeError("Qwen3-TTS could not be loaded: " + " | ".join(errors))
        if errors:
            self.log("TTS backends skipped: " + " | ".join(errors))
        self.log(f"TTS backend: {self.backend}")
        if self.budget is not None:
            self.log(f"TTS memory: {self.budget.describe()}; batch {self.max_batch()}")

    @staticmethod
    def candidates(use_cuda: bool, prefer: str = "auto", need_adapters: bool = False) -> List[Tuple[str, str]]:
        """Backends to try, first choice first.

        ``auto`` on CUDA without LoRA adapters puts ``graphs/sdpa`` first. Clip 3 (16 short lines) took 67.5 s that way and
        208.9 s on batched ``standard/sdpa``. The batched backend is selected when ``prefer`` is ``standard/sdpa`` (settings
        and the engine combo still list it) and remains the last fallback when Graphs cannot load. Adapters cannot use Graphs."""
        if prefer and prefer != "auto":
            kind, _, attn = prefer.partition("/")
            first = [(kind, attn or "sdpa")]
        else:
            first = []
        out = list(first)
        if use_cuda and not need_adapters and importlib.util.find_spec("faster_qwen3_tts") is not None:
            out.append(("graphs", "sdpa"))
        if use_cuda and importlib.util.find_spec("flash_attn") is not None:
            out.append(("standard", "flash_attention_2"))
        out.append(("standard", "sdpa"))
        seen, uniq = set(), []
        for c in out:
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        return uniq

    @property
    def graphs(self) -> bool:
        return self.backend.startswith("graphs")

    def max_batch(self) -> int:
        from dubber.infra import resources

        if self.graphs:
            return 1
        if not self.use_cuda:
            return resources.tts_batch("cpu", 0.0, resources.RAM_SHARE * resources.ram_gb()[1])
        try:
            return resources.tts_batch("cuda", self.budget.left(), self.budget.ram_budget)
        except Exception:  # noqa: BLE001
            return 1

    def _swap_out_other_voices(self, keep: str) -> None:
        """Budget too tight for several voices: drop every adapter except ``keep`` (they are reloaded when needed)."""
        for name in [n for n in self._adapters if n != keep]:
            try:
                self._peft.delete_adapter(name)
            except Exception as exc:  # noqa: BLE001 - keep going with what is loaded
                self.log(f"voice adapter {name} not unloaded: {type(exc).__name__}: {exc}")
                continue
            self._adapters.pop(name, None)
        self._free()

    def _inner(self):
        return self.model.model if self.graphs else self.model

    def _activate(self, voice: VoiceSpec) -> None:
        """Switch the LoRA adapter for a library voice (peft, unmerged so several voices can share the base model)."""
        if voice.kind != "library":
            if self._peft is not None:                     # a cloned voice runs on the plain base model
                self._peft.base_model.disable_adapter_layers()
            return
        from peft import PeftModel

        q = self._inner()
        if self._peft is None:
            self._peft = PeftModel.from_pretrained(q.model.talker, voice.adapter_dir, adapter_name=voice.key)
            q.model.talker = self._peft
            self._adapters[voice.key] = voice.adapter_dir
        elif voice.key not in self._adapters:
            self._peft.load_adapter(voice.adapter_dir, adapter_name=voice.key)
            self._adapters[voice.key] = voice.adapter_dir
        self._peft.base_model.enable_adapter_layers()
        self._peft.set_adapter(voice.key)
        set_lora_scale(self._peft, voice.key, voice.adapter_scale)
        if self.budget is not None and len(self._adapters) > 1:
            from dubber.infra import resources

            if not resources.keep_voices_loaded(self.budget.left()):
                self._swap_out_other_voices(voice.key)

    def _prompt(self, voice: VoiceSpec):
        key = voice.tag()
        if key not in self._prompts:
            kw: Dict[str, Any] = {"ref_audio": voice.ref_audio}
            if voice.ref_text:
                kw["ref_text"] = voice.ref_text
            else:
                kw["x_vector_only_mode"] = True
            prompt = self._inner().create_voice_clone_prompt(**kw)
            if voice.key in self._blends:
                use_centroid(prompt, self._blends[voice.key])          # actor-like voice: the blended embedding
            elif voice.kind == "library" and voice.adapter_dir:
                use_centroid(prompt, load_centroid(Path(voice.adapter_dir)))
            self._prompts[key] = prompt
        return self._prompts[key]

    def actor_embedding(self, voice: VoiceSpec) -> Optional[np.ndarray]:
        """The actor's x-vector from his reference clip (the model's own speaker encoder)."""
        try:
            prompt = self._inner().create_voice_clone_prompt(ref_audio=voice.ref_audio, x_vector_only_mode=True)
            return prompt[0].ref_spk_embedding.detach().float().cpu().numpy().reshape(-1)
        except Exception as exc:  # noqa: BLE001 - then the library voice is used
            self.log(f"actor voice: no speaker embedding ({type(exc).__name__}: {exc})")
            return None

    def synthesize_batch(self, texts: Sequence[str], voice: VoiceSpec, seed: Optional[int] = None) -> List[np.ndarray]:
        import torch

        if seed is not None:
            torch.manual_seed(int(seed))
        voice = self.resolve(voice)
        self._activate(voice)
        mnt = max(max_tokens_for(t) for t in texts)
        if self.graphs:
            out = []
            for t in texts:
                wavs, sr = self.model.generate_voice_clone(text=t, language=self.language, ref_audio=voice.ref_audio,
                                                           ref_text=voice.ref_text, max_new_tokens=max_tokens_for(t))
                out.append(np.asarray(wavs[0], dtype=np.float32).reshape(-1))
                self.sample_rate = int(sr)
            return out
        with torch.inference_mode():
            wavs, sr = self.model.generate_voice_clone(text=list(texts), language=[self.language] * len(texts),
                                                       voice_clone_prompt=self._prompt(voice), max_new_tokens=mnt)
        self.sample_rate = int(sr)
        return [np.asarray(w, dtype=np.float32).reshape(-1) for w in wavs]

    def close(self) -> None:
        self.model, self._peft, self._prompts = None, None, {}
        import gc

        gc.collect()
        self._free()


# ---------------------------------------------------------------------------------------------- library voice details
CENTROID_FILE = "speaker_centroid.safetensors"


def set_lora_scale(peft_model: Any, adapter: str, scale: float) -> int:
    """Strength of one LoRA adapter on every layer (peft ``LoraLayer.set_scale``: scaling = scale x alpha / r)."""
    from peft.tuners.lora import LoraLayer

    n = 0
    for module in peft_model.modules():
        if isinstance(module, LoraLayer) and adapter in getattr(module, "scaling", {}):
            module.set_scale(adapter, float(scale))
            n += 1
    return n


def load_centroid(folder: Path) -> Optional[np.ndarray]:
    """The voice's averaged speaker embedding (``speaker_centroid.safetensors`` from the Audiobook Builder) or None."""
    path = Path(folder) / CENTROID_FILE
    if not path.is_file():
        return None
    try:
        from safetensors.numpy import load_file

        return np.asarray(load_file(str(path))["speaker_embedding"], dtype=np.float32).reshape(-1)
    except Exception:  # noqa: BLE001 - then the reference clip's own embedding is used
        return None


def use_centroid(prompt: Any, centroid: Optional[np.ndarray]) -> bool:
    """Replace the x-vector of every voice-clone prompt item by the voice's centroid (the reference codes + text stay)."""
    if centroid is None:
        return False
    import torch

    items = list(prompt or [])
    if not items or any(int(getattr(it, "ref_spk_embedding").numel()) != int(centroid.size) for it in items):
        return False
    for it in items:
        t = it.ref_spk_embedding
        it.ref_spk_embedding = torch.from_numpy(np.asarray(centroid, dtype=np.float32)).to(device=t.device, dtype=t.dtype).reshape(t.shape)
    return True


def _library_spec(cand) -> Tuple[str, str]:
    """(reference clip, its text) of a library voice folder."""
    import json

    ref = Path(cand.path) / "ref_sample.wav"
    try:
        text = str(json.loads((Path(cand.path) / "training_meta.json").read_text(encoding="utf-8-sig")).get("ref_sample_text") or "")
    except (OSError, ValueError):
        text = ""
    return (str(ref) if ref.is_file() else "", text)

