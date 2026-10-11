"""Speech / background separation on the dialogue windows only (outside them the original is used as is)."""
from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from dubber import models
from dubber.core import audio

Window = Tuple[float, float]
CONTEXT_S = 0.5


def chunk_batch_size(total_gb: float, budget_gb: Optional[float] = None) -> int:
    """How many separation chunks to run at once.  4 on a 16 GB card, 8 on a 24 GB card.

    A tight VRAM budget (under 3 GB) halves that.  Cards below the 16 GB class take fewer.
    This sizes RoFormer copies. TIGER-DnR is always one block: its forward indexes tracks, not items.
    """
    if total_gb >= 20.0:
        n = 8
    elif total_gb >= 14.0:
        n = 4
    elif total_gb >= 8.0:
        n = 2
    else:
        n = 1
    if budget_gb is not None and budget_gb < 3.0:
        n = max(1, n // 2)
    return n


#: TIGER-DnR ``forward`` returns dialogue, effects, music. A batch of 1 is squeezed to this length.
N_TRACKS = 3


def stems_are_per_item(dialog: object, effect: object, music: object, n_items: int) -> bool:
    """True when each stem's leading axis is one row per mixture, not TIGER's track axis.

    TIGER-DnR squeezes a batch of 1 into ``[ntrack, nch, T]`` (``ntrack`` is 3) and ``forward``
    indexes that axis. A stack of N mixtures still comes back track-major, so ``stem[i]`` reads
    a track: N greater than 3 raises IndexError, and N of 3 or less silently swaps stems.
    A real batch has leading size ``n_items``. When ``n_items`` is 3 that shape is also the
    track axis, so a leading 3 is not treated as a batch.

    Args:
        dialog: Dialogue stem, array or tensor.
        effect: Effects stem.
        music: Music stem.
        n_items: How many mixtures were stacked.

    Returns:
        True when it is safe to index each stem with the mixture index.
    """
    if n_items <= 1:
        return True
    if n_items == N_TRACKS:
        return False
    for stem in (dialog, effect, music):
        shape = getattr(stem, "shape", None)
        if shape is None or len(shape) < 1 or int(shape[0]) != n_items:
            return False
        if int(shape[0]) == N_TRACKS:
            return False
    return True


def _as_numpy(stem: object) -> np.ndarray:
    """A tensor or array as float32. Tensors are detached and moved to CPU first."""
    if hasattr(stem, "detach"):
        stem = stem.detach()
    if hasattr(stem, "float"):
        stem = stem.float()
    if hasattr(stem, "cpu"):
        stem = stem.cpu()
    if hasattr(stem, "numpy"):
        return np.asarray(stem.numpy(), dtype=np.float32)
    return np.asarray(stem, dtype=np.float32)


def separate_with_forward(forward: Callable, segs: Sequence[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Separate each segment. Stack only when the model returns one row per item.

    A stacked call is kept only when :func:`stems_are_per_item` accepts it. Anything else,
    including an exception from the model, is separated one segment at a time. TIGER-DnR
    never passes the check, so it cannot swap stems or raise on the track axis.

    Args:
        forward: ``forward(batch) -> (dialog, effect, music)``. A single segment is ``[1, 1, T]``.
        segs: One waveform per mixture.

    Returns:
        ``(speech, background)`` per segment, background being effects plus music.
    """
    prepared = [np.asarray(seg, np.float32).reshape(-1) for seg in segs]
    if not prepared:
        return []

    def one(seg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dialog, effect, music = forward(seg[None, None, :])
        speech = _as_numpy(dialog).reshape(-1)[: len(seg)]
        back = (_as_numpy(effect) + _as_numpy(music)).reshape(-1)[: len(seg)]
        return speech, back

    if len(prepared) == 1:
        return [one(prepared[0])]
    width = max(len(seg) for seg in prepared)
    stacked = np.zeros((len(prepared), 1, width), np.float32)
    for i, seg in enumerate(prepared):
        stacked[i, 0, : len(seg)] = seg
    try:
        dialog, effect, music = forward(stacked)
    except Exception:  # noqa: BLE001 - a model that rejects a batch falls back to one chunk
        return [one(seg) for seg in prepared]
    if not stems_are_per_item(dialog, effect, music, len(prepared)):
        return [one(seg) for seg in prepared]
    outs: List[Tuple[np.ndarray, np.ndarray]] = []
    for i, seg in enumerate(prepared):
        speech = _as_numpy(dialog[i]).reshape(-1)[: len(seg)]
        back = (_as_numpy(effect[i]) + _as_numpy(music[i])).reshape(-1)[: len(seg)]
        outs.append((speech, back))
    return outs


def safe_compute_dtype(cuda: bool, bf16: bool) -> str:
    """Network dtype. Half precision only as bf16 on CUDA. RTX 40 / Ada and newer always have bf16.

    Callers cast the waveform back to fp32. The half-precision math stays inside the network."""
    if cuda and bf16:
        return "bf16"
    return "fp32"


class _BackgroundCopies:
    """Apply stem slices on a side thread so the next batch can compute while the copy runs."""

    def __init__(self) -> None:
        self._q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, fn: Callable[[], None]) -> None:
        self._q.put(fn)

    def finish(self) -> None:
        self._q.put(None)
        self._thread.join()

    def _run(self) -> None:
        while True:
            fn = self._q.get()
            if fn is None:
                return
            fn()


def _blocks(windows: Sequence[Window], total: float, max_s: float = 30.0) -> List[Window]:
    """Windows padded with context and grouped into blocks of at most ``max_s`` (bounded memory, fewer model calls)."""
    out: List[List[float]] = []
    for s, e in windows:
        s, e = max(0.0, s - CONTEXT_S), min(total, e + CONTEXT_S)
        if out and s <= out[-1][1] + 1.0 and e - out[-1][0] <= max_s:
            out[-1][1] = max(out[-1][1], e)
        else:
            while e - s > max_s:
                out.append([s, s + max_s])
                s += max_s
            out.append([s, e])
    return [(a, b) for a, b in out]


def separate_windows(mix44: Path, windows: Sequence[Window], out_speech: Path, out_bg: Path,
                     model_fn: Callable[[np.ndarray, int], Tuple[np.ndarray, np.ndarray]], log: Callable[[str], None] = lambda m: None,
                     batch_size: int = 1) -> None:
    """Run ``model_fn(segment, sr) -> (speech, background)`` on each block; stems are full length (silence outside blocks).

    ``batch_size`` blocks are computed together when ``model_fn`` has ``separate_batch``.  Copies into the stem
    buffers run on a side thread, and the final files are written there too, so the next batch is not waiting on disk.
    """
    import soundfile as sf

    x, sr = audio.read(mix44)
    total = len(x) / sr
    speech = np.zeros_like(x)
    bg = x.copy()
    blocks = _blocks(windows, total)
    step = max(1, int(batch_size))
    batches = [blocks[i:i + step] for i in range(0, len(blocks), step)]
    writer = _BackgroundCopies()
    batched = getattr(model_fn, "separate_batch", None)
    done = 0
    for batch in batches:
        pieces: List[Tuple[int, int, np.ndarray, np.ndarray]] = []
        spans = [(int(s * sr), int(e * sr)) for s, e in batch]
        if batched is not None and len(batch) > 1:
            outs = batched([x[a:b] for a, b in spans], sr)
        else:
            outs = [model_fn(x[a:b], sr) for a, b in spans]
        for (a, b), (sp, bk) in zip(spans, outs):
            pieces.append((a, b, sp, bk))

        def copy(rows: List[Tuple[int, int, np.ndarray, np.ndarray]] = pieces) -> None:
            for a, b, sp, bk in rows:
                n = min(b - a, len(sp), len(bk))
                speech[a:a + n] = sp[:n]
                bg[a:a + n] = bk[:n]

        writer.submit(copy)
        done += len(batch)
        log(f"separation: block {done}/{len(blocks)} (batch {len(batch)})")

    def dump() -> None:
        sf.write(str(out_speech), speech, sr, subtype="PCM_16")
        sf.write(str(out_bg), bg, sr, subtype="PCM_16")

    writer.submit(dump)
    writer.finish()


def build_tiger(device: str, allow_download: bool, log: Callable[[str], None]):
    """Construct TIGER-DnR (not cached).  Half precision stays inside the network; waveforms come back as fp32.

    One block at a time. ``wav_chunk_inference`` squeezes a batch of 1 into ``[ntrack, nch, T]``
    and ``forward`` indexes tracks. A stack of N blocks is still track-major, so a batched call
    swaps stems or raises. There is no ``separate_batch`` on the returned function.
    """
    import contextlib

    import torch
    from safetensors.torch import load_file

    from dubber.third_party.look2hear.models import TIGERDNR

    folder, _ = models.ensure(models.SPECS["sep"].repo, allow_download, log=log)
    cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    model = TIGERDNR(**{k: v for k, v in cfg.items() if k in TIGERDNR.__init__.__code__.co_varnames})
    model.load_state_dict(load_file(str(folder / "model.safetensors")), strict=False)
    model.eval().to(device)
    cuda = device == "cuda" and torch.cuda.is_available()
    bf16 = bool(cuda and getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    dtype = safe_compute_dtype(cuda, bf16)

    def _cast():
        if dtype == "bf16":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _forward(batch: torch.Tensor):
        with torch.no_grad(), _cast():
            return model(batch)

    def _one(seg: np.ndarray, sr: int):
        src_sr = sr
        if sr != 44100:
            seg = audio.resample(seg, sr, 44100)
            sr = 44100
        d, e, m = _forward(torch.from_numpy(np.asarray(seg, np.float32))[None, None, :].to(device))
        dialog = d.float().cpu().numpy().reshape(-1)[: len(seg)]
        back = (e + m).float().cpu().numpy().reshape(-1)[: len(seg)]
        if src_sr != 44100:
            dialog, back = audio.resample(dialog, 44100, src_sr), audio.resample(back, 44100, src_sr)
        return dialog, back

    def run(seg: np.ndarray, sr: int):
        return _one(seg, sr)

    run.dtype_name = dtype  # type: ignore[attr-defined]
    return run


def tiger_model(device: str, allow_download: bool, log: Callable[[str], None]):
    """TIGER-DnR (Apache-2.0 weights, vendored MIT code): dialogue vs effects+music at 44.1 kHz.

    The resident cache keeps the network loaded across stages and clips while it fits in the VRAM budget.
    """
    from dubber.infra import resident

    def build():
        return build_tiger(device, allow_download, log)

    if resident.enabled():
        return resident.slot("separation", 1.6, build)
    return build()


def _expose_roformer(folder: Path, model_file: str) -> str:
    """Name the pinned checkpoint the way audio-separator looks it up, so it does not fetch another copy of the weights."""
    src = folder / "MelBandRoformer.ckpt"
    name = model_file or "vocals_mel_band_roformer.ckpt"
    dest = folder / Path(name).name
    if dest.name != src.name and not dest.exists() and not dest.is_symlink():
        try:
            dest.symlink_to(src.name)
        except OSError:
            dest.hardlink_to(src)
    return dest.name if dest.exists() else src.name


def roformer_model(device: str, allow_download: bool, log: Callable[[str], None],
                   model_file: str = "vocals_mel_band_roformer.ckpt"):
    """Mel-Band RoFormer (MIT, KimberleyJSN/melbandroformer at the 2026-04-22 relicence) through ``audio-separator``.

    The checkpoint is the pinned file in the shared model store. audio-separator still loads its own yaml for that filename
    and skips the weight download when the file is already there. ``device`` is unused: the package picks the device.
    """
    import tempfile

    from audio_separator.separator import Separator

    folder, _ = models.ensure(models.SPECS["roformer"].repo, allow_download, log=log)
    filename = _expose_roformer(folder, model_file)
    sep = Separator(model_file_dir=str(folder), output_dir=tempfile.mkdtemp(prefix="vmd-rof-"), output_single_stem=None)
    sep.load_model(model_filename=filename)

    def run(seg: np.ndarray, sr: int):
        d = tempfile.mkdtemp(prefix="vmd-rofseg-")
        p = Path(d) / "seg.wav"
        audio.write(p, seg, sr)
        outs = [Path(sep.output_dir) / o for o in sep.separate(str(p))]
        voc = next(o for o in outs if "vocal" in o.name.lower())
        ins = next(o for o in outs if o != voc)
        v, _ = audio.read(voc, sr)
        i, _ = audio.read(ins, sr)
        return v[: len(seg)], i[: len(seg)]
    return run
