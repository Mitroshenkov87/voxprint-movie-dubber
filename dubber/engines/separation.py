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

    A tight VRAM budget (under 3 GB) halves that.  Cards below the 16 GB class take fewer."""
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
    """Construct TIGER-DnR (not cached).  Half precision stays inside the network; waveforms come back as fp32."""
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

    def separate_batch(segs: Sequence[np.ndarray], sr: int):
        prepared = []
        for seg in segs:
            if sr != 44100:
                seg = audio.resample(seg, sr, 44100)
            prepared.append(np.asarray(seg, np.float32).reshape(-1))
        width = max(len(s) for s in prepared)
        stacked = np.zeros((len(prepared), 1, width), np.float32)
        for i, seg in enumerate(prepared):
            stacked[i, 0, : len(seg)] = seg
        try:
            d, e, m = _forward(torch.from_numpy(stacked).to(device))
        except Exception:  # noqa: BLE001 - a model that rejects a batch falls back to one chunk
            return [_one(seg, 44100) for seg in prepared]
        outs = []
        for i, seg in enumerate(prepared):
            dialog = d[i].float().cpu().numpy().reshape(-1)[: len(seg)]
            back = (e[i] + m[i]).float().cpu().numpy().reshape(-1)[: len(seg)]
            if sr != 44100:
                dialog, back = audio.resample(dialog, 44100, sr), audio.resample(back, 44100, sr)
            outs.append((dialog, back))
        return outs

    def run(seg: np.ndarray, sr: int):
        return _one(seg, sr)

    run.separate_batch = separate_batch  # type: ignore[attr-defined]
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
