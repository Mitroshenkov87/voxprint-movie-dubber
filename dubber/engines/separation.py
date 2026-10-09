"""Speech / background separation on the dialogue windows only (outside them the original is used as is)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import numpy as np

from dubber.core import audio

Window = Tuple[float, float]
CONTEXT_S = 0.5


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
                     model_fn: Callable[[np.ndarray, int], Tuple[np.ndarray, np.ndarray]], log: Callable[[str], None] = lambda m: None) -> None:
    """Run ``model_fn(segment, sr) -> (speech, background)`` on each block; stems are full length (silence outside blocks)."""
    import soundfile as sf

    x, sr = audio.read(mix44)
    total = len(x) / sr
    speech = np.zeros_like(x)
    bg = x.copy()
    blocks = _blocks(windows, total)
    for i, (s, e) in enumerate(blocks):
        a, b = int(s * sr), int(e * sr)
        sp, bk = model_fn(x[a:b], sr)
        n = min(b - a, len(sp))
        speech[a:a + n], bg[a:a + n] = sp[:n], bk[:n]
        if i % 10 == 0:
            log(f"separation: block {i + 1}/{len(blocks)}")
    sf.write(str(out_speech), speech, sr, subtype="PCM_16")
    sf.write(str(out_bg), bg, sr, subtype="PCM_16")


def tiger_model(device: str, allow_download: bool, log: Callable[[str], None]):
    """TIGER-DnR (Apache-2.0 weights, vendored MIT code): dialogue vs effects+music at 44.1 kHz."""
    import torch
    from safetensors.torch import load_file

    from dubber import models
    from dubber.third_party.look2hear.models import TIGERDNR

    folder, _ = models.ensure(models.SPECS["sep"].repo, allow_download, log=log)
    cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    model = TIGERDNR(**{k: v for k, v in cfg.items() if k in TIGERDNR.__init__.__code__.co_varnames})
    model.load_state_dict(load_file(str(folder / "model.safetensors")), strict=False)
    model.eval().to(device)

    def run(seg: np.ndarray, sr: int):
        if sr != 44100:
            seg = audio.resample(seg, sr, 44100)
        with torch.no_grad():
            d, e, m = model(torch.from_numpy(seg)[None, None, :].to(device))
        d = d.float().cpu().numpy().reshape(-1)[: len(seg)]
        bg = (e + m).float().cpu().numpy().reshape(-1)[: len(seg)]
        if sr != 44100:
            d, bg = audio.resample(d, 44100, sr), audio.resample(bg, 44100, sr)
        return d, bg
    return run


def roformer_model(device: str, model_dir: Path, model_file: str, log: Callable[[str], None]):
    """Mel-Band RoFormer (MIT) through the ``audio-separator`` package (optional dependency; its models are kept in the shared
    models folder under ``audio-separator``).  Not verified on this machine (no GPU): see DECISIONS.md."""
    import tempfile

    from audio_separator.separator import Separator

    sep = Separator(model_file_dir=str(model_dir), output_dir=tempfile.mkdtemp(prefix="vmd-rof-"), output_single_stem=None)
    sep.load_model(model_filename=model_file)

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
