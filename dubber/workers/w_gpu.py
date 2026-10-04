"""Worker ``gpu``: what PyTorch (and the other CUDA libraries) can do on this machine.

Checks, each wrapped on its own so one failure never hides the others:
PyTorch / CUDA / cuDNN versions and devices, a bf16 matmul and a memory-copy micro-benchmark, the three PyTorch SDPA attention
backends (FlashAttention kernel built into PyTorch, memory-efficient, math), the separate ``flash_attn`` package (FlashAttention-2:
imported AND executed), CUDA Graphs (captured and replayed, speed-up measured), triton, CTranslate2 and onnxruntime CUDA support.
"""
from __future__ import annotations

import importlib.util
import time
import traceback
from typing import Any, Dict, List

from dubber.workers.common import WorkerContext


def _try(lines: List[str], tracebacks: List[str], label: str, fn):
    """Run ``fn`` and return its value; on error add a ``label: ERROR ...`` line and keep the traceback."""
    try:
        return fn()
    except BaseException as exc:  # noqa: BLE001
        lines.append(f"{label:<28}: ERROR {type(exc).__name__}: {str(exc)[:200]}")
        tracebacks.append(f"[{label}]\n{traceback.format_exc()}")
        return None


def run(args: Dict[str, Any], ctx: WorkerContext) -> Dict[str, Any]:
    lines: List[str] = []
    tbs: List[str] = []
    metrics: Dict[str, Any] = {"flash_attn": "unknown", "cuda_graphs": "unknown"}
    status = "OK"
    t0 = time.time()
    try:
        import torch
    except BaseException as exc:  # noqa: BLE001
        return {"status": "FAIL", "summary": f"PyTorch cannot be imported: {type(exc).__name__}: {str(exc)[:200]}",
                "traceback": traceback.format_exc(), "details": [], "metrics": metrics}
    lines.append(f"{'torch import':<28}: {time.time() - t0:.1f} s")
    metrics["torch"] = torch.__version__
    metrics["cuda"] = torch.version.cuda or "none (CPU-only build)"
    lines.append(f"{'torch version':<28}: {torch.__version__}  (built for CUDA {torch.version.cuda or 'none'}, "
                 f"cuDNN {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'n/a'})")
    cuda_ok = bool(torch.cuda.is_available())
    lines.append(f"{'cuda available':<28}: {cuda_ok}")
    if not cuda_ok:
        reason = []
        if not torch.version.cuda:
            reason.append("this PyTorch is a CPU-only build (install the CUDA build: uv pip install torch --torch-backend=auto)")
        else:
            reason.append("PyTorch has CUDA support but sees no usable GPU (driver missing/too old, GPU disabled, or Optimus/hybrid "
                          "graphics routing; check nvidia-smi and the NVIDIA driver version)")
        lines += ["no GPU test was run: " + reason[0]]
        _optional_libs(lines, tbs, metrics, ctx)
        short = "this PyTorch is a CPU-only build" if not torch.version.cuda else "PyTorch sees no usable GPU (driver/hybrid graphics?)"
        return {"status": "WARN", "summary": f"CUDA not available to PyTorch: {short}", "details": lines,
                "traceback": "\n".join(tbs), "metrics": metrics}

    # ---- devices
    n = torch.cuda.device_count()
    lines.append(f"{'cuda devices':<28}: {n}")
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        lines.append(f"  device {i}: {p.name}, compute capability {p.major}.{p.minor}, {total / 1024 ** 3:.1f} GB total, "
                     f"{free / 1024 ** 3:.1f} GB free now, {p.multi_processor_count} SMs")
        if i == 0:
            metrics.update(name=p.name, vram_total_gb=round(total / 1024 ** 3, 2), vram_free_gb=round(free / 1024 ** 3, 2),
                           capability=f"{p.major}.{p.minor}", sms=p.multi_processor_count)
    lines.append(f"{'bf16 supported':<28}: {torch.cuda.is_bf16_supported()}")
    dev = torch.device("cuda:0")

    # ---- micro benchmarks
    def matmul_bench() -> None:
        a = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
        b = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
        for _ in range(3):
            a @ b
        torch.cuda.synchronize()
        t = time.time()
        iters = 20
        for _ in range(iters):
            a @ b
        torch.cuda.synchronize()
        dt = time.time() - t
        tf = 2 * 4096 ** 3 * iters / dt / 1e12
        metrics["bf16_tflops"] = round(tf, 1)
        lines.append(f"{'bf16 matmul 4096^3':<28}: {tf:.1f} TFLOPS")

    def bandwidth_bench() -> None:
        x = torch.empty(256 * 1024 * 1024 // 4, device=dev, dtype=torch.float32)       # 256 MB
        x.fill_(1.0)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(10):
            y = x.clone()
        torch.cuda.synchronize()
        dt = time.time() - t
        gbs = 2 * x.numel() * 4 * 10 / dt / 1e9
        metrics["mem_gbs"] = round(gbs)
        lines.append(f"{'memory copy bandwidth':<28}: {gbs:.0f} GB/s")
        del y

    _try(lines, tbs, "matmul benchmark", matmul_bench)
    _try(lines, tbs, "bandwidth benchmark", bandwidth_bench)

    # ---- SDPA backends (PyTorch built-in kernels; the 'flash' one is FlashAttention inside PyTorch, works on Windows)
    def sdpa_backends() -> None:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        q = torch.randn(1, 16, 2048, 128, device=dev, dtype=torch.bfloat16)
        k, v = torch.randn_like(q), torch.randn_like(q)
        names = [("flash (PyTorch built-in)", SDPBackend.FLASH_ATTENTION), ("mem-efficient", SDPBackend.EFFICIENT_ATTENTION),
                 ("math (slow reference)", SDPBackend.MATH)]
        if hasattr(SDPBackend, "CUDNN_ATTENTION"):
            names.insert(2, ("cuDNN attention", SDPBackend.CUDNN_ATTENTION))
        res = {}
        for label, be in names:
            try:
                with sdpa_kernel(be):
                    for _ in range(2):
                        torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
                    torch.cuda.synchronize()
                    t = time.time()
                    for _ in range(10):
                        torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
                    torch.cuda.synchronize()
                ms = (time.time() - t) / 10 * 1000
                res[label] = ms
                lines.append(f"{'SDPA ' + label:<28}: works, {ms:.2f} ms (B1 H16 L2048 D128 bf16, causal)")
            except Exception as exc:  # noqa: BLE001
                lines.append(f"{'SDPA ' + label:<28}: NOT available ({str(exc).splitlines()[0][:100] if str(exc) else type(exc).__name__})")
        metrics["sdpa_flash_builtin"] = "works" if "flash (PyTorch built-in)" in res else "not available"

    _try(lines, tbs, "SDPA backends", sdpa_backends)

    # ---- flash_attn package (FlashAttention-2)
    def flash_attn_pkg() -> None:
        spec = importlib.util.find_spec("flash_attn")
        if spec is None:
            metrics["flash_attn"] = "not installed"
            lines.append(f"{'flash_attn package':<28}: NOT installed (normal on Windows: no official wheels; PyTorch's built-in flash SDPA is used instead)")
            return
        import flash_attn  # noqa: F401
        from flash_attn import flash_attn_func

        q = torch.randn(1, 2048, 16, 128, device=dev, dtype=torch.bfloat16)
        k, v = torch.randn_like(q), torch.randn_like(q)
        flash_attn_func(q, k, v, causal=True)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(10):
            flash_attn_func(q, k, v, causal=True)
        torch.cuda.synchronize()
        ms = (time.time() - t) / 10 * 1000
        ver = getattr(flash_attn, "__version__", "?")
        metrics["flash_attn"] = f"works ({ver})"
        lines.append(f"{'flash_attn package':<28}: installed {ver} and WORKS, {ms:.2f} ms (same shape as above)")

    r = _try(lines, tbs, "flash_attn package", flash_attn_pkg)
    if r is None and metrics["flash_attn"] == "unknown":
        metrics["flash_attn"] = "installed but fails (see errors)"

    # ---- CUDA Graphs
    def cuda_graphs() -> None:
        x = torch.randn(64, 2048, device=dev, dtype=torch.bfloat16)
        w = [torch.randn(2048, 2048, device=dev, dtype=torch.bfloat16) for _ in range(8)]

        def step(inp):
            for m in w:
                inp = torch.tanh(inp @ m) * 0.5
            return inp

        static_in = x.clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                step(static_in)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(200):
            step(static_in)
        torch.cuda.synchronize()
        eager = (time.time() - t) / 200 * 1000
        g = torch.cuda.CUDAGraph()
        tcap = time.time()
        with torch.cuda.graph(g):
            static_out = step(static_in)
        torch.cuda.synchronize()
        cap_ms = (time.time() - tcap) * 1000
        t = time.time()
        for _ in range(200):
            g.replay()
        torch.cuda.synchronize()
        graph = (time.time() - t) / 200 * 1000
        static_in.copy_(x)
        g.replay()
        ref = step(x)
        ok = bool(torch.allclose(static_out.float(), ref.float(), atol=1e-1, rtol=1e-1))
        metrics["cuda_graphs"] = f"works (x{eager / graph:.1f} on micro-test)" if ok else "captured but result differs"
        metrics["cuda_graph_speedup"] = round(eager / graph, 2)
        lines.append(f"{'CUDA Graphs':<28}: capture OK in {cap_ms:.0f} ms; 8 small matmuls: eager {eager:.3f} ms -> graph {graph:.3f} ms "
                     f"(x{eager / graph:.1f}); result matches eager: {ok}")

    r = _try(lines, tbs, "CUDA Graphs", cuda_graphs)
    if r is None and metrics["cuda_graphs"] == "unknown":
        metrics["cuda_graphs"] = "FAILED (see errors)"

    # ---- other
    lines.append(f"{'triton':<28}: " + ("installed" if importlib.util.find_spec("triton") else "not installed (torch.compile is not used by default)"))
    _optional_libs(lines, tbs, metrics, ctx)

    warn = []
    if metrics["cuda_graphs"].startswith(("FAILED", "captured")):
        warn.append("CUDA Graphs broken")
    if metrics["flash_attn"].startswith("installed but"):
        warn.append("flash_attn installed but fails")
    if metrics.get("sdpa_flash_builtin") != "works":
        warn.append("PyTorch flash SDPA unavailable")
    if tbs:
        warn.append("some sub-checks errored")
    status = "WARN" if warn else "OK"
    summary = (f"{metrics.get('name', '?')}, {metrics.get('vram_total_gb', '?')} GB, torch {torch.__version__}+CUDA {torch.version.cuda}; "
               f"flash_attn: {metrics['flash_attn']}; CUDA Graphs: {metrics['cuda_graphs']}" + (f"; PROBLEMS: {', '.join(warn)}" if warn else ""))
    return {"status": status, "summary": summary, "details": lines, "traceback": "\n".join(tbs), "metrics": metrics}


def _optional_libs(lines: List[str], tbs: List[str], metrics: Dict[str, Any], ctx: WorkerContext) -> None:
    """CTranslate2 (faster-whisper) and onnxruntime: do they see CUDA?"""
    def ct2() -> None:
        import ctranslate2

        n = ctranslate2.get_cuda_device_count()
        types = sorted(ctranslate2.get_supported_compute_types("cuda")) if n else []
        metrics["ct2_cuda_devices"] = n
        lines.append(f"{'ctranslate2':<28}: {ctranslate2.__version__}, CUDA devices {n}" + (f", compute types {', '.join(types)}" if types else ""))

    def ort() -> None:
        import onnxruntime as ort_

        lines.append(f"{'onnxruntime':<28}: {ort_.__version__}, providers {', '.join(ort_.get_available_providers())}")

    for label, fn in (("ctranslate2", ct2), ("onnxruntime", ort)):
        if importlib.util.find_spec(label) is None:
            lines.append(f"{label:<28}: not installed")
            continue
        _try(lines, tbs, label, fn)
