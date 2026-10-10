"""In-process checks that need no heavy libraries (the GUI process never imports torch): OS, hardware, power, disk, Python and package
versions, network reachability, ffmpeg, nvidia-smi and the Windows video controllers.  Each function returns a :class:`CheckResult`;
:func:`dubber.diag.runner.guarded` wraps them so an exception cannot stop the report.
"""
from __future__ import annotations

import importlib.metadata as md
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

from dubber import ffmpeg, paths
from dubber.appinfo import version_line
from dubber.diag.procs import find_nvidia_smi
from dubber.diag.report import CheckResult, Status
from dubber.infra.gpu_policy import GpuFact, evaluate_gpus, parse_compute, parse_driver_branch

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0

#: Packages whose versions matter for this project (looked up in the metadata of the environment, never imported).
PACKAGES = ["torch", "torchaudio", "transformers", "tokenizers", "huggingface_hub", "safetensors", "accelerate", "qwen-tts", "qwen-tts-hf",
            "faster-qwen3-tts", "flash-attn", "triton", "triton-windows", "faster-whisper", "ctranslate2", "onnxruntime", "silero-vad",
            "pyannote.audio", "sentencepiece", "numpy", "scipy", "soundfile", "librosa", "PySide6", "psutil", "imageio-ffmpeg"]


def _run(cmd: List[str], timeout: float = 15) -> str:
    """Output of a command (stdout, else stderr); '' on any failure."""
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=timeout, creationflags=_NO_WINDOW)
        return (_decode(p.stdout) or _decode(p.stderr) or "").strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _decode(raw: bytes) -> str:
    """UTF-8 if valid, else the console (OEM) code page on Windows: powercfg / PowerShell print localized text in cp866 etc."""
    if not raw:
        return ""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    if sys.platform == "win32":
        try:
            return raw.decode("oem", errors="replace")
        except LookupError:
            pass
    return raw.decode("utf-8", errors="replace")


def _ps(script: str, timeout: float = 25) -> str:
    """Run a PowerShell snippet (Windows only)."""
    exe = shutil.which("powershell") or shutil.which("pwsh")
    if not exe:
        return ""
    return _run([exe, "-NoProfile", "-NonInteractive", "-Command", script], timeout)


def check_app() -> CheckResult:
    r = CheckResult("system.app", "Program and run context", Status.INFO)
    r.summary = version_line()
    r.kv("program", version_line()).kv("executable", sys.executable).kv("frozen", bool(getattr(sys, "frozen", False)))
    r.kv("data folder", paths.app_home()).kv("models folder", paths.models_dir())
    from dubber.infra import gpu_lock, model_store

    users = model_store.read_users()
    r.kv("models folder users", ", ".join(sorted(users)) if users else "none registered")
    held = gpu_lock.read_lock()
    r.kv("shared GPU lock", "free" if held is None else f"{held.get('owner', '?')} / {held.get('job', '?')} (eta {held.get('eta', '?')})")
    r.kv("working dir", os.getcwd())
    env = {k: os.environ[k] for k in ("CUDA_VISIBLE_DEVICES", "HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "VOXPRINT_DUBBER_HOME", "VOXPRINT_HOME", "VOXPRINT_MODELS_DIR",
                                      "VOXPRINT_FFMPEG", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "PYTORCH_CUDA_ALLOC_CONF") if k in os.environ}
    r.kv("relevant env vars", ", ".join(f"{k}={v}" for k, v in env.items()) if env else "none set")
    return r


def check_os() -> CheckResult:
    r = CheckResult("system.os", "Operating system", Status.INFO)
    plat = platform.platform()
    r.kv("platform", plat).kv("machine", platform.machine()).kv("python", f"{platform.python_version()} ({platform.python_implementation()}, "
                                                                f"{'64' if sys.maxsize > 2**32 else '32'}-bit)")
    if sys.platform == "win32":
        try:
            v = sys.getwindowsversion()                                  # type: ignore[attr-defined]
            r.kv("windows build", f"{v.major}.{v.minor}.{v.build}")
            from dubber import platform_win

            chk = platform_win.check_os()
            r.kv("min. supported build", f"{platform_win.MIN_BUILD} (Windows 11 24H2) -> {'OK' if chk.ok else 'TOO OLD'}")
            if not chk.ok:
                r.status = Status.WARN
        except Exception as exc:  # noqa: BLE001
            r.kv("windows build", f"unknown ({exc})")
        ed = _ps("(Get-CimInstance Win32_OperatingSystem | Select-Object -ExpandProperty Caption)")
        if ed:
            r.kv("edition", ed.splitlines()[0])
        # long paths + developer mode matter for some Python packages
        lp = _ps("(Get-ItemProperty 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem' -ErrorAction SilentlyContinue).LongPathsEnabled")
        if lp:
            r.kv("long paths enabled", "yes" if lp.strip() == "1" else "no")
    r.summary = plat[:70]
    return r


def check_hardware() -> CheckResult:
    r = CheckResult("system.hw", "CPU, RAM, power", Status.INFO)
    cpu = platform.processor() or ""
    if sys.platform == "win32":
        name = _ps("(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)")
        cpu = name.splitlines()[0] if name else cpu
    elif os.path.exists("/proc/cpuinfo"):
        m = re.search(r"model name\s*:\s*(.+)", Path("/proc/cpuinfo").read_text(errors="replace"))
        cpu = m.group(1) if m else cpu
    r.metrics["cpu"] = cpu.strip() or "?"
    r.kv("cpu", r.metrics["cpu"])
    try:
        import psutil

        r.kv("cores / threads", f"{psutil.cpu_count(logical=False)} / {psutil.cpu_count(logical=True)}")
        vm = psutil.virtual_memory()
        r.metrics.update(ram_total_gb=round(vm.total / 1024 ** 3, 1), ram_free_gb=round(vm.available / 1024 ** 3, 1))
        r.kv("ram", f"{vm.total / 1024 ** 3:.1f} GB total, {vm.available / 1024 ** 3:.1f} GB available ({vm.percent:.0f}% used)")
        sw = psutil.swap_memory()
        r.kv("page file / swap", f"{sw.total / 1024 ** 3:.1f} GB total, {sw.used / 1024 ** 3:.1f} GB used")
        bat = psutil.sensors_battery()
        if bat is not None:
            plugged = "plugged in (AC)" if bat.power_plugged else "ON BATTERY"
            r.kv("power source", f"{plugged}, battery {bat.percent:.0f}%")
            r.metrics["power"] = plugged
            if not bat.power_plugged:
                r.status = Status.WARN
                r.summary = "ON BATTERY - laptop GPUs are throttled on battery, benchmark numbers will be low"
        else:
            r.metrics["power"] = "no battery (desktop)"
            r.kv("power source", "no battery detected (desktop or AC only)")
        if vm.available < 6 * 1024 ** 3:
            r.status = Status.WARN
            r.summary = (r.summary + "; " if r.summary else "") + "less than 6 GB RAM available"
    except Exception as exc:  # noqa: BLE001
        r.kv("psutil", f"unavailable ({type(exc).__name__}: {exc})")
    if sys.platform == "win32":
        plan = _run(["powercfg", "/getactivescheme"])
        if plan:
            r.kv("power plan", plan.replace("Power Scheme GUID:", "").strip())
            r.metrics["power"] = (r.metrics.get("power", "") + "; " + plan.split("(")[-1].rstrip(")")).strip("; ")
        # Windows per-app GPU preference (hybrid laptops: python.exe may be pinned to the integrated GPU)
        pref = _ps("(Get-ItemProperty 'HKCU:\\Software\\Microsoft\\DirectX\\UserGpuPreferences' -ErrorAction SilentlyContinue | Out-String)")
        if pref.strip():
            r.kv("per-app GPU preferences", "; ".join(ln.strip() for ln in pref.strip().splitlines() if "=" in ln or ":" in ln)[:300])
    r.summary = r.summary or f"{r.metrics.get('cpu', '?')[:40]}, {r.metrics.get('ram_total_gb', '?')} GB RAM, {r.metrics.get('power', '?')}"
    return r


def check_disk(extra: Optional[List[Path]] = None) -> CheckResult:
    r = CheckResult("system.disk", "Disk space", Status.OK)
    low = []
    for label, p in (("models folder", paths.models_dir()), ("temp folder", Path(__import__("tempfile").gettempdir())), ("desktop (report)", paths.desktop_dir())):
        try:
            u = shutil.disk_usage(p)
            r.kv(label, f"{p} -> {u.free / 1024 ** 3:.0f} GB free of {u.total / 1024 ** 3:.0f} GB")
            if u.free < 12 * 1024 ** 3 and label != "desktop (report)":
                low.append(label)
        except OSError as exc:
            r.kv(label, f"{p} -> cannot read ({exc})")
    if low:
        r.status, r.summary = Status.WARN, "less than 12 GB free on: " + ", ".join(low)
    else:
        r.summary = "enough free space"
    return r


def check_packages() -> CheckResult:
    r = CheckResult("system.packages", "Python packages (installed versions)", Status.INFO)
    found, missing = [], []
    for name in PACKAGES:
        try:
            found.append(f"{name} {md.version(name)}")
        except md.PackageNotFoundError:
            missing.append(name)
    for i in range(0, len(found), 3):
        r.line("  ".join(f"{s:<32}" for s in found[i:i + 3]).rstrip())
    r.kv("not installed", ", ".join(missing) if missing else "-", 16)
    r.summary = f"{len(found)} installed, {len(missing)} not installed"
    return r


def check_network(timeout: float = 5.0) -> CheckResult:
    """Reach the places models come from.  Only a TCP connect + HEAD request; nothing is uploaded."""
    r = CheckResult("system.network", "Network (model download hosts)", Status.OK)
    bad = []
    for host in ("huggingface.co", "github.com"):
        t = time.time()
        try:
            socket.create_connection((host, 443), timeout=timeout).close()
            r.kv(host, f"reachable ({(time.time() - t) * 1000:.0f} ms)")
        except OSError as exc:
            r.kv(host, f"NOT reachable: {type(exc).__name__}: {exc}")
            bad.append(host)
    try:
        import urllib.request

        req = urllib.request.Request("https://huggingface.co/api/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base", method="GET")
        t = time.time()
        with urllib.request.urlopen(req, timeout=timeout + 5) as resp:
            r.kv("HF API", f"HTTP {resp.status} ({(time.time() - t) * 1000:.0f} ms)")
    except Exception as exc:  # noqa: BLE001
        r.kv("HF API", f"failed: {type(exc).__name__}: {str(exc)[:120]}")
        bad.append("HF API")
    if bad:
        r.status, r.summary = Status.WARN, "cannot reach: " + ", ".join(bad) + " (models can only be used if already downloaded)"
    else:
        r.summary = "huggingface.co and github.com reachable"
    return r


def check_ffmpeg() -> CheckResult:
    r = CheckResult("system.ffmpeg", "ffmpeg / ffprobe", Status.OK)
    info = ffmpeg.ffmpeg_info(refresh=True)
    if not info.ffmpeg:
        r.status, r.summary = Status.FAIL, "ffmpeg NOT FOUND - the app cannot read or write movies without it"
        r.kv("problems", "; ".join(info.problems) or "none found anywhere (PATH, program folder, app tools folder, imageio-ffmpeg)")
        return r
    r.kv("ffmpeg", info.ffmpeg).kv("version", info.version).kv("found via", info.source).kv("ffprobe", info.ffprobe or "not found (the app falls back to parsing ffmpeg output)")
    out = _run([info.ffmpeg, "-hide_banner", "-encoders"], 20)
    enc = {name: bool(re.search(rf"^\s*A[.\w]+\s+{name}\b", out, re.M)) for name in ("aac", "libopus", "flac", "eac3", "ac3", "libmp3lame", "libfdk_aac")}
    r.kv("audio encoders", ", ".join(f"{k}={'yes' if v else 'no'}" for k, v in enc.items()))
    cfg = _run([info.ffmpeg, "-hide_banner", "-buildconf"], 20)
    lic = "GPL" if "--enable-gpl" in cfg else "LGPL (no --enable-gpl)" if cfg else "unknown"
    nf = " + NONFREE" if "--enable-nonfree" in cfg else ""
    r.kv("build licence", lic + nf)
    hw = _run([info.ffmpeg, "-hide_banner", "-hwaccels"], 20).replace("Hardware acceleration methods:", "").split()
    r.kv("hwaccels", ", ".join(hw) or "none")
    if not enc["aac"]:
        r.status, r.summary = Status.WARN, f"{info.version.split(' Copyright')[0]} - no AAC encoder"
    else:
        r.summary = f"{info.version.split(' Copyright')[0]} ({info.source}); ffprobe {'yes' if info.ffprobe else 'no'}"
    if info.source.startswith("imageio"):
        r.status = Status.WARN
        r.summary += " - fallback GPL build; install an LGPL build for distribution"
    return r


def _mib(text: str) -> Optional[float]:
    return round(float(text) / 1024, 2) if text.replace(".", "").isdigit() else None


def query_nvidia_smi() -> dict:
    """Static facts from nvidia-smi: name, compute capability, driver branch, VRAM, power and clocks.  {} when absent."""
    smi = find_nvidia_smi()
    if not smi:
        return {}
    q = _run([smi, "--query-gpu=name,compute_cap,driver_version,memory.total,memory.used,power.limit,power.max_limit,"
                   "clocks.max.sm,clocks.max.mem,pcie.link.gen.current,pcie.link.width.current,temperature.gpu,"
                   "utilization.gpu,persistence_mode,display_active",
              "--format=csv,noheader,nounits"])
    out: dict = {"smi": smi}
    rows = [ln for ln in q.splitlines() if ln.strip()]
    if not rows or "," not in rows[0]:
        out["error"] = q[:300]
        return out
    gpus = []
    for row in rows:
        c = [x.strip() for x in row.split(",")]
        if len(c) < 15:
            continue
        branch = parse_driver_branch(c[2])
        gpus.append(dict(name=c[0], compute_cap=c[1], driver=c[2], driver_branch=branch,
                         vram_total_gb=_mib(c[3]), vram_used_gb=_mib(c[4]),
                         power_limit_w=c[5], power_max_w=c[6], sm_max_mhz=c[7], mem_max_mhz=c[8],
                         pcie=f"gen{c[9]} x{c[10]}", temp_c=c[11], util=c[12], persistence=c[13], display_active=c[14]))
    out["gpus"] = gpus
    full = _run([smi])
    m = re.search(r"CUDA (?:UMD )?Version:\s*([\d.]+)", full)      # drivers 6xx write "CUDA UMD Version"
    out["cuda_driver_api"] = m.group(1) if m else ""
    return out


def _gpu_facts(rows: List[dict]) -> tuple[GpuFact, ...]:
    found: List[GpuFact] = []
    for row in rows:
        name = str(row.get("name") or "").strip()
        compute = parse_compute(str(row.get("compute_cap") or ""))
        branch = row.get("driver_branch")
        if not isinstance(branch, int):
            branch = parse_driver_branch(str(row.get("driver") or ""))
        if not name or compute is None or branch is None:
            continue
        found.append(GpuFact(name, compute, str(row.get("driver") or ""), branch))
    return tuple(found)


def check_gpu_smi() -> CheckResult:
    r = CheckResult("gpu.smi", "NVIDIA driver (nvidia-smi)", Status.OK)
    d = query_nvidia_smi()
    if not d:
        decision = evaluate_gpus(())
        r.status = Status.FAIL
        r.summary = decision.summary
        r.line("nvidia-smi was not found on PATH, in System32 or in 'C:\\Program Files\\NVIDIA Corporation\\NVSMI'.")
        r.line(decision.summary)
        return r
    if d.get("error") or not d.get("gpus"):
        decision = evaluate_gpus(())
        r.status = Status.FAIL
        r.summary = decision.summary
        r.line(d.get("error", "") or decision.summary)
        return r
    facts = _gpu_facts(d["gpus"])
    decision = evaluate_gpus(facts)
    shown = facts[0] if facts else None
    if shown is not None:
        r.metrics.update(name=shown.name, driver=shown.driver, compute_cap=f"{shown.compute[0]}.{shown.compute[1]}",
                         driver_branch=shown.driver_branch, vram_total_gb=d["gpus"][0]["vram_total_gb"],
                         cuda_driver_api=d.get("cuda_driver_api", ""))
    r.kv("nvidia-smi", d["smi"])
    for i, g in enumerate(d["gpus"]):
        cap = g.get("compute_cap") or "?"
        branch = g.get("driver_branch")
        r.line(f"GPU {i}: {g['name']}")
        r.kv("  compute capability", str(cap)).kv("  driver version", g["driver"]).kv("  driver branch", str(branch if branch is not None else "?"))
        r.kv("  CUDA version (driver API)", d.get("cuda_driver_api") or "?")
        r.kv("  VRAM total / used now", f"{g['vram_total_gb']} GB / {g['vram_used_gb']} GB")
        r.kv("  power limit (current/max)", f"{g['power_limit_w']} W / {g['power_max_w']} W")
        r.kv("  max clocks (SM / memory)", f"{g['sm_max_mhz']} / {g['mem_max_mhz']} MHz").kv("  PCIe link now", g["pcie"])
        r.kv("  temperature / util now", f"{g['temp_c']} C / {g['util']} %").kv("  display attached", g["display_active"])
    if not decision.ok:
        r.status = Status.FAIL
        r.summary = decision.summary
        r.line(decision.summary)
        return r
    if decision.qualifying:
        best = decision.qualifying[0]
        r.metrics.update(name=best.name, driver=best.driver, compute_cap=f"{best.compute[0]}.{best.compute[1]}",
                         driver_branch=best.driver_branch)
    try:
        used = float(d["gpus"][0]["vram_used_gb"])
        if used > 2.5:
            r.status = Status.WARN
            r.line(f"NOTE: {used:.1f} GB of VRAM are already in use by other programs (browser, games, other AI tools) - close them for a clean benchmark.")
    except (TypeError, ValueError):
        pass
    cap = r.metrics.get("compute_cap", "?")
    branch = r.metrics.get("driver_branch", "?")
    g = d["gpus"][0]
    if decision.qualifying:
        g = next((row for row in d["gpus"] if row.get("name") == decision.qualifying[0].name), g)
    r.summary = (f"{r.metrics.get('name', g['name'])}, compute capability {cap}, "
                 f"driver {r.metrics.get('driver', g['driver'])} (branch {branch}), "
                 f"{g['vram_total_gb']} GB, CUDA {d.get('cuda_driver_api') or '?'}")
    if r.status == Status.WARN and r.details:
        r.summary += "; " + str(r.details[-1])[6:100]
    return r


def check_ctranslate2() -> CheckResult:
    """CTranslate2 (faster-whisper) must see a CUDA device. The CUDA 12 DLLs are registered first."""
    from dubber.infra import cuda_dlls

    r = CheckResult("gpu.ctranslate2", "CTranslate2 CUDA", Status.OK)
    dirs = cuda_dlls.expose()
    if dirs:
        r.kv("CUDA 12 DLL folders", "; ".join(dirs))
        r.kv("cublas64_12.dll", "yes" if cuda_dlls.has_cublas12([Path(d) for d in dirs]) else "no")
    count = cuda_dlls.cuda_device_count()
    ok, summary = cuda_dlls.cuda_device_summary(count)
    r.summary = summary
    if count is not None:
        r.kv("CUDA devices", str(count))
    if not ok:
        r.status = Status.FAIL
        r.line(summary)
    return r


def check_gpu_wmi() -> CheckResult:
    """Windows: every video controller (hybrid ROG laptops have an integrated GPU + the NVIDIA one)."""
    r = CheckResult("gpu.adapters", "Video adapters (Windows)", Status.INFO)
    if sys.platform != "win32":
        r.status, r.summary = Status.SKIP, "not Windows"
        return r
    out = _ps("Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name + ' | driver ' + $_.DriverVersion + ' | ' + [string]([math]::Round($_.AdapterRAM/1GB,1)) + ' GB (WMI, capped at 4) | status ' + $_.Status }")
    if not out:
        r.status, r.summary = Status.WARN, "could not query Win32_VideoController (PowerShell unavailable?)"
        return r
    names = out.splitlines()
    for ln in names:
        r.line(ln.strip())
    r.summary = f"{len(names)} adapter(s): " + "; ".join(n.split("|")[0].strip() for n in names)
    return r
