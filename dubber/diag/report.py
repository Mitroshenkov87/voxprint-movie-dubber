"""The diagnostic report: data model, plain-text renderer, redaction and a crash-safe writer.

Design goals
------------
* **Readable and pasteable**: ASCII layout (the only non-ASCII text is real data such as a Russian test phrase), fixed-width
  tables, a SUMMARY block first, details below, all tracebacks collected at the end.
* **One failing check never stops the report**: every check yields a :class:`CheckResult`; exceptions become FAIL entries
  carrying the traceback (see :func:`dubber.diag.runner.guarded`).
* **Crash-safe**: the runner re-saves the file after every check (atomic replace), so even if the program, the GPU driver or
  Windows dies half-way the Desktop already has a report saying ``INCOMPLETE`` and everything measured so far.
* **No secrets / personal data**: :func:`redact` hides Hugging Face tokens, bearer tokens and the Windows user name.
"""
from __future__ import annotations

import datetime as _dt
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

REPORT_FORMAT = 1
WIDTH = 78


class Status(str, Enum):
    """Outcome of one check."""
    OK = "OK"        # worked as expected
    WARN = "WARN"    # worked, but something is suboptimal (e.g. fell back to a slower path)
    FAIL = "FAIL"    # did not work (error, crash, timeout)
    SKIP = "SKIP"    # not run (missing prerequisite, disabled by the user, nothing to test on this machine)
    INFO = "INFO"    # facts only


_ORDER = {Status.FAIL: 0, Status.WARN: 1, Status.OK: 2, Status.INFO: 3, Status.SKIP: 4}


@dataclass
class CheckResult:
    """Result of one check (one line in the SUMMARY, one block in DETAILS)."""
    id: str                                   # stable dotted id, e.g. "tts.graphs_sdpa"
    title: str
    status: Status = Status.INFO
    summary: str = ""                         # one line (shown in SUMMARY)
    details: List[str] = field(default_factory=list)   # free-form lines for the DETAILS block
    seconds: Optional[float] = None           # wall time of the whole check
    traceback: str = ""                       # full traceback / stderr tail, goes to ERRORS
    metrics: Dict[str, Any] = field(default_factory=dict)   # numbers for KEY NUMBERS (see :func:`key_numbers`)

    def kv(self, key: str, value: Any, width: int = 24) -> "CheckResult":
        """Append a ``key : value`` detail line."""
        self.details.append(f"{key:<{width}}: {value}")
        return self

    def line(self, text: str = "") -> "CheckResult":
        """Append a plain detail line."""
        self.details.append(text)
        return self


# ---------------------------------------------------------------------------------------------- redaction
_HF_TOKEN = re.compile(r"\bhf_[A-Za-z0-9]{8,}\b")
_BEARER = re.compile(r"(?i)\b(bearer|authorization:?)\s+[A-Za-z0-9._\-]{12,}")
_API_KEY = re.compile(r"(?i)\b(api[_-]?key|token|secret|password)\s*[=:]\s*['\"]?[A-Za-z0-9._\-]{12,}")
_WIN_USER = re.compile(r"(?i)([A-Z]:[\\/]+Users[\\/]+)([^\\/\s\"']+)")
_LINUX_HOME = re.compile(r"(?<![\w.\-])(/home/|/Users/)([^/\s\"']+)")


def redact(text: str, extra_secrets: Sequence[str] = ()) -> str:
    """Hide tokens and the user name in ``text`` (applied to the whole report just before it is written)."""
    for s in extra_secrets:
        if s and len(s) >= 6:
            text = text.replace(s, "***")
    text = _HF_TOKEN.sub("hf_***", text)
    text = _BEARER.sub(lambda m: f"{m.group(1)} ***", text)
    text = _API_KEY.sub(lambda m: f"{m.group(1)}=***", text)
    text = _WIN_USER.sub(lambda m: m.group(1) + "<user>", text)
    text = _LINUX_HOME.sub(lambda m: m.group(1) + "<user>", text)
    return text


# ---------------------------------------------------------------------------------------------- formatting helpers
def fmt_seconds(s: Optional[float]) -> str:
    """``0.42 s`` / ``12.3 s`` / ``3 min 05 s`` / ``-``."""
    if s is None:
        return "-"
    if s < 10:
        return f"{s:.2f} s"
    if s < 120:
        return f"{s:.1f} s"
    m, sec = divmod(int(round(s)), 60)
    if m < 120:
        return f"{m} min {sec:02d} s"
    h, m = divmod(m, 60)
    return f"{h} h {m:02d} min"


def fmt_gb(x: Optional[float]) -> str:
    """Gigabytes with one decimal, ``-`` when unknown."""
    return "-" if x is None else f"{x:.1f} GB"


def fmt_num(x: Optional[float], digits: int = 2) -> str:
    """Format ``x`` to ``digits`` decimal places, or ``-`` when ``x`` is missing."""
    return "-" if x is None else f"{x:.{digits}f}"


def table(headers: Sequence[str], rows: Iterable[Sequence[Any]], indent: str = "  ") -> List[str]:
    """Fixed-width text table (left aligned, two spaces between columns)."""
    rows = [[str(c) for c in r] for r in rows]
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))
    def fmt(r: Sequence[str]) -> str:
        return indent + "  ".join(c.ljust(widths[i]) for i, c in enumerate(r)).rstrip()
    out = [fmt(list(headers)), indent + "  ".join("-" * w for w in widths)]
    out += [fmt(r) for r in rows]
    return out


def _wrap_block(text: str, indent: str = "    ", limit: int = 140) -> List[str]:
    """Indent a multi-line block; very long lines are cut (a runaway stderr must not bloat the report)."""
    out = []
    for ln in text.rstrip().splitlines():
        out.append(indent + (ln if len(ln) <= limit else ln[:limit] + " ...[cut]"))
    return out


# ---------------------------------------------------------------------------------------------- the report
@dataclass
class Report:
    """A diagnostic report under construction.  Thread-compatible when only one thread calls :meth:`add`."""
    app_line: str = ""
    options: Dict[str, Any] = field(default_factory=dict)
    header: List[tuple] = field(default_factory=list)       # extra (key, value) lines in the header
    results: List[CheckResult] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    started: float = field(default_factory=time.time)
    finished: Optional[float] = None
    complete: bool = False
    max_traceback_lines: int = 60
    secrets: List[str] = field(default_factory=list)       # strings that must never appear (tokens passed in by the user)
    current_step: str = ""                                  # shown while INCOMPLETE: the check that was running when the file was last saved

    # -- building
    def add(self, result: CheckResult) -> CheckResult:
        """Add a result (a result with the same ``id`` replaces the old one, keeping its position)."""
        for i, old in enumerate(self.results):
            if old.id == result.id:
                self.results[i] = result
                break
        else:
            self.results.append(result)
        return result

    def get(self, check_id: str) -> Optional[CheckResult]:
        """Return the check with this id, or None when it has not been added."""
        for r in self.results:
            if r.id == check_id:
                return r
        return None

    def note(self, text: str) -> None:
        """Append a note printed under the summary totals."""
        self.notes.append(text)

    def finish(self, complete: bool = True) -> None:
        """Record the finish time. ``complete`` False leaves the rendered state as ``INCOMPLETE``."""
        self.finished = time.time()
        self.complete = complete

    def counts(self) -> Dict[str, int]:
        """Count checks for each status, including statuses that did not occur."""
        c = {s.value: 0 for s in Status}
        for r in self.results:
            c[r.status.value] += 1
        return c

    # -- rendering
    def render(self) -> str:
        """The whole report as one string (already redacted)."""
        lines: List[str] = []
        bar = "=" * WIDTH
        sub = "-" * WIDTH
        lines += [bar, " VOXPRINT AI MOVIE DUBBER - DIAGNOSTIC REPORT", bar]
        now = self.finished or time.time()
        lines.append(f"{'Report format':<14}: {REPORT_FORMAT}")
        if self.app_line:
            lines.append(f"{'App':<14}: {self.app_line}")
        lines.append(f"{'Started':<14}: {_local_iso(self.started)}")
        if self.finished:
            lines.append(f"{'Finished':<14}: {_local_iso(self.finished)}")
        lines.append(f"{'Duration':<14}: {fmt_seconds(now - self.started)}")
        state = "COMPLETE" if self.complete else "INCOMPLETE (still running, or the program/driver/Windows died - see the last check below)"
        lines.append(f"{'State':<14}: {state}")
        if not self.complete and self.current_step:
            lines.append(f"{'Now running':<14}: {self.current_step}   <-- if the report stops here, this step hung or crashed the program/driver")
        for k, v in self.header:
            lines.append(f"{k:<14}: {v}")
        if self.options:
            lines.append(f"{'Options':<14}: " + ", ".join(f"{k}={v}" for k, v in self.options.items()))
        lines.append("")
        lines.append("Legend: OK = works | WARN = works but suboptimal/fallback | FAIL = broken (see ERRORS) | SKIP = not run | INFO = facts")

        lines += ["", sub, " SUMMARY", sub]
        for r in self.results:
            sec = fmt_seconds(r.seconds) if r.seconds is not None else ""
            text = r.summary or r.title
            lines.append(f" [{r.status.value:<4}] {r.id:<24} {text}" + (f"   ({sec})" if sec else ""))
        c = self.counts()
        lines.append(" Totals: " + ", ".join(f"{c[s.value]} {s.value}" for s in (Status.OK, Status.WARN, Status.FAIL, Status.SKIP, Status.INFO) if c[s.value]))
        for n in self.notes:
            lines.append(f" NOTE: {n}")

        kn = key_numbers(self.results)
        if kn:
            lines += ["", sub, " KEY NUMBERS", sub] + kn

        lines += ["", sub, " DETAILS", sub]
        for r in self.results:
            head = f"### {r.id} - {r.title}  [{r.status.value}]"
            if r.seconds is not None:
                head += f"  ({fmt_seconds(r.seconds)})"
            lines.append(head)
            if r.summary:
                lines.append(f"    {r.summary}")
            for d in r.details:
                lines.append(("    " + d) if d else "")
            lines.append("")

        errs = [r for r in self.results if r.traceback]
        lines += [sub, " ERRORS AND TRACEBACKS" + ("" if errs else " (none)"), sub]
        for r in errs:
            lines.append(f"### {r.id} [{r.status.value}]")
            tb = r.traceback.rstrip().splitlines()
            if len(tb) > self.max_traceback_lines:
                half = self.max_traceback_lines // 2
                tb = tb[:half] + [f"    ... {len(tb) - 2 * half} lines cut ..."] + tb[-half:]
            lines += _wrap_block("\n".join(tb))
            lines.append("")
        lines.append("=== END OF REPORT (" + ("complete" if self.complete else "INCOMPLETE") + ") ===")
        return redact("\n".join(lines) + "\n", self.secrets)


def _local_iso(ts: float) -> str:
    """Local time with UTC offset, e.g. ``2026-10-04 14:30:12 +03:00``."""
    d = _dt.datetime.fromtimestamp(ts).astimezone()
    off = d.strftime("%z")
    off = f"{off[:3]}:{off[3:]}" if len(off) == 5 else off
    return d.strftime("%Y-%m-%d %H:%M:%S ") + off


# ---------------------------------------------------------------------------------------------- KEY NUMBERS
TTS_MODE_LABELS = {
    "standard_sdpa": "standard, SDPA (no FlashAttention, no graphs)",
    "standard_fa2": "standard + FlashAttention-2",
    "graphs_sdpa": "CUDA Graphs + SDPA",
    "graphs_fa2": "CUDA Graphs + FlashAttention-2",
}


def key_numbers(results: Sequence[CheckResult]) -> List[str]:
    """The few lines worth pasting into a chat: GPU, TTS speed per mode, per-stage times.  Built from ``CheckResult.metrics``."""
    by = {r.id: r for r in results}
    out: List[str] = []
    g = by.get("gpu.torch")
    gm = (g.metrics if g else {}) or {}
    smi_row = by.get("gpu.smi")
    smi = (smi_row.metrics if smi_row else {}) or {}
    if gm or smi:
        name = gm.get("name") or smi.get("name") or "no NVIDIA GPU detected"
        vram = gm.get("vram_total_gb") or smi.get("vram_total_gb")
        cc = smi.get("compute_cap") or "-"
        branch = smi.get("driver_branch")
        branch_txt = "-" if branch in (None, "") else str(branch)
        out.append(f"GPU: {name}; compute capability {cc}; VRAM {fmt_gb(vram)}; "
                   f"driver {smi.get('driver', '-')} (branch {branch_txt}); "
                   f"torch {gm.get('torch', '-')} (CUDA {gm.get('cuda', '-')}); "
                   f"FlashAttention pkg: {gm.get('flash_attn', '-')}; CUDA Graphs: {gm.get('cuda_graphs', '-')}")
    sysr = by.get("system.hw")
    if sysr and sysr.metrics:
        out.append(f"CPU: {sysr.metrics.get('cpu', '-')}; RAM {fmt_gb(sysr.metrics.get('ram_total_gb'))} "
                   f"(free {fmt_gb(sysr.metrics.get('ram_free_gb'))}); power: {sysr.metrics.get('power', '-')}")
    rows = []
    for mode, label in TTS_MODE_LABELS.items():
        r = by.get(f"tts.{mode}")
        if not r:
            continue
        m = r.metrics
        rows.append([label, r.status.value, fmt_num(m.get("load_s"), 1), fmt_num(m.get("warmup_s"), 1),
                     fmt_num(m.get("rtf"), 3), fmt_num(m.get("xrt"), 2), fmt_num(m.get("peak_vram_gb"), 1)])
    if rows:
        out += ["", "TTS speed (Qwen3-TTS, test phrase; RTF = generation time / audio time, lower is better; x-real-time > 1 means faster than playback):"]
        out += table(["mode", "status", "load s", "warmup s", "RTF", "x-real-time", "peak VRAM GB"], rows)
    st = [r for r in results if r.id.startswith("stage.")]
    if st:
        rows = [[r.id[len("stage."):], r.status.value, fmt_num(r.metrics.get("load_s"), 1), fmt_num(r.metrics.get("run_s"), 2),
                 r.metrics.get("device", "-"), fmt_num(r.metrics.get("peak_vram_gb"), 1), r.summary[:44]] for r in st]
        out += ["", "Pipeline stages on the bundled test clip (load = model load, run = processing the clip):"]
        out += table(["stage", "status", "load s", "run s", "device", "peak VRAM GB", "result"], rows)
    return out


# ---------------------------------------------------------------------------------------------- writing
def report_filename(ts: Optional[float] = None) -> str:
    """``Voxprint-MovieDubber-Diagnostics-20261004-143012.txt``"""
    return "Voxprint-MovieDubber-Diagnostics-" + time.strftime("%Y%m%d-%H%M%S", time.localtime(ts or time.time())) + ".txt"


def write_text_atomic(path: Path, text: str, retries: int = 3) -> None:
    """Write ``text`` (UTF-8) to ``path`` through a temp file + ``os.replace``; raises OSError after ``retries`` failures."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    last: Optional[BaseException] = None
    for attempt in range(retries):
        try:
            with open(tmp, "w", encoding="utf-8", newline=None) as fh:    # text mode: "\n" becomes "\r\n" on Windows (Notepad)
                fh.write(text)
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    pass
            os.replace(tmp, path)
            return
        except OSError as exc:            # AV scanner / OneDrive sync can hold the file for a moment
            last = exc
            time.sleep(0.2 * (attempt + 1))
    try:
        tmp.unlink()
    except OSError:
        pass
    raise OSError(f"cannot write {path}: {last}")


def save_report(report: Report, primary: Path, fallback_dir: Optional[Path] = None) -> Path:
    """Write the rendered report to ``primary``; if that fails, to ``fallback_dir`` under the same name.  Returns the real path."""
    text = report.render()
    try:
        primary.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(primary, text)
        return primary
    except OSError:
        if fallback_dir is None:
            raise
        fallback_dir.mkdir(parents=True, exist_ok=True)
        alt = fallback_dir / primary.name
        write_text_atomic(alt, text)
        return alt
