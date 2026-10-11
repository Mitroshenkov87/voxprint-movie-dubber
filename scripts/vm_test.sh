#!/usr/bin/env bash
# GPU virtual-machine check for Voxprint AI Movie Dubber.
#
#   ./scripts/vm_test.sh [--smoke|--full] [--json] [clips-dir]
#
# --smoke  dry-run JSON, selftest, a fetch-models check, and one short clip dub.
# --full   every clip in the folder, plus one actor-blend dub and one multi-voice dub.
# --json   print one JSON object on stdout. Progress stays on stderr.
#
# Exit 0 pass, 1 a case failed, 2 the machine is not set up, 3 no NVIDIA GPU.
# Clips default to ./clips. See docs/TESTING-GPU.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="smoke"
JSON=0
CLIPS=""

while [ $# -gt 0 ]; do
  case "$1" in
    --smoke) MODE="smoke" ;;
    --full) MODE="full" ;;
    --json) JSON=1 ;;
    -h|--help) sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    --*) printf 'unknown option: %s\n' "$1" >&2; exit 2 ;;
    *)
      if [ -n "$CLIPS" ]; then
        printf 'only one clips folder is accepted\n' >&2
        exit 2
      fi
      CLIPS="$1"
      ;;
  esac
  shift
done

CLIPS="${CLIPS:-$ROOT/clips}"
PY="${PYTHON:-python3}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
LOG="$WORK/run.log"
SUMMARY="$WORK/summary.json"
: >"$LOG"

say() { printf '%s\n' "$*" >&2; }

fail_setup() {
  say "setup: $1"
  if [ "$JSON" = 1 ]; then
    printf '{"mode":"%s","ok":false,"exit_code":2,"error":"%s"}\n' "$MODE" "$1"
  fi
  exit 2
}

if ! command -v "$PY" >/dev/null 2>&1; then
  fail_setup "python was not found"
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  say "no NVIDIA GPU: nvidia-smi is missing"
  if [ "$JSON" = 1 ]; then
    printf '{"mode":"%s","ok":false,"exit_code":3,"error":"nvidia-smi is missing"}\n' "$MODE"
  fi
  exit 3
fi
if ! nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then
  say "no NVIDIA GPU: nvidia-smi did not report a device"
  if [ "$JSON" = 1 ]; then
    printf '{"mode":"%s","ok":false,"exit_code":3,"error":"nvidia-smi reported no device"}\n' "$MODE"
  fi
  exit 3
fi
if [ ! -d "$CLIPS" ]; then
  fail_setup "clips folder does not exist: $CLIPS"
fi

mapfile -t VIDEOS < <(find "$CLIPS" -type f \( -iname '*.mp4' -o -iname '*.mkv' -o -iname '*.webm' -o -iname '*.mov' \) | sort)
if [ "${#VIDEOS[@]}" -eq 0 ]; then
  fail_setup "no video files in $CLIPS"
fi

shortest() {
  local best="" size best_size
  best_size=999999999999
  for path in "$@"; do
    size="$(wc -c < "$path" | tr -d ' ')"
    if [ "$size" -lt "$best_size" ]; then
      best="$path"
      best_size="$size"
    fi
  done
  printf '%s\n' "$best"
}

SHORT="$(shortest "${VIDEOS[@]}")"
PEAK_FILE="$WORK/vram-mib"
echo 0 >"$PEAK_FILE"
SAMPLER_PID=""

start_vram() {
  echo 0 >"$PEAK_FILE"
  (
    while true; do
      used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ' || true)"
      case "$used" in
        ''|*[!0-9]*) ;;
        *)
          current="$(cat "$PEAK_FILE")"
          if [ "$used" -gt "$current" ]; then
            echo "$used" >"$PEAK_FILE"
          fi
          ;;
      esac
      sleep 1
    done
  ) &
  SAMPLER_PID=$!
}

stop_vram() {
  if [ -n "${SAMPLER_PID:-}" ]; then
    kill "$SAMPLER_PID" >/dev/null 2>&1 || true
    wait "$SAMPLER_PID" >/dev/null 2>&1 || true
    SAMPLER_PID=""
  fi
}

RESULTS="$WORK/cases.jsonl"
: >"$RESULTS"

record() {
  local name="$1" code="$2" seconds="$3" note="$4"
  "$PY" - "$RESULTS" "$name" "$code" "$seconds" "$note" <<'PY'
import json, sys
path, name, code, seconds, note = sys.argv[1:]
row = {"name": name, "ok": int(code) == 0, "exit_code": int(code), "seconds": round(float(seconds), 3), "note": note}
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(row) + "\n")
PY
}

run_timed() {
  local name="$1"
  shift
  local start end code
  start="$(date +%s.%N)"
  set +e
  "$@" >>"$LOG" 2>&1
  code=$?
  set -e
  end="$(date +%s.%N)"
  "$PY" - "$name" "$code" "$start" "$end" "$RESULTS" <<'PY'
import json, sys
name, code, start, end, path = sys.argv[1:]
row = {"name": name, "ok": int(code) == 0, "exit_code": int(code), "seconds": round(float(end) - float(start), 3), "note": ""}
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(row) + "\n")
print(code)
PY
}

dub_video() {
  local name="$1" target="$2"
  start_vram
  local code
  code="$(run_timed "$name" "$PY" "$ROOT/main.py" run-project "$target" --json)"
  stop_vram
  return 0
}

make_project() {
  local name="$1" clip="$2" kind="$3"
  "$PY" - "$WORK/$name" "$clip" "$kind" <<'PY'
import sys
from pathlib import Path
from dubber.core.project import Project
folder, clip, kind = sys.argv[1:]
settings = {"target_lang": "ru", "source_lang": "en"}
if kind == "actor":
    settings["single_voice"] = {"kind": "actor", "id": "", "actor_weight": 0.5}
elif kind == "multi":
    settings["multi_voice"] = True
Project.create(Path(folder), Path(clip), **settings)
print(folder)
PY
}

say "mode $MODE, clips $CLIPS, short clip $SHORT"
run_timed "dry-run" "$PY" "$ROOT/main.py" --dry-run --json >/dev/null
run_timed "selftest" "$PY" "$ROOT/main.py" selftest --json >/dev/null
run_timed "fetch-models" "$PY" "$ROOT/main.py" fetch-models --dry-run --json >/dev/null

if [ "$MODE" = "smoke" ]; then
  dub_video "dub:$SHORT" "$SHORT"
else
  for clip in "${VIDEOS[@]}"; do
    dub_video "dub:$clip" "$clip"
  done
  actor="$(make_project actor "$SHORT" actor)"
  dub_video "actor-blend" "$actor"
  multi="$(make_project multi "$SHORT" multi)"
  dub_video "multi-voice" "$multi"
fi

"$PY" - "$RESULTS" "$LOG" "$PEAK_FILE" "$MODE" "$CLIPS" "$JSON" <<'PY'
import json, sys
from pathlib import Path
results, log_path, peak_path, mode, clips, as_json = sys.argv[1:]
cases = [json.loads(line) for line in Path(results).read_text(encoding="utf-8").splitlines() if line.strip()]
log = Path(log_path).read_text(encoding="utf-8", errors="replace")
backend = ""
graphs_note = ""
for line in log.splitlines():
    if "TTS backend:" in line:
        backend = line.split("TTS backend:", 1)[1].strip()
    if "TTS backends skipped:" in line or "CUDA Graphs" in line:
        graphs_note = line.strip()
cuda_graphs = backend.startswith("graphs") or "CUDA Graphs" in backend
if not cuda_graphs and not graphs_note:
    graphs_note = "the log did not name a TTS backend"
peak = 0
try:
    peak = int(Path(peak_path).read_text(encoding="utf-8").strip() or "0")
except ValueError:
    peak = 0
ok = all(case["ok"] for case in cases) and (cuda_graphs or bool(graphs_note))
# A missing explanation is a failure even when every process exited 0.
if not cuda_graphs and graphs_note == "the log did not name a TTS backend":
    ok = False
body = {
    "mode": mode,
    "ok": ok,
    "exit_code": 0 if ok else 1,
    "clips": clips,
    "cases": cases,
    "tts_backend": backend,
    "cuda_graphs": cuda_graphs,
    "graphs_note": graphs_note,
    "peak_vram_mib": peak,
}
text = json.dumps(body, indent=2)
if as_json == "1":
    print(text)
else:
    print(text, file=sys.stderr)
raise SystemExit(0 if ok else 1)
PY
