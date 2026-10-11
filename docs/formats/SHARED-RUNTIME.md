# Shared runtime manifest v1 (owner: Suite Lead, 2026-10-11)

Single schema for `%LOCALAPPDATA%\Voxprint\shared\manifest.json` (Linux `~/.local/share/voxprint/shared/manifest.json`).
Every app reads and writes THIS shape; no app-private variants. Blocker fix, applies in the current RC (RULES-FREEZE exception).

```json
{
  "schema": 1,
  "components": {
    "<component-id>": {
      "kind": "runtime | wheels | model | tool",
      "version": "2.11.0+cu130",
      "path": "runtime/py314-torch2.11-cu130",
      "sha256": "<hash of the archive or lock file it was installed from>",
      "installed_at": "2026-10-11T08:40:00+03:00",
      "refs": ["enlarge", "canvas"]
    }
  }
}
```

Rules
- Top-level keys are exactly `schema` (integer) and `components` (object keyed by id). Unknown keys are preserved on write, never dropped.
- Component ids are lowercase `[a-z0-9][a-z0-9._-]*`, e.g. `python-3.14`, `torch-2.11.0-cu130`, `onnxruntime-gpu-1.31`, `tensorrt-10`, `ffmpeg-8`, `model-<sha256-prefix>`.
- Reuse when id + version + sha256 match; otherwise install side by side under a new path. Never overwrite a component another app refs.
- `refs`: add your app id on install, remove it on uninstall; delete the component only when `refs` is empty.
- Writes are atomic: write `manifest.json.tmp`, fsync, rename; take the lock file `manifest.lock` (exclusive, 30 s timeout) around read-modify-write.
- Reader migration: if a file has `version` + `resources` (old Canvas shape), convert each resource into a component and write back schema 1. Readers MUST accept the old Enlarge shape too (it already is `schema`+`components`).
- The shared Python runtime includes torch (cu130) and `onnxruntime-gpu` only; installing CPU `onnxruntime` into it is forbidden (it shadows the GPU build). TensorRT 10 libs ship as component `tensorrt-10` for ORT 1.31.
