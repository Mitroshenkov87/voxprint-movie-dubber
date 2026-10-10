# Code audit

Automated pass used by pull requests, pushes to `main`, a weekly schedule, and the installer workflow.
The script is `tools/audit.py`. It writes one summary to the GitHub job summary and the full text to
`audit-report.txt` (uploaded as the `audit-report` artifact).

## What runs

| Tool | Scope | Release gate |
| --- | --- | --- |
| ruff | Pyflakes (`F`, including undefined names) and syntax (`E9`). Line length (`E501`) and pyupgrade (`UP`) are not selected. | Fails on any error |
| mypy | Non-strict (`mypy.ini`). Untyped functions are allowed. No global `ignore_errors`. | Fails on any error |
| bandit | Medium and high severity, medium and high confidence. | Fails on high only |
| pip-audit | Packages installed from `requirements.txt`, `installer/runtime-constraints.txt`, and CPU PyTorch `2.11.0` with torchaudio `2.11.0` and torchcodec `0.17.0` (CI only, `/TORCH=cpu`; a user install does not offer that build). A local version tag such as `+cpu` is removed so PyPI can see the release. | Fails on high or critical that are not in `tools/audit-allowlist.toml`. An advisory with no readable severity counts as high. |
| gitleaks | Secrets. Full git history on the weekly schedule, on a manual run, and when the installer workflow calls the audit. The commits of a push or pull request only, otherwise. | Fails on any finding |
| vulture | Dead code, confidence 80 and above. | Report only |
| radon | Cyclomatic complexity, rank D and worse. | Report only |

`dubber/third_party/` (vendored Look2Hear) is not scanned. It is upstream code copied in with its own licence.

When an advisory has a GitHub id (`GHSA`), that reviewed severity is the one that counts.
`MODERATE` is medium and does not block. Another database's label is used only when there is no GHSA.
When there is no reviewed label at all, the CVSS v3 base score is used (high is 7.0 and above, critical is 9.0 and above).

## How to run it locally

```
python -m pip install -r tools/requirements-audit.txt
python tools/audit.py --python <interpreter that has the app dependencies>
```

`--gitleaks-mode diff --gitleaks-range BASE..HEAD` matches a pull request. The default is a full-history secret scan.
`gitleaks` must be on `PATH` for that check. A missing blocking tool fails the run.

The installer workflow (`.github/workflows/build-installer.yml`) calls `.github/workflows/audit.yml` and does not
build when that job fails. CodeQL (Python, weekly and manual) is `.github/workflows/codeql.yml`. Dependabot opens
weekly pull requests for GitHub Actions only; Python pins stay in `requirements.txt` and the runtime lock.

## Fixed in this pass

- `dubber/engines/tts.py`, `dubber/core/project.py`: SHA-1 and MD5 are cache keys, project-folder names, and a
  deterministic mock-TTS seed. Marked `usedforsecurity=False` (bandit B324, high).
- `dubber/workers/w_tts.py`: the nested `synth` helper closed over `model`, and a later `del model` made ruff report
  the name as undefined (F821). The helper now binds the loaded objects as default arguments, same idea as the
  Audiobook Builder audit.
- Unused imports and unused timers in the diagnostics runner, the fetch worker, and the system checks.
- mypy findings in `dubber/` and `tools/` (optional values used without a local, ffmpeg argument lists, probe
  results typed as plain objects, Windows-only `msvcrt` / `os.add_dll_directory` not visible to mypy under
  `os.name == "nt"`). Fixed with annotations and narrower locals. No file-wide ignore was added. After merging
  the clip-regression work: `float()` in `_parse_vram_fraction` only sees `int`, `float`, or `str`, and the
  reference-clip ASR call passes a string model repo (the default when `asr_repo` is missing).
- `setuptools>=78.1.1` in `requirements.txt`. 78.1.0 is PYSEC-2025-49 / CVE-2025-47273 (high: path traversal in
  `PackageIndex.download`). 78.1.1 fixes that release. The program does not call that downloader; the floor keeps
  the vulnerable release out of the runtime. A current install resolves to 84.x, which also covers the later
  medium advisory PYSEC-2026-3447.

## Accepted (not changed)

| Tool | Item | Why it stays |
| --- | --- | --- |
| bandit B310 | `urllib.request.urlopen` in `dubber/core/subtitles.py`, `dubber/core/voice_catalog.py`, `dubber/diag/checks_system.py`, and `tools/audit.py` | Medium. The calls go to HTTPS hosts the program builds (subtitle services, `huggingface.co`, the OSV API), not to a `file:` URL taken from a user. |
| bandit B615 | `from_pretrained` / `snapshot_download` without a revision literal (`dubber/engines/translation.py`, `dubber/workers/w_mt.py`, `dubber/infra/model_store.py`) | Medium. Model files are checked against `model_manifest.json` (size and SHA-256) before use. `from_pretrained` then loads that local folder. Pinning every upstream tip is a separate product decision. |
| pip-audit | `accelerate==1.12.0` — PYSEC-2026-3804 / GHSA-4j2p-28q2-5m79 (sharded-checkpoint `weight_map` path traversal). GitHub rates it moderate. No fix version is published. | `requirements.txt` pins accelerate 1.12.0 on purpose (the verified stack). The advisory is not high or critical under the GitHub label, so it does not block and is not on the allowlist. |
| pip-audit | `torch==2.11.0` — PYSEC-2025-194 / GHSA-rrmf-rvhw-rf47 (`torch.jit.script` memory corruption). GitHub rates it low. The fix is torch 2.13.0. | Open at low severity. Torch 2.13 and 2.14 have no torchaudio wheel on the cu130 index, so the pin stays on the newest official pair (torch 2.11.0 + torchaudio 2.11.0). Low does not block and is not on the allowlist. |
| vulture | `dubber/workers/common.py` `default_rope` argument `seq_len` (100% confidence) | The name is part of the transformers RoPE initialiser signature. The shim has to accept it even though this workaround does not use the value. |
| radon | Rank D or E in `mixing.mix_range`, `voices.pick_reference_lines`, `segment.segment`, `script.translate_with_context`, `report.render`, `translation.complete_translate`, `model_store.ensure_model`, `stages.st_tts`, `window._update_steps`, `window._on_done`, `w_gpu.run`, `w_tts.run`, `w_asr.run`, and `tools/audit.py` `classify_pip` | Report only. These are the long pipeline, UI, and audit functions. Splitting them is not part of this pass. |
| ruff E501 / UP | Style and `pyupgrade` rewrites | The tree uses long lines and the existing typing style. Selecting those rules would bury the real errors. |
| mypy | `ignore_missing_imports` | Third-party libraries in this stack (torch, PySide6, and others) ship no stubs. Our own modules are still type-checked. There is no `ignore_errors`. |

`tools/audit-allowlist.toml` is empty: the high setuptools advisory is fixed by the version floor above, and the
accelerate advisory is moderate. An entry added later needs an id and a reason. The id matches the advisory or any alias.

## Tests

`python -m pytest -q` (the suite in `tests/`, including `tests/test_audit.py` for the gate rules).
