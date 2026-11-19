"""The command-line reference is rendered from the argparse parsers, not from a second copy of the flags."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from dubber.cli import (
    EXIT_CODES,
    _COMMANDS,
    _SWITCHES,
    _VALUE_FLAGS,
    build_parser,
    build_window_parser,
    flag_sets,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_generator():
    path = ROOT / "tools" / "gen_cli_docs.py"
    spec = importlib.util.spec_from_file_location("gen_cli_docs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["gen_cli_docs"] = module
    spec.loader.exec_module(module)
    return module


def test_headless_flags_come_from_the_parser():
    switches, values, commands = flag_sets(build_parser())
    assert switches == _SWITCHES
    assert values == _VALUE_FLAGS
    assert commands == _COMMANDS
    assert "--json" in switches and "--dry-run" in switches
    assert "--models" in values and "--languages" in values
    assert "project-info" in commands and "run-project" in commands


def test_window_entry_points_are_on_their_own_parser():
    switches, values, commands = flag_sets(build_window_parser())
    assert commands == set()
    documented = switches | values
    for flag in (
        "--diagnose", "--worker", "--selftest", "--register-models-user", "--unregister-models-user",
        "--register-runtime-user", "--unregister-runtime-user", "--sync-suite-settings", "--out",
    ):
        assert flag in documented
    assert "--worker" not in _SWITCHES and "--worker" not in _VALUE_FLAGS


def test_generated_reference_lists_every_option_and_the_exit_codes():
    text = _load_generator().render()
    assert text == (ROOT / "docs" / "CLI.md").read_text(encoding="utf-8")
    for code, name, when in EXIT_CODES:
        assert f"| {code} | {name} | {when} |" in text
    assert "Code 1 is not used." in text
    for flag in _SWITCHES | _VALUE_FLAGS:
        assert flag in text
    for flag in ("--diagnose", "--worker", "--selftest", "--sync-suite-settings"):
        assert flag in text
