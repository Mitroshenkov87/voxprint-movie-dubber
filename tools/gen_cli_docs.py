"""Write ``docs/CLI.md`` from the argparse parsers in ``dubber.cli``.

    python tools/gen_cli_docs.py          # rewrite docs/CLI.md
    python tools/gen_cli_docs.py --check  # fail if the file is stale
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dubber.cli import EXIT_CODE_NOTE, EXIT_CODES, build_parser, build_window_parser

OUT = ROOT / "docs" / "CLI.md"
_SKIP_GROUP = {"positional arguments", "options", "optional arguments"}


def _is_switch(action: argparse.Action) -> bool:
    return isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)) or action.nargs == 0


def _label(action: argparse.Action) -> str:
    parts: list[str] = []
    for opt in action.option_strings:
        if _is_switch(action):
            parts.append(f"`{opt}`")
            continue
        meta = action.metavar or "VALUE"
        if isinstance(meta, tuple):
            meta = " ".join(str(item) for item in meta)
        parts.append(f"`{opt} {meta}`")
    return ", ".join(parts)


def _option_rows(parser: argparse.ArgumentParser) -> list[tuple[str, list[tuple[str, str]]]]:
    groups: list[tuple[str, list[tuple[str, str]]]] = []
    for group in parser._action_groups:
        rows: list[tuple[str, str]] = []
        for action in group._group_actions:
            if isinstance(action, (argparse._SubParsersAction, argparse._HelpAction)):
                continue
            if not action.option_strings:
                continue
            rows.append((_label(action), " ".join((action.help or "").split())))
        if rows:
            groups.append((group.title or "Options", rows))
    return groups


def _subcommands(parser: argparse.ArgumentParser):
    for action in parser._actions:
        if not isinstance(action, argparse._SubParsersAction):
            continue
        seen: set[int] = set()
        for name, sub in action.choices.items():
            if id(sub) in seen:
                continue
            seen.add(id(sub))
            aliases = [alias for alias, other in action.choices.items() if other is sub and alias != name]
            yield name, aliases, sub


def _table(rows: list[tuple[str, str]]) -> list[str]:
    lines = ["| Flag | Meaning |", "| --- | --- |"]
    lines.extend(f"| {label} | {help_text} |" for label, help_text in rows)
    return lines


def _usage(name: str, sub: argparse.ArgumentParser) -> str:
    extra = []
    for action in sub._actions:
        if action.option_strings or isinstance(action, argparse._HelpAction):
            continue
        extra.append(str(action.metavar or action.dest))
    return "python main.py " + " ".join([name, *extra])


def render() -> str:
    """Markdown command-line reference for the parsers in ``dubber.cli``."""
    parser = build_parser()
    lines = ["# Command line", "", (parser.description or "").rstrip(), ""]
    lines.extend([
        "```",
        "python main.py version",
        "python main.py diagnose",
        "python main.py fetch-models [--models KEY,KEY]",
        "python main.py run-project PATH [--languages SRC,TGT] [--stages NAME,NAME]",
        "python main.py info PATH",
        "python main.py selftest",
        "python main.py --dry-run --json",
        "```",
        "",
    ])
    for title, rows in _option_rows(parser):
        if title not in _SKIP_GROUP:
            lines.extend([f"## {title}", ""])
        lines.extend(_table(rows))
        lines.append("")
    lines.extend(["## Exit codes", "", "| Code | Name | When |", "| --- | --- | --- |"])
    for code, name, when in EXIT_CODES:
        lines.append(f"| {code} | {name} | {when} |")
    lines.extend(["", EXIT_CODE_NOTE, ""])
    for name, aliases, sub in _subcommands(parser):
        heading = name if not aliases else f"{name} ({', '.join(aliases)})"
        lines.extend([f"## {heading}", ""])
        if sub.description:
            lines.extend([sub.description.strip(), ""])
        lines.extend(["```", _usage(name, sub)])
        for alias in aliases:
            lines.append(_usage(alias, sub))
        lines.extend(["```", ""])
    if parser.epilog:
        lines.extend(["## --dry-run", "", "```", "python main.py --dry-run --json", "```", "", parser.epilog.strip(), ""])
    window = build_window_parser()
    lines.extend(["## Window and installer", ""])
    if window.description:
        lines.extend([window.description.strip(), ""])
    lines.extend([
        "The desktop window returns 3 (gpu) when the graphics card is not an RTX 40-series or newer, "
        "or the NVIDIA driver is older than branch 600. "
        "`--selftest` uses the same codes. A repaired component is exit 0 with outcome `repair`.",
        "",
    ])
    for _title, rows in _option_rows(window):
        lines.extend(_table(rows))
        lines.append("")
    for action in window._actions:
        if action.option_strings or isinstance(action, argparse._HelpAction):
            continue
        meta = action.metavar or action.dest
        lines.extend([f"Positional `{meta}`: {action.help}", ""])
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    """Rewrite ``docs/CLI.md``, or exit 1 when ``--check`` finds a stale copy."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--check", action="store_true", help="Fail when docs/CLI.md does not match the parsers.")
    args = parser.parse_args(argv)
    text = render()
    check = args.check
    if check:
        current = OUT.read_text(encoding="utf-8") if OUT.is_file() else ""
        if current != text:
            print(f"{OUT.relative_to(ROOT)} is stale. Run: python tools/gen_cli_docs.py", file=sys.stderr)
            return 1
        print(f"{OUT.relative_to(ROOT)} matches the parsers")
        return 0
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
