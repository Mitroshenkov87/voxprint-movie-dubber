"""Pages generated while MkDocs builds: the command-line reference, the roadmap, and the API reference."""
from __future__ import annotations

import sys
from pathlib import Path

import mkdocs_gen_files

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from gen_cli_docs import render


def _modules() -> list[str]:
    """Importable modules under ``dubber``, excluding the vendored Look2Hear tree, plus ``main``."""
    names: list[str] = []
    for path in sorted((ROOT / "dubber").rglob("*.py")):
        rel = path.relative_to(ROOT)
        if "third_party" in rel.parts:
            continue
        if rel.name == "__init__.py":
            names.append(".".join(rel.parent.parts))
        else:
            names.append(".".join(rel.with_suffix("").parts))
    names.append("main")
    return names


def main() -> None:
    """Write the generated pages into the MkDocs file set."""
    with mkdocs_gen_files.open("CLI.md", "w") as handle:
        handle.write(render())
    mkdocs_gen_files.set_edit_path("CLI.md", "dubber/cli.py")

    roadmap = ROOT / "ROADMAP.md"
    if roadmap.is_file():
        with mkdocs_gen_files.open("roadmap.md", "w") as handle:
            handle.write(roadmap.read_text(encoding="utf-8"))
        mkdocs_gen_files.set_edit_path("roadmap.md", "ROADMAP.md")

    lines = [
        "# API reference",
        "",
        "Generated from the docstrings of the public modules, classes, and functions.",
        "The vendored Look2Hear tree under `dubber/third_party` is not included.",
        "",
    ]
    for name in _modules():
        lines.extend([f"::: {name}", ""])
    with mkdocs_gen_files.open("api.md", "w") as handle:
        handle.write("\n".join(lines))
    mkdocs_gen_files.set_edit_path("api.md", "dubber/__init__.py")


main()
