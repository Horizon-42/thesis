"""The package's layout rules, enforced rather than described.

`outputs/<path>/` holds each prediction path's own code behind one strategy (review §4.2);
`outputs/control/` exists because twelve `control_*.py` files at the top level once named
their subject but not their role. What keeps that from coming back is checkable:

1. no new `control_*.py` may appear at the top level;
2. a module belongs under `outputs/control/` only if EVERY consumer of it is
   control-specific — which is why `terminal_state_loss`, `arc_length_geometry`,
   `fixed_dt_supervision` and `flyability` stay outside it;
3. the direction: only a path's strategy seam reaches the spine, nothing under `outputs/`
   imports the loop, and `dataset` reaches only the lazy registry.

The second rule is the one worth testing: without it the package slowly absorbs shared
modules and starts claiming ownership it does not have, which is worse than the flat
listing because the name now lies.
"""

from __future__ import annotations

import ast

from pathlib import Path

import re

import sys

TS_DIR = Path(__file__).resolve().parents[1]

def _module_files() -> list[Path]:
    return [
        path
        for path in TS_DIR.rglob("*.py")
        if "__pycache__" not in path.parts
        and "vendor" not in path.parts
        and "tests" not in path.parts
        and "archive" not in path.parts
    ]

def _imported_names(path: Path) -> set[str]:
    """Every module name a file imports, with relative imports resolved.

    `from .common import x` inside `cli/` reads as `node.module == "common"` and
    `node.level == 1`; without the resolution below the layering checks would see a
    top-level `common` that does not exist and miss a real edge.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    # Root `run_ts_*.py` runners live outside the package and have no relative imports.
    package = (
        path.parent.relative_to(TS_DIR).parts
        if path.is_relative_to(TS_DIR) else ()
    )
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                prefix = package[: len(package) - node.level + 1]
                base = ".".join((*prefix, node.module)) if node.module else ".".join(prefix)
                names.add(base)
                names.update(f"{base}.{alias.name}" for alias in node.names)
            elif node.module:
                # `from ts_transformer.training import train` names the loop module as
                # surely as `from ts_transformer.training.train import …` does.
                names.add(node.module)
                names.update(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    # The layering rules below are written in package-relative names (`control.envelope`,
    # `dataset`); every live import is qualified (`ts_transformer.outputs.envelope`).
    return {_package_relative(name) for name in names}

PACKAGE = "ts_transformer"

def _package_relative(name: str) -> str:
    return name[len(PACKAGE) + 1:] if name.startswith(PACKAGE + ".") else name

MANOEUVRE = TS_DIR / "manoeuvre"

def test_only_the_runners_reach_the_manoeuvre_package():
    """Every `manoeuvre` module imports the control path, the guidance layer, the data plane and
    the inference helpers (plan §4.2), and nothing imports it back: only the runners under
    `experiments/` consume this package (the control path's edge to it went with the intent-code
    archive, 2026-09-20). A new edge from `outputs/` is a layering decision, not an import."""
    for path in _module_files():
        if path.is_relative_to(MANOEUVRE):
            continue
        imported = {
            ".".join(name.split(".")[:2])   # the module, not the name imported from it
            for name in _imported_names(path) if name.split(".")[0] == "manoeuvre"
        }
        imported.discard("manoeuvre")       # `from ts_transformer.manoeuvre import lockstep`
        if not imported:
            continue
        rel = path.relative_to(TS_DIR).as_posix()
        assert rel.startswith("experiments/"), (
            f"{rel} imports {sorted(imported)}; only the runners consume the manoeuvre package"
        )

def test_the_manoeuvre_package_is_not_a_runner_and_imports_none():
    """`manoeuvre/` is package code: runners under `experiments/` and the CLI call it, never
    the reverse (layout rule L3)."""
    for path in MANOEUVRE.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        offending = {name for name in _imported_names(path) if name.split(".")[0] in {"experiments", "cli"}}
        assert not offending, f"{path.relative_to(TS_DIR)} imports {sorted(offending)}"
