"""The control package's membership rule, enforced rather than described.

`control/` exists because twelve `control_*.py` files at the top level named their subject
but not their role. Two things keep that from coming back, and both are checkable:

1. no new `control_*.py` may appear at the top level;
2. a module belongs in `control/` only if EVERY consumer of it is control-specific — which
   is why `prediction_outputs`, `terminal_state_loss`, `arc_length_geometry`,
   `fixed_dt_supervision` and `flyability` stay outside it.

The second rule is the one worth testing: without it the package slowly absorbs shared
modules and starts claiming ownership it does not have, which is worse than the flat
listing because the name now lies.
"""
from __future__ import annotations

import ast
from pathlib import Path
import sys

TS_DIR = Path(__file__).resolve().parents[1]
if str(TS_DIR) not in sys.path:
    sys.path.insert(0, str(TS_DIR))

CONTROL = TS_DIR / "control"
# Shared with the state path through `fixed_anchor_validation`, `dataset` or `batching`.
SHARED_BY_DESIGN = {
    "prediction_outputs",
    "terminal_state_loss",
    "arc_length_geometry",
    "fixed_dt_supervision",
    "flyability",
}


def _module_files() -> list[Path]:
    return [
        path
        for path in TS_DIR.rglob("*.py")
        if "__pycache__" not in path.parts
        and "vendor" not in path.parts
        and "tests" not in path.parts
        and "docs" not in path.parts
    ]


def _imported_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


def test_no_control_prefixed_module_returns_to_the_top_level():
    stragglers = sorted(p.name for p in TS_DIR.glob("control_*.py"))
    assert not stragglers, (
        f"{stragglers} belong under control/ by role, not at the top level behind a prefix"
    )
    assert (CONTROL / "__init__.py").is_file()
    for sub in ("dynamics", "loss", "training", "oracle"):
        assert (CONTROL / sub / "__init__.py").is_file(), f"control/{sub} is not a package"


def test_shared_modules_stay_outside_the_control_package():
    """Each name here has at least one consumer that is NOT control-specific."""
    for name in SHARED_BY_DESIGN:
        assert (TS_DIR / f"{name}.py").is_file(), (
            f"{name} moved into control/, but the state path reaches it — check its "
            f"consumers before claiming it as control-only"
        )
        consumers = {
            path.relative_to(TS_DIR).as_posix()
            for path in _module_files()
            if name in _imported_names(path)
        }
        outside = {c for c in consumers if not c.startswith("control/")}
        assert outside, (
            f"{name} is now imported only from control/ — it may have become genuinely "
            f"control-specific, in which case move it in and drop it from SHARED_BY_DESIGN"
        )


def test_the_control_package_does_not_import_the_training_loop():
    """control/ is imported BY the training loop; it must never reach back up into it.

    `dataset` is deliberately NOT on this list. `Normalizer` and the window types are
    data-plane value types the loss and the oracle genuinely consume, and `Normalizer.fit`
    balances over `FlightSeries`, so it belongs with the data plane rather than under
    `control`. The direction that matters is this one: a loss module that imported `train`
    would make the package unusable outside the loop it was extracted from.
    """
    consumers = {"train", "forecast", "__main__", "models", "batching"}
    for path in CONTROL.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        offending = _imported_names(path) & consumers
        assert not offending, (
            f"{path.relative_to(TS_DIR)} imports {sorted(offending)}; control/ is imported "
            f"BY the training loop, never the other way round"
        )


def test_the_conditioning_names_and_their_scalings_are_one_source():
    """`mass_100t` must actually be divided by 100 t, and nothing used to check that.

    The names sat in `dataset` and the divisors in `dynamics_arrays` fifty lines apart, so
    renaming a channel without changing its divisor was a silent mislabel of the vector the
    head is conditioned on.
    """
    from types import SimpleNamespace

    from control.conditioning import (
        CONDITION_CHANNELS,
        DYNAMICS_CONDITION_NAMES,
        condition_vector,
    )

    assert DYNAMICS_CONDITION_NAMES == tuple(name for name, _ in CONDITION_CHANNELS)

    aero = SimpleNamespace(S=500.0, Cl_max=3.0, Cd0=0.1, k=0.1, stall_threshold=0.8,
                           k_stall=0.2)
    vector = condition_vector(mass_kg=100_000.0, max_thrust_n=1_000_000.0, aero=aero)

    # Every channel fed exactly the quantity its name declares, so each must read 1.0 —
    # except stall_threshold, which the name says is already dimensionless.
    named = dict(zip(DYNAMICS_CONDITION_NAMES, vector))
    assert named["stall_threshold"] == 0.8
    for name, value in named.items():
        if name != "stall_threshold":
            assert value == 1.0, f"{name} does not divide by the unit its name states"
