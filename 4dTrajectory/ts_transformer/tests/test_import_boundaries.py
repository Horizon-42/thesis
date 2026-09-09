"""Dependency boundaries promised by the TS transformer's lean requirements."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
TS_ROOT = REPO_ROOT / "4dTrajectory" / "ts_transformer"


def _import_without(banned: str, *modules: str) -> subprocess.CompletedProcess:
    """Import `modules` in a fresh interpreter where importing `banned` raises.

    The canary matters more than it looks: if the poison stopped working (a stale
    `sys.modules` entry, an import hook installed too late) every test below would pass by
    importing the banned module happily. So the child proves the ban is live on itself
    first, and only then imports what is under test.
    """
    code = f"""
import builtins
import sys

sys.path[:0] = [{str(TS_ROOT.parent)!r}, {str(REPO_ROOT)!r}, {str(REPO_ROOT / 'geokit' / 'src')!r}]
real_import = builtins.__import__
banned = {banned!r}

def without_it(name, globals=None, locals=None, fromlist=(), level=0):
    if name == banned or name.startswith(banned + '.'):
        raise ModuleNotFoundError(banned + " deliberately unavailable in lean TS environment")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = without_it
try:
    __import__(banned)
except ModuleNotFoundError:
    pass
else:
    raise AssertionError("the " + banned + " ban is not in effect; this test proves nothing")
""" + "".join(f"import {module}\n" for module in modules)
    return subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True
    )


def test_dataset_import_does_not_require_pandas() -> None:
    completed = _import_without("pandas", "ts_transformer.dataset")
    assert completed.returncode == 0, completed.stderr


def test_the_provenance_and_protocol_modules_do_not_reach_the_data_plane() -> None:
    """`data_provenance` is JSON and hashes; nothing about it needs a tensor.

    `evaluation_protocol` compares two fingerprints for equality, and it used to pull
    `require_matching_data_provenance` through `dataset` — which meant the test-release
    ledger could not be read without torch. `splits` is on the list for the same reason and
    keeps its `dataset` import under `TYPE_CHECKING`: it hashes flight identities, and
    `BuildReport` / `FlightSeries` appear in it only as annotations.
    """
    completed = _import_without(
        "torch", "ts_transformer.data_provenance", "ts_transformer.evaluation_protocol",
        "ts_transformer.splits",
    )
    assert completed.returncode == 0, completed.stderr


def test_the_remaining_path_axis_is_a_leaf_under_dataset() -> None:
    """`anchor_strata` owns the grid's VALUES so both sides of one import edge can read them.

    `anchor_grid` imports `dataset` (`FlightSeries`, `truth_duration_s`) while `dataset`
    needs the same kilometres for the training-anchor draw, so the values cannot live in
    `anchor_grid` without either a cycle or a second copy. They live in this leaf,
    `anchor_grid` re-exports them, and `dataset` imports it at module scope — which stays
    legal only while the leaf reaches neither `dataset` nor torch.
    """
    for banned in ("ts_transformer.dataset", "torch"):
        completed = _import_without(banned, "ts_transformer.anchor_strata")
        assert completed.returncode == 0, completed.stderr
