"""Dependency boundaries promised by the TS transformer's lean requirements."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
TS_ROOT = REPO_ROOT / "4dTrajectory" / "ts_transformer"


def _import_without(banned: str, *modules: str) -> subprocess.CompletedProcess:
    code = f"""
import builtins
import sys

sys.path[:0] = [{str(TS_ROOT)!r}, {str(REPO_ROOT)!r}, {str(REPO_ROOT / 'geokit' / 'src')!r}]
real_import = builtins.__import__
banned = {banned!r}

def without_it(name, globals=None, locals=None, fromlist=(), level=0):
    if name == banned or name.startswith(banned + '.'):
        raise ModuleNotFoundError(banned + " deliberately unavailable in lean TS environment")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = without_it
""" + "".join(f"import {module}\n" for module in modules)
    return subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True
    )


def test_dataset_import_does_not_require_pandas() -> None:
    completed = _import_without("pandas", "dataset")
    assert completed.returncode == 0, completed.stderr


def test_the_provenance_and_protocol_modules_do_not_reach_the_data_plane() -> None:
    """`data_provenance` is JSON and hashes; nothing about it needs a tensor.

    `evaluation_protocol` compares two fingerprints for equality, and it used to pull
    `require_matching_data_provenance` through `dataset` — which meant the test-release
    ledger could not be read without torch, numpy, openap and the whole harvest stack.
    """
    completed = _import_without("torch", "data_provenance", "evaluation_protocol")
    assert completed.returncode == 0, completed.stderr
