"""Dependency boundaries promised by the TS transformer's lean requirements."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
TS_ROOT = REPO_ROOT / "4dTrajectory" / "ts_transformer"


def test_dataset_import_does_not_require_pandas() -> None:
    code = f"""
import builtins
import sys

sys.path[:0] = [{str(TS_ROOT)!r}, {str(REPO_ROOT)!r}, {str(REPO_ROOT / 'geokit' / 'src')!r}]
real_import = builtins.__import__

def without_pandas(name, globals=None, locals=None, fromlist=(), level=0):
    if name == 'pandas' or name.startswith('pandas.'):
        raise ModuleNotFoundError("pandas deliberately unavailable in lean TS environment")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = without_pandas
import dataset
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
