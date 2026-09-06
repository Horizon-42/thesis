"""Put the package and the repository root on ``sys.path`` once, for every test here.

``ts_transformer`` is a flat module tree collected from the repository root (see
``run_all_tests.sh``), so its modules are only importable with its own directory on the
path. Older test files each carry their own preamble doing this; a new one does not need
to — this runs first for the whole directory.
"""

from __future__ import annotations

from pathlib import Path
import sys

_TS_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[3]
for _path in (_TS_DIR, _REPO_ROOT, _REPO_ROOT / "geokit" / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
