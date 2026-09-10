"""Put the package's parent and the repository root on ``sys.path`` once, for every test here.

``ts_transformer`` is a regular package under ``4dTrajectory/``; its modules are imported by
their qualified names (``ts_transformer.data.dataset``), so what goes on the path is
``4dTrajectory/`` — never ``ts_transformer/`` itself, which would make the flat names
importable beside the qualified ones and load every module twice. Older test files still
carry their own preamble doing the same; a new one does not need to.
"""

from __future__ import annotations

from pathlib import Path
import sys

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[3]
for _path in (_PACKAGE_ROOT, _REPO_ROOT, _REPO_ROOT / "geokit" / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
