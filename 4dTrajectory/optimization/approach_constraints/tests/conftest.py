"""Put the optimization dir (for ``import approach_constraints``) and the repo root (for ``geokit``) on the path."""

import sys
from pathlib import Path

_OPT_DIR = Path(__file__).resolve().parents[2]   # 4dTrajectory/optimization
_REPO_ROOT = Path(__file__).resolve().parents[4]  # repo root
for _p in (_OPT_DIR, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
