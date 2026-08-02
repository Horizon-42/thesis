"""Bootstrap the isolated ts_transformer approach-clustering package."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
TS_ROOT = REPO_ROOT / "4dTrajectory" / "ts_transformer"
sys.path[:0] = [str(TS_ROOT), str(REPO_ROOT)]

from approach_clustering.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
