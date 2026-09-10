#!/usr/bin/env python
"""The ts_transformer experiment runners: `python run_ts.py <name> [args]`, `--list` for the names.

The runners live in `4dTrajectory/ts_transformer/experiments/` (review §4.5); this is the
repository-root door to them, and it puts `4dTrajectory/` on the path the way the package CLI
does.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PACKAGE_PARENT = Path(__file__).resolve().parent / "4dTrajectory"
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))

from ts_transformer.experiments.__main__ import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
