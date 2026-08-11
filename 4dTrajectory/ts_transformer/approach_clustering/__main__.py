"""Run the approach-clustering command line as a module."""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
TS_ROOT = Path(__file__).resolve().parents[1]
for source_root in (REPO_ROOT, TS_ROOT):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
