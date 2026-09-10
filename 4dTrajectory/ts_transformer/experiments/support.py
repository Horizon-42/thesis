"""What every runner shares: where the repository is, and the helpers five of them carried as
byte-identical copies (review §4.5). The paths are `repo_layout`'s and `parse_airports` is
`cli.common`'s (the benchmark CLI reads it too), both re-exported here so a runner has one
import; `series_digest` lives here; `write_json_atomic` / `file_sha256` are `io_utils`'s —
the private copies are gone."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Sequence

from ts_transformer.cli.common import parse_airports
from ts_transformer.repo_layout import REPO_ROOT, TS_DIR, TS_SCRIPT

if TYPE_CHECKING:
    from ts_transformer.dataset import FlightSeries

__all__ = [
    "EXPERIMENTS_MAIN", "REPO_ROOT", "RUN_TS", "TS_DIR", "TS_SCRIPT",
    "parse_airports", "series_digest",
]

#: The runners' own entry point, for a runner that spawns another runner.
EXPERIMENTS_MAIN = TS_DIR / "experiments" / "__main__.py"
RUN_TS = REPO_ROOT / "run_ts.py"


def series_digest(series: Sequence[FlightSeries]) -> str:
    """The cohort's identity for a runner's resume check: sha256 over the sorted dataset ids."""
    payload = "\n".join(sorted(item.dataset_id for item in series)).encode()
    return hashlib.sha256(payload).hexdigest()
