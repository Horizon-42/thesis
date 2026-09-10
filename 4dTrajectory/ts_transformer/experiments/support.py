"""What every runner shares: where the repository is, and the helpers five of them carried as
byte-identical copies (review §4.5). The paths are `repo_layout`'s and `parse_airports` is
`cli.common`'s (the benchmark CLI reads it too), both re-exported here so a runner has one
import; `series_digest` lives here; `write_json_atomic` / `file_sha256` are `io_utils`'s —
the private copies are gone; `forecast_geometry` is the time-free reading of one forecast
against its truth that the anytime curve and the plan oracle both take."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Sequence

import numpy as np

from ts_transformer.cli.common import parse_airports
from ts_transformer.data.channels import POSITION_IDX
import ts_transformer.geometry.geometric_metrics as gm
from ts_transformer.repo_layout import REPO_ROOT, TS_DIR, TS_SCRIPT

if TYPE_CHECKING:
    from ts_transformer.data.dataset import FlightSeries
    from ts_transformer.inference.forecast import Forecast

__all__ = [
    "EXPERIMENTS_MAIN", "REPO_ROOT", "RUN_TS", "TS_DIR", "TS_SCRIPT",
    "forecast_geometry", "parse_airports", "series_digest",
]

#: The runners' own entry point, for a runner that spawns another runner.
EXPERIMENTS_MAIN = TS_DIR / "experiments" / "__main__.py"
RUN_TS = REPO_ROOT / "run_ts.py"


def series_digest(series: Sequence[FlightSeries]) -> str:
    """The cohort's identity for a runner's resume check: sha256 over the sorted dataset ids."""
    payload = "\n".join(sorted(item.dataset_id for item in series)).encode()
    return hashlib.sha256(payload).hexdigest()


def forecast_geometry(series: FlightSeries, forecast: Forecast) -> dict[str, float]:
    """The time-free metrics for one forecast against the truth AFTER its anchor.

    Both paths are ``[N, 4]`` ``(e, n, u, t)`` in the flight's own chart, which is all
    `geometric_metrics` needs — chamfer and Fréchet are relative, so the chart origin
    (threshold or airport) does not enter.
    """
    anchor_time = float(series.times[forecast.anchor])
    future = series.supervision_times > anchor_time
    truth = np.column_stack([
        np.asarray(series.supervision_values, dtype=np.float64)[future][:, list(POSITION_IDX)],
        np.asarray(series.supervision_times, dtype=np.float64)[future] - anchor_time,
    ])
    predicted = np.column_stack([
        np.asarray(forecast.values, dtype=np.float64)[:, list(POSITION_IDX)],
        np.cumsum(forecast.sample_durations_s),
    ])
    return gm.path_metrics(predicted, truth)
