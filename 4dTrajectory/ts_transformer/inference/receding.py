"""Receding-horizon pieces: a forecast flown for Δ, the rolled history a re-ask reads, and the
displacement of a forecast from its truth at a lead.

Shared by `experiments/chain_sensitivity.py` (two-tier T0(b): a checkpoint chained on its own
rollout) and `experiments/tracker_lockstep.py` (T1: a plan-given tracker re-asked every Δ).
A re-ask on `rolled_series` IS the model's ordinary predict path on that history — the window,
the anchor state and the lagged actuators' initial condition inverted from the rolled lookback —
which `tests/test_chain_sensitivity.py` pins with a displaced flown history.
"""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np

from ts_transformer.data.channels import POSITION_IDX
from ts_transformer.data.dataset import FlightSeries
from ts_transformer.inference.forecast import Forecast, cut_rows

#: A cut lead must fall on a rollout query row: the dense grid is the integrator step plus the
#: segment boundaries (`dense_query_offsets`), so a Δ that is a whole multiple of the step has
#: a row exactly there. This is the tolerance that row is found at.
ROW_TOLERANCE_S = 1e-6


def rolled_series(series: FlightSeries, anchor: int, flown: Forecast, until: int, dt_s: float) -> FlightSeries:
    """``series`` with every row after ``anchor`` replaced by the ``flown`` rows, sampled on
    the series' own uniform grid up to index ``until`` — the history a chained re-ask at
    ``until`` is shown. The flown rows start one query step after the anchor, so the anchor's
    own observed row stands in before them. The supervision arrays are dropped (a rolled
    series has no truth): a caller whose checkpoint reads a CTA or a plan supplies them itself
    (`tracker_lockstep` reads both off the ORIGINAL series' truth expert), and nothing else on
    the hook-free control forecast path reads the truth."""
    if until <= anchor:
        raise ValueError(f"a rolled series continues past its anchor {anchor}, not to {until}")
    origin = float(series.times[anchor])
    grid = origin + dt_s * np.arange(1, until - anchor + 1, dtype=np.float64)
    if grid[-1] > float(flown.times[-1]) + ROW_TOLERANCE_S:
        raise ValueError(
            f"the flown rows end at {float(flown.times[-1]) - origin:.3f} s, the rolled series needs "
            f"{grid[-1] - origin:.3f} s"
        )
    times = np.concatenate(([origin], np.asarray(flown.times, dtype=np.float64)))
    values = np.concatenate((np.asarray(series.values[anchor : anchor + 1], dtype=np.float64),
                             np.asarray(flown.values, dtype=np.float64)))
    rows = np.stack([np.interp(grid, times, values[:, c]) for c in range(values.shape[1])], axis=1)
    return replace(
        series,
        times=np.concatenate((np.asarray(series.times[: anchor + 1], dtype=np.float64), grid)),
        values=np.concatenate((np.asarray(series.values[: anchor + 1], dtype=np.float64), rows)),
        supervision_times=None, supervision_values=None, supervision_weights=None,
    )


def cut_at_lead(forecast: Forecast, lead_s: float) -> Forecast:
    """The forecast's rows up to and including the query row at ``lead_s`` from its anchor."""
    offsets = np.cumsum(forecast.sample_durations_s)
    rows = np.flatnonzero(np.abs(offsets - lead_s) <= ROW_TOLERANCE_S)
    if not rows.size:
        raise ValueError(
            f"no rollout row at {lead_s:g} s (the dense grid is the integrator step plus the "
            "segment boundaries; the step must be a whole multiple of the integrator step)"
        )
    return cut_rows(forecast, int(rows[0]) + 1)


def displacement_at(series: FlightSeries, forecast: Forecast, origin_index: int, time_s: float) -> float | None:
    """The 3D chart displacement between ``forecast`` (the observed row at ``origin_index``
    standing in before its first row) and the observed track at absolute ``time_s``; None
    when either ends before it."""
    truth_times = np.asarray(series.times, dtype=np.float64)
    if time_s > float(forecast.times[-1]) + ROW_TOLERANCE_S or time_s > float(truth_times[-1]) + ROW_TOLERANCE_S:
        return None
    position = list(POSITION_IDX)
    times = np.concatenate(([float(series.times[origin_index])], np.asarray(forecast.times, dtype=np.float64)))
    values = np.concatenate((np.asarray(series.values[origin_index : origin_index + 1], dtype=np.float64)[:, position],
                             np.asarray(forecast.values, dtype=np.float64)[:, position]))
    truth = np.asarray(series.values, dtype=np.float64)[:, position]
    predicted = np.array([np.interp(time_s, times, values[:, c]) for c in range(3)])
    observed = np.array([np.interp(time_s, truth_times, truth[:, c]) for c in range(3)])
    value = float(np.linalg.norm(predicted - observed))
    if not math.isfinite(value):
        raise ValueError(f"{series.dataset_id}: non-finite displacement at {time_s - float(series.times[origin_index]):g} s")
    return value


__all__ = ["ROW_TOLERANCE_S", "cut_at_lead", "displacement_at", "rolled_series"]
