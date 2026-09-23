"""Receding-horizon pieces: a forecast flown for Δ, the rolled history the next prediction reads, and the
displacement of a forecast from its truth at a lead.

Shared by `experiments/chain_sensitivity.py` (a checkpoint chained on its own rollout) and the
no-token lockstep (`manoeuvre/lockstep.py`, two-tier v3 §3.1; the archived two-tier tracker and
the archived intent-code protocols read it the same way).
A prediction on `rolled_series` IS the model's ordinary predict path on that history — the window,
the anchor state and the lagged actuators' initial condition inverted from the rolled lookback —
which `tests/test_chain_sensitivity.py` pins with a displaced flown history.
"""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np

from ts_transformer.data.channels import POSITION_IDX
from ts_transformer.data.dataset import FlightSeries
from ts_transformer.data.time_grids import ROW_TOLERANCE_S
from ts_transformer.inference.forecast import Forecast, cut_rows

# A cut lead must fall on a rollout query row: the dense grid is the integrator step plus the
# segment boundaries (`dense_query_offsets`), so a Δ that is a whole multiple of the step has
# a row exactly there. `ROW_TOLERANCE_S` (one definition, `data/time_grids.py`) is the
# tolerance that row is found at.


def rolled_series(series: FlightSeries, anchor: int, flown: Forecast, until: int, dt_s: float) -> FlightSeries:
    """``series`` with every row after ``anchor`` replaced by the ``flown`` rows, sampled on
    the series' own uniform grid up to index ``until`` — the history a chained prediction at
    ``until`` is shown. The flown rows start one query step after the anchor, so the anchor's
    own observed row stands in before them. The supervision arrays are dropped (a rolled
    series has no truth): a caller whose checkpoint reads a CTA or a plan supplies them itself
    (a lockstep: the truth's, or the prior's), and nothing else on the hook-free control forecast
    path reads the truth."""
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


def mean_displacement_to(series: FlightSeries, forecast: Forecast, origin_index: int, horizon_s: float,
                         *, grid_dt_s: float = 1.0) -> float:
    """ADE[0, horizon]: the mean 3D chart displacement between ``forecast`` and the observed track
    on the ``grid_dt_s`` grid from the origin to ``horizon_s`` INCLUSIVE — `lead_time_error`'s
    accounting (the t=0 point, identically zero, is in the mean). Both must reach the horizon."""
    grid = np.arange(0.0, horizon_s + ROW_TOLERANCE_S, grid_dt_s)
    origin_time = float(series.times[origin_index])
    if origin_time + grid[-1] > float(forecast.times[-1]) + ROW_TOLERANCE_S:
        raise ValueError(f"the forecast ends {float(forecast.times[-1]) - origin_time:.3f} s after the origin, "
                         f"before the {horizon_s:g} s horizon")
    if origin_time + grid[-1] > float(series.times[-1]) + ROW_TOLERANCE_S:
        raise ValueError(f"{series.dataset_id}: the truth ends before the {horizon_s:g} s horizon")
    position = list(POSITION_IDX)
    times = np.concatenate(([origin_time], np.asarray(forecast.times, dtype=np.float64)))
    values = np.concatenate((np.asarray(series.values[origin_index : origin_index + 1], dtype=np.float64)[:, position],
                             np.asarray(forecast.values, dtype=np.float64)[:, position]))
    truth_times = np.asarray(series.times, dtype=np.float64)
    truth = np.asarray(series.values, dtype=np.float64)[:, position]
    query = origin_time + grid
    predicted = np.column_stack([np.interp(query, times, values[:, c]) for c in range(3)])
    observed = np.column_stack([np.interp(query, truth_times, truth[:, c]) for c in range(3)])
    value = float(np.linalg.norm(predicted - observed, axis=1).mean())
    if not math.isfinite(value):
        raise ValueError(f"{series.dataset_id}: non-finite displacement inside {horizon_s:g} s of the origin")
    return value


def displacement_at(series: FlightSeries, forecast: Forecast, origin_index: int, time_s: float,
                    *, hold_forecast_end: bool = False) -> float | None:
    """The 3D chart displacement between ``forecast`` (the observed row at ``origin_index``
    standing in before its first row) and the observed track at absolute ``time_s``; None
    when either ends before it. With ``hold_forecast_end`` the forecast's LAST row stands in
    past its end — a plan that says it arrived is at the threshold, and a reading past that
    claim measures the claim — so only the truth's end makes the reading absent."""
    truth_times = np.asarray(series.times, dtype=np.float64)
    if time_s > float(truth_times[-1]) + ROW_TOLERANCE_S:
        return None
    if not hold_forecast_end and time_s > float(forecast.times[-1]) + ROW_TOLERANCE_S:
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


__all__ = ["cut_at_lead", "displacement_at", "mean_displacement_to", "rolled_series"]
