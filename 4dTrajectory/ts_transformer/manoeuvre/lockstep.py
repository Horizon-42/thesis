"""Lockstep (two-tier v3 §3.1): the executor predicts again every round on its own flown rows,
with NO second layer — protocol ``none``, the only protocol there is.

The executor (`plan_conditioning = off`) forecasts its horizon from the anchor, flies the first
``executed_step_s`` of that forecast, and predicts again from the rows it just flew, until the
forecast crosses the threshold ON THE FINAL (`inference.forecast.cut_at_threshold_crossing`: the
flight ends, ``crossed``) or the flight's budget runs out (``horizon``). The intent-code
protocols C / A / A-truth — the truth's code, the prior's top-1 on the flown history, the
prior's top-1 on the truth history — are ARCHIVED 2026-09-20
(`archive/manoeuvre_codes_2026_09/`, README there; their readings:
`docs/2026-09-18_manoeuvre_token_results.zh.md` §9–§11).

The first prediction is made at the executor's fixed anchor, L−1 (`config.default_anchor`: no floor
under v3 — the first row with a complete lookback, L seconds after entry). The second reading of
v3 §3.1 starts from a REMAINING-PATH bin instead (`from_remaining_path`): each flight is first seen
at the row `anchor_grid.bin_anchor` places at the bin, so the same lockstep from L−1 of the cut
flight IS the lockstep from that row of the whole flight. The third (`from_row`) starts every
flight at ONE common row, so cells of different lookback fly the same segment.

One round = one prediction, flown for `executed_step_s`: the executor's horizon, or — with
``execute_s`` (v3 A3-a) — only the first step of each forecast before the next prediction. The
budget is `closing_horizon_s` of the truth's duration at the first prediction — T₀ + max(30 s,
0.1·T₀), the rule copied from the archived plan oracle so "established" is read over one budget in
every campaign — and is a CAP.

The reference verdicts are the plan oracle's (`geometry.flyability`: fully flyable over the
flown geodetic states; established = crossed the threshold on the final), and the row carries
`observed_series_metrics`' ADE / FDE and the time-free geometry (chamfer, Fréchet).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Sequence

import numpy as np
import torch

from ts_transformer.config import TSConfig, default_anchor
from ts_transformer.data.anchor_grid import anchors_for_bin, remaining_path_profiles
from ts_transformer.data.approach_difficulty import approach_difficulty
from ts_transformer.data.channels import POSITION_IDX
from ts_transformer.data.dataset import FlightSeries, Normalizer, effective_min_future_s, series_from_row, truth_duration_s
from ts_transformer.data.time_grids import ROW_TOLERANCE_S
from ts_transformer.geometry import geometric_metrics
from ts_transformer.geometry.flyability import flyability_summary, required_controls
from ts_transformer.inference.export import observed_series_metrics
from ts_transformer.inference.forecast import Forecast, concatenate, cut_at_threshold_crossing, cut_rows
from ts_transformer.inference.receding import cut_at_lead, displacement_at, rolled_series
from ts_transformer.outputs.control.forecast import forecast_control_batch
from ts_transformer.outputs.dynamics.context import dynamics_arrays

#: The one protocol: no second layer. Kept as a name because every payload carries it and the
#: gates refuse a payload flown under anything else (`gates.gate_grid`).
PROTOCOL_NONE = "none"
#: How a flight ended.
ENDED_CROSSED = "crossed"            # the threshold on the final, inside a leg
ENDED_HORIZON = "horizon"            # the budget
#: The closing budget — the archived `plan_oracle.closing_horizon_s`, copied here as the plan
#: says (§2.7): T₀ + max(HORIZON_SLACK_S, HORIZON_SLACK_FRACTION·T₀).
HORIZON_SLACK_S = 30.0
HORIZON_SLACK_FRACTION = 0.1
#: The leads (s from the first prediction) the displacement is read at (plan §3.2).
LEADS_S = (60.0, 120.0, 180.0, 300.0)


def closing_horizon_s(T_s: float) -> float:
    return T_s + max(HORIZON_SLACK_S, HORIZON_SLACK_FRACTION * T_s)


@dataclass(frozen=True)
class Executor:
    model: torch.nn.Module
    config: TSConfig
    normalizer: Normalizer


@dataclass
class FlightRun:
    series: FlightSeries
    anchor: int                                  # the first prediction's row
    horizon_s: float
    legs: list[Forecast] = field(default_factory=list)
    flown_s: float = 0.0
    ended: str | None = None
    truncated: bool = False
    predictions: int = 0                         # rounds flown
    rounds: list[dict[str, Any]] = field(default_factory=list)       # one record per round: lead, e_track


def from_remaining_path(series: Sequence[FlightSeries], config: TSConfig, target_m: float) -> tuple[list[FlightSeries], dict[str, int]]:
    """The cohort first seen at the remaining-path bin ``target_m``: every flight cut
    (`series_from_row`) so that the row `anchor_grid.bin_anchor` places at the bin — the sample
    nearest ``target_m`` of path left to fly, with a complete lookback before it and the
    executor's horizon of truth after it — becomes the executor's fixed anchor (`default_anchor`:
    L−1, or a floored config's floor). ``(the cut flights in cohort order, {dataset id: that row
    in the whole flight})``; a flight with no admissible row at the bin is absent from both (the
    caller counts it)."""
    anchors = anchors_for_bin(series, remaining_path_profiles(series), target_m, seq_len=config.seq_len,
                              min_future_s=effective_min_future_s(config), minimum_anchor_index=config.anchor_floor_index)
    # the bin's row lands on the executor's own fixed anchor (L−1, or its floor under a floored config)
    cut = [series_from_row(series[index], anchor - default_anchor(config)) for index, anchor in anchors.items()]
    return cut, {series[index].dataset_id: anchor for index, anchor in anchors.items()}


def from_row(series: Sequence[FlightSeries], config: TSConfig, row: int) -> tuple[list[FlightSeries], dict[str, int]]:
    """The cohort first seen at ONE common row ``row`` of every flight (two-tier v3's reading (c):
    every cell flies the same segment of the approach and differs only in its lookback): each
    flight cut (`series_from_row`) so that ``row`` becomes the executor's fixed anchor, by the
    rule `anchor_grid.bin_anchor` applies to a bin's row — the executor's horizon of truth after
    it; a flight that ends earlier is absent (the caller counts it). ``row`` is at or after the
    executor's own first row (`default_anchor`), which the caller checks."""
    kept = [item for item in series if item.n_samples > row and truth_duration_s(item, row) >= effective_min_future_s(config)]
    cut = [series_from_row(item, row - default_anchor(config)) for item in kept]
    return cut, {item.dataset_id: row for item in kept}


def executed_step_s(config: TSConfig, execute_s: float | None) -> float:
    """The seconds of each forecast the closed loop flies before predicting again: the whole
    horizon, or ``execute_s`` (v3 A3-a) — which must be a whole number of series rows AND of
    integrator steps inside the horizon, because the leg is cut on the dense rollout grid
    (`cut_at_lead`: the integrator step plus the segment boundaries)."""
    horizon_s = config.control_horizon_s
    if execute_s is None:
        return horizon_s
    step_s = float(execute_s)
    grids = (config.dt_s, config.control_rollout_integrator_dt_s)
    if not (0 < step_s <= horizon_s) or any(abs(round(step_s / h) * h - step_s) > ROW_TOLERANCE_S for h in grids):
        raise ValueError(f"execute_s must be a whole number of rows ({config.dt_s:g} s) and of integrator steps "
                         f"({config.control_rollout_integrator_dt_s:g} s) inside the horizon {horizon_s:g} s: got {step_s:g} s")
    return step_s


def _polyline(run: FlightRun) -> tuple[np.ndarray, np.ndarray]:
    """The flown path on the flight's clock: the anchor's observed row, then every leg's rows."""
    a0 = run.anchor
    times = [float(run.series.times[a0])]
    values = [np.asarray(run.series.values[a0], dtype=np.float64)]
    for leg in run.legs:
        times.extend(np.asarray(leg.times, dtype=np.float64).tolist())
        values.extend(np.asarray(leg.values, dtype=np.float64))
    return np.asarray(times), np.stack(values)


def _history(run: FlightRun, anchor: int, dt_s: float) -> FlightSeries:
    if not run.legs:
        return run.series
    return rolled_series(run.series, run.anchor, concatenate(run.legs, run.anchor, 0.0), anchor, dt_s)


def _dynamics(executor: Executor, histories: Sequence[FlightSeries], anchor: int,
              device: torch.device) -> dict[str, torch.Tensor]:
    """The executor's dynamics rows at ``anchor`` — a no-token executor's, so nothing is added
    beside them."""
    config = executor.config
    rows = [
        dynamics_arrays(history, anchor, parameterization=config.control_thrust_parameterization,
                        condition_features=config.control_condition_features)
        for history in histories
    ]
    return {name: torch.from_numpy(np.stack([row[name] for row in rows])).to(device) for name in rows[0]}


def _fly(executor: Executor, histories: Sequence[FlightSeries], anchor: int, device: torch.device,
         batch_size: int) -> list[Forecast]:
    out: list[Forecast] = []
    for start in range(0, len(histories), batch_size):
        chunk = list(histories[start : start + batch_size])
        out.extend(forecast_control_batch(
            executor.model, chunk, executor.config, executor.normalizer, anchor, device,
            dynamics=_dynamics(executor, chunk, anchor, device),
        ))
    return out


def fly(
    executor: Executor, series: Sequence[FlightSeries], *, device: torch.device, batch_size: int,
    log=None, execute_s: float | None = None,
) -> list[FlightRun]:
    """Every flight of ``series`` from the executor's fixed anchor, one round at a time for the
    whole cohort. ``execute_s`` (v3 A3-a, D43) flies only the first ``execute_s`` of each forecast
    and predicts again there — the executor still forecasts its whole horizon."""
    config = executor.config
    dt_s = config.dt_s
    if not config.control_horizon_s:
        raise ValueError("the lockstep flies one fixed segment per round: the executor needs control_horizon_s > 0")
    step_s = executed_step_s(config, execute_s)
    a0 = default_anchor(config)
    step_rows = int(round(step_s / dt_s))
    runs = [FlightRun(series=item, anchor=a0, horizon_s=closing_horizon_s(truth_duration_s(item, a0)))
            for item in series]
    round_index = 0
    while any(run.ended is None for run in runs):
        active = [run for run in runs if run.ended is None]
        anchor = a0 + round_index * step_rows
        histories = [_history(run, anchor, dt_s) for run in active]
        forecasts = _fly(executor, histories, anchor, device, batch_size)
        for run, history, forecast in zip(active, histories, forecasts, strict=True):
            _fly_leg(run, history, forecast, step_s=step_s, round_index=round_index)
        if log is not None:
            log(f"    {PROTOCOL_NONE} round {round_index}: {len(active)} flights at anchor {anchor}, "
                f"{sum(1 for run in runs if run.ended is None)} continue")
        round_index += 1
    return runs


def _fly_leg(run: FlightRun, history: FlightSeries, forecast: Forecast, *, step_s: float,
             round_index: int) -> Forecast | None:
    """Fly this round's step of the forecast; end at the crossing on the final or at the budget.
    Returns the leg appended, or None when the budget left no row to fly."""
    crossed = cut_at_threshold_crossing(forecast, history)
    remaining_s = run.horizon_s - run.flown_s
    offsets = np.cumsum(forecast.sample_durations_s)
    if crossed.truncated_at_threshold and crossed.final_time_s <= min(step_s, remaining_s) + ROW_TOLERANCE_S:
        leg, run.ended, run.truncated = crossed, ENDED_CROSSED, True
    elif remaining_s <= step_s + ROW_TOLERANCE_S:
        rows = int(np.searchsorted(offsets, remaining_s + ROW_TOLERANCE_S, side="right"))
        leg, run.ended = (cut_rows(forecast, rows) if rows else None), ENDED_HORIZON
    else:
        leg = cut_at_lead(forecast, step_s)
    if leg is None:
        return None
    flown = float(np.sum(leg.sample_durations_s))
    run.legs.append(leg)
    run.predictions += 1
    run.flown_s += flown
    run.rounds.append({
        "round": round_index, "lead_s": run.flown_s,
        "e_track_m": displacement_at(run.series, leg, run.anchor, float(leg.times[-1])),
    })
    return leg


# ── the rows ─────────────────────────────────────────────────────────────────

def whole_forecast(run: FlightRun) -> Forecast:
    """The flight's legs as one forecast from the anchor. Its predicted end is the flown span."""
    span = float(sum(np.sum(leg.sample_durations_s) for leg in run.legs))
    whole = concatenate(run.legs, run.anchor, span)
    return replace(whole, truncated_at_threshold=run.truncated, horizon_capped=run.ended == ENDED_HORIZON)


def _forecast_geometry(series: FlightSeries, forecast: Forecast) -> dict[str, float]:
    # MUST match `experiments/support.forecast_geometry` (a runner module the package cannot
    # import): both paths [N, 4] (e, n, u, t) in the flight's chart, the truth after the anchor.
    anchor_time = float(series.times[forecast.anchor])
    future = series.supervision_times > anchor_time
    truth = np.column_stack([
        np.asarray(series.supervision_values, dtype=np.float64)[future][:, list(POSITION_IDX)],
        np.asarray(series.supervision_times, dtype=np.float64)[future] - anchor_time,
    ])
    predicted = np.column_stack([
        np.asarray(forecast.values, dtype=np.float64)[:, list(POSITION_IDX)], np.cumsum(forecast.sample_durations_s),
    ])
    return geometric_metrics.path_metrics(predicted, truth)


def reference_verdicts(series: FlightSeries, forecast: Forecast) -> dict[str, Any]:
    """The plan oracle's reading: fully flyable over the flown geodetic states, established =
    crossed the threshold on the final."""
    rows = [
        {"t": float(t), "lat": float(g[0]), "lon": float(g[1]), "alt": float(g[2]),
         "V": float(g[3]), "psi": float(g[4]), "gamma": float(g[5]), "m": float(g[6])}
        for t, g in zip(np.cumsum(forecast.sample_durations_s), forecast.geodetic_values, strict=True)
    ]
    aircraft, aero = series.scenario.dynamics("the lockstep's flyability verdict")
    controls = required_controls(rows, aircraft, aero=aero)
    summary = flyability_summary(controls, aircraft_code=str(aircraft.code))
    return {
        "fully_flyable": bool(summary["fully_flyable"]), "violations": dict(summary["violations"]),
        "established": bool(forecast.truncated_at_threshold),
    }


def flight_row(run: FlightRun, points: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """One flight's row (and the record metrics): the reference verdicts, ADE / FDE, the
    geometry, the displacement at every lead, every round's step error, how it ended."""
    forecast = whole_forecast(run)
    metrics = observed_series_metrics(run.series, forecast, points=points)
    geometry = _forecast_geometry(run.series, forecast)
    origin = float(run.series.times[run.anchor])
    difficulty = approach_difficulty(run.series, run.anchor)
    row = {
        "flight_id": run.series.flight_id,
        "reference": reference_verdicts(run.series, forecast),
        "ended": run.ended, "truncated_at_threshold": run.truncated, "predictions": run.predictions,
        "ade_m": float(metrics["ade_m"]), "fde_m": float(metrics["fde_m"]),
        "final_time_error_s": float(metrics["final_time_error_s"]),
        "chamfer_m": float(geometry["chamfer_m"]), "frechet_m": float(geometry["frechet_m"]),
        # plan §3.1's absence rule: the forecast's last row is HELD past its end (a flight that
        # says it has arrived is at the threshold, and a reading past that claim measures the
        # claim); only the truth's end makes a reading absent — the rule B61's 768 / 1396 were
        # read under (review 2026-09-18 B3)
        "at": {f"{lead:g}": displacement_at(run.series, forecast, run.anchor, origin + lead, hold_forecast_end=True)
               for lead in LEADS_S},
        "rounds": run.rounds,
        "route_tortuosity": difficulty.route_tortuosity, "established_at_anchor": difficulty.established_at_anchor,
        "remaining_path_m": difficulty.remaining_path_m,
    }
    return row, metrics


__all__ = [
    "ENDED_CROSSED", "ENDED_HORIZON", "HORIZON_SLACK_FRACTION", "HORIZON_SLACK_S", "LEADS_S",
    "PROTOCOL_NONE", "Executor", "FlightRun", "closing_horizon_s", "executed_step_s", "flight_row",
    "fly", "from_remaining_path", "from_row", "reference_verdicts", "whole_forecast",
]
