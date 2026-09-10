"""The state path's inference: one deterministic pass per flight under the horizon
contract, the fixed-time postprocessors, and the corridor projection."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from ts_transformer.data.batch_contract import model_forward
from ts_transformer.data.channels import IDX, horizontal_distance_m
from ts_transformer.config import (
    CORRIDOR_GATE_ON_FINAL,
    HORIZON_FULL,
    HORIZON_NORMALIZED,
    HORIZON_WINDOW,
    PREDICTION_STATE,
    TSConfig,
)
from ts_transformer.data.dataset import FlightSeries, Normalizer, series_conditioning
from ts_transformer.geometry.final_approach_geometry import (
    alignment_cosine,
    bound_to_final,
    chart_from_axes,
    final_approach_arrays,
    membership,
    position_direction,
    runway_axes,
)
from ts_transformer.inference.forecast import Forecast, history_at_anchor
from ts_transformer.geometry.metrics import states_with_derived_velocity
from ts_transformer.data.target_conditioning import conditioned_history
from ts_transformer.data.time_grids import output_time_grid


def _final_approach_context(
    series: FlightSeries, config: TSConfig, device: torch.device
) -> dict[str, torch.Tensor] | None:
    """The one-flight context a corridor-bounded output needs; None for every other recipe."""
    if not config.uses_final_approach_context:
        return None
    rows = final_approach_arrays(series)
    return {
        name: torch.from_numpy(np.asarray(value)[None]).to(device)
        for name, value in rows.items()
    }


def _forward(
    model: nn.Module,
    history: np.ndarray,
    device: torch.device,
    context: dict[str, torch.Tensor] | None = None,
) -> tuple[np.ndarray, float]:
    """Run one deterministic pass in normalized channel space."""
    model.eval()
    tensor = torch.from_numpy(history[None, ...].astype(np.float32)).to(device)
    with torch.no_grad():
        prediction = model_forward(model, tensor, context)
    return (
        prediction.states[0].cpu().numpy().astype(np.float64),
        float(prediction.final_time_s[0].cpu()),
    )


def _forecast_from_fixed_states(
    states: np.ndarray,
    predicted_final_time_s: float,
    series: FlightSeries,
    config: TSConfig,
    normalizer: Normalizer,
    anchor: int,
    passes: int,
) -> Forecast:
    offsets = np.arange(1, len(states) + 1, dtype=np.float64) * config.dt_s
    final_time_s = float(offsets[-1]) if len(offsets) else 0.0
    progress = offsets / final_time_s if final_time_s else np.zeros_like(offsets)
    durations = np.full(len(offsets), config.dt_s, dtype=np.float64)
    physical = states_with_derived_velocity(
        series.values[anchor], normalizer.decode(states), durations
    )
    return Forecast(
        times=float(series.times[anchor]) + offsets,
        values=physical,
        normalized_progress=progress,
        anchor=anchor,
        final_time_s=final_time_s,
        predicted_final_time_s=predicted_final_time_s,
        horizon_mode=config.horizon_mode,
        passes=passes,
        truncated_at_threshold=False,
        horizon_capped=False,
        sample_durations_s=durations,
        segment_durations_s=durations,
        prediction_output=config.prediction_output,
    )


def _forecast_normalized(
    model: nn.Module,
    series: FlightSeries,
    config: TSConfig,
    normalizer: Normalizer,
    anchor: int,
    device: torch.device,
) -> Forecast:
    history = history_at_anchor(series, config, normalizer, anchor)
    states, predicted_final_time_s = _forward(
        model, history, device, _final_approach_context(series, config, device)
    )
    time_grid = output_time_grid(predicted_final_time_s, config)
    offsets = time_grid.offsets_s
    final_time_s = float(offsets[-1]) if len(offsets) else 0.0
    progress = offsets / final_time_s if final_time_s else np.zeros_like(offsets)
    physical = states_with_derived_velocity(
        series.values[anchor], normalizer.decode(states), time_grid.segment_durations_s
    )
    return Forecast(
        times=float(series.times[anchor]) + offsets,
        values=physical,
        normalized_progress=progress,
        anchor=anchor,
        final_time_s=final_time_s,
        predicted_final_time_s=predicted_final_time_s,
        horizon_mode=config.horizon_mode,
        passes=1,
        truncated_at_threshold=False,
        horizon_capped=False,
        sample_durations_s=time_grid.segment_durations_s,
        segment_durations_s=time_grid.segment_durations_s,
        prediction_output=config.prediction_output,
    )


def _forecast_full(
    model: nn.Module,
    series: FlightSeries,
    config: TSConfig,
    normalizer: Normalizer,
    anchor: int,
    device: torch.device,
) -> Forecast:
    history = history_at_anchor(series, config, normalizer, anchor)
    states, predicted_final_time_s = _forward(
        model, history, device, _final_approach_context(series, config, device)
    )
    return _forecast_from_fixed_states(
        states, predicted_final_time_s, series, config, normalizer, anchor, passes=1
    )


def _forecast_window(
    model: nn.Module,
    series: FlightSeries,
    config: TSConfig,
    normalizer: Normalizer,
    anchor: int,
    device: torch.device,
) -> Forecast:
    # The recursion feeds predicted STATE rows back as history; the conditioning row is
    # a per-flight constant, so it is re-appended to every pass rather than recursed.
    conditioning = series_conditioning(series, config, normalizer, anchor=anchor)
    context = _final_approach_context(series, config, device)
    encoded = normalizer.encode(series.values)
    history = encoded[anchor - config.seq_len + 1 : anchor + 1]
    chunks: list[np.ndarray] = []
    predicted_final_time_s = 0.0
    produced = 0
    while produced < config.full_horizon_steps:
        states, pass_final_time_s = _forward(
            model, conditioned_history(history, conditioning), device, context
        )
        if not chunks:
            predicted_final_time_s = pass_final_time_s
        physical_anchor = normalizer.decode(history[-1:])[0]
        physical_states = states_with_derived_velocity(
            physical_anchor,
            normalizer.decode(states),
            np.full(len(states), config.dt_s, dtype=np.float64),
        )
        states = normalizer.encode(physical_states)
        chunks.append(states)
        produced += len(states)
        history = np.concatenate((history, states), axis=0)[-config.seq_len :]
    future = np.concatenate(chunks, axis=0)[: config.full_horizon_steps]
    return _forecast_from_fixed_states(
        future,
        predicted_final_time_s,
        series,
        config,
        normalizer,
        anchor,
        passes=len(chunks),
    )


_FORECASTERS: dict[str, Callable[..., Forecast]] = {
    HORIZON_NORMALIZED: _forecast_normalized,
    HORIZON_FULL: _forecast_full,
    HORIZON_WINDOW: _forecast_window,
}


def _keep_complete_forecast(forecast: Forecast, series: FlightSeries) -> Forecast:
    del series
    return forecast


def _truncate_fixed_forecast(forecast: Forecast, series: FlightSeries) -> Forecast:
    truncated = truncate_at_threshold(forecast, series.target_chart)
    if truncated.truncated_at_threshold:
        return truncated
    return replace(truncated, horizon_capped=True)


_POSTPROCESSORS = {
    HORIZON_NORMALIZED: _keep_complete_forecast,
    HORIZON_FULL: _truncate_fixed_forecast,
    HORIZON_WINDOW: _truncate_fixed_forecast,
}


def project_onto_final(forecast: Forecast, series: FlightSeries, gate: str) -> Forecast:
    """Clamp a state forecast into the final-approach corridor and glidepath window.

    Every row the forecast itself places on the final is bound (``on-final``: inside the
    membership cone and the predicted path aligned with the course) — row by row, the same
    gate the corridor-bounded output layer applies softly, so the post-hoc projection is
    the hard counterpart of that layer. ``gate`` is the label ``predict --project-final``
    takes; the FAF-distance gate it used to name beside this one is deleted (2026-09-09).  (An earlier version bound only the suffix from which every
    later row was on the final; on arm A that moved 1.35 % of the rows, because one off-final
    row near the end cancelled the whole tail, and was not comparable to the layer.)
    Along-track distance is untouched, cross-track is clamped into ``±k·hw(d)``, height
    into the glidepath window, and the velocity channels are re-derived from the moved
    positions so the record stays one trajectory.  Nothing is learned here: it is what the
    constraint recovers after the fact, and the deployment fallback — it satisfies the rows
    and pays for it in kinks.

    The corridor geometry (`final_approach_geometry`) is written about the THRESHOLD as the
    origin, so the positions are translated by ``series.target_chart`` on the way in and
    back on the way out — the same translation `cut_at_threshold_crossing` makes. Under
    ``enu`` / ``runway-aligned`` the target IS the origin and this is the identity; under
    ``airport-enu`` it is 1.3–1.9 km, and until 2026-09-09 the projection clamped about the
    airport reference point instead (review C-11).
    """
    if forecast.prediction_output != PREDICTION_STATE:
        raise ValueError("project_onto_final applies to state forecasts only")
    if gate != CORRIDOR_GATE_ON_FINAL:
        raise ValueError(f"unknown corridor gate {gate!r}; the one gate is {CORRIDOR_GATE_ON_FINAL!r}")
    rows = final_approach_arrays(series)
    target = np.asarray(series.target_chart, dtype=np.float64)
    values = torch.as_tensor(forecast.values, dtype=torch.float64)[None]
    e = values[..., IDX["e"]] - float(target[0])
    n = values[..., IDX["n"]] - float(target[1])
    u = values[..., IDX["u"]] - float(target[2])
    psi = torch.as_tensor(rows["runway_heading_rad"], dtype=torch.float64)[None]
    tan_gpa = torch.as_tensor(rows["glidepath_tan"], dtype=torch.float64)[None]
    d, xt = runway_axes(e, n, psi)
    anchor = torch.as_tensor(series.values[forecast.anchor], dtype=torch.float64)
    step_e, step_n = position_direction(
        e, n,
        (anchor[IDX["e"]] - float(target[0]))[None],
        (anchor[IDX["n"]] - float(target[1]))[None],
    )
    cos_align = alignment_cosine(step_e, step_n, psi)
    on_final = membership(d=d, xt=xt, cos_align=cos_align, hard=True)
    xt_bounded, u_bounded = bound_to_final(
        d=d, xt=xt, u=u, weight=on_final.to(torch.float64), tan_gpa=tan_gpa, hard=True,
    )
    e_bounded, n_bounded = chart_from_axes(d, xt_bounded, psi)
    projected = np.array(forecast.values, dtype=np.float64, copy=True)
    projected[:, IDX["e"]] = e_bounded[0].numpy() + target[0]
    projected[:, IDX["n"]] = n_bounded[0].numpy() + target[1]
    projected[:, IDX["u"]] = u_bounded[0].numpy() + target[2]
    projected = states_with_derived_velocity(
        series.values[forecast.anchor], projected, forecast.sample_durations_s
    )
    return replace(forecast, values=projected, projected_onto_final=gate)


def forecast_state(
    model: nn.Module,
    series: FlightSeries,
    config: TSConfig,
    normalizer: Normalizer,
    anchor: int,
    device: torch.device,
    truncate: bool,
    project_final: str | None,
) -> Forecast:
    forecast = _FORECASTERS[config.horizon_mode](
        model, series, config, normalizer, anchor, device
    )
    if truncate:
        forecast = _POSTPROCESSORS[config.horizon_mode](forecast, series)
    if project_final is not None:
        forecast = project_onto_final(forecast, series, project_final)
    return forecast


def truncate_at_threshold(forecast: Forecast, target_chart: np.ndarray) -> Forecast:
    """Cut a fixed-time forecast at its closest horizontal approach to the threshold.

    ``target_chart`` is the threshold's chart position (``FlightSeries.target_chart``) —
    the closest approach is to the TARGET, which is the origin only under the
    threshold-anchored frames.
    """
    distance = horizontal_distance_m(forecast.values, target_chart)
    closest = int(np.argmin(distance))
    if closest == len(distance) - 1:
        return forecast
    times = forecast.times[: closest + 1]
    step_s = forecast.final_time_s / len(forecast.times)
    final_time_s = float((closest + 1) * step_s)
    progress = np.arange(1, closest + 2, dtype=np.float64) / (closest + 1)
    return replace(
        forecast,
        times=times,
        values=forecast.values[: closest + 1],
        normalized_progress=progress,
        final_time_s=final_time_s,
        truncated_at_threshold=True,
        sample_durations_s=forecast.sample_durations_s[: closest + 1],
        segment_durations_s=forecast.segment_durations_s[: closest + 1],
    )
