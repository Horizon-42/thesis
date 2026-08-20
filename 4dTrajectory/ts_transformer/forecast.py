"""Inference strategies for normalized, full-horizon, and recursive-window prediction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn

from channels import horizontal_distance_m
from config import (
    HORIZON_FULL,
    HORIZON_NORMALIZED,
    HORIZON_WINDOW,
    PREDICTION_CONTROL,
    PREDICTION_STATE,
    TSConfig,
)
import control_rollout
from control_envelope import physical_controls
from dataset import FlightSeries, Normalizer, dynamics_arrays
from metrics import states_with_derived_velocity
from prediction_outputs import ControlPrediction
from time_grids import output_time_grid


@dataclass(frozen=True)
class Forecast:
    """A predicted approach in physical state units and physical wall-clock time."""

    times: np.ndarray
    values: np.ndarray
    normalized_progress: np.ndarray
    anchor: int
    final_time_s: float
    predicted_final_time_s: float
    horizon_mode: str
    passes: int
    truncated_at_threshold: bool
    horizon_capped: bool
    # State-sample intervals align 1:1 with ``times``/``values``. Control segment
    # durations remain sparse and align 1:1 with ``controls``; the two clocks are
    # deliberately separate because a dynamics trajectory is denser than its controls.
    sample_durations_s: np.ndarray
    segment_durations_s: np.ndarray
    controls: np.ndarray | None = None
    geodetic_values: np.ndarray | None = None
    prediction_output: str = PREDICTION_STATE

    @property
    def n_steps(self) -> int:
        return len(self.times)


def default_anchor(config: TSConfig) -> int:
    """Use the earliest anchor with a complete observed lookback."""
    return config.seq_len - 1


def _history_at_anchor(
    series: FlightSeries,
    config: TSConfig,
    normalizer: Normalizer,
    anchor: int,
) -> np.ndarray:
    if anchor < config.seq_len - 1:
        raise ValueError(
            f"anchor {anchor} has no full lookback window (needs at least {config.seq_len - 1})"
        )
    encoded = normalizer.encode(series.values)
    return encoded[anchor - config.seq_len + 1 : anchor + 1]


def _history_batch(
    series: Sequence[FlightSeries],
    config: TSConfig,
    normalizer: Normalizer,
    anchor: int,
) -> np.ndarray:
    return np.stack([
        _history_at_anchor(item, config, normalizer, anchor) for item in series
    ]).astype(np.float32, copy=False)


def _forward(
    model: nn.Module,
    history: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    """Run one deterministic pass in normalized channel space."""
    model.eval()
    tensor = torch.from_numpy(history[None, ...].astype(np.float32)).to(device)
    with torch.no_grad():
        prediction = model(tensor)
    return (
        prediction.states[0].cpu().numpy().astype(np.float64),
        float(prediction.final_time_s[0].cpu()),
    )


def _dynamics_batch(
    series: Sequence[FlightSeries],
    anchor: int,
    device: torch.device,
    config: TSConfig,
) -> dict[str, torch.Tensor]:
    rows = [dynamics_arrays(item, anchor, config) for item in series]
    return {
        name: torch.from_numpy(np.stack([row[name] for row in rows])).to(device)
        for name in rows[0]
    }


def _padded_dense_queries(
    segment_durations_s: np.ndarray,
    output_dt_s: float,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    offsets = [
        _dense_control_query_offsets(row, output_dt_s)
        for row in segment_durations_s
    ]
    width = max(len(row) for row in offsets)
    padded = np.zeros((len(offsets), width), dtype=np.float64)
    valid = np.zeros((len(offsets), width), dtype=bool)
    for row, values in enumerate(offsets):
        padded[row, : len(values)] = values
        valid[row, : len(values)] = True
    return offsets, padded, valid


def _control_prediction_batch(
    model: nn.Module,
    histories: np.ndarray,
    dynamics: dict[str, torch.Tensor],
    device: torch.device,
) -> ControlPrediction:
    """Preserve the original per-flight network arithmetic, then stack its schedules."""
    predictions: list[ControlPrediction] = []
    model.eval()
    with torch.no_grad():
        for row, history in enumerate(histories):
            row_dynamics = {
                name: value[row : row + 1] for name, value in dynamics.items()
            }
            predictions.append(
                model(torch.from_numpy(history[None]).to(device), row_dynamics)
            )
    return ControlPrediction(
        controls=torch.cat([item.controls for item in predictions], dim=0),
        segment_durations=torch.cat(
            [item.segment_durations for item in predictions], dim=0
        ),
        final_time_s=torch.cat([item.final_time_s for item in predictions], dim=0),
    )


def _forecast_control_batch(
    model: nn.Module,
    series: Sequence[FlightSeries],
    config: TSConfig,
    normalizer: Normalizer,
    anchor: int,
    device: torch.device,
) -> list[Forecast]:
    """Predict and densely roll a heterogeneous batch of bounded control schedules."""
    histories = _history_batch(series, config, normalizer, anchor)
    dynamics = _dynamics_batch(series, anchor, device, config)
    prediction = _control_prediction_batch(model, histories, dynamics, device)
    durations = prediction.segment_durations.detach().cpu().numpy().astype(np.float64)
    offsets, padded_offsets, query_valid = _padded_dense_queries(
        durations, config.control_rollout_integrator_dt_s
    )
    with torch.no_grad():
        rollout = control_rollout.rollout_control_dense(
            prediction.controls,
            prediction.segment_durations,
            dynamics,
            torch.from_numpy(padded_offsets),
            torch.from_numpy(query_valid),
            config,
        )
    query_channels = rollout.query_channels.detach().cpu().numpy().astype(np.float64)
    query_geodetic = (
        rollout.query_geodetic_states.detach().cpu().numpy().astype(np.float64)
    )
    # The head predicts in the dimensionless envelope; the exported record contract is
    # newtons, and is shared with the CasADi optimizer and the evaluation package.
    controls = (
        physical_controls(prediction.controls, dynamics["max_thrust_n"])
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    predicted_final_time = (
        prediction.final_time_s.detach().cpu().numpy().astype(np.float64)
    )
    forecasts: list[Forecast] = []
    for row, (item, row_offsets) in enumerate(zip(series, offsets, strict=True)):
        count = len(row_offsets)
        final_time_s = float(row_offsets[-1])
        forecasts.append(Forecast(
            times=float(item.times[anchor]) + row_offsets,
            values=query_channels[row, :count],
            normalized_progress=row_offsets / final_time_s,
            anchor=anchor,
            final_time_s=final_time_s,
            predicted_final_time_s=float(predicted_final_time[row]),
            horizon_mode=config.horizon_mode,
            passes=1,
            truncated_at_threshold=False,
            horizon_capped=False,
            controls=controls[row],
            sample_durations_s=np.diff(np.concatenate(([0.0], row_offsets))),
            segment_durations_s=durations[row],
            geodetic_values=query_geodetic[row, :count],
            prediction_output=config.prediction_output,
        ))
    return forecasts


def _dense_control_query_offsets(
    segment_durations_s: np.ndarray, output_dt_s: float
) -> np.ndarray:
    """Return regular output times plus every exact control-switch boundary."""
    durations = np.asarray(segment_durations_s, dtype=np.float64)
    if durations.ndim != 1 or not len(durations):
        raise ValueError("control forecast needs at least one segment duration")
    if not np.isfinite(durations).all() or np.any(durations <= 0.0):
        raise ValueError("control segment durations must be finite and positive")
    if not np.isfinite(output_dt_s) or output_dt_s <= 0.0:
        raise ValueError("dense control output interval must be finite and positive")
    boundaries = np.cumsum(durations)
    total = float(boundaries[-1])
    regular = np.arange(output_dt_s, total, output_dt_s, dtype=np.float64)
    candidates = np.sort(np.concatenate((regular, boundaries)))
    tolerance = np.finfo(np.float64).eps * max(total, 1.0) * 16.0
    keep = np.concatenate(([True], np.diff(candidates) > tolerance))
    offsets = candidates[keep]
    offsets[-1] = total
    return offsets


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
    history = _history_at_anchor(series, config, normalizer, anchor)
    states, predicted_final_time_s = _forward(model, history, device)
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
    history = _history_at_anchor(series, config, normalizer, anchor)
    states, predicted_final_time_s = _forward(model, history, device)
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
    history = _history_at_anchor(series, config, normalizer, anchor)
    chunks: list[np.ndarray] = []
    predicted_final_time_s = 0.0
    produced = 0
    while produced < config.full_horizon_steps:
        states, pass_final_time_s = _forward(model, history, device)
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


def _keep_complete_forecast(forecast: Forecast) -> Forecast:
    return forecast


def _truncate_fixed_forecast(forecast: Forecast) -> Forecast:
    truncated = truncate_at_threshold(forecast)
    if truncated.truncated_at_threshold:
        return truncated
    return replace(truncated, horizon_capped=True)


_POSTPROCESSORS = {
    HORIZON_NORMALIZED: _keep_complete_forecast,
    HORIZON_FULL: _truncate_fixed_forecast,
    HORIZON_WINDOW: _truncate_fixed_forecast,
}


def _forecast_state(
    model: nn.Module,
    series: FlightSeries,
    config: TSConfig,
    normalizer: Normalizer,
    anchor: int,
    device: torch.device,
    truncate: bool,
) -> Forecast:
    forecast = _FORECASTERS[config.horizon_mode](
        model, series, config, normalizer, anchor, device
    )
    if not truncate:
        return forecast
    return _POSTPROCESSORS[config.horizon_mode](forecast)


def forecast_approaches(
    model: nn.Module,
    series: Sequence[FlightSeries],
    config: TSConfig,
    normalizer: Normalizer,
    *,
    anchor: int | None = None,
    device: torch.device | None = None,
    truncate: bool = True,
) -> list[Forecast]:
    """Predict one inference batch through the same dense path used by fit evaluation."""
    if not series:
        return []
    device = device or next(model.parameters()).device
    anchor = default_anchor(config) if anchor is None else anchor
    if config.prediction_output == PREDICTION_CONTROL:
        return _forecast_control_batch(
            model, series, config, normalizer, anchor, device
        )
    return [
        _forecast_state(
            model, item, config, normalizer, anchor, device, truncate
        )
        for item in series
    ]


def forecast_approach(
    model: nn.Module,
    series: FlightSeries,
    config: TSConfig,
    normalizer: Normalizer,
    *,
    anchor: int | None = None,
    device: torch.device | None = None,
    truncate: bool = True,
) -> Forecast:
    """Predict from one observed anchor using the configured output strategy."""
    return forecast_approaches(
        model,
        [series],
        config,
        normalizer,
        anchor=anchor,
        device=device,
        truncate=truncate,
    )[0]


def truncate_at_threshold(forecast: Forecast) -> Forecast:
    """Cut a fixed-time forecast at its closest horizontal approach to the threshold."""
    distance = horizontal_distance_m(forecast.values)
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
