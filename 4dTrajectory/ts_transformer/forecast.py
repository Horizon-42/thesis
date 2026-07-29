"""Inference strategies for normalized, full-horizon, and recursive-window prediction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from channels import horizontal_distance_m
from config import (
    HORIZON_FULL,
    HORIZON_NORMALIZED,
    HORIZON_WINDOW,
    PREDICTION_CONTROL,
    TSConfig,
)
from dataset import FlightSeries, Normalizer, dynamics_arrays
from prediction_outputs import ControlPrediction
from time_grids import output_time_grid
from train import control_rollout_channels


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
    segment_durations_s: np.ndarray
    controls: np.ndarray | None = None
    geodetic_values: np.ndarray | None = None

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


def _forecast_control(
    model: nn.Module,
    series: FlightSeries,
    config: TSConfig,
    normalizer: Normalizer,
    anchor: int,
    device: torch.device,
) -> Forecast:
    """Predict bounded controls, then obtain states only through the shared dynamics."""
    history = _history_at_anchor(series, config, normalizer, anchor)
    tensor = torch.from_numpy(history[None, ...].astype(np.float32)).to(device)
    dynamics = {
        name: torch.from_numpy(value[None, ...]).to(device)
        for name, value in dynamics_arrays(series, anchor).items()
    }
    model.eval()
    with torch.no_grad():
        prediction = model(tensor, dynamics)
        if not isinstance(prediction, ControlPrediction):
            raise TypeError("control checkpoint did not return ControlPrediction")
        channels, geodetic = control_rollout_channels(prediction, dynamics, config)
    durations = prediction.segment_durations[0].cpu().numpy().astype(np.float64)
    offsets = np.cumsum(durations)
    final_time_s = float(offsets[-1])
    return Forecast(
        times=float(series.times[anchor]) + offsets,
        values=channels[0].cpu().numpy().astype(np.float64),
        normalized_progress=offsets / final_time_s,
        anchor=anchor,
        final_time_s=final_time_s,
        predicted_final_time_s=float(prediction.final_time_s[0].cpu()),
        horizon_mode=config.horizon_mode,
        passes=1,
        truncated_at_threshold=False,
        horizon_capped=False,
        controls=prediction.controls[0].cpu().numpy().astype(np.float64),
        segment_durations_s=durations,
        geodetic_values=geodetic[0].cpu().numpy().astype(np.float64),
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
    return Forecast(
        times=float(series.times[anchor]) + offsets,
        values=normalizer.decode(states),
        normalized_progress=progress,
        anchor=anchor,
        final_time_s=final_time_s,
        predicted_final_time_s=predicted_final_time_s,
        horizon_mode=config.horizon_mode,
        passes=passes,
        truncated_at_threshold=False,
        horizon_capped=False,
        segment_durations_s=np.full(len(offsets), config.dt_s, dtype=np.float64),
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
    return Forecast(
        times=float(series.times[anchor]) + offsets,
        values=normalizer.decode(states),
        normalized_progress=progress,
        anchor=anchor,
        final_time_s=final_time_s,
        predicted_final_time_s=predicted_final_time_s,
        horizon_mode=config.horizon_mode,
        passes=1,
        truncated_at_threshold=False,
        horizon_capped=False,
        segment_durations_s=time_grid.segment_durations_s,
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
    """Predict from one observed anchor using the configured, independently dispatched mode."""
    device = device or next(model.parameters()).device
    anchor = default_anchor(config) if anchor is None else anchor
    if config.prediction_output == PREDICTION_CONTROL:
        return _forecast_control(model, series, config, normalizer, anchor, device)
    forecast = _FORECASTERS[config.horizon_mode](
        model, series, config, normalizer, anchor, device
    )
    if not truncate:
        return forecast
    return _POSTPROCESSORS[config.horizon_mode](forecast)


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
        segment_durations_s=forecast.segment_durations_s[: closest + 1],
    )
