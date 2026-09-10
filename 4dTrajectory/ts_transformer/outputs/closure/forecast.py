"""The closure path's inference: a decision vector drawn into a path, a clock and heights."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from ts_transformer.config import TSConfig, default_anchor
from ts_transformer.data.dataset import FlightSeries, Normalizer
from ts_transformer.inference.forecast import Forecast, history_batch
from ts_transformer.outputs.closure.model import (
    ClosureLabels,
    check_airport,
    decision_from_label,
    reconstruct,
)


def closure_forecast(item: FlightSeries, vector: np.ndarray, config: TSConfig, anchor: int,
                     *, from_labels: bool = False) -> Forecast:
    """One flight drawn from a decision vector: the closed-form path, clock and heights
    (no controls — the export takes the reference-shaped record branch). The record
    carries which construction drew it and whether the vector was the flight's label."""
    drawn = reconstruct(vector, item.values[anchor], float(item.scenario.target.psi), config)
    offsets = drawn.offsets_s
    durations = np.diff(np.concatenate(([0.0], offsets)))
    return Forecast(
        times=float(item.times[anchor]) + offsets,
        values=drawn.values,
        normalized_progress=offsets / drawn.final_time_s,
        anchor=anchor,
        final_time_s=drawn.final_time_s,
        predicted_final_time_s=drawn.final_time_s,
        horizon_mode=config.horizon_mode,
        passes=1,
        truncated_at_threshold=False,
        horizon_capped=False,
        sample_durations_s=durations,
        segment_durations_s=durations,
        prediction_output=config.prediction_output,
        closure_construction=drawn.construction,
        closure_from_labels=from_labels,
    )


def forecast_closure_batch(
    model: nn.Module,
    series: Sequence[FlightSeries],
    config: TSConfig,
    normalizer: Normalizer,
    anchor: int,
    device: torch.device,
) -> list[Forecast]:
    histories = history_batch(series, config, normalizer, anchor)
    model.eval()
    with torch.no_grad():
        prediction = model(torch.from_numpy(histories).to(device))
    decisions = prediction.decision.detach().cpu().numpy().astype(np.float64)
    return [closure_forecast(item, vector, config, anchor) for item, vector in zip(series, decisions, strict=True)]


def forecast_closure_from_labels(
    series: Sequence[FlightSeries], config: TSConfig, labels: ClosureLabels, *, anchor: int | None = None,
) -> list[Forecast]:
    """The closure family's own ceiling: every flight drawn from its LABEL instead of a
    model output (the oracle arm) — every label, the non-canonical and above-cap ones
    training leaves out included (each is still the family's best fit of that flight;
    the record's ``closureFromLabels`` marks the arm). A flight without a label has
    nothing to draw."""
    anchor = default_anchor(config) if anchor is None else anchor
    for item in series:
        check_airport(item, labels)
    missing = [item.flight_id for item in series if item.flight_id not in labels.flights]
    if missing:
        raise KeyError(f"{len(missing)} flight(s) have no closure label; first: {missing[0]!r}")
    decisions = [decision_from_label(labels.flights[item.flight_id], config) for item in series]
    return [closure_forecast(item, vector, config, anchor, from_labels=True) for item, vector in zip(series, decisions, strict=True)]
