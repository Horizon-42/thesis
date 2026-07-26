"""One-pass prediction on a fixed normalized progress grid.

The model emits N state endpoints over ``tau in (0, 1]`` plus the physical duration from
the observed anchor to ``tau=1``.  Wall-clock timestamps are reconstructed only after the
forward pass, so there is no fixed-step horizon or geometric post-truncation rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from config import TSConfig
from dataset import FlightSeries, Normalizer


@dataclass(frozen=True)
class Forecast:
    """A predicted approach in physical state units and physical wall-clock time."""

    times: np.ndarray                 # [N], absolute seconds in the source track
    values: np.ndarray                # [N, C], decoded channel space
    normalized_progress: np.ndarray   # [N], 1/N ... 1
    anchor: int                       # last observed sample shown to the model
    final_time_s: float               # predicted seconds from anchor to endpoint

    @property
    def n_steps(self) -> int:
        return len(self.times)


def default_anchor(config: TSConfig) -> int:
    """Use the earliest anchor with a complete observed lookback."""
    return config.seq_len - 1


def forecast_approach(
    model: nn.Module,
    series: FlightSeries,
    config: TSConfig,
    normalizer: Normalizer,
    *,
    anchor: int | None = None,
    device: torch.device | None = None,
) -> Forecast:
    """Predict one complete remaining approach in one forward pass."""
    device = device or next(model.parameters()).device
    anchor = default_anchor(config) if anchor is None else anchor
    if anchor < config.seq_len - 1:
        raise ValueError(
            f"anchor {anchor} has no full lookback window (needs at least {config.seq_len - 1})"
        )

    encoded = normalizer.encode(series.values)
    history = encoded[anchor - config.seq_len + 1 : anchor + 1]
    model.eval()
    tensor = torch.from_numpy(history[None, ...].astype(np.float32)).to(device)
    with torch.no_grad():
        prediction = model(tensor)

    normalized_states = prediction.states[0].cpu().numpy().astype(np.float64)
    final_time_s = float(prediction.final_time_s[0].cpu())
    progress = np.arange(1, config.n_segments + 1, dtype=np.float64) / config.n_segments
    times = float(series.times[anchor]) + progress * final_time_s
    return Forecast(
        times=times,
        values=normalizer.decode(normalized_states),
        normalized_progress=progress,
        anchor=anchor,
        final_time_s=final_time_s,
    )
