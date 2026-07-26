"""Turning a trained model into a predicted approach.

Two ways to cover a whole approach, matching the two horizon modes:

**full** — one forward pass emits ``pred_len`` steps covering the rest of the approach.
No compounding: every predicted step was produced from real observed history.

**window** — ``pred_len`` is short, so :func:`recursive_forecast` chains passes, appending
each prediction to the history and sliding the input window forward. From the second pass
on, the model is reading its own output. Error compounds, and measuring how much is the
reason both modes exist.

Note the naming: ``4dTrajectory/optimization`` already uses "rollout" for forward-integrating
optimizer controls through the true dynamics. That is a different operation on a different
object, so nothing here is called a rollout.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn as nn

from channels import horizontal_distance_m
from config import DEFAULT_PRED_LEN_FULL, HORIZON_FULL, TSConfig
from dataset import FlightSeries, Normalizer


@dataclass
class Forecast:
    """A predicted approach, in physical units, on the series' own time grid."""

    times: np.ndarray        # [K] seconds, continuing the series' grid past the anchor
    values: np.ndarray       # [K, C] channel space
    anchor: int              # index of the last OBSERVED sample the model was shown
    passes: int              # forward passes used (1 in full mode, >1 when chained)
    truncated_at_threshold: bool
    # True when the fixed horizon ended the forecast BEFORE it reached the threshold —
    # the final state is then a horizon artifact, not a prediction, and grading it against
    # the gates measures the cap, not the model. Set by forecast_approach; carried into
    # the record/summary so a gate failure can be told apart from a capped flight.
    horizon_capped: bool = False

    @property
    def n_steps(self) -> int:
        return len(self.times)


def default_anchor(config: TSConfig) -> int:
    """The earliest anchor with a full lookback window — predict as much as possible.

    Anchoring at ``seq_len - 1`` means the model sees the first ``lookback_s`` seconds of the
    arrival and forecasts everything after it, which is the comparison the optimizer makes
    (it plans the whole trajectory from an initial state).
    """
    return config.seq_len - 1


def _forward(
    model: nn.Module, history: np.ndarray, device: torch.device
) -> np.ndarray:
    """One pass. ``history`` is ``[L, C]`` NORMALISED; returns ``[H, C]`` normalised.

    Forces eval mode: with the default dropout a train-mode model returns a different
    trajectory every call, and in window mode that noise compounds through the chain.
    """
    model.eval()
    x = torch.from_numpy(history[None, ...].astype(np.float32)).to(device)
    with torch.no_grad():
        return model(x)[0].cpu().numpy().astype(np.float64)


def recursive_forecast(
    model: nn.Module,
    series: FlightSeries,
    config: TSConfig,
    normalizer: Normalizer,
    *,
    anchor: int | None = None,
    max_steps: int,
    device: torch.device | None = None,
) -> Forecast:
    """Chain forward passes until ``max_steps`` future samples exist.

    Chaining happens entirely in NORMALISED space — the model's input and output live
    there, so decoding and re-encoding between passes would only add float round-trips.
    One decode at the end.
    """
    device = device or next(model.parameters()).device
    anchor = default_anchor(config) if anchor is None else anchor
    if anchor < config.seq_len - 1:
        raise ValueError(
            f"anchor {anchor} has no full lookback window (needs at least {config.seq_len - 1})"
        )

    encoded = normalizer.encode(series.values)
    history = encoded[anchor - config.seq_len + 1 : anchor + 1]

    produced: list[np.ndarray] = []
    passes = 0
    while sum(len(chunk) for chunk in produced) < max_steps: # TODO use a more resonable stopping criterion than max_steps, e.g. distance to threshold
        step = _forward(model, history, device)
        produced.append(step)
        passes += 1
        # Slide: the last L samples of (history + this prediction) become the next input.
        history = np.concatenate([history, step], axis=0)[-config.seq_len :] # Take the last seq_len samples from the concatenated history and step 
                                                                             # to form the new history for the next iteration. 

    future = np.concatenate(produced, axis=0)[:max_steps]
    times = series.times[anchor] + np.arange(1, len(future) + 1, dtype=np.float64) * config.dt_s
    return Forecast(
        times=times, values=normalizer.decode(future), anchor=anchor, passes=passes,
        truncated_at_threshold=False,
    )


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
    """Predict the rest of the approach, dispatching on ``config.horizon_mode``.

    Both modes run through :func:`recursive_forecast` (one code path, one anchor
    validation): in full mode ``pred_len`` covers ``max_steps``, so it is a single pass;
    in window mode the chain runs long enough to cover a full-mode horizon
    (``DEFAULT_PRED_LEN_FULL`` steps). Either way the forecast is then cut at the
    threshold, so both modes produce a comparable object regardless of how many passes
    it took to get there.
    """
    max_steps = (
        config.pred_len if config.horizon_mode == HORIZON_FULL
        else max(DEFAULT_PRED_LEN_FULL, config.pred_len)
    )
    forecast = recursive_forecast(
        model, series, config, normalizer, anchor=anchor, max_steps=max_steps, device=device,
    )

    if truncate:
        forecast = truncate_at_threshold(forecast)
        if not forecast.truncated_at_threshold:
            # The horizon ran out before the forecast turned back past the threshold —
            # ~2% of harvested flights outlast the 600 s cap (config.py's distribution).
            # Never silent: the flag reaches the eval record and the summary row, and
            # predict prints a warning with the count.
            forecast = replace(forecast, horizon_capped=True)
    return forecast


def truncate_at_threshold(forecast: Forecast) -> Forecast:
    """Cut the forecast at its closest horizontal approach to the runway threshold.

    The ENU frame is anchored AT the threshold, so the approach ends where horizontal
    distance from the origin is smallest. Everything after that is the model flying past
    the runway — an artifact of a fixed-length horizon, not a prediction anyone wants
    judged, and keeping it would put the final state (which the evaluation gates read)
    somewhere beyond the airport.

    This is an explicit, stated rule rather than a silent cap: ``truncated_at_threshold``
    records whether it fired, and passing ``truncate=False`` to
    :func:`forecast_approach` disables it.
    """
    distance = horizontal_distance_m(forecast.values)
    closest = int(np.argmin(distance))
    if closest == len(distance) - 1:
        return forecast  # never turned back outbound; nothing to cut
    return Forecast(
        times=forecast.times[: closest + 1],
        values=forecast.values[: closest + 1],
        anchor=forecast.anchor,
        passes=forecast.passes,
        truncated_at_threshold=True,
    )
