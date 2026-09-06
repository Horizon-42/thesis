"""Inference strategies for normalized, full-horizon, and recursive-window prediction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn

from batch_contract import model_forward
from channels import IDX, horizontal_distance_m
from config import (
    CTA_CONDITIONING_GIVEN,
    CORRIDOR_GATES,
    HORIZON_FULL,
    HORIZON_NORMALIZED,
    HORIZON_WINDOW,
    PREDICTION_CLOSURE,
    PREDICTION_CONTROL,
    PREDICTION_STATE,
    TSConfig,
)
from closure_output import ClosureLabels, check_airport, decision_from_label, reconstruct
from control.constraints import build_command_hook
from control.dynamics import rollout as control_rollout
from control.envelope import physical_controls
from dataset import (
    FixedAnchorTrajectoryWindows,
    truth_duration_s,
    FlightSeries,
    Normalizer,
    bounded_output_gate,
    dynamics_arrays,
    final_approach_arrays,
    final_approach_fix_distance,
    series_conditioning,
)
from final_approach_geometry import (
    alignment_cosine,
    bound_to_final,
    chart_from_axes,
    membership,
    position_direction,
    runway_axes,
)
from metrics import states_with_derived_velocity
from prediction_outputs import ControlPrediction
from target_conditioning import conditioned_history
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
    # The corridor gate the inference-time projection applied (``project_onto_final``),
    # or None: the values are the model's own.
    projected_onto_final: str | None = None
    # The rollout command hook that rewrote the schedule (``hook/saturation``), or None.
    command_hook: str | None = None
    # Closure output only: which construction drew the path (via-Dubins, or a fallback),
    # and whether it was drawn from the flight's LABEL rather than a model output.
    closure_construction: str | None = None
    closure_from_labels: bool = False
    # Latent control output only: which prior SAMPLE this forecast decodes (None = the
    # deterministic top-1 the record contract carries) and its probability — the mixture
    # weight of the component it was drawn from, or 1/samples under a single Gaussian.
    mode_index: int | None = None
    mode_probability: float | None = None
    # Latent control output only: decoded from ANOTHER flight's top-1 latent (the
    # posterior-collapse diagnostic — a decoder that ignores z barely moves).
    latent_shuffled: bool = False
    # CTA-conditioned control output only: the arrival time the decoder was given (truth +
    # the counterfactual offset) — the record says what it was asked for.
    cta_s: float | None = None
    cta_offset_s: float | None = None
    # Latent control output only: decoded from the POSTERIOR mean q(z | this flight's own
    # future) — the z-oracle upper bound. Reads the future; never a prediction result.
    z_from_posterior: bool = False

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
    return conditioned_history(
        encoded[anchor - config.seq_len + 1 : anchor + 1],
        series_conditioning(series, config, normalizer, anchor=anchor),
    )


def _history_batch(
    series: Sequence[FlightSeries],
    config: TSConfig,
    normalizer: Normalizer,
    anchor: int,
) -> np.ndarray:
    return np.stack([
        _history_at_anchor(item, config, normalizer, anchor) for item in series
    ]).astype(np.float32, copy=False)


def _final_approach_context(
    series: FlightSeries, config: TSConfig, device: torch.device
) -> dict[str, torch.Tensor] | None:
    """The one-flight context a corridor-bounded output needs; None for every other recipe."""
    if not config.uses_final_approach_context:
        return None
    rows = final_approach_arrays(
        series,
        fix_distance_m=final_approach_fix_distance(series, gate=bounded_output_gate(config)),
    )
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


def _dynamics_batch(
    series: Sequence[FlightSeries], anchor: int, device: torch.device,
    config: TSConfig, cta_offset_s: float = 0.0,
) -> dict[str, torch.Tensor]:
    """The per-flight context; under ``cta_conditioning=given`` it carries the CTA =
    the truth duration + ``cta_offset_s`` (the counterfactual a scheduler asks for)."""
    rows = [dynamics_arrays(item, anchor) for item in series]
    if config.cta_conditioning == CTA_CONDITIONING_GIVEN:
        for item, row in zip(series, rows, strict=True):
            row["cta_s"] = np.array(truth_duration_s(item, anchor) + cta_offset_s, dtype=np.float64)
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
    latent: torch.Tensor | None = None,
) -> ControlPrediction:
    """Preserve the original per-flight network arithmetic, then stack its schedules.

    ``latent`` (``[B, Z]``) decodes a latent control model from these latents instead of
    its prior's top-1 — the K-sample and shuffled-z forecasts.
    """
    predictions: list[ControlPrediction] = []
    model.eval()
    with torch.no_grad():
        for row, history in enumerate(histories):
            row_dynamics = {
                name: value[row : row + 1] for name, value in dynamics.items()
            }
            history_row = torch.from_numpy(history[None]).to(device)
            predictions.append(
                model(history_row, row_dynamics) if latent is None
                else model(history_row, row_dynamics, latent=latent[row : row + 1])
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
    *,
    latent: torch.Tensor | None = None,
    mode: tuple[int, np.ndarray] | None = None,
    latent_shuffled: bool = False,
    z_from_posterior: bool = False,
    histories: np.ndarray | None = None,
    dynamics: dict[str, torch.Tensor] | None = None,
    cta_offset_s: float = 0.0,
) -> list[Forecast]:
    """Predict and densely roll a heterogeneous batch of bounded control schedules.

    ``latent`` decodes from given latents (``[B, Z]``); ``mode`` = (sample index, per-flight
    probability) stamps the forecasts as that prior sample; ``latent_shuffled`` stamps them
    as the collapse diagnostic. ``histories`` / ``dynamics`` accept the batch's inputs when
    the caller decodes the same flights several times.
    """
    if histories is None:
        histories = _history_batch(series, config, normalizer, anchor)
    if dynamics is None:
        dynamics = _dynamics_batch(series, anchor, device, config, cta_offset_s)
    prediction = _control_prediction_batch(model, histories, dynamics, device, latent=latent)
    cta = (
        dynamics["cta_s"].detach().cpu().numpy().astype(np.float64)
        if config.cta_conditioning == CTA_CONDITIONING_GIVEN else None
    )
    durations = prediction.segment_durations.detach().cpu().numpy().astype(np.float64)
    offsets, padded_offsets, query_valid = _padded_dense_queries(
        durations, config.control_rollout_integrator_dt_s
    )
    command_hook = build_command_hook(config, dynamics)
    with torch.no_grad():
        rollout = control_rollout.rollout_control_dense(
            prediction.controls,
            prediction.segment_durations,
            dynamics,
            torch.from_numpy(padded_offsets),
            torch.from_numpy(query_valid),
            config,
            command_hook=command_hook,
        )
    query_channels = rollout.query_channels.detach().cpu().numpy().astype(np.float64)
    query_geodetic = (
        rollout.query_geodetic_states.detach().cpu().numpy().astype(np.float64)
    )
    # The head predicts in the dimensionless envelope; the exported record contract is
    # newtons, and is shared with the CasADi optimizer and the evaluation package.
    # The schedule FLOWN (a hook may have rewritten the network's commands), in newtons.
    controls = (
        physical_controls(
            rollout.controls.to(prediction.controls.dtype), dynamics["max_thrust_n"]
        )
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
            command_hook=(
                None if command_hook is None
                else f"{config.control_command_hook}/{config.control_hook_saturation}"
            ),
            mode_index=None if mode is None else mode[0],
            mode_probability=None if mode is None else float(mode[1][row]),
            latent_shuffled=latent_shuffled,
            z_from_posterior=z_from_posterior,
            cta_s=None if cta is None else float(cta[row]),
            cta_offset_s=None if cta is None else float(cta_offset_s),
        ))
    return forecasts


def _latent_prior_batch(
    model: nn.Module,
    histories: np.ndarray,
    dynamics: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The prior's (logits, mean, logvar) and its top-1 latent for a batch, per flight."""
    logits, means, logvars, top1 = [], [], [], []
    model.eval()
    with torch.no_grad():
        for row, history in enumerate(histories):
            row_dynamics = {name: value[row : row + 1] for name, value in dynamics.items()}
            prediction = model(torch.from_numpy(history[None]).to(device), row_dynamics)
            logits.append(prediction.prior_logits)
            means.append(prediction.prior_mean)
            logvars.append(prediction.prior_logvar)
            top1.append(prediction.latent)
    return (torch.cat(logits), torch.cat(means), torch.cat(logvars), torch.cat(top1))


def _latent_batch(
    model: nn.Module,
    series: Sequence[FlightSeries],
    config: TSConfig,
    normalizer: Normalizer,
    anchor: int | None,
    device: torch.device | None,
    cta_offset_s: float,
) -> tuple[int, torch.device, np.ndarray, dict[str, torch.Tensor]]:
    """The anchor, device and batch inputs every latent decode below starts from."""
    device = device or next(model.parameters()).device
    anchor = default_anchor(config) if anchor is None else anchor
    return (
        anchor,
        device,
        _history_batch(series, config, normalizer, anchor),
        _dynamics_batch(series, anchor, device, config, cta_offset_s),
    )


def _latent_forecasts(
    model: nn.Module,
    series: Sequence[FlightSeries],
    config: TSConfig,
    normalizer: Normalizer,
    anchor: int,
    device: torch.device,
    *,
    latents: torch.Tensor,
    probabilities: np.ndarray | None,
    histories: np.ndarray,
    dynamics: dict[str, torch.Tensor],
    latent_shuffled: bool = False,
    z_from_posterior: bool = False,
    cta_offset_s: float = 0.0,
) -> list[list[Forecast]]:
    """Decode and roll the batch once per latent: ``latents`` is ``[S, B, Z]``.

    ``probabilities`` is ``[S, B]`` for prior samples and ``None`` when the latents are not
    samples at all (the z-oracle, the shuffle), in which case the forecasts carry no mode
    index. That is the only difference between the four public entries below.

    ``config.latent_dim > 0`` is deliberately NOT re-checked here. ``__main__`` refuses
    every latent flag against a non-latent checkpoint — twice, once for
    ``--z-from-posterior`` and once for the prior-sample options — and a second copy of
    that rule is a second place to forget.
    """
    return [
        _forecast_control_batch(
            model, series, config, normalizer, anchor, device,
            latent=latents[index],
            mode=None if probabilities is None else (index, probabilities[index]),
            latent_shuffled=latent_shuffled,
            z_from_posterior=z_from_posterior,
            histories=histories, dynamics=dynamics, cta_offset_s=cta_offset_s,
        )
        for index in range(len(latents))
    ]


def latent_mode_forecasts(
    model: nn.Module,
    series: Sequence[FlightSeries],
    config: TSConfig,
    normalizer: Normalizer,
    *,
    samples: int,
    seed: int,
    anchor: int | None = None,
    device: torch.device | None = None,
    cta_offset_s: float = 0.0,
) -> list[list[Forecast]]:
    """K prior samples per flight, decoded and rolled out: ``[sample][flight]``.

    The record contract stays single-trajectory — the top-1 goes through
    ``forecast_approaches`` as always; these are the additional modes a multimodal
    readout scores (minADE_K, miss rate, calibration). Each forecast carries its sample
    index and probability: the mixture weight of the component it was drawn from, or
    ``1/samples`` under a single Gaussian prior (which has no discrete weight to report).
    """
    if samples < 1:
        raise ValueError("samples must be positive")
    anchor, device, histories, dynamics = _latent_batch(
        model, series, config, normalizer, anchor, device, cta_offset_s
    )
    logits, mean, logvar, _top1 = _latent_prior_batch(model, histories, dynamics, device)
    generator = torch.Generator(device=device).manual_seed(seed)
    latents, components = model.sample_latents(logits, mean, logvar, samples, generator=generator)
    weights = torch.softmax(logits, dim=-1)                       # [B, K]
    if weights.shape[-1] == 1:
        probabilities = np.full((samples, len(series)), 1.0 / samples)
    else:
        probabilities = weights.gather(1, components.transpose(0, 1)).transpose(0, 1).cpu().numpy()
    return _latent_forecasts(
        model, series, config, normalizer, anchor, device,
        latents=latents, probabilities=probabilities,
        histories=histories, dynamics=dynamics, cta_offset_s=cta_offset_s,
    )


def random_latent_forecasts(
    model: nn.Module,
    series: Sequence[FlightSeries],
    config: TSConfig,
    normalizer: Normalizer,
    *,
    samples: int,
    seed: int,
    anchor: int | None = None,
    device: torch.device | None = None,
    cta_offset_s: float = 0.0,
) -> list[list[Forecast]]:
    """K latents drawn from N(0, I) — not the prior — decoded and rolled out: ``[sample][flight]``.

    The same-K CONTROL for minADE_K: K chances at a latent the prior did not choose. A
    prior whose modes only beat the top-1 because there are K of them will not beat this.
    Every forecast is stamped with its sample index and a probability of ``1/samples``.
    """
    if samples < 1:
        raise ValueError("samples must be positive")
    anchor, device, histories, dynamics = _latent_batch(
        model, series, config, normalizer, anchor, device, cta_offset_s
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(
        (samples, len(series), config.latent_dim), generator=generator, device=device
    )
    return _latent_forecasts(
        model, series, config, normalizer, anchor, device,
        latents=latents,
        probabilities=np.full((samples, len(series)), 1.0 / samples),
        histories=histories, dynamics=dynamics, cta_offset_s=cta_offset_s,
    )


def posterior_latent_forecasts(
    model: nn.Module,
    series: Sequence[FlightSeries],
    config: TSConfig,
    normalizer: Normalizer,
    *,
    anchor: int | None = None,
    device: torch.device | None = None,
    cta_offset_s: float = 0.0,
) -> list[Forecast]:
    """Every flight decoded from the MEAN of q(z | its own future): the z-oracle.

    The upper bound the latent can reach when the intent is known — L2's gate 3 reads it
    against the truth-intent arm. The future is the same target rows and true duration
    the training loop hands the posterior, built by the dataset itself (no restated
    target logic). It READS THE FUTURE: the records say ``zFromPosterior`` and this is
    never a prediction result.
    """
    anchor, device, histories, dynamics = _latent_batch(
        model, series, config, normalizer, anchor, device, cta_offset_s
    )
    windows = FixedAnchorTrajectoryWindows(list(series), config, normalizer)
    if len(windows) != len(series) or any(windows.index[i][1] != anchor for i in range(len(series))):
        raise RuntimeError("the fixed-anchor windows do not sit at the forecast anchor")
    batch = windows.batch(np.arange(len(series)))
    y, final_time_s = batch[1].to(device), batch[3].to(device)
    model.eval()
    with torch.no_grad():
        posterior_mean, _logvar = model.posterior(y, final_time_s)
    return _latent_forecasts(
        model, series, config, normalizer, anchor, device,
        latents=posterior_mean[None], probabilities=None, z_from_posterior=True,
        histories=histories, dynamics=dynamics, cta_offset_s=cta_offset_s,
    )[0]


def shuffled_latent_forecasts(
    model: nn.Module,
    series: Sequence[FlightSeries],
    config: TSConfig,
    normalizer: Normalizer,
    *,
    seed: int,
    anchor: int | None = None,
    device: torch.device | None = None,
    cta_offset_s: float = 0.0,
) -> list[Forecast]:
    """Every flight decoded from ANOTHER flight's top-1 latent (a seeded permutation).

    The posterior-collapse reading: if the trajectory error barely moves when each flight
    is handed someone else's intent, the decoder is not reading z, and every other number
    would still say "converged". A batch of one cannot be shuffled and raises.
    """
    if len(series) < 2:
        raise ValueError("shuffling latents needs at least two flights in the batch")
    anchor, device, histories, dynamics = _latent_batch(
        model, series, config, normalizer, anchor, device, cta_offset_s
    )
    _logits, _mean, _logvar, top1 = _latent_prior_batch(model, histories, dynamics, device)
    source = torch.from_numpy(latent_derangement(len(series), seed)).to(device)
    return _latent_forecasts(
        model, series, config, normalizer, anchor, device,
        latents=top1[source][None], probabilities=None, latent_shuffled=True,
        histories=histories, dynamics=dynamics, cta_offset_s=cta_offset_s,
    )[0]


def latent_derangement(count: int, seed: int) -> np.ndarray:
    """``source[i]`` = the flight whose latent flight ``i`` decodes from; never ``i`` itself.

    A seeded random cycle over the flights: each takes the next one's latent, so there is
    no fixed point by construction (a random permutation with fixed points rolled away is
    not one — the roll can create new ones).
    """
    if count < 2:
        raise ValueError("a derangement needs at least two flights")
    order = np.random.default_rng(seed).permutation(count)
    source = np.empty(count, dtype=np.int64)
    source[order] = np.roll(order, -1)
    return source


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


def _forecast_closure_batch(
    model: nn.Module,
    series: Sequence[FlightSeries],
    config: TSConfig,
    normalizer: Normalizer,
    anchor: int,
    device: torch.device,
) -> list[Forecast]:
    histories = _history_batch(series, config, normalizer, anchor)
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
    history = _history_at_anchor(series, config, normalizer, anchor)
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

    Every row the forecast itself places on the final under ``gate`` is bound
    (``on-final``: inside the membership cone and the predicted path aligned with the
    course; ``faf``: inside the coded FAF distance) — row by row, the same gate the
    corridor-bounded output layer applies softly, so the post-hoc projection is the hard
    counterpart of that layer.  (An earlier version bound only the suffix from which every
    later row was on the final; on arm A that moved 1.35 % of the rows, because one off-final
    row near the end cancelled the whole tail, and was not comparable to the layer.)
    Along-track distance is untouched, cross-track is clamped into ``±k·hw(d)``, height
    into the glidepath window, and the velocity channels are re-derived from the moved
    positions so the record stays one trajectory.  Nothing is learned here: it is what the
    constraint recovers after the fact, and the deployment fallback — it satisfies the rows
    and pays for it in kinks.
    """
    if forecast.prediction_output != PREDICTION_STATE:
        raise ValueError("project_onto_final applies to state forecasts only")
    if gate not in CORRIDOR_GATES:
        raise ValueError(f"unknown corridor gate {gate!r}; expected one of {CORRIDOR_GATES}")
    rows = final_approach_arrays(
        series, fix_distance_m=final_approach_fix_distance(series, gate=gate)
    )
    values = torch.as_tensor(forecast.values, dtype=torch.float64)[None]
    psi = torch.as_tensor(rows["runway_heading_rad"], dtype=torch.float64)[None]
    tan_gpa = torch.as_tensor(rows["glidepath_tan"], dtype=torch.float64)[None]
    d_faf = torch.as_tensor(rows["final_approach_fix_m"], dtype=torch.float64)[None]
    d, xt = runway_axes(values[..., IDX["e"]], values[..., IDX["n"]], psi)
    anchor = torch.as_tensor(series.values[forecast.anchor], dtype=torch.float64)
    step_e, step_n = position_direction(
        values[..., IDX["e"]], values[..., IDX["n"]],
        anchor[IDX["e"]][None], anchor[IDX["n"]][None],
    )
    cos_align = alignment_cosine(step_e, step_n, psi)
    on_final = membership(gate, d=d, xt=xt, cos_align=cos_align, d_faf=d_faf, hard=True)
    xt_bounded, u_bounded = bound_to_final(
        d=d, xt=xt, u=values[..., IDX["u"]], weight=on_final.to(torch.float64),
        tan_gpa=tan_gpa, hard=True,
    )
    e_bounded, n_bounded = chart_from_axes(d, xt_bounded, psi)
    projected = np.array(forecast.values, dtype=np.float64, copy=True)
    projected[:, IDX["e"]] = e_bounded[0].numpy()
    projected[:, IDX["n"]] = n_bounded[0].numpy()
    projected[:, IDX["u"]] = u_bounded[0].numpy()
    projected = states_with_derived_velocity(
        series.values[forecast.anchor], projected, forecast.sample_durations_s
    )
    return replace(forecast, values=projected, projected_onto_final=gate)


def _forecast_state(
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


def forecast_approaches(
    model: nn.Module,
    series: Sequence[FlightSeries],
    config: TSConfig,
    normalizer: Normalizer,
    *,
    anchor: int | None = None,
    device: torch.device | None = None,
    truncate: bool = True,
    project_final: str | None = None,
    cta_offset_s: float = 0.0,
) -> list[Forecast]:
    """Predict one inference batch through the same dense path used by fit evaluation.

    ``project_final`` names a corridor gate to clamp each state forecast into the
    final-approach corridor after truncation (``project_onto_final``); None = the
    model's own output.
    """
    if cta_offset_s and config.cta_conditioning != CTA_CONDITIONING_GIVEN:
        raise ValueError("a CTA offset applies to a checkpoint trained with cta_conditioning=given only")
    if not series:
        return []
    device = device or next(model.parameters()).device
    anchor = default_anchor(config) if anchor is None else anchor
    if config.prediction_output == PREDICTION_CONTROL:
        if project_final is not None:
            raise ValueError("the final-approach projection applies to state forecasts only")
        return _forecast_control_batch(
            model, series, config, normalizer, anchor, device, cta_offset_s=cta_offset_s
        )
    if config.prediction_output == PREDICTION_CLOSURE:
        if project_final is not None:
            raise ValueError("the final-approach projection applies to state forecasts only")
        return _forecast_closure_batch(model, series, config, normalizer, anchor, device)
    return [
        _forecast_state(
            model, item, config, normalizer, anchor, device, truncate, project_final
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
    project_final: str | None = None,
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
        project_final=project_final,
    )[0]


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
