"""The control path's inference: the schedule predicted per flight and rolled densely
through the twin under the command hook; the CTA and calibrated-interval plumbing; the
latent decodes (top-1, K prior samples, the posterior, the shuffled-z collapse diagnostic)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from ts_transformer.data.approach_difficulty import approach_difficulty
from ts_transformer.inference.calibration import conformal_intervals, interval_stratum
from ts_transformer.config import (
    CTA_CONDITIONING_OFF,
    PLAN_CONDITIONING_OFF,
    DURATION_HEADS_WITH_QUANTILES,
    TSConfig,
    default_anchor,
)
from ts_transformer.data.dataset import (
    FixedAnchorTrajectoryWindows,
    FlightSeries,
    Normalizer,
    truth_duration_s,
)
from ts_transformer.inference.forecast import Forecast, history_batch
from ts_transformer.outputs.constraints import build_command_hook
from ts_transformer.outputs.dynamics import rollout as control_rollout
from ts_transformer.outputs.dynamics.rollout import padded_dense_queries
from ts_transformer.outputs.dynamics.hooks import per_flight_hook_diagnostics
from ts_transformer.outputs.envelope import control_contract
from ts_transformer.outputs.control.heads import ControlPrediction
from ts_transformer.outputs.dynamics.context import dynamics_arrays
from ts_transformer.outputs.control.plan_token import PLAN_TOKEN_KEY, training_plan_token


def record_newton_controls(
    flown: torch.Tensor,
    segment_end_geodetic_states: torch.Tensor,
    dynamics: dict[str, torch.Tensor],
    config: TSConfig,
) -> torch.Tensor:
    """The schedule flown, in the record's newton contract, ``[B, N, 3]``: the contract's law
    resolving each segment's command at the state where the segment BEGINS
    (``_ControlLaw.geodetic_controls``, the same two resolvers the lag RHS calls at every
    stage, so the record can never clamp where the dynamics did not).

    Under thrust-fraction that is ``δ·T_max`` and the command's own bank and load — exact. Under
    the other contracts the RHS re-solves the thrust (and, under the path angle, the load) at
    every stage inside the hold, so one number per segment is a stand-in, chosen so the
    record's per-state controls stay a zero-order hold of this schedule as the contract says.
    **Under speed-command it is the hold's EXTREME, not its mean**: the loop's speed error
    decays across the whole segment (τ_V ≫ τ_T), so the thrust flown relaxes away from this
    start-of-segment demand (review N1: −0.047 T_max on average on a stepped descent). A
    readout of the record's thrust fraction reads it so; the specific force and the speed read
    off the STATES are exact.
    """
    initial = dynamics["initial_state"].to(flown)
    starts = torch.cat(
        (initial.unsqueeze(1), segment_end_geodetic_states.to(flown)[:, :-1]), dim=1
    )
    return control_contract(config.control_thrust_parameterization).law.geodetic_controls(
        starts, flown,
        max_thrust_n=dynamics["max_thrust_n"].to(flown),
        initial_geodetic_states=initial,
        aero_params=dynamics["aero_params"].to(flown),
    )


def _dynamics_batch(
    series: Sequence[FlightSeries], anchor: int, device: torch.device,
    config: TSConfig, cta_offset_s: float = 0.0, cta_s: np.ndarray | None = None,
) -> dict[str, torch.Tensor]:
    """The per-flight context, plus the CTA when the decoder takes one.

    Two sources, and which one it is decides whether the run READS THE FUTURE: under
    ``given`` the CTA is the truth duration + ``cta_offset_s`` (the counterfactual a
    scheduler asks for), and under B3 the caller supplies ``cta_s`` — the model's own
    duration quantile — which is the whole point of ``cta=self-q``.
    """
    rows = [
        dynamics_arrays(
            item, anchor, parameterization=config.control_thrust_parameterization,
            condition_features=config.control_condition_features,
        )
        for item in series
    ]
    if config.cta_conditioning != CTA_CONDITIONING_OFF:
        given = (
            np.asarray(cta_s, dtype=np.float64) if cta_s is not None
            else np.array(
                [truth_duration_s(item, anchor) + cta_offset_s for item in series],
                dtype=np.float64,
            )
        )
        for row, value in zip(rows, given, strict=True):
            row["cta_s"] = np.array(value, dtype=np.float64)
    if config.plan_conditioning != PLAN_CONDITIONING_OFF:
        # the truth's plan at the anchor, as training reads it (two-tier; reads the future).
        # A caller with another source (the lockstep's rolled asks) passes `dynamics` itself.
        for row, item in zip(rows, series, strict=True):
            row[PLAN_TOKEN_KEY] = training_plan_token(item, anchor, config)
    return {
        name: torch.from_numpy(np.stack([row[name] for row in rows])).to(device)
        for name in rows[0]
    }


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
        # The contract's own discriminator: a point head emits None for every row.
        duration_quantiles_s=(
            torch.cat([item.duration_quantiles_s for item in predictions], dim=0)
            if predictions[0].duration_quantiles_s is not None else None
        ),
    )


def forecast_control_batch(
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
    conformal: dict | None = None,
    cta_s: np.ndarray | None = None,
    cta_quantile: float | None = None,
    cta_interval: dict[str, object] | None = None,
) -> list[Forecast]:
    """Predict and densely roll a heterogeneous batch of bounded control schedules.

    ``latent`` decodes from given latents (``[B, Z]``); ``mode`` = (sample index, per-flight
    probability) stamps the forecasts as that prior sample; ``latent_shuffled`` stamps them
    as the collapse diagnostic. ``histories`` / ``dynamics`` accept the batch's inputs when
    the caller decodes the same flights several times.
    """
    if histories is None:
        histories = history_batch(series, config, normalizer, anchor)
    if dynamics is None:
        dynamics = _dynamics_batch(series, anchor, device, config, cta_offset_s, cta_s)
    prediction = _control_prediction_batch(model, histories, dynamics, device, latent=latent)
    cta = (
        dynamics["cta_s"].detach().cpu().numpy().astype(np.float64)
        if config.cta_conditioning != CTA_CONDITIONING_OFF else None
    )
    durations = prediction.segment_durations.detach().cpu().numpy().astype(np.float64)
    offsets, padded_offsets, query_valid = padded_dense_queries(
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
    # The head predicts in its contract; the exported record contract is newtons, and is
    # shared with the CasADi optimizer and the evaluation package. The schedule FLOWN (a
    # hook may have rewritten the network's commands) is exported at the head's own
    # precision (a hook's float64 rewrite is rounded to it), so bank and load read the same
    # bits under every law.
    flown = rollout.controls.to(prediction.controls.dtype)
    controls = record_newton_controls(
        flown.to(rollout.controls.dtype), rollout.segment_end_geodetic_states, dynamics, config,
    ).detach().cpu().numpy().astype(np.float64)
    commands = flown.detach().cpu().numpy().astype(np.float64)
    predicted_final_time = (
        prediction.final_time_s.detach().cpu().numpy().astype(np.float64)
    )
    duration_quantiles = (
        prediction.duration_quantiles_s.detach().cpu().numpy().astype(np.float64)
        if prediction.duration_quantiles_s is not None else None
    )
    hook_diagnostics = (
        None if command_hook is None else per_flight_hook_diagnostics(command_hook)
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
            commands=commands[row],
            control_parameterization=config.control_thrust_parameterization,
            sample_durations_s=np.diff(np.concatenate(([0.0], row_offsets))),
            segment_durations_s=durations[row],
            geodetic_values=query_geodetic[row, :count],
            prediction_output=config.prediction_output,
            command_hook=(
                None if command_hook is None
                else f"{config.control_command_hook}/{config.control_hook_saturation}"
            ),
            command_hook_diagnostics=(
                None if hook_diagnostics is None else hook_diagnostics[row]
            ),
            mode_index=None if mode is None else mode[0],
            mode_probability=None if mode is None else float(mode[1][row]),
            latent_shuffled=latent_shuffled,
            z_from_posterior=z_from_posterior,
            cta_s=None if cta is None else float(cta[row]),
            # There is no OFFSET under `self-q`: the CTA is not the truth plus anything, and
            # writing 0.0 there would read as "the truth, unshifted".
            cta_offset_s=None if cta is None or cta_s is not None else float(cta_offset_s),
            cta_from_quantiles=cta_s is not None,
            cta_quantile=cta_quantile,
            cta_interval=cta_interval,
            duration_quantiles_s=(
                None if duration_quantiles is None else duration_quantiles[row]
            ),
            **_calibrated_interval_fields(
                item, anchor, None if duration_quantiles is None else duration_quantiles[row],
                conformal,
            ),
        ))
    return forecasts


def _calibrated_interval_fields(
    series: FlightSeries, anchor: int, quantiles_s: np.ndarray | None, conformal: dict | None
) -> dict[str, object]:
    """The record's calibrated interval, or nothing at all (B2).

    The stratum is decided from the flight's OWN difficulty covariates at THIS anchor,
    through `calibration.interval_stratum` — the same `strata_masks` cut the table was
    fitted per, so a record and the delta it was widened by name one population.
    """
    if quantiles_s is None or conformal is None:
        return {}
    stratum = interval_stratum(conformal, approach_difficulty(series, anchor).to_dict())
    return {
        "duration_interval_s": conformal_intervals(quantiles_s, conformal, stratum),
        "duration_interval_stratum": stratum,
        "duration_interval_cohort": {
            "split": conformal["split"],
            "airports": list(conformal["airports"]),
            "smokeTest": bool(conformal["smoke_test"]),
        },
    }


def duration_quantile_predictions(
    model: nn.Module,
    series: Sequence[FlightSeries],
    config: TSConfig,
    normalizer: Normalizer,
    *,
    anchor: int | None = None,
    device: torch.device | None = None,
) -> np.ndarray:
    """The duration head's five quantiles per flight, ``[B, Q]`` seconds — nothing else.

    No rollout, no controls, and NO CTA: the quantile head reads the history alone
    (`control.heads.ControlFeatureModel.duration_quantiles`), which is what lets B2 calibrate
    and B3 decode a ``cta_conditioning=given`` checkpoint without first handing it an arrival
    time it would have had to read from the future. Under ``two-head`` (B1.b) it is the
    quantile head that answers here — never the point head that drove the rollout — so the
    interval B2 calibrates is the one the records publish. Batched per flight like every other
    forward here, so the arithmetic is the one `predict` runs.
    """
    if config.duration_head not in DURATION_HEADS_WITH_QUANTILES:
        raise ValueError(
            "duration quantiles need a checkpoint trained with duration_head in "
            f"{DURATION_HEADS_WITH_QUANTILES}; this one has {config.duration_head!r}"
        )
    device = device or next(model.parameters()).device
    anchor = default_anchor(config) if anchor is None else anchor
    histories = history_batch(series, config, normalizer, anchor)
    model.eval()
    with torch.no_grad():
        rows = [
            model.duration_quantiles(torch.from_numpy(history[None]).to(device))
            for history in histories
        ]
    return torch.cat(rows, dim=0).cpu().numpy().astype(np.float64)


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
    """The anchor, device and batch inputs every latent decode below starts from.

    The ONE ``latent_dim`` refusal lives here, not in the fan-out: two of the four entries
    read the prior (`_latent_prior_batch`) between this call and the fan-out, and on a
    non-latent model that dies on an ``AttributeError`` naming ``prior_logits`` instead of
    the contract. Every entry calls this first, so this is the earliest shared point.
    ``__main__`` refuses the flags earlier still, at the CLI boundary; this is the library's
    own contract, for the callers that are not the CLI.
    """
    if config.latent_dim < 1:
        raise ValueError(
            "latent forecasts need a latent control checkpoint (latent_dim > 0); this one "
            f"has latent_dim={config.latent_dim}"
        )
    device = device or next(model.parameters()).device
    anchor = default_anchor(config) if anchor is None else anchor
    return (
        anchor,
        device,
        history_batch(series, config, normalizer, anchor),
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

    ``latent_dim`` is refused once, in `_latent_batch`, which every entry calls before it
    reaches this point (see there for why it cannot be here).
    """
    if probabilities is not None and len(probabilities) != len(latents):
        raise ValueError(
            f"{len(probabilities)} probability rows for {len(latents)} latent samples"
        )
    return [
        forecast_control_batch(
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
