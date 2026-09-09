"""Inference strategies for normalized, full-horizon, and recursive-window prediction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn

from ts_transformer.approach_difficulty import approach_difficulty
from ts_transformer.batch_contract import model_forward
from ts_transformer.calibration import conformal_intervals, interval_stratum
from ts_transformer.channels import IDX, horizontal_distance_m
from ts_transformer.config import (
    CORRIDOR_GATES,
    CTA_CONDITIONING_GIVEN,
    CTA_CONDITIONING_OFF,
    DURATION_HEADS_WITH_QUANTILES,
    HORIZON_FULL,
    HORIZON_NORMALIZED,
    HORIZON_WINDOW,
    PREDICTION_CLOSURE,
    PREDICTION_CONTROL,
    PREDICTION_STATE,
    TSConfig,
    # `L-1`, defined in `config` because `dataset` reserves a share of its random draws for
    # it (A2b) and cannot import this module; used here, and re-exported by being imported.
    default_anchor,
)
from ts_transformer.closure_output import ClosureLabels, check_airport, decision_from_label, reconstruct
from ts_transformer.control.constraints import build_command_hook
from ts_transformer.control.dynamics import rollout as control_rollout
from ts_transformer.control.dynamics.hooks import (
    HOOK_DIAGNOSTIC_PREFIX, HOOK_STEPS_KEY, CommandHook,
)
from ts_transformer.control.envelope import physical_controls
from ts_transformer.dataset import (
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
from ts_transformer.final_approach_geometry import (
    alignment_cosine,
    bound_to_final,
    chart_from_axes,
    membership,
    position_direction,
    runway_axes,
    threshold_crossing_index,
)
from ts_transformer.metrics import states_with_derived_velocity
from ts_transformer.prediction_outputs import ControlPrediction
from ts_transformer.target_conditioning import conditioned_history
from ts_transformer.time_grids import output_time_grid


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
    # ...and THIS flight's own hook counts: `steps`, then every other count as a share of
    # it (`_per_flight_hook_diagnostics`). None when no hook ran. A batch share written
    # onto every record would say the same thing about a flight the hook never touched and
    # one it rewrote at every step, so the record carries the per-flight number. Strings
    # appear where a module ran a NAMED variant of itself (`CommandHook.diagnostic_labels`);
    # those are the same for every row and are never divided by `steps`.
    command_hook_diagnostics: dict[str, float | str] | None = None
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
    # B1 / B1.b, quantile-bearing duration heads only: the five `DURATION_QUANTILES` in
    # seconds, in level order. `predicted_final_time_s` above is still the ONE duration this
    # trajectory was rolled over — under `quantile` the MEDIAN of these, under `two-head`
    # the POINT head's (a different number), under any CTA arm the CTA. These are the
    # interval, and §六 6 — quantiles of the DURATION, never of the trajectory.
    duration_quantiles_s: np.ndarray | None = None
    # B2: the CALIBRATED interval per alpha, `[{"alpha", "lo", "hi"}]`, and the stratum whose
    # conformal delta widened it. None = this checkpoint has no calibration table, and the
    # record then says `calibrated: false` rather than passing raw quantiles off as an
    # interval (design §六 4).
    duration_interval_s: list[dict[str, float]] | None = None
    duration_interval_stratum: str | None = None
    # WHICH calibration produced that interval: the table's own split, airports and smoke
    # flag. A delta fitted at one airport and applied at another is a different calibration,
    # and a --limit table deployed deliberately has to stay visible in the record.
    duration_interval_cohort: dict[str, object] | None = None
    # B3: this trajectory was decoded at the model's OWN duration quantile, not at a CTA read
    # from the future. `cta_quantile` is the level (None at a calibrated interval endpoint,
    # which `cta_interval` then names). §六 5 — `cta=self-q` and `cta=given` are two arms.
    cta_from_quantiles: bool = False
    cta_quantile: float | None = None
    cta_interval: dict[str, object] | None = None

    @property
    def n_steps(self) -> int:
        return len(self.times)


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
    config: TSConfig, cta_offset_s: float = 0.0, cta_s: np.ndarray | None = None,
) -> dict[str, torch.Tensor]:
    """The per-flight context, plus the CTA when the decoder takes one.

    Two sources, and which one it is decides whether the run READS THE FUTURE: under
    ``given`` the CTA is the truth duration + ``cta_offset_s`` (the counterfactual a
    scheduler asks for), and under B3 the caller supplies ``cta_s`` — the model's own
    duration quantile — which is the whole point of ``cta=self-q``.
    """
    rows = [dynamics_arrays(item, anchor) for item in series]
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
        # The contract's own discriminator: a point head emits None for every row.
        duration_quantiles_s=(
            torch.cat([item.duration_quantiles_s for item in predictions], dim=0)
            if predictions[0].duration_quantiles_s is not None else None
        ),
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
        histories = _history_batch(series, config, normalizer, anchor)
    if dynamics is None:
        dynamics = _dynamics_batch(series, anchor, device, config, cta_offset_s, cta_s)
    prediction = _control_prediction_batch(model, histories, dynamics, device, latent=latent)
    cta = (
        dynamics["cta_s"].detach().cpu().numpy().astype(np.float64)
        if config.cta_conditioning != CTA_CONDITIONING_OFF else None
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
    duration_quantiles = (
        prediction.duration_quantiles_s.detach().cpu().numpy().astype(np.float64)
        if prediction.duration_quantiles_s is not None else None
    )
    hook_diagnostics = (
        None if command_hook is None else _per_flight_hook_diagnostics(command_hook)
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


def _per_flight_hook_diagnostics(command_hook: CommandHook) -> list[dict[str, float | str]]:
    """Each flight's own hook counts: ``steps``, then every other count as a share of it.

    The same normalisation ``train.fit_model`` writes into an epoch record
    (``value / hook_steps``), one level down — an epoch reports the batch, a prediction
    record reports the flight. Keys drop the ``hook_`` prefix and arrive camelCased like
    every other ``source`` field. A module's LABELS (which named variant of itself it ran)
    join the same bag under the same naming, undivided: they are strings, and a share of a
    name means nothing.
    """
    counts = {
        name: value.tolist()
        for name, value in command_hook.per_flight_diagnostics().items()
    }
    labels = {
        _camel_case(name.removeprefix(HOOK_DIAGNOSTIC_PREFIX)): value
        for name, value in command_hook.diagnostic_labels().items()
    }
    steps = counts.pop(HOOK_STEPS_KEY)
    return [
        {
            "steps": row_steps,
            **{
                _camel_case(name.removeprefix(HOOK_DIAGNOSTIC_PREFIX)): value[row] / row_steps
                for name, value in counts.items()
            },
            **labels,
        }
        for row, row_steps in enumerate(steps)
    ]


def _camel_case(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(word.title() for word in rest)


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
    histories = _history_batch(series, config, normalizer, anchor)
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

    ``latent_dim`` is refused once, in `_latent_batch`, which every entry calls before it
    reaches this point (see there for why it cannot be here).
    """
    if probabilities is not None and len(probabilities) != len(latents):
        raise ValueError(
            f"{len(probabilities)} probability rows for {len(latents)} latent samples"
        )
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

    The corridor geometry (`final_approach_geometry`) is written about the THRESHOLD as the
    origin, so the positions are translated by ``series.target_chart`` on the way in and
    back on the way out — the same translation `cut_at_threshold_crossing` makes. Under
    ``enu`` / ``runway-aligned`` the target IS the origin and this is the identity; under
    ``airport-enu`` it is 1.3–1.9 km, and until 2026-09-09 the projection clamped about the
    airport reference point instead (review C-11).
    """
    if forecast.prediction_output != PREDICTION_STATE:
        raise ValueError("project_onto_final applies to state forecasts only")
    if gate not in CORRIDOR_GATES:
        raise ValueError(f"unknown corridor gate {gate!r}; expected one of {CORRIDOR_GATES}")
    rows = final_approach_arrays(
        series, fix_distance_m=final_approach_fix_distance(series, gate=gate)
    )
    target = np.asarray(series.target_chart, dtype=np.float64)
    values = torch.as_tensor(forecast.values, dtype=torch.float64)[None]
    e = values[..., IDX["e"]] - float(target[0])
    n = values[..., IDX["n"]] - float(target[1])
    u = values[..., IDX["u"]] - float(target[2])
    psi = torch.as_tensor(rows["runway_heading_rad"], dtype=torch.float64)[None]
    tan_gpa = torch.as_tensor(rows["glidepath_tan"], dtype=torch.float64)[None]
    d_faf = torch.as_tensor(rows["final_approach_fix_m"], dtype=torch.float64)[None]
    d, xt = runway_axes(e, n, psi)
    anchor = torch.as_tensor(series.values[forecast.anchor], dtype=torch.float64)
    step_e, step_n = position_direction(
        e, n,
        (anchor[IDX["e"]] - float(target[0]))[None],
        (anchor[IDX["n"]] - float(target[1]))[None],
    )
    cos_align = alignment_cosine(step_e, step_n, psi)
    on_final = membership(gate, d=d, xt=xt, cos_align=cos_align, d_faf=d_faf, hard=True)
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
    conformal: dict | None = None,
    cta_s: np.ndarray | None = None,
    cta_quantile: float | None = None,
    cta_interval: dict[str, object] | None = None,
    # `--truncate-at-threshold`. The module-level `truncate_at_threshold` (the fixed-time
    # postprocessor's closest-approach rule) is deliberately NOT called from this function,
    # so the shadow costs nothing; `cut_at_threshold_crossing` is the rule this flag runs.
    truncate_at_threshold: bool = False,
) -> list[Forecast]:
    """Predict one inference batch through the same dense path used by fit evaluation.

    ``project_final`` names a corridor gate to clamp each state forecast into the
    final-approach corridor after truncation (``project_onto_final``); None = the
    model's own output. ``truncate_at_threshold`` additionally cuts every forecast where it
    first crosses the threshold ON THE FINAL (:func:`cut_at_threshold_crossing`) — for any
    output kind, the control rollout included, which the fixed-time rule never reaches.
    """
    if cta_offset_s and config.cta_conditioning != CTA_CONDITIONING_GIVEN:
        raise ValueError("a CTA offset applies to a checkpoint trained with cta_conditioning=given only")
    if cta_s is not None and cta_offset_s:
        raise ValueError(
            "a self-quantile CTA and a counterfactual offset are two different arms: the "
            "first reads the model's own duration, the second the truth's"
        )
    if not series:
        return []
    device = device or next(model.parameters()).device
    anchor = default_anchor(config) if anchor is None else anchor
    if config.prediction_output == PREDICTION_CONTROL:
        if project_final is not None:
            raise ValueError("the final-approach projection applies to state forecasts only")
        forecasts = _forecast_control_batch(
            model, series, config, normalizer, anchor, device,
            cta_offset_s=cta_offset_s, conformal=conformal, cta_s=cta_s,
            cta_quantile=cta_quantile, cta_interval=cta_interval,
        )
    elif config.prediction_output == PREDICTION_CLOSURE:
        if project_final is not None:
            raise ValueError("the final-approach projection applies to state forecasts only")
        forecasts = _forecast_closure_batch(model, series, config, normalizer, anchor, device)
    else:
        forecasts = [
            _forecast_state(
                model, item, config, normalizer, anchor, device, truncate, project_final
            )
            for item in series
        ]
    if not truncate_at_threshold:
        return forecasts
    return [
        cut_at_threshold_crossing(forecast, flight)
        for forecast, flight in zip(forecasts, series, strict=True)
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


def cut_at_threshold_crossing(forecast: Forecast, series: FlightSeries) -> Forecast:
    """Cut a forecast where it FIRST crosses the landing threshold on the final.

    L3.d's geometry readout is why this exists. With the speed floor holding the commanded
    speed up, a rollout that is asked to arrive late reaches the threshold EARLY, flies on
    and turns: endpoint ``|xt|`` p95 of 43-63 km, pooled ADE 840 -> 3443 m at offset 0. All
    of that is post-landing flying, and every flyability, corridor and CTA readout taken
    over the whole record is reading it. Cut there and the same arms are read on the
    approach proper: at +60 s, fully-flyable 1.35 % -> 48.9 % on the identical rollout.

    WHERE it crosses is ``final_approach_geometry.threshold_crossing_index``, shared with the
    trombone's reference path: the first row at or past the threshold plane (``d <= 0`` in
    runway axes) that is also ON the final there — inside the ``on-final`` gate's membership
    cone and aligned with the course. The plane alone is not the crossing: a vectored flight
    passes ``d = 0`` on its DOWNWIND, several kilometres abeam, and this rule used to cut it
    there (2026-09-09: 96.5 % of the vectored cuts more than 1 km from the threshold, median
    ``|xt|`` 8.7 km at a median 1.75 km above it, against a straight-in ``|xt|`` p95 of 50 m).

    The cut point is then the closest horizontal approach to the target within that first
    crossing run — "first" so a rollout that wanders off and later passes near the threshold
    again is cut on its real arrival, and "closest approach" so a laterally displaced
    crossing is cut where it was nearest rather than a step early. A forecast that never
    crosses on the final never landed: it is returned WHOLE and still says
    ``truncatedAtThreshold: false`` — cutting it somewhere would invent an arrival, and the
    flag is what a reader checks. A forecast that crosses on its LAST row is also returned
    whole, but with the flag TRUE: nothing needed cutting and the record does end at the
    crossing, so the flag reads "this record ends at the threshold" and "whole" is not
    evidence of "never got there".

    ``final_time_s`` moves to the cut, which is the point: an early arrival stops being
    invisible in the CTA readout and becomes the ``final_time_error_s`` it always was. The
    two clocks stay aligned (``export`` requires it) — the control SEGMENTS are cut to the
    one containing the new end and its duration shortened to land exactly on it, so the
    schedule still ends where the states do and stays 1:1 ZOH-aligned with them. Both clocks
    are rebuilt by ``np.cumsum`` of the durations, which is what ``export`` does too, so the
    two agree by construction rather than by luck — ``cumsum(diff(x)) == x`` is NOT an
    identity in general, and it is the shared reconstruction, not the arithmetic, that makes
    the ends meet.

    On a fixed-time STATE forecast the postprocessor's own closest-approach truncation
    (:func:`truncate_at_threshold`) has already run and also sets ``truncated_at_threshold``.
    That rule has no plane in it at all — it is the closest approach anywhere — so on a
    vectored flight it stamps the flag on a record that ends kilometres abeam. Where this one
    runs, IT owns the flag: the flag then answers "does this record end at the threshold
    crossing", and a record with no crossing is returned whole with the flag CLEARED. Whether
    the fixed-time cut happened stays recoverable — ``horizon_capped`` marks the records it
    never reached — but only on a full/window STATE record, which is the only kind that rule
    runs on; everywhere else both fields are false and the run's output kind is what tells
    "control record left whole" from "state record cut at a closest approach that was not a
    crossing".
    """
    # The runway axes, from the one definition of them, about the target as the origin.
    psi = torch.tensor([float(series.scenario.target.psi)], dtype=torch.float64)
    e = torch.from_numpy(
        np.ascontiguousarray(forecast.values[:, IDX["e"]] - series.target_chart[0])
    )[None]
    n = torch.from_numpy(
        np.ascontiguousarray(forecast.values[:, IDX["n"]] - series.target_chart[1])
    )[None]
    d, xt = runway_axes(e, n, psi)
    # The path's own direction at every row, from the POSITIONS and with the anchor standing
    # in before the first — the gate's rule and `project_onto_final`'s call, because the
    # velocity channels are free outputs a state model could steer the gate with.
    anchor = torch.as_tensor(series.values[forecast.anchor], dtype=torch.float64)
    step_e, step_n = position_direction(
        e, n,
        (anchor[IDX["e"]] - float(series.target_chart[0]))[None],
        (anchor[IDX["n"]] - float(series.target_chart[1]))[None],
    )
    first_row, run_end, crossed = threshold_crossing_index(
        d, xt, alignment_cosine(step_e, step_n, psi)
    )
    if not bool(crossed[0]):
        # It never got onto the final. On a fixed-time STATE forecast the postprocessor's
        # own closest-approach truncation has already run and set the flag; clearing it here
        # is the point of the flag — under this switch it answers "does this record end at
        # the threshold crossing", and a record that ends at a closest approach eight
        # kilometres abeam does not. Nothing is lost: `horizon_capped` marks the fixed-time
        # records that rule never reached, so whether it cut this one is still recoverable.
        return replace(forecast, truncated_at_threshold=False)
    first, end = int(first_row[0]), int(run_end[0])
    # The horizontal distance to the target, from the one definition of THAT.
    distance = horizontal_distance_m(forecast.values, series.target_chart)
    cut = first + int(np.argmin(distance[first:end]))
    if cut == len(distance) - 1:
        # It reached the threshold on its last row: there is nothing to cut, but the record
        # DOES end at the crossing and must not read as one that never got there.
        return replace(forecast, truncated_at_threshold=True)
    sample_durations_s = forecast.sample_durations_s[: cut + 1]
    # The clock `export` reconstructs, so the two agree to the bit rather than to a sum.
    offsets = np.cumsum(sample_durations_s)
    final_time_s = float(offsets[-1])
    if forecast.controls is None:
        # Every non-control forecast carries one clock: segments ARE samples.
        segment_durations_s = forecast.segment_durations_s[: cut + 1]
        controls = None
    else:
        boundaries = np.cumsum(forecast.segment_durations_s)
        # `side="left"` gives the first boundary at or after the new end, so the shortened
        # duration below is strictly positive. `cut < n-1` puts `final_time_s` strictly under
        # the last boundary, so the search cannot run off the end.
        last = int(np.searchsorted(boundaries, final_time_s, side="left"))
        segment_durations_s = forecast.segment_durations_s[: last + 1].copy()
        segment_durations_s[-1] = final_time_s - (0.0 if last == 0 else boundaries[last - 1])
        controls = forecast.controls[: last + 1]
    return replace(
        forecast,
        times=forecast.times[: cut + 1],
        values=forecast.values[: cut + 1],
        normalized_progress=offsets / final_time_s,
        final_time_s=final_time_s,
        truncated_at_threshold=True,
        sample_durations_s=sample_durations_s,
        segment_durations_s=segment_durations_s,
        controls=controls,
        geodetic_values=(
            None if forecast.geodetic_values is None else forecast.geodetic_values[: cut + 1]
        ),
    )
