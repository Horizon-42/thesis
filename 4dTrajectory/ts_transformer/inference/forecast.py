"""The forecast contract and the batch entry point every predict-side caller uses.

`Forecast` is what a strategy returns per flight; `forecast_approaches` hands a batch to
the configured output strategy (`outputs/<path>/forecast.py` holds each path's own
inference) and applies the output-agnostic threshold cut on top.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from ts_transformer.data.channels import IDX, horizontal_distance_m
from ts_transformer.config import (
    CTA_CONDITIONING_GIVEN,
    PREDICTION_STATE,
    TSConfig,
    default_anchor,
)
from ts_transformer.data.dataset import FlightSeries, Normalizer, series_conditioning
from ts_transformer.geometry.final_approach_geometry import (
    alignment_cosine,
    position_direction,
    runway_axes,
    threshold_crossing_index,
)
from ts_transformer.outputs import ForecastOptions, strategy
from ts_transformer.outputs.envelope import control_contract
from ts_transformer.data.target_conditioning import conditioned_history


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
    # The schedule FLOWN in its contract's own units, ``[N, 3]`` 1:1 with ``controls`` (whose
    # newton columns are the contract law's resolution of it at each segment's START state),
    # and the ``control_thrust_parameterization`` naming the columns. Present exactly when
    # ``controls`` is; a record carries the contract's ``record_command_columns`` of it, so
    # the command the head emitted is in the record wherever the newton column is not it.
    commands: np.ndarray | None = None
    control_parameterization: str | None = None
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

    def __post_init__(self) -> None:
        if not (self.controls is None) == (self.commands is None) == (self.control_parameterization is None):
            raise ValueError(
                "a forecast carries controls, their commands and the contract naming them "
                "together or not at all"
            )
        if self.controls is not None and self.commands.shape != self.controls.shape:
            raise ValueError(
                f"commands {self.commands.shape} and controls {self.controls.shape} must align "
                "segment for segment; a cut or a join that shortens one shortens both"
            )

    @property
    def n_steps(self) -> int:
        return len(self.times)


def history_at_anchor(
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


def history_batch(
    series: Sequence[FlightSeries],
    config: TSConfig,
    normalizer: Normalizer,
    anchor: int,
) -> np.ndarray:
    return np.stack([
        history_at_anchor(item, config, normalizer, anchor) for item in series
    ]).astype(np.float32, copy=False)


def forecast_approaches(
    model: nn.Module,
    series: Sequence[FlightSeries],
    config: TSConfig,
    normalizer: Normalizer,
    *,
    anchor: int | None = None,
    device: torch.device | None = None,
    options: ForecastOptions = ForecastOptions(),
) -> list[Forecast]:
    """Predict one inference batch through the same dense path used by fit evaluation.

    ``options`` is what the caller asks beyond the checkpoint's contract
    (`outputs.ForecastOptions`); the output strategy refuses the ones that do not apply to
    its path. ``truncate_at_threshold`` additionally cuts every forecast where it first
    crosses the threshold ON THE FINAL (:func:`cut_at_threshold_crossing`) — for any output
    kind, the control rollout included, which the fixed-time rule never reaches.
    """
    if options.cta_offset_s and config.cta_conditioning != CTA_CONDITIONING_GIVEN:
        raise ValueError("a CTA offset applies to a checkpoint trained with cta_conditioning=given only")
    if options.cta_s is not None and options.cta_offset_s:
        raise ValueError(
            "a self-quantile CTA and a counterfactual offset are two different arms: the "
            "first reads the model's own duration, the second the truth's"
        )
    if not series:
        return []
    device = device or next(model.parameters()).device
    anchor = default_anchor(config) if anchor is None else anchor
    forecasts = strategy(config).forecast(
        model, series, config, normalizer, anchor, device, options
    )
    if not options.truncate_at_threshold:
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
    options: ForecastOptions = ForecastOptions(),
) -> Forecast:
    """Predict from one observed anchor using the configured output strategy."""
    return forecast_approaches(
        model, [series], config, normalizer, anchor=anchor, device=device, options=options,
    )[0]


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
        controls = commands = None
    else:
        boundaries = np.cumsum(forecast.segment_durations_s)
        # `side="left"` gives the first boundary at or after the new end, so the shortened
        # duration below is strictly positive. `cut < n-1` puts `final_time_s` strictly under
        # the last boundary, so the search cannot run off the end.
        last = int(np.searchsorted(boundaries, final_time_s, side="left"))
        segment_durations_s = forecast.segment_durations_s[: last + 1].copy()
        segment_durations_s[-1] = final_time_s - (0.0 if last == 0 else boundaries[last - 1])
        controls = forecast.controls[: last + 1]
        commands = forecast.commands[: last + 1]
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
        commands=commands,
        geodetic_values=(
            None if forecast.geodetic_values is None else forecast.geodetic_values[: cut + 1]
        ),
    )


def cut_rows(forecast: Forecast, count: int) -> Forecast:
    """The forecast's first ``count`` rows, its control segments cut to the one containing
    the new end and that one shortened to land on it — :func:`cut_at_threshold_crossing`'s
    clock rule, without its crossing rule (and like it, a forecast with no controls carries
    one clock: its segments ARE its samples). A hooked forecast's `steps` becomes the
    segments kept; its shares stay the rollout's (the cut steps' own counts are not
    recoverable from a per-flight share). Moved here from `outputs/plan/forecast.py`
    (2026-09-16): the control path's chained re-ask (`experiments/chain_sensitivity.py`)
    cuts hook-free forecasts too."""
    if count >= len(forecast.times):
        return forecast
    sample_durations_s = forecast.sample_durations_s[:count]
    offsets = np.cumsum(sample_durations_s)
    final_time_s = float(offsets[-1])
    if forecast.controls is None:
        segment_durations_s, controls, commands, last = (
            forecast.segment_durations_s[:count], None, None, count - 1
        )
    else:
        boundaries = np.cumsum(forecast.segment_durations_s)
        last = int(np.searchsorted(boundaries, final_time_s, side="left"))
        segment_durations_s = forecast.segment_durations_s[: last + 1].copy()
        segment_durations_s[-1] = final_time_s - (0.0 if last == 0 else boundaries[last - 1])
        controls = forecast.controls[: last + 1]
        commands = forecast.commands[: last + 1]
    diagnostics = (
        None if forecast.command_hook_diagnostics is None
        else {**forecast.command_hook_diagnostics, "steps": float(last + 1)}
    )
    return replace(
        forecast, times=forecast.times[:count], values=forecast.values[:count],
        normalized_progress=offsets / final_time_s, final_time_s=final_time_s,
        sample_durations_s=sample_durations_s, segment_durations_s=segment_durations_s,
        controls=controls, commands=commands,
        geodetic_values=None if forecast.geodetic_values is None else forecast.geodetic_values[:count],
        command_hook_diagnostics=diagnostics,
    )


def concatenate(legs: Sequence[Forecast], anchor: int, predicted_final_time_s: float) -> Forecast:
    """The legs as one forecast: the rows run on (a leg's rows start one query step after
    its anchor, so nothing repeats), ``passes`` counts the legs; a hooked leg set's counts
    are summed over the legs' steps (`rolledLegs` says how many legs), a hook-free one
    carries none, and a set that mixes the two is refused, as is one that mixes contracts."""
    first = legs[0]
    if len({leg.command_hook_diagnostics is None for leg in legs}) > 1:
        raise ValueError("cannot concatenate hooked and hook-free legs into one forecast")
    if len({leg.control_parameterization for leg in legs}) > 1:
        raise ValueError("cannot concatenate legs flown under different control contracts")
    if len(legs) > 1 and control_contract(first.control_parameterization).relative_to_anchor_speed:
        # each leg's command is relative to ITS OWN ask's anchor airspeed; joined, the record's
        # column would mix references and say nothing about which
        raise ValueError(
            f"cannot concatenate {first.control_parameterization!r} legs: their commands are "
            "relative to each leg's own anchor airspeed"
        )
    times = np.concatenate([leg.times for leg in legs])
    values = np.concatenate([leg.values for leg in legs])
    geodetic = np.concatenate([leg.geodetic_values for leg in legs])
    samples = np.concatenate([leg.sample_durations_s for leg in legs])
    segments = np.concatenate([leg.segment_durations_s for leg in legs])
    controls = np.concatenate([leg.controls for leg in legs])
    commands = np.concatenate([leg.commands for leg in legs])
    final_time_s = float(np.sum(samples))
    diagnostics: dict[str, float | str] | None = None
    if first.command_hook_diagnostics is not None:
        steps = [float(leg.command_hook_diagnostics["steps"]) for leg in legs]
        diagnostics = {"steps": float(sum(steps))}
        for key in first.command_hook_diagnostics:
            if key == "steps":
                continue
            values_by_leg = [leg.command_hook_diagnostics[key] for leg in legs]
            if all(isinstance(v, (int, float)) for v in values_by_leg):
                diagnostics[key] = float(np.average(values_by_leg, weights=steps))
            else:
                diagnostics[key] = values_by_leg[0]
        diagnostics["rolledLegs"] = float(len(legs))
    return replace(
        first, times=times, values=values, geodetic_values=geodetic, sample_durations_s=samples,
        segment_durations_s=segments, controls=controls, commands=commands, normalized_progress=np.cumsum(samples) / max(final_time_s, 1e-9),
        final_time_s=final_time_s, predicted_final_time_s=float(predicted_final_time_s), anchor=int(anchor),
        passes=len(legs), command_hook_diagnostics=diagnostics,
    )
