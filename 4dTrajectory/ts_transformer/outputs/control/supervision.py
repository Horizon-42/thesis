"""The control path's two supervision references — the inverse-dynamics imitation schedule
and the heading rate — and the batch-size probe's representative batch. The physical
context a rollout needs (`dynamics_arrays`, `anchor_controls`) is `outputs.dynamics.context`
since 2026-09-10: the plan path's guidance rolls the same dynamics."""

from __future__ import annotations


import numpy as np
import torch

from aircraft.aero_params import AeroParams
from ts_transformer.data.channels import states_from_channels
from ts_transformer.config import CTA_CONDITIONING_GIVEN, TSConfig
from ts_transformer.data.dataset import FlightSeries
from ts_transformer.geometry.final_approach_geometry import final_approach_arrays, probe_final_approach
from ts_transformer.outputs.conditioning import condition_vector
from ts_transformer.outputs.dynamics.inverse import MINIMUM_INVERSE_STATES, segment_controls
from ts_transformer.outputs.envelope import control_contract


def reference_control_supervision(
    series: FlightSeries,
    anchor: int,
    config: TSConfig,
    total_duration_s: float,
    last_measured_time_s: float,
) -> dict[str, np.ndarray]:
    """Return the control schedule the flown track implies, as a TRAINING TARGET.

    This is supervision, not dynamics: it reads the future, so it is deliberately built
    here and not in :func:`dynamics_arrays`, which forecast and predict also call and which
    must stay deployable from the lookback alone.

    The schedule comes from the SAME inverse registry the forward rollout dispatches
    through, keyed on ``config.control_dynamics_model``, so the lagged model is supervised
    on COMMANDS and the point-mass model on actual controls -- a target can never be the
    solution of equations the training rollout does not integrate.

    ``total_duration_s`` is the full supervised horizon (fitted tail included) because the
    model's segments span it, but the fitted tail has no measured velocity to differentiate.
    Segments whose midpoint falls past ``last_measured_time_s`` therefore get weight zero —
    the same cut the velocity term already makes, not the same numbers (this is a hard 0/1
    step, that one an interpolated per-channel weight).

    Those midpoints sit at the UNIFORM ``(k+0.5)·T/N``, which pairs index-for-index with the
    predicted schedule only through one chain: this target is built for the
    ``true-time-position`` objective, which ``TSConfig`` admits only with
    ``control_duration_parameterization="uniform"``. Under a non-uniform partition segment k
    of the two sides would cover different physical times.

    The weight vector can be ALL ZERO: an anchor at or after the last measured velocity
    with a fitted tail behind it supervises no segment. Such a flight contributes exactly
    zero to :func:`objective.control_imitation_mse` (its denominator is clamped at one) —
    no gradient, and a zero that DILUTES the reported per-flight mean, never a "perfect
    imitation" (review C-17). Under the default 60 s future floor no training anchor
    reaches it.
    """
    n_segments = int(config.n_segments)
    if series.n_samples - anchor < MINIMUM_INVERSE_STATES:
        # The all-zero case above, reached before the inversion can run: from an anchor on
        # the flight's last or second-last observed sample there is no measured velocity to
        # differentiate, so no segment is supervised. Never at L-1 on the stored cohorts;
        # two-tier T0(c)'s common anchor 119 put 38 KRDU train+val flights there and the
        # inversion's own refusal killed the campaign's first build (2026-09-16).
        return {
            "reference_controls": np.zeros((n_segments, 3), dtype=np.float64),
            "reference_control_weight": np.zeros(n_segments, dtype=np.float64),
        }
    mass_kg = float(series.scenario.initial.m)
    anchor_time = float(series.times[anchor])
    times = series.times[anchor:] - anchor_time
    samples = states_from_channels(
        times, series.values[anchor:], series.frame, mass_kg=mass_kg
    )
    states = np.asarray(
        [[s.latitude, s.longitude, s.altitude, s.V, s.psi, s.gamma, s.m]
         for _t, s in samples],
        dtype=np.float64,
    )
    aero = series.scenario.aero
    contract = control_contract(config.control_thrust_parameterization)
    inverted = segment_controls(
        states,
        times,
        config=config,
        aero_params=np.array(
            [aero.S, aero.Cl_max, aero.Cd0, aero.k, aero.stall_threshold, aero.k_stall],
            dtype=np.float64,
        ),
        max_thrust_n=float(series.scenario.aircraft.engine.max_thrust_total_n),
        control_lower=contract.lower_array,
        control_upper=contract.upper_array,
        n_segments=n_segments,
        total_duration_s=total_duration_s,
    )
    midpoints = (np.arange(n_segments, dtype=np.float64) + 0.5) * (
        total_duration_s / n_segments
    )
    return {
        "reference_controls": inverted.controls.astype(np.float64),
        "reference_control_weight": (
            midpoints <= last_measured_time_s
        ).astype(np.float64),
    }


# The observed heading is differentiated over a WINDOW, never between neighbouring
# samples: ADS-B reports a quantised track angle, this package's psi comes from a
# least-squares velocity fit of it, and a single dt_s step of heading difference is
# therefore dominated by that quantisation rather than by the turn. A central difference
# over this window IS the boxcar average of the one-step slopes inside it — long enough to
# bury the quantisation, short enough that a roll-in (~5 s) is not smeared across the
# straight legs either side. Stated as a constant because it is a smoothing choice the
# target depends on, not a free parameter.
#
# The constant is NOMINAL: the difference is taken between samples, so the half-width is
# `int(round(W / 2 / dt_s))` samples and the REALIZED span is `2 * half * dt_s`. At the
# package default dt_s = 2 s that is half = round(2.5) = 2 (Python rounds a .5 tie to even)
# and a span of 8 s, not 10. Quote the realized span, never the constant, when reading how
# much a target was smoothed. Rounding goes both ways: at a coarse dt_s the realized span
# can also EXCEED the constant (dt_s = 3 s -> half 2 -> 12 s; dt_s = 5.05 s -> half 1 ->
# 10.1 s). That widening is stated rather than guarded — only a dt_s too coarse for even
# one sample of half-width is refused below.
HEADING_RATE_SMOOTHING_WINDOW_S = 10.0


def _smoothed_heading_rate_dps(
    times_s: np.ndarray, heading_rad: np.ndarray, dt_s: float
) -> np.ndarray:
    """Central difference of an UNWRAPPED heading over the smoothing window, deg/s.

    A sample step too coarse for even one sample of half-width is REFUSED with its numbers
    rather than silently floored to one: a ``max(1, ...)`` would keep running with a window
    of ``2 * dt_s``, arbitrarily wider than the constant that names it, and nothing
    downstream would say so. The narrower rounding wobble either side of the nominal width
    is stated at the constant.
    """
    half = int(round(HEADING_RATE_SMOOTHING_WINDOW_S / (2.0 * dt_s)))
    if half < 1:
        raise ValueError(
            f"dt_s={dt_s:g}s cannot realize the {HEADING_RATE_SMOOTHING_WINDOW_S:g}s "
            f"heading-rate smoothing window: its half-width rounds to {half} samples "
            f"({HEADING_RATE_SMOOTHING_WINDOW_S / (2.0 * dt_s):.3g} before rounding), so "
            f"the term needs dt_s < {HEADING_RATE_SMOOTHING_WINDOW_S:g}s"
        )
    rows = np.arange(len(heading_rad))
    right = np.minimum(rows + half, len(heading_rad) - 1)
    left = np.maximum(rows - half, 0)
    return np.degrees(
        (heading_rad[right] - heading_rad[left]) / (times_s[right] - times_s[left])
    )


def reference_heading_rate_supervision(
    series: FlightSeries,
    anchor: int,
    config: TSConfig,
    total_duration_s: float,
    last_measured_time_s: float,
) -> dict[str, np.ndarray]:
    """The flown track's turn rate at the N segment ENDPOINTS, as a TRAINING TARGET.

    Supervision, not dynamics: it reads the future, so it is built here rather than in
    :func:`dynamics_arrays`, which predict also calls and which must stay deployable from
    the lookback alone. The same shape as :func:`reference_control_supervision`, with two
    deliberate differences — no inverse dynamics is solved (the observed track's own
    heading IS the target, and the rollout supplies the model side, so the two can never
    be solutions of different equations), and the mask is read at segment ENDPOINTS rather
    than midpoints, because a turn rate is a quantity at an instant while a
    piecewise-constant control is a quantity over a segment. That is the same cut the
    velocity term makes past the last measured velocity — not the same numbers: this is a
    hard 0/1 step, the velocity term's is an interpolated per-channel weight.

    The endpoints are placed at the UNIFORM ``(k+1)·T/N``, which is the rollout's own
    ``cumsum(segment_durations)`` only through one chain: this target is built for the
    ``true-time-position`` objective, which ``TSConfig`` admits only with
    ``control_duration_parameterization="uniform"``. Under a non-uniform partition row k of
    the two sides would be different physical instants.

    ``psi`` comes from :func:`channels.states_from_channels`, i.e. the modeling layer's
    math-ENU heading, unwrapped before differencing so the +/-pi branch cut cannot read as
    a turn. The sign convention is therefore the rollout's own: counter-clockwise is
    positive, which is what a positive bank produces under the shared RHS.
    """
    anchor_time = float(series.times[anchor])
    times = series.times[anchor:] - anchor_time
    if len(times) < 2:
        # The central difference below would be 0/0: a NaN target at weight one, not an
        # error (review B-4). Reachable under `random_train_anchor_min_future_s=0` when the
        # remainder past the anchor is the fitted tail alone; the imitation target's
        # inverse refuses the same anchor, so this term refuses it too.
        raise ValueError(
            f"flight {series.flight_id}: the heading-rate target needs at least two "
            f"observed samples from the anchor on, and anchor {anchor} leaves {len(times)} "
            "(the remainder is the fitted tail alone, which has no measured turn rate)"
        )
    samples = states_from_channels(
        times,
        series.values[anchor:],
        series.frame,
        mass_kg=float(series.scenario.initial.m),
    )
    heading_rad = np.unwrap(np.asarray([s.psi for _t, s in samples], dtype=np.float64))
    n_segments = int(config.n_segments)
    endpoints = (np.arange(n_segments, dtype=np.float64) + 1.0) * (
        total_duration_s / n_segments
    )
    return {
        "reference_heading_rate_dps": np.interp(
            endpoints,
            times,
            _smoothed_heading_rate_dps(times, heading_rad, config.dt_s),
        ),
        "reference_heading_rate_weight": (
            endpoints <= last_measured_time_s
        ).astype(np.float64),
    }


# The per-flight final-approach context (final_approach_geometry.FINAL_APPROACH_KEYS):
# what the corridor-bounded state output and the procedure penalty need to place a chart
# row on the runway's final. One row per flight, NEVER a model input (it rides in the
# batch's context slot beside the control dynamics).


def probe_dynamics(
    batch_size: int, device: torch.device, config: TSConfig
) -> dict[str, torch.Tensor]:
    """One representative dynamics batch for shape/throughput probes.

    Kept beside :func:`dynamics_arrays`, and its SUPERVISION keys are added under the same
    conditions ``TrajectoryWindows._dynamics_arrays`` adds them, so the batch-size probe
    and the gradient diagnostics cannot carry a stale copy of the contract —
    ``tests/test_supervision_terms.py`` pins the probe's key set equal to a real training
    batch's. Until 2026-09-09 the probe omitted the imitation and heading-rate targets
    (review B-1): ``--batch-size auto`` then died with a bare ``KeyError`` inside the
    objective, after the dataset build, on every custom arm that weighted either term.
    """
    contract = control_contract(config.control_thrust_parameterization)
    # A mid-size narrowbody. Its condition row is written by the feature set the head reads,
    # from the same airframe the aero row and the installed thrust below describe.
    aero = AeroParams(S=122.6, Cl_max=2.7, Cd0=0.02, k=0.04, stall_threshold=0.9, k_stall=0.1)
    mass_kg, max_thrust_n = 66_000.0, 240_000.0
    rows = {
        "condition": condition_vector(
            mass_kg, max_thrust_n, aero, features=config.control_condition_features
        ).tolist(),
        "initial_state": [35.9, -78.8, 1000.0, 80.0, 2.0, -0.05, mass_kg],
        "aero_params": [aero.S, aero.Cl_max, aero.Cd0, aero.k, aero.stall_threshold, aero.k_stall],
        "control_lower": list(contract.lower),
        "control_upper": list(contract.upper),
        "initial_controls": list(contract.neutral),
        "frame_params": [35.9, -78.8, 100.0, 0.0],
    }
    dynamics = {
        name: torch.tensor([value], dtype=torch.float32, device=device).expand(
            batch_size, -1
        )
        for name, value in rows.items()
    }
    dynamics["max_thrust_n"] = torch.full(
        (batch_size,), max_thrust_n, dtype=torch.float32, device=device
    )
    probe = probe_final_approach(batch_size, device)
    dynamics["runway_heading_rad"] = probe["runway_heading_rad"]
    dynamics["glidepath_tan"] = probe["glidepath_tan"]
    n_segments = int(config.n_segments)
    if config.cta_conditioning == CTA_CONDITIONING_GIVEN:
        # The probe's target duration is the scale, so the given CTA matches it and the
        # probe's final_time term stays zero, as in a real given run.
        dynamics["cta_s"] = torch.full(
            (batch_size,), config.final_time_scale_s, dtype=torch.float64, device=device
        )
    if config.control_imitation_loss_weight:
        dynamics["reference_controls"] = torch.tensor(
            [[list(contract.neutral)]], dtype=torch.float64, device=device
        ).expand(batch_size, n_segments, -1)
        dynamics["reference_control_weight"] = torch.ones(
            (batch_size, n_segments), dtype=torch.float64, device=device
        )
    if config.control_heading_rate_loss_weight:
        dynamics["reference_heading_rate_dps"] = torch.zeros(
            (batch_size, n_segments), dtype=torch.float64, device=device
        )
        dynamics["reference_heading_rate_weight"] = torch.ones(
            (batch_size, n_segments), dtype=torch.float64, device=device
        )
    return dynamics
