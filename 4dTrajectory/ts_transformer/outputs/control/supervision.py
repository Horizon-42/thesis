"""The per-flight physical tensors the control path reads: the dynamics context a model
and its rollout need (`dynamics_arrays`), the anchor's actual controls, and the two
supervision references — the inverse-dynamics imitation schedule and the heading rate."""

from __future__ import annotations


import numpy as np
import torch

from ts_transformer.data.channels import states_from_channels
from ts_transformer.config import CTA_CONDITIONING_GIVEN, TSConfig
from ts_transformer.data.dataset import FlightSeries
from ts_transformer.geometry.final_approach_geometry import final_approach_arrays, probe_final_approach
from ts_transformer.outputs.conditioning import condition_vector
from ts_transformer.outputs.dynamics.inverse import actual_controls, segment_controls
from ts_transformer.outputs.envelope import CONTROL_LOWER, CONTROL_UPPER


# How much observed lookback the anchor-state control inversion differentiates. It needs
# at least three samples for a second-order gradient; a few more absorb ADS-B jitter
# without reaching back into a different phase of flight (11 x 2 s = 20 s).
ANCHOR_CONTROL_SAMPLES = 11


def anchor_controls(series: FlightSeries, anchor: int, mass_kg: float) -> np.ndarray:
    """Return the controls the observed lookback implies are in effect at ``anchor``.

    This is the lagged model's actuator initial condition. It reads only samples at or
    before the anchor, so it is as deployable as the history window itself, and it is the
    ACTUAL control (never a command) for every flight model — the commands that produced
    it are a separate inversion in ``control_inverse_dynamics``.
    """
    start = max(0, anchor + 1 - ANCHOR_CONTROL_SAMPLES)
    window = slice(start, anchor + 1)
    times = series.times[window]
    samples = states_from_channels(
        times, series.values[window], series.frame, mass_kg=mass_kg
    )
    states = np.asarray(
        [
            [s.latitude, s.longitude, s.altitude, s.V, s.psi, s.gamma, s.m]
            for _t, s in samples
        ],
        dtype=np.float64,
    )
    aero = series.scenario.aero
    return actual_controls(
        states,
        times,
        aero_params=np.array(
            [aero.S, aero.Cl_max, aero.Cd0, aero.k, aero.stall_threshold, aero.k_stall],
            dtype=np.float64,
        ),
        max_thrust_n=float(series.scenario.aircraft.engine.max_thrust_total_n),
    )[-1]


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
    n_segments = int(config.n_segments)
    inverted = segment_controls(
        states,
        times,
        config=config,
        aero_params=np.array(
            [aero.S, aero.Cl_max, aero.Cd0, aero.k, aero.stall_threshold, aero.k_stall],
            dtype=np.float64,
        ),
        max_thrust_n=float(series.scenario.aircraft.engine.max_thrust_total_n),
        control_lower=CONTROL_LOWER,
        control_upper=CONTROL_UPPER,
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


def dynamics_arrays(series: FlightSeries, anchor: int) -> dict[str, np.ndarray]:
    """Physical per-flight tensors required by a control model and its rollout."""
    scenario = series.scenario
    mass_kg = float(scenario.initial.m)
    initial = states_from_channels(
        np.array([0.0], dtype=np.float64),
        series.values[anchor : anchor + 1],
        series.frame,
        mass_kg=mass_kg,
    )[0][1]
    aero = scenario.aero
    max_thrust = float(scenario.aircraft.engine.max_thrust_total_n)
    condition = condition_vector(mass_kg, max_thrust, aero)
    heading = float(getattr(series.frame, "heading_rad", 0.0))
    return {
        "condition": condition,
        "initial_state": np.array(
            [
                initial.latitude,
                initial.longitude,
                initial.altitude,
                initial.V,
                initial.psi,
                initial.gamma,
                initial.m,
            ],
            dtype=np.float64,
        ),
        "aero_params": np.array(
            [aero.S, aero.Cl_max, aero.Cd0, aero.k, aero.stall_threshold, aero.k_stall],
            dtype=np.float64,
        ),
        "max_thrust_n": np.array(max_thrust, dtype=np.float64),
        # The dimensionless envelope is the same box on every airframe (see
        # control_envelope); the flight's installed thrust enters through max_thrust_n.
        "control_lower": CONTROL_LOWER.astype(np.float32),
        "control_upper": CONTROL_UPPER.astype(np.float32),
        # The lagged model's actuator initial condition. Emitted unconditionally so the
        # batch contract does not depend on the flight model, and clipped to the same box
        # the head predicts in — an anchor whose implied thrust is outside the envelope is
        # a starting point the model could not have commanded.
        "initial_controls": np.clip(
            anchor_controls(series, anchor, mass_kg), CONTROL_LOWER, CONTROL_UPPER
        ).astype(np.float64),
        "frame_params": np.array(
            [series.frame.lat0, series.frame.lon0, series.frame.alt0, heading],
            dtype=np.float64,
        ),
        # The rollout frame rotation and the runway heading coincide only for the
        # runway-aligned coordinate frame.  Keep the terminal-loss reference separate
        # so ENU rollouts are decomposed along/across the actual runway, not east/north.
        # Runway course and glidepath: one definition with the state path's
        # final-approach context (the procedure penalty on the rollout reads both). The
        # FAF distance is not carried: no control recipe gates at the FAF.
        **{
            key: value
            for key, value in final_approach_arrays(series).items()
            if key in ("runway_heading_rad", "glidepath_tan")
        },
    }


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
    rows = {
        "condition": [0.66, 0.24, 0.2452, 0.9, 0.2, 0.4, 0.9, 0.5],
        "initial_state": [35.9, -78.8, 1000.0, 80.0, 2.0, -0.05, 66_000.0],
        "aero_params": [122.6, 2.7, 0.02, 0.04, 0.9, 0.1],
        "control_lower": CONTROL_LOWER.tolist(),
        "control_upper": CONTROL_UPPER.tolist(),
        "initial_controls": [0.2, 0.0, 1.0],
        "frame_params": [35.9, -78.8, 100.0, 0.0],
    }
    dynamics = {
        name: torch.tensor([value], dtype=torch.float32, device=device).expand(
            batch_size, -1
        )
        for name, value in rows.items()
    }
    dynamics["max_thrust_n"] = torch.full(
        (batch_size,), 240_000.0, dtype=torch.float32, device=device
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
            [[[0.2, 0.0, 1.0]]], dtype=torch.float64, device=device
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
