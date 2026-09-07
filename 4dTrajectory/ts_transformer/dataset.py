"""Observed arrivals -> uniform histories -> configured trajectory-time targets.

Pipeline per flight::

    flight dict  ({id, runway, waypoints: [[t, lon, lat, alt], ...]})
      -> flight_scenarios.build_scenario(..., target_from_threshold=True)   # target, aircraft, mass
      -> flight_scenarios.state_samples_from_track(...)                     # V/psi/gamma per sample
      -> channels.channels_from_states(...)                                 # ENU metres, threshold origin
      -> channels.resample_uniform(...)                                     # regular dt grid
      -> measured FlightSeries + position-only fitted-tail supervision

Every one of those steps is an existing, tested seam except the last two. That is on
purpose: the reference records the predictions get judged against are built by the same
functions, so a divergence here would read as model error rather than as a bug.

Two questions this module deliberately does NOT answer: which arrival rosters a run was
trained on (`data_provenance.py`) and which split a flight belongs to (`splits.py`). Both
are pure functions over identities and digests, and keeping them out here is what lets
`evaluation_protocol` compare two fingerprints without importing torch.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from aircraft.identity import get_default_identity_resolver
from aircraft.query_aircraft_parameters import openap_support_kind

# flight_key is the identity ``id_runway_icao24_landingTime`` — single-sourced in
# flight_scenarios.identity because the optimizer batch derives its record filenames from
# the SAME function; the split key here and both writers' filename stems cannot drift.
from final_approach import bracket_fraction
from flight_scenarios import (
    FittedApproach,
    FlightScenario,
    build_scenario,
    fit_flight_final_approach,
    flight_key,
    state_samples_from_track,
)
from flight_scenarios.datum import flight_to_msl
from trajectory_data_process.harvest.arrivals import load_arrival_flights

from channels import (
    CHANNELS,
    POSITION_IDX,
    VELOCITY_IDX,
    channels_from_states,
    resample_uniform,
    states_from_channels,
    target_chart_position,
)
from anchor_eligibility import (
    eligible_random_train_anchors,
    random_train_anchor_eligibility_policy,
)
from config import (
    CTA_CONDITIONING_GIVEN,
    AIRCRAFT_FILTER_OPENAP_DIRECT,
    uses_closure_labels,
    CONTROL_STATE_LOSS_GRID_FIXED_DT,
    CORRIDOR_GATE_FAF,
    DEFAULT_AIRCRAFT_TYPE,
    HORIZON_FULL,
    HORIZON_NORMALIZED,
    HORIZON_WINDOW,
    RANDOM_TRAIN_ANCHOR_SAMPLING_STRATA,
    RANDOM_TRAIN_ANCHOR_SAMPLING_UNIFORM,
    STATE_POSITION_CORRIDOR_BOUNDED,
    TSConfig,
    uses_control_dynamics,
)
from data_provenance import manifest_paths
from control.basis_fit import FittedTeacherTable
from control.conditioning import condition_vector
from control.envelope import CONTROL_LOWER, CONTROL_UPPER
from control.dynamics.inverse import actual_controls, segment_controls
from coordinate_frames import (
    COORDINATE_FRAME_AIRPORT_ENU,
    AirportReference,
    CoordinateFrame,
    frame_for_state,
)
from final_approach_geometry import FINAL_APPROACH_KEYS
from flight_scenarios.procedure_final import final_approach_fix
from flight_scenarios.runway_target import airport_reference_point
from fixed_dt_supervision import (
    FixedDTControlSupervision,
    FixedDTSupervisionRow,
    build_fixed_dt_supervision,
    cache_fixed_dt_supervision_rows,
    pack_fixed_dt_supervision_rows,
)
from intent_conditioning import LeadLanding, intent_vector, lead_landings
from target_conditioning import (
    TARGET_CONDITIONING_NONE,
    conditioned_history,
    conditioning_vector,
)
from time_grids import output_time_grid
from reference_velocity import rebuild_reference_velocities



def dataset_flight_key(source: dict[str, Any], index: int) -> str:
    """Airport-qualified identity used by splits and checkpoint membership."""
    airport = str(source.get("arr_airport") or "").strip().upper()
    key = flight_key(source, index)
    return f"{airport}:{key}" if airport else key



@dataclass
class FlightSeries:
    """One observed arrival plus its training-only fitted position supervision."""

    flight_id: str
    scenario: FlightScenario
    frame: CoordinateFrame
    times: np.ndarray        # [N] seconds, uniform dt, rebased to 0 at the first sample
    values: np.ndarray       # [N, C] channel space (see channels.CHANNELS)
    # The observed arrays above remain the only model INPUT and the only arrays exposed to
    # forecast/export.  These arrays extend them with a fitted tail for training TARGETS.
    supervision_times: np.ndarray | None = None    # [M], M >= N
    supervision_values: np.ndarray | None = None   # [M, C]
    supervision_weights: np.ndarray | None = None  # [M, C], fitted velocities are zero
    # Scene context, from the tracks roster: the previous landing on the SAME runway.
    # ``None`` = the roster was never consulted (a series built outside
    # ``load_flight_dicts``); ``LeadLanding(None)`` = consulted, no earlier landing. Read
    # only by the Phase 0 intent channels (intent_conditioning); Phase 1 replaces it
    # with the neighbour set.
    lead_landing: LeadLanding | None = None

    def __post_init__(self) -> None:
        supplied = (
            self.supervision_times is not None,
            self.supervision_values is not None,
            self.supervision_weights is not None,
        )
        if not any(supplied):
            # Backward-compatible measured-only construction for small fixtures and generic
            # consumers that do not need fitted labels.
            self.supervision_times = self.times
            self.supervision_values = self.values
            self.supervision_weights = np.full(
                self.values.shape, 1.0 / self.values.shape[1], dtype=np.float64
            )
        elif not all(supplied):
            raise ValueError("supervision_times/values/weights must be supplied together")
        if not (
            len(self.supervision_times) == len(self.supervision_values)
            == len(self.supervision_weights)
        ):
            raise ValueError("supervision times, values, and weights must align")

    @property
    def n_samples(self) -> int:
        return len(self.times)

    @property
    def n_supervision_samples(self) -> int:
        return len(self.supervision_times)

    @property
    def airport(self) -> str:
        """Arrival airport carried by the canonical manifest record."""
        return str(self.scenario.source.get("arr_airport") or "").strip().upper()

    @property
    def dataset_id(self) -> str:
        """Cross-airport split/checkpoint identity; export stems remain ``flight_id``."""
        return f"{self.airport}:{self.flight_id}" if self.airport else self.flight_id

    @property
    def target_chart(self) -> np.ndarray:
        """The runway target's chart position ``(e, n, u)`` — where "distance to go" is
        measured from. ``(0, 0, 0)`` under the threshold-anchored frames; the frame's
        anchor and the target are different points under ``airport-enu``."""
        return target_chart_position(self.scenario.target, self.frame)




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


def bounded_output_gate(config: TSConfig) -> str | None:
    """The gate the trained output layer applies, or ``None`` when it bounds nothing."""
    if config.state_position_reference == STATE_POSITION_CORRIDOR_BOUNDED:
        return config.corridor_gate
    return None


def final_approach_fix_distance(series: FlightSeries, *, gate: str | None) -> float | None:
    """The coded FAF distance when ``gate`` is the FAF gate, else ``None``.

    Only ``corridor_gate="faf"`` reads it (the bounded output layer, or the inference-time
    projection); the deployable ``on-final`` gate and the penalty's truth gate need no
    procedure document at all. Raises when the flight cannot name its runway's RNAV(GPS)
    FAF — a FAF gate on a guessed distance would be silently wrong.
    """
    if gate != CORRIDOR_GATE_FAF:
        return None
    runway = str(series.scenario.source.get("runway") or "").strip().upper()
    if not series.airport or not runway:
        raise ValueError(
            f"flight {series.flight_id}: corridor_gate={CORRIDOR_GATE_FAF!r} needs the "
            "arrival airport and runway to read the coded FAF, and the flight carries "
            f"arr_airport={series.airport!r} runway={runway!r}"
        )
    return final_approach_fix(series.airport, runway).distance_to_threshold_m


def final_approach_arrays(
    series: FlightSeries, *, fix_distance_m: float | None
) -> dict[str, np.ndarray]:
    """``FINAL_APPROACH_KEYS`` for one flight: the runway course (math-ENU, the direction
    of travel on final), tan of the coded glidepath, and the FAF distance (NaN = unresolved)."""
    target = series.scenario.target
    rows = {
        # The rollout frame rotation and the runway heading coincide only for the
        # runway-aligned coordinate frame.  Keep the terminal-loss reference separate
        # so ENU rollouts are decomposed along/across the actual runway, not east/north.
        "runway_heading_rad": np.array(float(target.psi), dtype=np.float64),
        # The target's gamma is the coded glidepath DESCENT (negative); the chart height of
        # the glidepath at distance d back from the threshold is d · tan(GPA).
        "glidepath_tan": np.array(math.tan(-float(target.gamma)), dtype=np.float64),
        "final_approach_fix_m": np.array(
            math.nan if fix_distance_m is None else float(fix_distance_m), dtype=np.float64
        ),
    }
    assert tuple(rows) == FINAL_APPROACH_KEYS
    return rows


def probe_final_approach(batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    """One representative final-approach context for shape/throughput probes."""
    rows = {
        "runway_heading_rad": 0.0,
        "glidepath_tan": math.tan(math.radians(3.0)),
        "final_approach_fix_m": 10_000.0,
    }
    return {
        name: torch.full((batch_size,), value, dtype=torch.float32, device=device)
        for name, value in rows.items()
    }


def truth_duration_s(series: FlightSeries, anchor: int) -> float:
    """Seconds from the anchor to the truth's end — the value the CTA is given as."""
    return float(series.supervision_times[-1] - series.times[anchor])


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
            for key, value in final_approach_arrays(series, fix_distance_m=None).items()
            if key in ("runway_heading_rad", "glidepath_tan")
        },
    }


def probe_dynamics(batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    """One representative dynamics batch for shape/throughput probes.

    Kept beside :func:`dynamics_arrays` so the batch-size probe and the gradient
    diagnostics cannot carry a stale copy of the contract: a key added to the real batch
    appears here in the same commit or the probe fails immediately.
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
    return dynamics


@dataclass(frozen=True)
class Normalizer:
    """Per-channel standardisation, fit on the TRAINING split only.

    Both architectures already normalise each window internally (iTransformer's ``use_norm``,
    PatchTST's RevIN), so this is not about conditioning the attention — it is about the
    LOSS. Predictions come back in physical units, where ``e``/``n`` span ~2.5e4 m and
    ``udot`` spans ~1e1 m/s; an unweighted MSE over raw channels is ~99% a
    horizontal-position loss and the vertical channel never trains. Standardising first
    makes the loss weight channels comparably.
    """

    mean: np.ndarray   # [C]
    std: np.ndarray    # [C]

    @classmethod
    def fit(
        cls,
        series: Sequence[FlightSeries],
        *,
        balance_airports_and_flights: bool = False,
    ) -> Normalizer:
        """Fit on training only, optionally matching the hierarchical sampler's measure."""
        if balance_airports_and_flights:
            by_airport: dict[str, list[FlightSeries]] = {}
            for item in series:
                by_airport.setdefault(item.airport or "<unknown>", []).append(item)
            airport_means, airport_second_moments = [], []
            for group in by_airport.values():
                airport_means.append(np.mean([item.values.mean(axis=0) for item in group], axis=0))
                airport_second_moments.append(np.mean([
                    np.square(item.values).mean(axis=0) for item in group
                ], axis=0))
            mean = np.mean(airport_means, axis=0)
            variance = np.maximum(np.mean(airport_second_moments, axis=0) - mean**2, 0.0)
            std = np.sqrt(variance)
            std = np.where(std > 1e-9, std, 1.0)
            return cls(mean=mean, std=std)
        # series is a list of FlightSeries, each with values [N, C]; stack them to [N_total, C]
        stacked = np.concatenate([s.values for s in series], axis=0) # [N, C]
        mean = stacked.mean(axis=0) # [C]
        std = stacked.std(axis=0) # [C]
        # A channel that never varies across the training set carries no signal; leaving its
        # std at 0 would produce inf on the first divide. Scale it by 1 and let it ride as a
        # constant (the model can still use it as a bias).
        std = np.where(std > 1e-9, std, 1.0) # normal Z-score standardization: z=(x-mean)/std;
        # For channel with std=0, there is no need to standardize, so we set std=1.0 to avoid division by zero.
        return cls(mean=mean, std=std)

    def encode(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def decode(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, data: dict[str, list[float]]) -> Normalizer:
        return cls(mean=np.asarray(data["mean"], dtype=np.float64),
                   std=np.asarray(data["std"], dtype=np.float64))


# ── Loading ──────────────────────────────────────────────────────────────────

def load_flight_dicts(
    paths: str | Path | Sequence[str | Path],
    *,
    include_flight_keys: set[str] | None = None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Model-ready flights from authoritative manifests, optionally split-filtered.

    ``include_flight_keys`` uses the airport-qualified checkpoint identity
    ``AIRPORT:flight_key``. The set is reduced to each manifest's local keys before
    :func:`load_arrival_flights` opens source tracks, so an excluded split's trajectory
    values are never read merely to discard them later.
    """
    requested = None if include_flight_keys is None else set(include_flight_keys)
    flights: list[dict[str, Any]] = []
    loaded_keys: set[str] = set()
    for manifest_path in manifest_paths(paths):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        airport = str(manifest.get("airport") or "").strip().upper()
        local_keys = None
        if requested is not None:
            prefix = f"{airport}:"
            local_keys = {
                key[len(prefix):] for key in requested if key.startswith(prefix)
            }
            if not local_keys:
                if verbose:
                    print(f"  {manifest_path}: 0 selected manifest-rostered arrival(s)")
                continue
        manifest_flights = load_arrival_flights(
            manifest_path,
            include_flight_keys=local_keys,
        )
        # Scene context from the same roster the flights came through: the previous
        # same-runway landing, keyed by the manifest's own (airport-local) flight_key.
        keys = [flight_key(flight, index) for index, flight in enumerate(manifest_flights)]
        leads = lead_landings(manifest, manifest_path=manifest_path, flight_keys=keys)
        for key, flight in zip(keys, manifest_flights):
            flight["lead_landing"] = leads[key]
        flights.extend(manifest_flights)
        loaded_keys.update(
            dataset_flight_key(flight, index)
            for index, flight in enumerate(manifest_flights)
        )
        if verbose:
            qualifier = "selected " if requested is not None else ""
            print(
                f"  {manifest_path}: {len(manifest_flights)} "
                f"{qualifier}manifest-rostered arrival(s)"
            )
    if requested is not None:
        missing = requested - loaded_keys
        if missing:
            raise ValueError(
                f"arrival manifests do not roster requested flight {min(missing)!r}"
            )
    return flights


@dataclass
class BuildReport:
    """What survived the build, and why the rest did not.

    Skips are counted and reported rather than silently dropped: a run that quietly trains
    on 20% of the data because most thresholds were missing from the config should be
    obvious from the console, not from a confusing loss curve.
    """

    built: int = 0
    skipped: dict[str, int] = field(default_factory=dict)  # reason -> count
    selected_typecodes: dict[str, int] = field(default_factory=dict)
    rejected_aircraft: dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def select_typecode(self, typecode: str) -> None:
        self.selected_typecodes[typecode] = self.selected_typecodes.get(typecode, 0) + 1

    def reject_aircraft(self, reason: str) -> None:
        self.rejected_aircraft[reason] = self.rejected_aircraft.get(reason, 0) + 1
        self.skip("aircraft filter rejected")

    @property
    def total(self) -> int:
        return self.built + sum(self.skipped.values())

    def format(self) -> str:
        lines = [f"built {self.built}/{self.total} series"]
        for reason, count in sorted(self.skipped.items(), key=lambda kv: -kv[1]):
            lines.append(f"  skipped {count:5d}  {reason}")
        if self.selected_typecodes:
            selected = ", ".join(
                f"{code}:{count}"
                for code, count in sorted(
                    self.selected_typecodes.items(), key=lambda item: (-item[1], item[0])
                )
            )
            lines.append(f"  selected aircraft  {selected}")
        rejected = sorted(
            self.rejected_aircraft.items(), key=lambda item: (-item[1], item[0])
        )
        for reason, count in rejected[:20]:
            lines.append(f"  rejected {count:5d}  {reason}")
        if len(rejected) > 20:
            omitted = sum(count for _reason, count in rejected[20:])
            lines.append(
                f"  rejected {omitted:5d}  {len(rejected) - 20} additional type reasons "
                "(see data_selection.json)"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "built": self.built,
            "total": self.total,
            "skipped": dict(sorted(self.skipped.items())),
            "selected_typecodes": dict(sorted(self.selected_typecodes.items())),
            "rejected_aircraft": dict(sorted(self.rejected_aircraft.items())),
        }


@functools.lru_cache(maxsize=None)
def _airport_reference(airport: str) -> AirportReference:
    """The airport-fixed frame's anchor, from the harvest's own airport roster."""
    point = airport_reference_point(airport)
    return AirportReference(
        code=airport,
        lat=point["lat"],
        lon=point["lon"],
        elevation_msl_m=point["elevation_m"],
    )


def _frame_for_scenario(scenario: FlightScenario, config: TSConfig) -> CoordinateFrame:
    """Resolve the chart once per flight; ``airport-enu`` needs the arrival airport."""
    airport_ref = None
    if config.coordinate_frame == COORDINATE_FRAME_AIRPORT_ENU:
        code = str(scenario.source.get("arr_airport") or "").strip().upper()
        if not code:
            raise ValueError(
                f"flight {scenario.source.get('id')!r}: coordinate frame "
                f"{config.coordinate_frame!r} needs the arrival airport, but the flight "
                "carries no arr_airport and build_series was given none"
            )
        airport_ref = _airport_reference(code)
    return frame_for_state(scenario.target, config.coordinate_frame, airport_ref=airport_ref)


def build_series(
    flights: Sequence[dict[str, Any]],
    config: TSConfig,
    *,
    airport: str | None = None,
    aircraft_type: str = DEFAULT_AIRCRAFT_TYPE,
) -> tuple[list[FlightSeries], BuildReport]:
    """Flight dicts -> :class:`FlightSeries`, skipping what cannot be built.

    A flight is skipped when it has no published runway threshold (no ENU frame and no
    target to judge against), when the track is too short to resample onto the grid, or
    when it cannot furnish one full window (``seq_len + 1`` samples minimum).

    ``aircraft_type`` is the fallback when the flight dict does not name a resolvable type.
    Every harvested arrival currently carries ``"type": "UNK"`` (``czml_export`` hardcodes
    it), and ``flight_scenarios._resolve_aircraft`` RAISES rather than guessing — so
    without this the whole batch dies on the first flight. The choice is not cosmetic: it
    sets the target state's Vref and threshold-crossing height, which is what the
    evaluation gates measure the final state against.
    """
    minimum_samples = config.seq_len + 1
    series: list[FlightSeries] = []
    report = BuildReport()

    for index, flight in enumerate(flights):
        # Real inputs come through the harvest arrival manifest and must carry the
        # published CIFP target that made them model-ready. Synthetic fixtures are built
        # from the static threshold configuration and intentionally have no manifest
        # metadata, so they retain that explicit alternate path.
        if flight.get("altitude_source") != "synthetic":
            target_meta = flight.get("runway_target") or {}
            if target_meta.get("threshold_crossing_height_m") is None:
                report.skip("no published runway TCH")
                continue
            if target_meta.get("published_glidepath_deg") is None:
                report.skip("no published runway glidepath")
                continue
        waypoints = flight.get("waypoints") or []
        if len(waypoints) < 2:
            report.skip("track has fewer than 2 waypoints")
            continue

        # Waypoints are [t, lon, lat, alt] and state_samples_from_track keeps every
        # waypoint at time t - t0, so the span is known from the raw dict — check it
        # BEFORE the expensive build (aircraft resolution + per-sample least-squares
        # velocity fits), which too-short flights would otherwise pay for in full.
        span = float(waypoints[-1][0]) - float(waypoints[0][0])
        if span < config.dt_s * (minimum_samples - 1):
            report.skip(f"track shorter than one window ({config.lookback_s:.0f}s)")
            continue

        # Apply the strict fleet contract before geoid conversion, scenario construction,
        # velocity fitting or resampling. Split identity is a per-flight hash assigned from
        # manifest metadata, so rejecting a row here removes it *within* its original split;
        # it cannot reshuffle any other flight between train/validation/test.
        if config.aircraft_filter == AIRCRAFT_FILTER_OPENAP_DIRECT:
            identity = get_default_identity_resolver().resolve(
                declared_type=flight.get("type"),
                icao24=flight.get("icao24"),
            )
            support_kind = openap_support_kind(identity.typecode)
            if support_kind != "direct":
                if identity.typecode is None:
                    reason = "unresolved ICAO Doc 8643 typecode"
                elif support_kind == "synonym":
                    reason = f"OpenAP synonym model ({identity.typecode})"
                else:
                    reason = f"no native OpenAP model ({identity.typecode})"
                report.reject_aircraft(reason)
                continue
            report.select_typecode(identity.typecode)

        # Into the modeling plane: harvested altitude is ellipsoidal (HAE) while the
        # threshold-anchored channels and the evaluation gates are MSL. Converted HERE, not
        # inside build_scenario alone, because state_samples_from_track below takes the bare
        # waypoint list and so cannot convert itself. Idempotent (see flight_scenarios/datum).
        # Placed after the cheap skips (which read no altitude) so rejected flights don't
        # pay the conversion; ``waypoints`` must be rebound to the converted rows.
        flight = flight_to_msl(flight)
        waypoints = flight["waypoints"]

        scenario = build_scenario(flight,
            None if config.aircraft_filter == AIRCRAFT_FILTER_OPENAP_DIRECT
            else aircraft_type,
            airport=airport,
            target_from_threshold=True,
            aircraft_provider=(
                "openap"
                if config.aircraft_filter == AIRCRAFT_FILTER_OPENAP_DIRECT
                else "auto"
            ),
        )
        if scenario.target is None:
            runway = flight.get("runway") or "?"
            report.skip(f"no published threshold for runway {runway}")
            continue

        samples = state_samples_from_track(waypoints, mass_kg=scenario.initial.m)
        frame = _frame_for_scenario(scenario, config)
        target_chart = target_chart_position(scenario.target, frame)
        times, values = channels_from_states(samples, frame)
        grid, resampled = resample_uniform(times, values, config.dt_s)
        # Not redundant with the span pre-check: for a non-dyadic dt the multiply and
        # resample_uniform's floor-divide can round differently at the boundary.
        if len(grid) < minimum_samples:
            report.skip(f"track shorter than one window ({config.lookback_s:.0f}s)")
            continue

        fitted = fit_flight_final_approach(flight)
        observed_crossing = _observed_threshold_crossing(
            times,
            values,
            frame,
            target_chart=target_chart,
            runway_heading_rad=scenario.target.psi,
            fitted=fitted,
        )
        if observed_crossing is not None:
            crossing_time, _crossing_values = observed_crossing
            before_crossing = grid < crossing_time
            grid = grid[before_crossing]
            resampled = resampled[before_crossing]
            if len(grid) < config.seq_len:
                report.skip(f"track shorter than one window ({config.lookback_s:.0f}s)")
                continue

        supervision_times, supervision_values, supervision_weights = _build_supervision(
            samples,
            frame,
            grid,
            resampled,
            config,
            fitted=fitted,
            observed_crossing=observed_crossing,
        )
        measured_velocity_rows = (
            np.arange(len(supervision_times)) < len(grid)
        ) & np.all(
            supervision_weights[:, list(VELOCITY_IDX)] > 0.0, axis=1
        )
        supervision_values = rebuild_reference_velocities(
            supervision_times,
            supervision_values,
            source=config.reference_velocity_source,
            valid_rows=measured_velocity_rows,
        )
        resampled = supervision_values[: len(grid)].copy()
        series.append(FlightSeries(
            flight_id=flight_key(scenario.source, index), scenario=scenario, frame=frame,
            times=grid, values=resampled,
            supervision_times=supervision_times,
            supervision_values=supervision_values,
            supervision_weights=supervision_weights,
            lead_landing=flight.get("lead_landing"),
        ))
        report.built += 1

    return series, report


def _observed_threshold_crossing(
    measured_times: np.ndarray,
    measured_values: np.ndarray,
    frame: CoordinateFrame,
    *,
    target_chart: np.ndarray,
    runway_heading_rad: float,
    fitted: FittedApproach | None,
) -> tuple[float, np.ndarray] | None:
    """Interpolate the measured final's first crossing of the runway threshold plane.

    The plane passes through the TARGET (``target_chart``), normal to the runway course —
    not through the frame origin, which only coincides with the threshold under the
    threshold-anchored frames.
    """
    cosine = np.cos(runway_heading_rad)
    sine = np.sin(runway_heading_rad)
    along = np.asarray([
        east * cosine + north * sine
        for east, north in (
            frame.to_world_horizontal(
                float(row[0]) - float(target_chart[0]),
                float(row[1]) - float(target_chart[1]),
            )
            for row in measured_values
        )
    ])
    first_right_index = fitted.fit.last_sample_index + 1 if fitted is not None else 1
    for right_index in range(first_right_index, len(along)):
        left_index = right_index - 1
        left_along = along[left_index]
        right_along = along[right_index]
        if left_along <= 0.0 <= right_along and right_along > left_along:
            # The project's single crossing-fraction definition (final_approach) —
            # the same two-point operation the harvest bracket and the evaluator use.
            fraction = bracket_fraction(float(left_along), float(right_along))
            crossing_time = measured_times[left_index] + fraction * (
                measured_times[right_index] - measured_times[left_index]
            )
            crossing_values = measured_values[left_index] + fraction * (
                measured_values[right_index] - measured_values[left_index]
            )
            return float(crossing_time), crossing_values
    return None


def _build_supervision(
    measured_samples,
    frame: CoordinateFrame,
    grid: np.ndarray,
    measured_values: np.ndarray,
    config: TSConfig,
    *,
    fitted: FittedApproach | None,
    observed_crossing: tuple[float, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Observed six-channel labels plus low-weight, position-only fitted tail labels."""
    channel_count = len(CHANNELS)
    # Row weights sum to one, preserving the previous all-channel mean-MSE scale.
    measured_weights = np.full(
        measured_values.shape, 1.0 / channel_count, dtype=np.float64
    )
    if observed_crossing is not None:
        crossing_time, crossing_values = observed_crossing
        return (
            np.concatenate([grid, np.asarray([crossing_time])]),
            np.concatenate([measured_values, crossing_values[None, :]], axis=0),
            np.concatenate([
                measured_weights,
                np.full((1, channel_count), 1.0 / channel_count, dtype=np.float64),
            ], axis=0),
        )
    if (
        config.fitted_tail_position_weight == 0.0
        and config.fitted_terminal_position_weight == 0.0
    ):
        return grid, measured_values, measured_weights

    if fitted is None:
        return grid, measured_values, measured_weights
    tail = fitted.uniform_tail(after_time_s=float(grid[-1]), dt_s=config.dt_s)
    if not tail:
        return grid, measured_values, measured_weights

    # Kinematics are placeholders required by the fixed six-channel tensor shape.  Their
    # supervision weights below are zero, so no extrapolated velocity enters the loss.
    terminal_state = measured_samples[-1][1]
    tail_states = [
        (
            row.time_s,
            type(terminal_state)(
                latitude=row.point.lat,
                longitude=row.point.lon,
                altitude=row.point.alt_m,
                V=terminal_state.V,
                psi=terminal_state.psi,
                gamma=terminal_state.gamma,
                m=terminal_state.m,
            ),
        )
        for row in tail
    ]
    tail_times, tail_values = channels_from_states(tail_states, frame)
    tail_weights = np.zeros_like(tail_values)
    for index in POSITION_IDX:
        tail_weights[:, index] = config.fitted_tail_position_weight / len(POSITION_IDX)
        tail_weights[-1, index] += (
            config.fitted_terminal_position_weight / len(POSITION_IDX)
        )

    return (
        np.concatenate([grid, tail_times]),
        np.concatenate([measured_values, tail_values], axis=0),
        np.concatenate([measured_weights, tail_weights], axis=0),
    )


# ── Windowing ────────────────────────────────────────────────────────────────

def series_conditioning(
    series: FlightSeries, config: TSConfig, normalizer: Normalizer, *, anchor: int
) -> np.ndarray | None:
    """The flight's constant input-only conditioning row at ``anchor``, or ``None`` when
    all off.

    Built here, beside the windows, because it is a per-flight INPUT value in the
    normalizer's space — the same row is appended to every history window of the flight
    at training time (``TrajectoryWindows``) and at inference (``forecast``). Column
    order is ``config.input_channels``: the target conditioning, then the intent. Only
    the intent's lead channel depends on the anchor (an ETA is relative to a moment).
    """
    position = list(POSITION_IDX)
    parts: list[np.ndarray] = []
    if config.target_conditioning != TARGET_CONDITIONING_NONE:
        parts.append(conditioning_vector(
            series.target_chart,
            float(series.scenario.target.psi),
            position_mean=normalizer.mean[position],
            position_std=normalizer.std[position],
        ))
    intent = intent_vector(
        series,
        config.intent_conditioning,
        anchor_time_s=float(series.times[anchor]),
        position_mean=normalizer.mean[position],
        position_std=normalizer.std[position],
        remaining_time_scale_s=config.final_time_scale_s,
    )
    if intent is not None:
        parts.append(intent)
    if not parts:
        return None
    return np.concatenate(parts).astype(np.float32)


def window_anchors(
    series: FlightSeries,
    config: TSConfig,
    *,
    minimum_anchor_index: int | None = None,
    minimum_future_s: float = 0.0,
) -> range:
    """Valid anchor indices for ``series``.

    An anchor ``i`` is the index of the LAST observed sample: the model is shown
    ``values[i - seq_len + 1 : i + 1]`` and must predict what follows.

    Normalized and full modes accept any non-empty remainder; normalized resamples the
    complete remainder and full masks a short padded suffix. Window mode requires a complete
    fixed-dt short horizon because those targets train one recursive forecasting pass.
    """
    if minimum_anchor_index is not None and minimum_anchor_index < 0:
        raise ValueError("minimum_anchor_index must be non-negative")
    if minimum_future_s < 0.0:
        raise ValueError("minimum_future_s must be non-negative")
    first = max(config.seq_len - 1, minimum_anchor_index or 0)
    # An anchor is always observed; fitted rows can be targets but never model inputs.
    last_with_remainder = series.n_supervision_samples - 2
    required_window_s = config.pred_len * config.dt_s
    complete_window_indices = np.flatnonzero(
        series.supervision_times[-1] - series.times >= required_window_s - 1e-9
    )
    last_with_complete_window = (
        int(complete_window_indices[-1]) if len(complete_window_indices) else -1
    )
    last_by_mode = {
        HORIZON_NORMALIZED: last_with_remainder,
        HORIZON_FULL: last_with_remainder,
        HORIZON_WINDOW: last_with_complete_window,
    }
    future_eligible = np.flatnonzero(
        series.supervision_times[-1] - series.times >= minimum_future_s - 1e-9
    )
    last_with_minimum_future = (
        int(future_eligible[-1]) if len(future_eligible) else -1
    )
    last = min(
        series.n_samples - 1,
        last_by_mode[config.horizon_mode],
        last_with_minimum_future,
    )
    return range(first, last + 1) if last >= first else range(0)


class TrajectoryWindows(Dataset, ABC):
    """Shared target-grid and batch mechanics for an explicit anchor population.

    Measured rows supervise all six channels. Fitted rows supervise only ``e/n/u`` at lower
    weight. Linear interpolation maps the remainder either to a complete normalized grid or
    to a physical full/window grid with a masked suffix. Concrete subclasses own anchor
    population and per-epoch index selection; this class contains no fixed-versus-random
    anchor-policy branch.
    """

    anchor_description: str
    anchor_policy: str
    sampling_version: str
    #: Whether the batch carries the CONTROL SUPERVISION targets (the imitation schedule and
    #: the heading-rate reference). Every training and validation window set does; a
    #: replay-only set (`ExplicitAnchorTrajectoryWindows(supervision=False)`) does not,
    #: because nothing reads them there and the imitation target is anchor-bound.
    control_supervision: bool = True

    def __init__(
        self,
        series: Sequence[FlightSeries],
        config: TSConfig,
        normalizer: Normalizer,
        *,
        minimum_anchor_index: int | None = None,
        minimum_future_s: float = 0.0,
        fitted_teacher: FittedTeacherTable | None = None,
    ):
        self.series = list(series)
        self.config = config
        self.normalizer = normalizer
        self.minimum_anchor_index = minimum_anchor_index
        self.minimum_future_s = float(minimum_future_s)
        self.index: list[tuple[int, int]] = []
        self.series_ranges: dict[int, tuple[int, int]] = {}
        self.temporal_candidate_anchors = 0
        self.eligible_candidate_anchors = 0
        for s_idx, item in enumerate(self.series):
            anchors = window_anchors(
                item,
                config,
                minimum_anchor_index=minimum_anchor_index,
                minimum_future_s=self.minimum_future_s,
            )
            self.temporal_candidate_anchors += len(anchors)
            anchors = self._eligible_anchors(item, anchors)
            self.eligible_candidate_anchors += len(anchors)
            chosen = self._select_anchors(anchors)
            start = len(self.index)
            self.index.extend((s_idx, anchor) for anchor in chosen)
            self.series_ranges[s_idx] = (start, len(self.index) - start)
        self.range_starts = np.array(
            [self.series_ranges[index][0] for index in range(len(self.series))],
            dtype=np.int64,
        )
        self.range_counts = np.array(
            [self.series_ranges[index][1] for index in range(len(self.series))],
            dtype=np.int64,
        )
        self.eligible_series = np.flatnonzero(self.range_counts)
        # Standardise once up front rather than per __getitem__ — the series are small
        # (a few hundred rows each) and this is read on every epoch.
        self.encoded = [
            normalizer.encode(s.supervision_values).astype(np.float32) for s in self.series
        ]
        # Input-only conditioning, one constant row per flight built at the flight's
        # anchor (None when off, or when the flight has no window). A random-anchor
        # population varies the anchor per sample; the only anchor-dependent column,
        # the intent's lead ETA, refuses that policy at TSConfig.
        self.conditioning = [
            series_conditioning(
                item, config, normalizer, anchor=self.index[self.range_starts[s_idx]][1]
            )
            if self.range_counts[s_idx]
            else None
            for s_idx, item in enumerate(self.series)
        ]
        # The per-flight final-approach context (never a model input), when the recipe
        # bounds or penalises the corridor. Resolved once: the FAF read is a document load.
        self.final_approach = (
            [
                final_approach_arrays(
                    s,
                    fix_distance_m=final_approach_fix_distance(
                        s, gate=bounded_output_gate(config)
                    ),
                )
                for s in self.series
            ]
            if config.uses_final_approach_context
            else None
        )
        # The closure output's per-flight labels (never a model input): the decision
        # vector, its validity, the label's path length and the runway course, from the
        # config's labels file. Imported here because closure_output reaches this module
        # through the arc-length geometry.
        self.closure = None
        self.closure_coverage: tuple[int, int, int] | None = None
        if uses_closure_labels(config.prediction_output):
            from closure_output import CONTEXT_VALID, label_context, load_labels
            labels = load_labels(config.closure_labels_path)
            self.closure = [label_context(s, labels, config) for s in self.series]
            present = sum(s.flight_id in labels.flights for s in self.series)
            valid = sum(int(row[CONTEXT_VALID]) for row in self.closure)
            # Stated by the caller under its own verbosity; refused here when nothing
            # could train (the two zeros mean different things).
            self.closure_coverage = (present, valid, len(self.series))
            if self.series and present == 0:
                raise ValueError(
                    f"{config.closure_labels_path} carries none of these {len(self.series)} "
                    "flights — another cohort's labels?"
                )
            if self.series and valid == 0:
                raise ValueError(
                    f"{config.closure_labels_path} carries these {len(self.series)} flights but "
                    "marks every label non-canonical or above the residual cap: nothing to regress"
                )
        # The imitation term's per-flight teacher table (control_imitation_target=
        # "fitted"). It is a TRAINING-TIME INPUT, handed in by `train.fit_model` for the
        # train and validation window sets it supervises — never opened here. Every other
        # consumer of this class replays a checkpoint (evaluate-fit, the z-oracle forecast,
        # the approach-cohort comparison) and needs no teacher at all; loading it here
        # would make those paths depend on a training artifact that may be gone, and on it
        # covering a cohort it was never fitted for.
        #
        # What IS checked here is coverage, because this is where the flights are known: a
        # fitted schedule reproduces its truth only at the width, the anchor and the total
        # duration it was fitted under, and a flight the table does not carry has no
        # teacher at all. A table that does not cover this cohort refuses the build — there
        # is no partial mode, which would train part of every batch on nothing while the
        # loss still reported an imitation number.
        self.fitted_teacher = fitted_teacher
        if fitted_teacher is not None:
            covered = [
                (self.series[int(index)], self.index[int(self.range_starts[int(index)])][1])
                for index in self.eligible_series
            ]
            fitted_teacher.require_cover(
                [
                    (item.flight_id, truth_duration_s(item, anchor))
                    for item, anchor in covered
                ],
                airports={item.airport for item, _anchor in covered},
                anchor_indices={int(anchor) for _item, anchor in covered},
                n_segments=int(config.n_segments),
            )
        # Public diagnostic for normalized-time experiments. Actual query times come from
        # the shared clock below, which also defines fixed-time loss and inference timing.
        self.progress = (
            np.arange(1, config.pred_len + 1, dtype=np.float64) / config.pred_len
        )
        self.kinematic_channels = np.array(
            [channel for channel in range(len(config.channels)) if channel not in POSITION_IDX]
        )
        # Fitted rows retain placeholder velocity values for tensor shape only. Cache the
        # final valid supervision time per flight/channel once; recomputing flatnonzero for
        # every sampled anchor was a substantial part of host-side batch preparation.
        self.last_supervised_times = [
            np.array([
                item.supervision_times[
                    np.flatnonzero(item.supervision_weights[:, channel] > 0.0)[-1]
                ]
                for channel in range(len(config.channels))
            ])
            for item in self.series
        ]
        eligible_airports = [
            item.airport or "<unknown>"
            for series_index, item in enumerate(self.series)
            if self.series_ranges[series_index][1]
        ]
        airport_flights = Counter(eligible_airports)
        eligible_count = len(eligible_airports)
        airport_count = len(airport_flights)
        # The mean weight is one, and summing a complete epoch gives the same
        # eligible_count / airport_count weight to every airport. This lets a normal batch
        # mean estimate the airport-macro objective without repeating smaller airports.
        self.flight_weights = np.array([
            eligible_count / (
                airport_count * airport_flights[item.airport or "<unknown>"]
            )
            if self.series_ranges[series_index][1]
            else 0.0
            for series_index, item in enumerate(self.series)
        ], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.index)

    @abstractmethod
    def _select_anchors(self, anchors: Sequence[int]) -> Sequence[int]:
        """Return the concrete mode's stored anchor population for one flight."""

    def _eligible_anchors(
        self, _series: FlightSeries, anchors: Sequence[int]
    ) -> Sequence[int]:
        """Apply mode-specific non-temporal eligibility before anchor sampling."""
        return anchors

    @abstractmethod
    def epoch_indices(self, seed: int) -> np.ndarray:
        """Return the concrete mode's one-flight-per-epoch sample indices."""

    def _sampling_extras(self, _indices: np.ndarray) -> dict[str, Any]:
        """Per-epoch audit rows only one anchor policy can produce.

        Empty here: a policy that stores ONE anchor per flight has no distribution to
        report — where its anchors sit is the definition of the policy.
        """
        return {}

    def anchor_statistics(self, seed: int) -> dict[str, Any]:
        """Audit the exact one-window-per-flight sample selected for an epoch."""
        indices = self.epoch_indices(seed)
        anchors = np.array([self.index[int(index)][1] for index in indices], dtype=np.int64)
        series_indices = np.array(
            [self.index[int(index)][0] for index in indices], dtype=np.int64
        )
        extras = self._sampling_extras(indices)
        if not len(indices):
            return {
                "policy": self.anchor_policy,
                "sampling_version": self.sampling_version,
                "minimum_future_s": self.minimum_future_s,
                "eligibility_policy": getattr(
                    self, "anchor_eligibility_policy", "temporal-only-v1"
                ),
                "temporal_candidate_anchors": self.temporal_candidate_anchors,
                "eligible_candidate_anchors": self.eligible_candidate_anchors,
                "excluded_candidate_anchors": (
                    self.temporal_candidate_anchors - self.eligible_candidate_anchors
                ),
                "samples": 0,
                **extras,
            }
        anchor_times = np.array([
            self.series[int(series_index)].times[int(anchor)]
            for series_index, anchor in zip(series_indices, anchors)
        ])
        remaining_times = np.array([
            self.series[int(series_index)].supervision_times[-1]
            - self.series[int(series_index)].times[int(anchor)]
            for series_index, anchor in zip(series_indices, anchors)
        ])
        digest_rows = sorted(
            f"{self.series[int(series_index)].dataset_id}:{int(anchor)}"
            for series_index, anchor in zip(series_indices, anchors)
        )
        digest = hashlib.sha256("\n".join(digest_rows).encode()).hexdigest()

        def distribution(values: np.ndarray) -> dict[str, float]:
            return {
                "min": float(np.min(values)),
                "p50": float(np.median(values)),
                "max": float(np.max(values)),
            }

        return {
            "policy": self.anchor_policy,
            "sampling_version": self.sampling_version,
            "minimum_future_s": self.minimum_future_s,
            "eligibility_policy": getattr(
                self, "anchor_eligibility_policy", "temporal-only-v1"
            ),
            "temporal_candidate_anchors": self.temporal_candidate_anchors,
            "eligible_candidate_anchors": self.eligible_candidate_anchors,
            "excluded_candidate_anchors": (
                self.temporal_candidate_anchors - self.eligible_candidate_anchors
            ),
            "samples": len(indices),
            "sample_sha256": digest,
            "anchor_index": distribution(anchors),
            "anchor_time_s": distribution(anchor_times),
            "remaining_time_s": distribution(remaining_times),
            "fixed_anchor_fraction": float(np.mean(anchors == self.config.seq_len - 1)),
            **extras,
        }

    def _sample_arrays(
        self, i: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.float32, np.float32]:
        s_idx, anchor = self.index[i]
        values = self.encoded[s_idx]
        series = self.series[s_idx]
        L = self.config.seq_len

        x = conditioned_history(
            values[anchor - L + 1 : anchor + 1], self.conditioning[s_idx]
        )
        anchor_time = float(series.times[anchor])
        final_time_s = float(series.supervision_times[-1] - anchor_time)
        time_grid = output_time_grid(final_time_s, self.config)
        query_times = anchor_time + time_grid.offsets_s

        # One search locates interpolation neighbours for every output time. The same
        # [N] neighbour arrays then interpolate all C state and weight channels together;
        # this replaces 2*C separate np.interp calls without hiding the interpolation math.
        right = np.searchsorted(series.supervision_times, query_times, side="left")
        right = np.clip(right, 1, len(series.supervision_times) - 1)
        left = right - 1
        interval = series.supervision_times[right] - series.supervision_times[left]
        fraction = ((query_times - series.supervision_times[left]) / interval)[:, None]
        y = values[left] + fraction * (values[right] - values[left])
        source_weights = series.supervision_weights
        weights = source_weights[left] + fraction * (
            source_weights[right] - source_weights[left]
        )
        weights[~time_grid.active] = 0.0

        # An observed crossing remains supervised even when it lies off the regular input
        # grid, but fitted-tail kinematics beyond the last measured velocity stay masked.
        weights[:, self.kinematic_channels] = np.where(
            query_times[:, None]
            > self.last_supervised_times[s_idx][self.kinematic_channels][None, :],
            0.0,
            weights[:, self.kinematic_channels],
        )

        return (
            x,
            y.astype(np.float32),
            weights.astype(np.float32),
            np.float32(final_time_s),
            self.flight_weights[s_idx],
        )

    def _dynamics_arrays(self, i: int) -> dict[str, np.ndarray]:
        s_idx, anchor = self.index[i]
        series = self.series[s_idx]
        arrays = dynamics_arrays(series, anchor)
        if self.config.cta_conditioning == CTA_CONDITIONING_GIVEN:
            # Training feeds the truth as the controlled time of arrival.
            arrays["cta_s"] = np.array(truth_duration_s(series, anchor), dtype=np.float64)
        if self.config.control_imitation_loss_weight and self.control_supervision:
            anchor_time = float(series.times[anchor])
            # The fitted teacher replaces the inversion outright — its schedule was fitted
            # over the WHOLE supervised horizon through the rollout, so every segment
            # carries weight one, where the inversion has to mask the fitted tail it has no
            # measured velocity to differentiate.
            arrays.update(
                self.fitted_teacher.supervision(series.flight_id)
                if self.fitted_teacher is not None
                else reference_control_supervision(
                    series,
                    anchor,
                    self.config,
                    total_duration_s=float(
                        series.supervision_times[-1] - anchor_time
                    ),
                    last_measured_time_s=float(
                        self.last_supervised_times[s_idx][
                            self.kinematic_channels
                        ].min()
                        - anchor_time
                    ),
                )
            )
        if self.config.control_heading_rate_loss_weight and self.control_supervision:
            # Same supervised horizon and same last-measured instant as the imitation
            # target above; the heading-rate target only masks at the endpoints instead of
            # the midpoints, and costs no inverse-dynamics solve.
            anchor_time = float(series.times[anchor])
            arrays.update(
                reference_heading_rate_supervision(
                    series,
                    anchor,
                    self.config,
                    total_duration_s=float(
                        series.supervision_times[-1] - anchor_time
                    ),
                    last_measured_time_s=float(
                        self.last_supervised_times[s_idx][
                            self.kinematic_channels
                        ].min()
                        - anchor_time
                    ),
                )
            )
        return arrays

    def _fixed_dt_supervision(
        self, indices: Sequence[int] | np.ndarray
    ) -> FixedDTControlSupervision:
        return build_fixed_dt_supervision(
            self.series,
            self.encoded,
            [self.index[int(index)] for index in indices],
            dt_s=self.config.dt_s,
        )

    def __getitem__(
        self, i: int
    ) -> tuple:
        x, y, weights, final_time_s, flight_weight = self._sample_arrays(i)
        result = (
            torch.from_numpy(x.copy()),
            torch.from_numpy(y),
            torch.from_numpy(weights),
            torch.from_numpy(np.asarray(final_time_s)),
            torch.from_numpy(np.asarray(flight_weight)),
        )
        if self.closure is not None:
            return (*result, {
                key: torch.from_numpy(np.asarray(value))
                for key, value in self.closure[self.index[i][0]].items()
            })
        if not uses_control_dynamics(self.config.prediction_output):
            return result
        dynamics = {
            key: torch.from_numpy(value)
            for key, value in self._dynamics_arrays(i).items()
        }
        if self.config.control_state_loss_grid != CONTROL_STATE_LOSS_GRID_FIXED_DT:
            return (*result, dynamics)
        dense = self._fixed_dt_supervision([i])
        return (*result, dynamics, dense)

    def batch(
        self, indices: Sequence[int] | np.ndarray
    ) -> tuple:
        """Build one contiguous batch without per-sample Tensor creation and collation."""
        batch_size = len(indices)
        L, N, C = self.config.seq_len, self.config.pred_len, len(self.config.channels)
        # The history carries the input contract (state channels + any conditioning);
        # the targets carry the state contract only.
        x = np.empty((batch_size, L, len(self.config.input_channels)), dtype=np.float32)
        y = np.empty((batch_size, N, C), dtype=np.float32)
        weights = np.empty_like(y)
        final_time_s = np.empty(batch_size, dtype=np.float32)
        flight_weights = np.empty(batch_size, dtype=np.float32)

        # Flights are ragged, so locating each source array remains a short explicit loop.
        # All expensive work inside a sample is vectorized over N progress points and C
        # channels, and conversion to Torch happens once per complete batch below.
        for row, index in enumerate(indices):
            sample_x, sample_y, sample_weights, sample_time, flight_weight = (
                self._sample_arrays(int(index))
            )
            x[row] = sample_x
            y[row] = sample_y
            weights[row] = sample_weights
            final_time_s[row] = sample_time
            flight_weights[row] = flight_weight

        result = tuple(
            torch.from_numpy(array)
            for array in (x, y, weights, final_time_s, flight_weights)
        )
        # The context slot: the control dynamics (which already carry the final-approach
        # keys), or the final-approach keys alone for a state recipe that needs them.
        if uses_control_dynamics(self.config.prediction_output):
            context_rows = [self._dynamics_arrays(int(index)) for index in indices]
        elif self.closure is not None:
            context_rows = [self.closure[self.index[int(index)][0]] for index in indices]
        elif self.final_approach is not None:
            context_rows = [self.final_approach[self.index[int(index)][0]] for index in indices]
        else:
            return result
        context = {
            key: torch.from_numpy(np.stack([row[key] for row in context_rows]))
            for key in context_rows[0]
        }
        if self.config.control_state_loss_grid != CONTROL_STATE_LOSS_GRID_FIXED_DT:
            return (*result, context)
        dense = self._fixed_dt_supervision(indices)
        return (*result, context, dense)


class FixedAnchorTrajectoryWindows(TrajectoryWindows):
    """One deterministic `L-1` (or experiment-supplied common) anchor per flight."""

    anchor_description = "fixed train anchor L-1"
    anchor_policy = "fixed"
    sampling_version = "fixed-anchor-v1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Every fixed-anchor row is reused on every validation epoch and on final replay.
        # Cache the ragged source rows, not padded batches, so duration bucketing remains a
        # pure batching decision and cannot change any target or dynamics value.
        self._sample_cache = [
            TrajectoryWindows._sample_arrays(self, index)
            for index in range(len(self.index))
        ]
        self._dynamics_cache: list[dict[str, np.ndarray]] | None = None
        self._fixed_dt_cache: tuple[FixedDTSupervisionRow, ...] | None = None
        if uses_control_dynamics(self.config.prediction_output):
            self._dynamics_cache = [
                TrajectoryWindows._dynamics_arrays(self, index)
                for index in range(len(self.index))
            ]
            if self.config.control_state_loss_grid == CONTROL_STATE_LOSS_GRID_FIXED_DT:
                self._fixed_dt_cache = cache_fixed_dt_supervision_rows(
                    self.series,
                    self.encoded,
                    self.index,
                    dt_s=self.config.dt_s,
                )

    def _sample_arrays(self, i: int):
        return self._sample_cache[i]

    def _dynamics_arrays(self, i: int) -> dict[str, np.ndarray]:
        if self._dynamics_cache is None:
            return super()._dynamics_arrays(i)
        return self._dynamics_cache[i]

    def _fixed_dt_supervision(
        self, indices: Sequence[int] | np.ndarray
    ) -> FixedDTControlSupervision:
        if self._fixed_dt_cache is None:
            return super()._fixed_dt_supervision(indices)
        return pack_fixed_dt_supervision_rows(
            [self._fixed_dt_cache[int(index)] for index in indices],
            channels=len(self.config.channels),
        )

    def _select_anchors(self, anchors: Sequence[int]) -> Sequence[int]:
        return [anchors[0]] if len(anchors) else []

    def epoch_indices(self, seed: int) -> np.ndarray:
        indices = self.range_starts[self.eligible_series].copy()
        np.random.default_rng(seed).shuffle(indices)
        return indices


class ExplicitAnchorTrajectoryWindows(FixedAnchorTrajectoryWindows):
    """One CALLER-SUPPLIED anchor per flight — still one deterministic window each.

    The fixed-anchor policy places every flight at the same index (``L-1``, or a common
    ``minimum_anchor_index``); a remaining-path bin places each flight where ITS OWN
    geometry put the bin, so the anchors differ flight by flight. Everything downstream is
    unchanged — the same caching, the same batches, the same validation batch plan — which
    is why this subclasses the fixed policy rather than restating it.

    ``anchors`` maps ``FlightSeries.dataset_id`` to the anchor index, and must cover every
    flight with an index the flight can actually be anchored at: the caller
    (:mod:`anchor_grid`) has already decided which flights have a reading at this bin, so a
    gap here is a cohort bug, not a case to skip.

    ``supervision=False`` skips the per-flight CONTROL SUPERVISION targets — the
    inverse-dynamics (or fitted) imitation schedule and the heading-rate reference. A
    replay-only consumer reads neither, and building them here would be worse than
    wasteful: the imitation target is ANCHOR-BOUND, so a
    ``control_imitation_target="fitted"`` run would silently carry the inverse-dynamics
    teacher at these anchors instead of its own table.
    """

    anchor_description = "one explicit anchor per flight"
    anchor_policy = "explicit"
    sampling_version = "explicit-anchor-v1"

    def __init__(
        self,
        series: Sequence[FlightSeries],
        config: TSConfig,
        normalizer: Normalizer,
        *,
        anchors: Mapping[str, int],
        minimum_anchor_index: int | None = None,
        fitted_teacher: FittedTeacherTable | None = None,
        supervision: bool = True,
    ):
        self._anchor_by_flight = dict(anchors)
        self.control_supervision = supervision
        super().__init__(
            series, config, normalizer,
            minimum_anchor_index=minimum_anchor_index,
            fitted_teacher=fitted_teacher,
        )

    def _eligible_anchors(
        self, series: FlightSeries, anchors: Sequence[int]
    ) -> Sequence[int]:
        wanted = self._anchor_by_flight[series.dataset_id]
        if wanted not in anchors:
            raise ValueError(
                f"flight {series.dataset_id!r} cannot be anchored at {wanted} "
                f"(admissible anchors: {anchors})"
            )
        return [wanted]


class RandomAnchorTrajectoryWindows(TrajectoryWindows):
    """All valid anchors available; each epoch selects one uniformly per flight."""

    anchor_description = "one random valid train anchor per flight and epoch"
    anchor_policy = "uniform-random"
    sampling_version = "per-flight-hash-v2-output-eligibility"

    def __init__(
        self,
        series: Sequence[FlightSeries],
        config: TSConfig,
        normalizer: Normalizer,
        *,
        minimum_anchor_index: int | None = None,
        fitted_teacher: FittedTeacherTable | None = None,
    ):
        self.anchor_eligibility_policy = random_train_anchor_eligibility_policy(config)
        # The config refuses a fitted teacher with random anchors (the table is fitted AT the
        # fixed anchor), so this is always None here; it is forwarded, not special-cased, so
        # the two window classes keep one constructor contract — train() passes it to both.
        super().__init__(
            series,
            config,
            normalizer,
            minimum_anchor_index=minimum_anchor_index,
            minimum_future_s=config.random_train_anchor_min_future_s,
            fitted_teacher=fitted_teacher,
        )
        # The remaining-path stratum of every STORED anchor, aligned with `self.index`.
        # `remaining-path-strata` draws on it; `uniform` only COUNTS it, which is the point:
        # a uniform draw over samples is uniform over time, and the skew that produces (a
        # short flight contributes only near-runway anchors) is invisible until an epoch's
        # anchors are counted per stratum.
        #
        # Imported here rather than at module scope because `anchor_grid` reads THIS module
        # (`FlightSeries`, `truth_duration_s`), so the grid can only be read back at call
        # time — the same reason as the closure-label import above. The grid stays the one
        # definition of the boundaries either way.
        from anchor_grid import REMAINING_PATH_STRATA_LABELS, remaining_path_strata
        self.strata_labels = REMAINING_PATH_STRATA_LABELS
        self.anchor_strata = np.zeros(len(self.index), dtype=np.int64)
        for s_idx, (start, count) in self.series_ranges.items():
            if count:
                self.anchor_strata[start : start + count] = remaining_path_strata(
                    self.series[s_idx]
                )[[anchor for _s_idx, anchor in self.index[start : start + count]]]

    def _select_anchors(self, anchors: Sequence[int]) -> Sequence[int]:
        return anchors

    def _eligible_anchors(
        self, series: FlightSeries, anchors: Sequence[int]
    ) -> Sequence[int]:
        return eligible_random_train_anchors(series, anchors, self.config)

    def _sampling_extras(self, indices: np.ndarray) -> dict[str, Any]:
        """The epoch's REALISED anchors, counted per remaining-path stratum.

        One draw per flight, so the counts sum to the epoch's flights. Recorded under both
        policies: the uniform one's counts ARE the skew the stratified one exists to remove,
        and a per-epoch record is the only place it can be read while a run is training.
        """
        counts = np.bincount(
            self.anchor_strata[indices], minlength=len(self.strata_labels)
        )
        return {
            "remaining_path_strata": {
                label: int(count)
                for label, count in zip(self.strata_labels, counts, strict=True)
            }
        }

    def epoch_indices(self, seed: int) -> np.ndarray:
        starts = self.range_starts[self.eligible_series]
        counts = self.range_counts[self.eligible_series]
        offsets = np.array([
            int.from_bytes(
                hashlib.sha256(
                    f"{self.sampling_version}:{seed}:"
                    f"{self.series[int(series_index)].dataset_id}".encode()
                ).digest()[:8],
                "big",
            ) % int(count)
            for series_index, count in zip(self.eligible_series, counts)
        ], dtype=np.int64)
        indices = starts + offsets
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        return indices


class StratifiedRandomAnchorTrajectoryWindows(RandomAnchorTrajectoryWindows):
    """One random valid anchor per flight and epoch, uniform over REMAINING-PATH STRATA.

    The uniform policy draws over a flight's admissible SAMPLES, which is uniform over time
    and therefore biased toward the runway: every flight contributes near-runway samples and
    only the long ones contribute far anchors. Measured consequence on the KRDU A0-random
    arm (design §2.4c): the 6 km validation anchor set improved for 180 epochs while L−1 and
    12 km degraded after epoch 10.

    Here a flight's anchors are grouped into the `anchor_grid` strata, one of the strata the
    flight ACTUALLY HAS anchors in is drawn uniformly, and then a sample inside it — so a
    flight's far anchors are drawn as often as its near ones. Both draws come from the
    flight's own per-epoch hash, exactly as the uniform policy's single draw does, so the
    epoch's anchors are a function of (seed, epoch, flight) and nothing else.

    Admissibility is UNCHANGED (`eligible_random_train_anchors`, the same future contract),
    so this policy trains the same cohort on the same anchor population; only which of them
    each epoch sees changes.
    """

    anchor_description = (
        "one random valid train anchor per flight and epoch, uniform over remaining-path strata"
    )
    anchor_policy = "remaining-path-strata-random"
    sampling_version = "per-flight-hash-v3-remaining-path-strata"

    @functools.cached_property
    def stratum_offsets(self) -> list[list[np.ndarray]]:
        """Per flight, its stored offsets grouped by stratum — ascending, non-empty only.

        The two draws are ``groups[h1 % len(groups)]`` then ``group[h2 % len(group)]``:
        uniform over the strata this flight HAS (an absent stratum is never drawn and never
        wastes a draw), then uniform inside the chosen one.
        """
        groups: list[list[np.ndarray]] = []
        for index in range(len(self.series)):
            start, count = self.series_ranges[index]
            strata = self.anchor_strata[start : start + count]
            groups.append(
                [np.flatnonzero(strata == value) for value in np.unique(strata)]
            )
        return groups

    def epoch_indices(self, seed: int) -> np.ndarray:
        groups_by_flight = self.stratum_offsets
        offsets = np.array([
            self._draw_offset(
                groups_by_flight[int(series_index)],
                hashlib.sha256(
                    f"{self.sampling_version}:{seed}:"
                    f"{self.series[int(series_index)].dataset_id}".encode()
                ).digest(),
            )
            for series_index in self.eligible_series
        ], dtype=np.int64)
        indices = self.range_starts[self.eligible_series] + offsets
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        return indices

    @staticmethod
    def _draw_offset(groups: list[np.ndarray], digest: bytes) -> int:
        """The stratum from the digest's first 8 bytes, the sample inside it from the next 8."""
        group = groups[int.from_bytes(digest[:8], "big") % len(groups)]
        return int(group[int.from_bytes(digest[8:16], "big") % len(group)])


#: The training window class each ``random_train_anchor_sampling`` value names. One table,
#: read by `train.fit_model` and the batch benchmark alike, so "which sampler did that run
#: use" has a single answer.
RANDOM_ANCHOR_WINDOW_CLASSES = {
    RANDOM_TRAIN_ANCHOR_SAMPLING_UNIFORM: RandomAnchorTrajectoryWindows,
    RANDOM_TRAIN_ANCHOR_SAMPLING_STRATA: StratifiedRandomAnchorTrajectoryWindows,
}


def training_window_class(config: TSConfig) -> type[TrajectoryWindows]:
    """The window class a training run's two anchor axes select."""
    if not config.random_train_anchor:
        return FixedAnchorTrajectoryWindows
    return RANDOM_ANCHOR_WINDOW_CLASSES[config.random_train_anchor_sampling]


class FlightEpochSampler(Sampler[int]):
    """Select one example per flight, then reshuffle the complete epoch."""

    def __init__(self, dataset: TrajectoryWindows, *, seed: int):
        self.dataset = dataset
        self.seed = seed

    def __len__(self) -> int:
        return len(self.dataset.eligible_series)

    def __iter__(self):
        return iter(self.dataset.epoch_indices(self.seed).tolist())


def iter_batches(
    dataset: TrajectoryWindows,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
) -> Iterator:
    """Yield contiguous in-RAM batches with deterministic index selection."""
    if shuffle:
        sampler = FlightEpochSampler(dataset, seed=seed)
        indices = np.fromiter(sampler, dtype=np.int64, count=len(sampler))
    else:
        indices = np.arange(len(dataset), dtype=np.int64)

    for start in range(0, len(indices), batch_size):
        yield dataset.batch(indices[start : start + batch_size])
