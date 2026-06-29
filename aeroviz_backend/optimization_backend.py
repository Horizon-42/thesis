from __future__ import annotations

import sys
import time
from typing import Any

from aeroviz_backend import paths  # noqa: F401
from aeroviz_backend.simulation_backend import (
    DEFAULT_AIRCRAFT_TYPE,
    DEFAULT_DT,
    DEFAULT_STATE,
    MAX_DT,
    clamp,
    format_control,
    format_geodetic_state,
    read_aircraft,
    read_float,
    read_geodetic_state,
    read_positive_int,
    read_required_mapping,
)

from geodetic_simulator import GeodeticSimulator, GeodeticState
from casadi_optimizer import CasadiOptimizer
from casadi_direct_collocation_optimizer import CasadiDirectCollocationOptimizer
from multiphase_collocation_optimizer import MultiphaseCollocationOptimizer
from common import LoadFactorControl
from least_squares_transcription_optimizor import LeastSquaresTranscriptionOptimizor
from single_shooting_optimizor import SingleShootingOptimizor
from simulator import Control
from aeroviz_backend.procedure_constraint import ProcedureConstraint
from aeroviz_backend.procedure_segments import build_constraint_segments
from aeroviz_backend.trajectory_playback import build_optimized_trajectory_playback
from transcription_optimizor import TranscriptionOptimizor
from variable_time_warm_start_transcription_optimizor import (
    VariableTimeWarmStartTranscriptionOptimizor,
)
from warm_start_transcription_optimizor import WarmStartTranscriptionOptimizor


DEFAULT_N_SEGMENTS = 10
DEFAULT_MAX_ITERATIONS = 1000
MIN_ARRIVAL_TIME_S = 1.0
MAX_ARRIVAL_TIME_S = 1000.0
MIN_OPTIMIZATION_DT = 0.001
DEFAULT_OPTIMIZER = "casadiDirectCollocation"

# Direct-collocation variants exposed as distinct optimizer names, each
# selecting a defect "fitting equation" (see casadi_direct_collocation_optimizer).
# The bare name keeps the default (Hermite-Simpson) for backward compatibility.
DIRECT_COLLOCATION_SCHEMES = {
    "casadiDirectCollocation": "hermiteSimpson",
    "casadiDirectCollocationTrapezoidal": "trapezoidal",
    "casadiDirectCollocationHermiteSimpson": "hermiteSimpson",
    "casadiDirectCollocationRk4": "rk4",
    "casadiDirectCollocationReanchoredEnu": "reanchoredEnu",
    "casadiDirectCollocationLocalEnu": "localEnu",
    "casadiDirectCollocationLocalEnuTrapezoidal": "localEnuTrapezoidal",
    "casadiDirectCollocationLocalEnuHermiteSimpson": "localEnuHermiteSimpson",
    # Full-transport geodetic: same geodetic RHS but the EXACT transport (adds
    # the psi cross term the default geodetic schemes drop); the bare name keeps
    # the Hermite-Simpson default.
    "casadiDirectCollocationFullTransport": "hermiteSimpsonFullTransport",
    "casadiDirectCollocationFullTransportTrapezoidal": "trapezoidalFullTransport",
    "casadiDirectCollocationFullTransportRk4": "rk4FullTransport",
    # Normalized geodetic schemes: same geodetic RHS, but the decision STATE is
    # metric position offsets from the target, which conditions the NLP well so
    # the solve is robust on loose arrival windows / finer meshes (the bare name
    # keeps the Hermite-Simpson default).
    "casadiDirectCollocationNormalized": "hermiteSimpsonNormalized",
    "casadiDirectCollocationNormalizedTrapezoidal": "trapezoidalNormalized",
    "casadiDirectCollocationNormalizedRk4": "rk4Normalized",
    # Normalized + FULL transport: well-conditioned metric-position state AND the
    # exact transport (the two compose); the bare name keeps Hermite-Simpson.
    "casadiDirectCollocationNormalizedFullTransport": "hermiteSimpsonNormalizedFullTransport",
    "casadiDirectCollocationNormalizedFullTransportTrapezoidal": "trapezoidalNormalizedFullTransport",
    "casadiDirectCollocationNormalizedFullTransportRk4": "rk4NormalizedFullTransport",
}
# Multiphase transcription: one phase per leg (start->IAF transition + each procedure leg), fixes
# pinned at the phase boundaries, exact per-leg constraints (no along-track partition). Each name
# maps to the per-phase collocation scheme; these schemes REQUIRE a procedureConstraint.
MULTIPHASE_SCHEMES = {
    "casadiMultiphaseNormalizedFullTransport": "hermiteSimpsonNormalizedFullTransport",
    "casadiMultiphaseNormalizedFullTransportTrapezoidal": "trapezoidalNormalizedFullTransport",
    "casadiMultiphaseNormalizedFullTransportRk4": "rk4NormalizedFullTransport",
}
SUPPORTED_OPTIMIZERS = (
    *DIRECT_COLLOCATION_SCHEMES,
    *MULTIPHASE_SCHEMES,
    "casadiIpopt",
    "transcription",
    "leastSquaresTranscription",
    "warmStartTranscription",
    "variableTimeWarmStartTranscription",
    "singleShooting",
)
# CasADi optimisers are cached (their NLP is compiled once); the direct-
# collocation variants additionally use the single-solve free-time path.
CASADI_OPTIMIZERS = ("casadiIpopt", *DIRECT_COLLOCATION_SCHEMES)


class OptimizationBackend:
    def __init__(self) -> None:
        self._casadi_optimizer_key = None
        self._casadi_optimizer = None

    def optimize(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        initial_payload = read_required_mapping(payload, "initialState")
        target_payload = read_required_mapping(payload, "targetState")
        n_segments = read_positive_int(payload, "nSegments", DEFAULT_N_SEGMENTS)
        max_iterations = read_positive_int(
            payload,
            "maxIterations",
            DEFAULT_MAX_ITERATIONS,
        )
        optimizer_name = read_optimizer(payload)
        arrival_time_s = read_arrival_time_s(payload)
        dt = clamp(read_float(payload, "dtS", DEFAULT_DT), MIN_OPTIMIZATION_DT, MAX_DT)
        aircraft = read_aircraft(initial_payload, DEFAULT_AIRCRAFT_TYPE)

        initial_state = read_geodetic_state(initial_payload, DEFAULT_STATE, aircraft)
        target_state = read_geodetic_state(target_payload, initial_state, aircraft)

        # Canonical procedure constraint (the front↔back shared path/altitude structure). It is
        # ENFORCED only by the explicit MULTIPHASE schemes (which require it — one phase per leg);
        # any other scheme parses and echoes it but does not enforce it.
        procedure_constraint = ProcedureConstraint.from_payload(
            payload.get("procedureConstraint")
        )
        is_multiphase = optimizer_name in MULTIPHASE_SCHEMES
        constraint_segments = None
        constraints_enforced = False
        if is_multiphase:
            if procedure_constraint is None:
                raise ValueError("the multiphase optimizer requires a procedureConstraint")
            # The multiphase optimiser models start->IAF as phase 0, so the start need NOT equal
            # the procedure's first fix.
            constraint_segments, _constraint_spans = build_constraint_segments(
                procedure_constraint,
                target_state.latitude,
                target_state.longitude,
                target_state.altitude,
            )
            if not constraint_segments:
                raise ValueError("the procedureConstraint produced no legs")
            constraints_enforced = True

        # Time the WHOLE optimization flow, not just the solver call: building
        # (compiling) the NLP, the solve (which for direct collocation is itself
        # a cold-start solve + a free-time solve), and rolling the controls
        # forward into the playback CZML.  The breakdown is logged to the server
        # log (stderr) below.
        flow_started = time.perf_counter()
        segment_durations = None
        if is_multiphase:
            # Multiphase transcription: one phase per leg (start->IAF transition + each procedure
            # leg), fixes pinned at the phase boundaries, exact per-leg constraints (replaces the
            # single-phase along-track partition). One NLP, free per-phase time, min total time.
            optimizer = MultiphaseCollocationOptimizer(
                aircraft, constraint_segments, scheme=MULTIPHASE_SCHEMES[optimizer_name],
            )
            build_s = time.perf_counter() - flow_started
            solve_started = time.perf_counter()
            final_time, node_control, node_state = optimizer.optimize_free_time(
                initial_state, target_state, arrival_time_s,
            )
            segment_durations = optimizer.segment_durations_s
            solve_s = time.perf_counter() - solve_started
        else:
            optimizer = self.make_optimizer(
                optimizer_name,
                GeodeticSimulator(aircraft),
                n_segments,
                dt,
                max_iterations,
                arrival_time_s=arrival_time_s,
            )
            build_s = time.perf_counter() - flow_started
            solve_started = time.perf_counter()
            if optimizer_name in DIRECT_COLLOCATION_SCHEMES:
                # Direct collocation includes T as a decision variable, so
                # one solve returns both the optimal trajectory and the
                # optimal arrival time -- no outer bisection needed.
                final_time, node_control, node_state = optimizer.optimize_free_time(
                    initial_state,
                    target_state,
                    arrival_time_s,
                )
            elif optimizer_name == "casadiIpopt":
                # Multiple shooting uses a fixed-time NLP; finding the
                # shortest feasible duration still requires bisecting on T.
                final_time, node_control, node_state = optimizer.optimize_time_to_target(
                    initial_state,
                    target_state,
                    arrival_time_s,
                )
            else:
                final_time, node_control, node_state = optimizer.optimize_trajectory(
                    initial_state,
                    target_state,
                )
            solve_s = time.perf_counter() - solve_started

        result = {
            "ok": True,
            "finalTimeS": float(final_time),
            # Multiphase emits its own per-phase control mesh, so the actual control count is the
            # one to report (not the request's nSegments, which it ignores).
            "nSegments": len(node_control) if is_multiphase else n_segments,
            "dtS": dt,
            "optimizer": optimizer_name,
            "controls": [
                format_optimizer_control(optimizer_name, control_values)
                for control_values in node_control
            ],
            "states": format_node_states(node_state, initial_state.m, aircraft.code),
        }

        # Roll the controls forward once to produce a CZML the frontend plays on
        # Cesium's own clock (like a downloaded trajectory) plus a dense sample
        # series for the live readout.  Kept separate from the solve so it can be
        # stubbed in tests via ``build_optimized_trajectory_playback``.
        playback_started = time.perf_counter()
        playback = build_optimized_trajectory_playback(
            optimizer_name,
            initial_state,
            node_control,
            float(final_time),
            aircraft,
            segment_durations=segment_durations,
        )
        playback_s = time.perf_counter() - playback_started
        if playback is not None:
            result["playback"] = playback

        # Echo the canonical procedure constraint summary (parsed above) plus whether it was
        # enforced as NLP path constraints (only on the normalized full-transport schemes).
        if procedure_constraint is not None:
            result["procedureConstraintSummary"] = procedure_constraint.summary()
            result["procedureConstraintEnforced"] = constraints_enforced

        log_optimization_timing(
            optimizer_name,
            build_s=build_s,
            solve_s=solve_s,
            playback_s=playback_s,
            total_s=time.perf_counter() - flow_started,
            solve_breakdown=getattr(optimizer, "last_solve_timings", None),
        )
        return result

    def make_optimizer(
        self,
        optimizer_name: str,
        geodetic_simulator: GeodeticSimulator,
        n_segments: int,
        dt: float,
        max_iterations: int,
        arrival_time_s: float,
        constraint_segments=None,
        constraint_spans=None,
    ) -> Any:
        if optimizer_name in CASADI_OPTIMIZERS:
            # CasADi optimisers recompile a (potentially large) NLP at
            # construction.  Caching one instance per (aircraft, mesh,
            # dt, arrival_time, optimizer_name) avoids the cost on every
            # request.  ``optimizer_name`` is included in the key
            # because the multiple-shooting and direct-collocation
            # solvers have different NLP layouts.
            #
            # Procedure-constrained NLPs bake request-specific geometry into the
            # constraint vector, so they cannot be cached across requests -- build
            # fresh and skip the cache.
            if constraint_segments:
                return make_optimizer(
                    optimizer_name,
                    geodetic_simulator,
                    n_segments,
                    dt,
                    max_iterations,
                    arrival_time_s,
                    constraint_segments=constraint_segments,
                    constraint_spans=constraint_spans,
                )
            aircraft = geodetic_simulator.simulator.aircraft
            key = (optimizer_name, aircraft.code, n_segments, dt, arrival_time_s)
            if self._casadi_optimizer_key != key:
                self._casadi_optimizer_key = key
                self._casadi_optimizer = make_optimizer(
                    optimizer_name,
                    geodetic_simulator,
                    n_segments,
                    dt,
                    max_iterations,
                    arrival_time_s,
                )
            return self._casadi_optimizer

        return make_optimizer(
            optimizer_name,
            geodetic_simulator,
            n_segments,
            dt,
            max_iterations,
            arrival_time_s,
        )


def log_optimization_timing(
    optimizer_name: str,
    build_s: float,
    solve_s: float,
    playback_s: float,
    total_s: float,
    solve_breakdown: dict[str, float] | None = None,
) -> None:
    """Write the whole-flow optimization timing to the server log (stderr).

    ``build`` is the NLP compile (≈0 on a cache hit), ``solve`` the optimiser
    call, ``playback`` the control-rollout CZML build, ``total`` the sum.  When
    the optimiser exposes a per-phase ``solve_breakdown`` (direct collocation:
    the cold-start solve vs the free-time solve), those are interleaved so the
    log shows the full pipeline rather than one lumped solve time.
    """
    parts = [
        "[aeroviz-backend] optimization timing",
        f"optimizer={optimizer_name}",
        f"build={build_s:.3f}s",
    ]
    if solve_breakdown:
        parts.append(f"coldStart={solve_breakdown.get('coldStartS', 0.0):.3f}s")
        parts.append(f"freeTime={solve_breakdown.get('freeTimeSolveS', 0.0):.3f}s")
    parts.extend([
        f"solve={solve_s:.3f}s",
        f"playback={playback_s:.3f}s",
        f"total={total_s:.3f}s",
    ])
    sys.stderr.write(" ".join(parts) + "\n")
    sys.stderr.flush()


def read_optimizer(payload: dict[str, Any]) -> str:
    value = payload.get("optimizer", DEFAULT_OPTIMIZER)
    if not isinstance(value, str):
        raise ValueError("optimizer must be a string")

    optimizer = value.strip()
    if optimizer not in SUPPORTED_OPTIMIZERS:
        valid_values = ", ".join(SUPPORTED_OPTIMIZERS)
        raise ValueError(f"optimizer must be one of {valid_values}")
    return optimizer


def make_optimizer(
    optimizer_name: str,
    geodetic_simulator: GeodeticSimulator,
    n_segments: int,
    dt: float,
    max_iterations: int,
    arrival_time_s: float,
    constraint_segments=None,
    constraint_spans=None,
) -> Any:
    if optimizer_name == "casadiIpopt":
        return CasadiOptimizer(
            n_segments=n_segments,
            dt=dt,
            max_duration=arrival_time_s,
            aircraft=geodetic_simulator.simulator.aircraft,
        )

    if optimizer_name in DIRECT_COLLOCATION_SCHEMES:
        return CasadiDirectCollocationOptimizer(
            n_segments=n_segments,
            dt=dt,
            max_duration=arrival_time_s,
            aircraft=geodetic_simulator.simulator.aircraft,
            collocation_scheme=DIRECT_COLLOCATION_SCHEMES[optimizer_name],
            constraint_segments=constraint_segments,
            constraint_spans=constraint_spans,
        )

    if optimizer_name == "singleShooting":
        return SingleShootingOptimizor(
            geodetic_simulator,
            n_control_segments=n_segments,
            dt=dt,
            max_iterations=max_iterations,
        )

    if optimizer_name == "leastSquaresTranscription":
        return LeastSquaresTranscriptionOptimizor(
            geodetic_simulator,
            n_segments=n_segments,
            dt=dt,
            arrival_time_s=arrival_time_s,
            max_iterations=max_iterations,
        )

    if optimizer_name == "warmStartTranscription":
        return WarmStartTranscriptionOptimizor(
            geodetic_simulator,
            n_segments=n_segments,
            dt=dt,
            arrival_time_s=arrival_time_s,
            max_iterations=max_iterations,
        )

    if optimizer_name == "variableTimeWarmStartTranscription":
        return VariableTimeWarmStartTranscriptionOptimizor(
            geodetic_simulator,
            n_segments=n_segments,
            dt=dt,
            arrival_time_s=arrival_time_s,
            max_iterations=max_iterations,
        )

    return TranscriptionOptimizor(
        geodetic_simulator,
        n_segments=n_segments,
        dt=dt,
        arrival_time_s=arrival_time_s,
        max_iterations=max_iterations,
    )


def format_optimizer_control(
    optimizer_name: str,
    control_values: Any,
) -> dict[str, float]:
    # The CasADi + multiphase optimisers use the load-factor parameterisation
    # (T, mu, n_cmd); the alpha-based optimisers use (T, mu, alpha).
    if optimizer_name in CASADI_OPTIMIZERS or optimizer_name in MULTIPHASE_SCHEMES:
        return format_control(LoadFactorControl(*control_values))
    return format_control(Control(*control_values))


def read_arrival_time_s(payload: dict[str, Any]) -> float:
    if "arrivalTimeS" not in payload:
        raise ValueError("arrivalTimeS must be a number")
    arrival_time_s = read_float(payload, "arrivalTimeS", 0.0)
    if not MIN_ARRIVAL_TIME_S <= arrival_time_s <= MAX_ARRIVAL_TIME_S:
        raise ValueError(
            f"arrivalTimeS must be between {MIN_ARRIVAL_TIME_S} and "
            f"{MAX_ARRIVAL_TIME_S}"
        )
    return arrival_time_s


def format_node_states(
    node_state: Any,
    mass: float,
    aircraft_code: str,
) -> list[dict[str, Any]]:
    if node_state is None:
        return []
    return [
        format_geodetic_state(
            _array_to_geodetic_state(state_values, mass),
            aircraft_code,
        )
        for state_values in node_state
    ]


def _array_to_geodetic_state(values: Any, mass: float) -> GeodeticState:
    latitude, longitude, altitude, V, psi, gamma = [float(value) for value in values]
    return GeodeticState(
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        V=V,
        psi=psi,
        gamma=gamma,
        m=mass,
    )
