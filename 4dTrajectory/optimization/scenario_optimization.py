"""Optimize flight scenarios: one trajectory per scenario, unconstrained or procedure-constrained.

For each :class:`FlightScenario` (from the ``flight_scenarios`` package) this solves the trajectory
from the scenario's initial state to its target with :class:`collocation.CollocationOptimizer` and
replays the optimizer's controls through the real simulator (``scenario_replay``):

  * :func:`optimize_scenario` — one free phase to the target (the ``runway`` / ``fitted_adsb`` modes);
  * :func:`optimize_scenario_shortest_iaf` / :func:`optimize_scenario_min_time_iaf` — the constrained
    mode: from the observed start through one IAF of the runway's RNAV(GPS) procedure (``procedure``);
  * :func:`optimize_scenarios` / :func:`optimize_scenarios_constrained_iaf` — the batches, both fronts
    over ``scenario_batch.run_batch``; the reference records come from ``scenario_references``.

The CLI (:func:`main`) is what ``run_scenario_optimization.py`` runs. Each scenario gets a
``*_states.json`` (the plan and the replay) and a ``*_eval.json`` (the neutral evaluation record,
``evaluation_export``); a failed scenario gets an eval record with empty lists.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_OPT_DIR = Path(__file__).resolve().parent
if str(_OPT_DIR) not in sys.path:
    sys.path.insert(0, str(_OPT_DIR))

from flight_scenarios import FlightScenario, flight_key, load_scenarios  # noqa: E402
from flight_scenarios.procedure_final import (  # noqa: E402
    DEFAULT_PROCEDURE_ROOT as _PROCEDURE_ROOT,
    rnav_gps_procedure_path,
)
from aircraft.aero_params import stall_speed_ms  # noqa: E402
from aerodynamic_model.common import GeodeticState  # noqa: E402
from collocation import CollocationOptimizer  # noqa: E402
from collocation.components import DEFAULT_MAX_ITERATIONS as _DEFAULT_MAX_ITERATIONS  # noqa: E402
# Single source for the control-mesh defaults (mirrored by the CLI + the pipeline) and for
# the target-anchored frame tolerance the constrained path validates its procedure against.
from collocation.optimizer import (  # noqa: E402
    _FRAME_ANCHOR_TOLERANCE_M,
    DEFAULT_N_SEGMENTS,
    DEFAULT_N_SEG_PER_PHASE,
    no_extra_rows,
)
from geokit import haversine_m  # noqa: E402
from procedure.iaf import iaf_paths, path_length_m  # noqa: E402
from procedure.segments import build_constraint_segments  # noqa: E402
from evaluation_export import REFERENCES_DIR, evaluation_record, file_sha256 as _file_sha256  # noqa: E402
from optimization_run_config import (  # noqa: E402
    DEFAULT_MAX_DURATION_S,
    DEFAULT_ROLLOUT_DT_S,
    FITTING_SCHEMES,
    build_optimization_config,
)
from scenario_batch import run_batch  # noqa: E402
from scenario_references import remove_legacy_category_references, write_reference_records  # noqa: E402
from scenario_replay import (  # noqa: E402
    ScenarioOptimization,
    StateSample,
    node_states_to_samples,
    require_usable_rollout,
    rollout_controls,
    rollout_guard_altitude_m,
)

# DEFAULT_N_SEGMENTS / the constrained DEFAULT_N_SEG_PER_PHASE are imported above from the
# optimizer (its own construction defaults).

# Fitting (transcription) selection for BOTH solve paths (unconstrained + constrained-IAF);
# each composes with the normalized full-transport dynamics. "hs" (Hermite-Simpson,
# 4th order) is the default — see the comment in optimize_scenario for why trapezoidal
# (2nd order) collapsed the batch success rate; it stays selectable for comparison runs.
# "rk4" is the 4th-order EXPLICIT shooting defect (same one-step map family as the replay
# integrator — playback-consistent by construction) at trapezoidal-like per-node cost.
DEFAULT_FITTING = "hs"

# IPOPT iteration cap for the batch: `components.DEFAULT_MAX_ITERATIONS` (3000), the optimizer's
# own default (the backend's HTTP request default is its own 1000). 3000 is the right budget when
# a human is waiting on ONE hard solve, and the wrong one for a 70k-solve batch, where every unsolvable scenario pays the
# cap in full before being skipped. Measured on 120 random KRDU arrivals (serial, HS,
# n_segments=8): the 8 that ended `Maximum_Iterations_Exceeded` cost 448 s (~56 s each)
# while the 8 solved ones cost 45 s (~4.3 s each) — 6.7% of the flights, ~48% of the CPU.
# The batch therefore exposes `--max-iterations`; it does NOT change the default, because
# lowering it converts slow successes into failures and that is a research decision, not a
# performance one. Pass it explicitly to buy the time back.
DEFAULT_MAX_ITERATIONS = _DEFAULT_MAX_ITERATIONS


def _scheme_for_fitting(fitting: str) -> str:
    try:
        return FITTING_SCHEMES[fitting]
    except KeyError:
        raise ValueError(
            f"unknown fitting {fitting!r}; choose from {sorted(FITTING_SCHEMES)}"
        ) from None

# Velocity floor = STALL_MARGIN x stall speed (at the scenario's mass), so the optimizer admits
# realistic touchdown-speed targets instead of forcing V >= Vref. It used to be capped at V_ref;
# since V_ref is the published approach speed (2026-09-24) the cap cannot bind: 1.10 x V_s1g is at
# most 0.93 x the published lower edge on every buildable airframe (pinned by
# tests/test_scenario_optimization.py), so it was removed.
_STALL_MARGIN = 1.10

def _stall_speed_ms(mass_kg: float, aero: Any) -> float:
    """The project stall model's 1-g stall speed (``aircraft.aero_params`` is the source)."""
    return stall_speed_ms(mass_kg, wing_area_m2=aero.S, cl_max=aero.Cl_max)


# ── The core: optimize one scenario into two state sequences ──────────────────

def optimize_scenario(
    scenario: FlightScenario,
    *,
    n_segments: int = DEFAULT_N_SEGMENTS,
    max_duration: float = DEFAULT_MAX_DURATION_S,
    rollout_dt_s: float = DEFAULT_ROLLOUT_DT_S,
    fitting: str = DEFAULT_FITTING,
    state_substeps: int | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    verbose: bool = False,
) -> ScenarioOptimization:
    """Optimize ``scenario`` and return its optimizer + simulator state sequences.

    ``fitting`` picks the transcription (a :data:`FITTING_SCHEMES` key).
    ``state_substeps`` fixes the per-control-segment state density M (``None`` =
    auto: ~3 s state step, capped at 16 — ``components.select_state_substeps``).
    """
    initial = scenario.initial
    target = scenario.target
    aircraft, aero = scenario.dynamics("the optimizer")
    if target is None:
        raise ValueError(
            "scenario has no target state; build scenarios with flight_scenarios (its "
            "build_scenario populates target) before optimizing."
        )

    # Run the optimizer (initial -> target). Floor the velocity at a stall margin (not Vref)
    # so observed touchdown-speed targets are admissible.
    min_speed_ms = _STALL_MARGIN * _stall_speed_ms(initial.m, aero)
    # Default fitting is Hermite-Simpson (4th order), matching the constrained path and
    # the frontend default. Trapezoidal (2nd order) produced node-feasible plans whose
    # TRUE-dynamics replays drifted km-scale on aggressive min-time floor-riding solves —
    # the evaluation gates judge the replay, so the batch success rate collapsed (KRDU
    # runway: 14% success with a 0.00 m plan-vs-target error but 5-15 km rollout-vs-target
    # error; the same flight re-solved with HS lands 3.4 m out).
    optimizer = CollocationOptimizer(
        aircraft,
        scheme=_scheme_for_fitting(fitting),
        n_segments=n_segments,
        max_duration=max_duration,
        min_speed_ms=min_speed_ms,
        state_substeps=state_substeps,
        max_iterations=max_iterations,
        verbose=verbose,
    )
    final_time, node_control, _node_endpoints = optimizer.optimize_free_time(
        initial, target, max_duration)
    #   • node_control rows are [thrust_N, bank_rad, load_factor]
    #   • optimizer.last_dense_states_geo: the DENSE (N*M, 6) collocation nodes the optimiser
    #     actually solved — [lat, lon, alt, V, psi, gamma] per row. We export these (prefixed
    #     with the initial state at t=0) so the planned trajectory is smooth; the returned
    #     N segment endpoints alone draw as a coarse, kinked polyline.
    #
    initial_row = [initial.latitude, initial.longitude, initial.altitude,
                   initial.V, initial.psi, initial.gamma]
    dense_rows = [list(row) for row in optimizer.last_dense_states_geo]
    dense_times = [0.0] + [float(t) for t in optimizer.last_dense_state_times_s]
    optimizer_states = node_states_to_samples([initial_row] + dense_rows, dense_times, initial.m)
    rollout = require_usable_rollout(rollout_controls(
        initial, node_control, final_time, aircraft, dt=rollout_dt_s,
        min_altitude_m=rollout_guard_altitude_m(target.altitude),
    ))
    return ScenarioOptimization(
        scenario.source, float(final_time),
        optimizer_states,
        [StateSample.from_state(s.t, s.state) for s in rollout],
        evaluation=evaluation_record(
            initial, target, rollout, scenario.source, subject="optimized"
        ),
    )


# ── Batch + IO (wired) ────────────────────────────────────────────────────────

def _optimize_one_scenario(
    payload: tuple[int, FlightScenario, dict[str, Any]],
) -> tuple[int, str, dict[str, Any] | None, dict[str, Any] | None, str | None]:
    """Solve one scenario and return a picklable record (process-pool worker).

    Returns ``(index, flight_id, states_dict | None, eval_dict | None, error | None)``.
    Per-scenario failures are captured (not raised) so one infeasible scenario never
    kills the pool; the parent writes/logs from the returned record. Both dicts are
    pure JSON types so they cross the process boundary cheaply.
    """
    index, scenario, params = payload
    flight_id = flight_key(scenario.source, index)
    try:
        result = optimize_scenario(scenario, **params)
    except Exception as exc:  # noqa: BLE001 — batch tool: skip + log per-scenario failures
        return (index, flight_id, None, None,
                f"{type(exc).__name__}: {str(exc).splitlines()[0][:90]}")
    return (index, flight_id, result.to_dict(), shipped_evaluation(result), None)


def shipped_evaluation(result: ScenarioOptimization) -> dict[str, Any]:
    """The eval record as the worker ships it to the parent: states emptied.

    The parent writes the rollout once (the states file) and points the eval record at
    it via ``states_ref``, so pickling the same ~90 KB array a second time inside the
    eval dict was pure IPC waste. ``final_time_s`` was already read off the array."""
    evaluation = dict(result.evaluation)
    evaluation["states"] = []
    return evaluation


def optimize_scenarios(
    scenarios: list[FlightScenario],
    *,
    output_dir: str | Path,
    n_segments: int = DEFAULT_N_SEGMENTS,
    max_duration: float = DEFAULT_MAX_DURATION_S,
    rollout_dt_s: float = DEFAULT_ROLLOUT_DT_S,
    fitting: str = DEFAULT_FITTING,
    state_substeps: int | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    jobs: int = 0,
    verbose: bool = False,
    scenarios_label: str | None = None,
    references_dir: str | None = None,
    resume: bool = False,
) -> list[Path]:
    """Optimize each scenario (unconstrained) and write one ``*_states.json`` per scenario.

    ``references_dir`` (a directory name under ``output_dir``, see
    :func:`write_reference_records`) makes every eval record — solved and failed —
    carry a ``reference_file`` pointer at its observed-track reference record.
    Batch mechanics (pooling, resume, stale sweep, summary): :func:`run_batch`.
    """
    return run_batch(
        scenarios,
        output_dir=output_dir,
        worker=_optimize_one_scenario,
        params={
            "n_segments": n_segments, "max_duration": max_duration,
            "rollout_dt_s": rollout_dt_s, "fitting": fitting,
            "state_substeps": state_substeps, "max_iterations": max_iterations,
            "verbose": verbose,
        },
        optimization_config=build_optimization_config(
            constrained_iaf=False,
            fitting=fitting,
            n_segments=n_segments,
            n_seg_per_phase=DEFAULT_N_SEG_PER_PHASE,
            state_substeps=state_substeps,
            max_duration_s=max_duration,
            rollout_dt_s=rollout_dt_s,
            max_iterations=max_iterations,
        ),
        mode=None,
        progress="",
        jobs=jobs,
        scenarios_label=scenarios_label,
        references_dir=references_dir,
        resume=resume,
    )


# ── Constrained IAF optimization ──────────────────────────────────────────────
#
# For one scenario, a CONSTRAINED trajectory from its observed start through one IAF of its
# runway's RNAV(GPS) procedure to the runway: the shortest IAF path (``"shortest"``, the batch
# default) or the fastest of all IAFs (``"minTime"``). Each scenario yields exactly one trajectory.
# The legs come from ``procedure.segments.build_constraint_segments``; the solve is the
# **multiphase** optimiser (one phase per leg); the record is ``scenario_replay``'s.

DEFAULT_PROCEDURE_ROOT = _PROCEDURE_ROOT  # flight_scenarios.procedure_final owns the path


@dataclass
class IafSolve:
    """One feasible IAF candidate's solve, kept while searching for the fastest."""
    final_time: float
    pc: Any                 # the IAF→runway ProcedureConstraint (carries the chosen IAF)
    initial: GeodeticState  # the IAF initial state
    controls: Any           # node controls (for the rollout)
    dense_states: Any       # the optimizer's dense planned states
    dense_times: Any        # the dense states' OWN times (non-uniform across phases)
    segment_durations: Any  # per-control-segment durations (multiphase non-uniform)
    decision_vector: Any    # the raw NLP solution: the warm start of a re-solve of the same IAF


def _resolve_procedure_path(procedure_root: str | Path, airport: str, runway: str) -> Path:
    """Path to the runway's RNAV(GPS) procedure detail document, via the airport's index.json.

    Resolution lives in ``flight_scenarios.procedure_final`` so the learned model's FAF
    read and this batch read the same document the same way."""
    return rnav_gps_procedure_path(airport, runway, root=procedure_root)


def _require_procedure_threshold_agrees(target: GeodeticState, paths: list) -> float:
    """Check the procedure ends where the scenario target is; return the gap in metres.

    The constrained solve is anchored on ``target`` (``TargetFrame`` origin = LTP), so the
    procedure's LAST waypoint — the CIFP landing threshold every IAF path of one procedure
    shares — must describe the SAME point. It is a second rendering of the same CIFP datum
    the arrival manifest's ``runway_target`` carries, and the two round differently:
    measured over the 25 runways in service here the gap is 0.05–0.22 m (KRDU 32 = 2.98 m,
    KSMF 35R = 39.45 m, neither in the arrival set).

    This USED to snap ``target`` horizontally onto the waypoint, which is why constrained
    records were rejected by ``evaluation.arrival._require_target_agrees_with_runway_data``
    (1 cm) once that check gained its position half on 2026-08-17: the snap moved an
    already-authoritative target off the threshold the evaluator grades against. The target
    is now left alone — the optimizer, the evaluator and the arrival manifest all use
    ``harvest.airports.Runway`` — and the procedure is checked against it instead. The
    tolerance is the optimizer's own ``_FRAME_ANCHOR_TOLERANCE_M``, i.e. exactly the
    displaced-threshold mis-anchor (KSJC 12L was 390 m against the NASR config) the snap was
    introduced to catch, so nothing that used to be caught stops being caught.
    """
    runway_wp = paths[0].waypoints[-1]
    gap_m = haversine_m(
        target.latitude, target.longitude, runway_wp.lat_deg, runway_wp.lon_deg
    )
    if gap_m > _FRAME_ANCHOR_TOLERANCE_M:
        raise ValueError(
            f"procedure threshold '{runway_wp.ident}' is {gap_m:.1f} m from the scenario "
            f"target (limit {_FRAME_ANCHOR_TOLERANCE_M:.0f} m); the scenario and the "
            "procedure were built against different runway data"
        )
    return gap_m


def iaf_setup(scenario: FlightScenario, procedure_root: str | Path):
    """Shared prologue for the IAF optimizers: resolve the runway's RNAV(GPS) procedure and return
    ``(target, iaf_paths, aircraft, min_speed_ms)``. ``target`` is the scenario's own
    authoritative threshold state, unmodified — the procedure is validated against it (see
    :func:`_require_procedure_threshold_agrees`). Raises if the procedure / paths are missing."""
    target = scenario.target
    if target is None:
        raise ValueError("scenario has no target state; build it with flight_scenarios first.")
    runway = scenario.source["runway"]
    apt = scenario.source["arr_airport"]
    document = json.loads(
        _resolve_procedure_path(procedure_root, apt, runway).read_text(encoding="utf-8")
    )
    paths = iaf_paths(document)
    if not paths:
        raise ValueError(f"no IAF->runway paths in the procedure for {apt} {runway}")
    _require_procedure_threshold_agrees(target, paths)
    aircraft, aero = scenario.dynamics("the optimizer")
    min_speed_ms = _STALL_MARGIN * _stall_speed_ms(scenario.initial.m, aero)
    return target, paths, aircraft, min_speed_ms


def solve_iaf(
    pc, scenario: FlightScenario, target: GeodeticState, aircraft: Any, min_speed_ms: float,
    *, max_duration: float, verbose: bool,
    fitting: str = DEFAULT_FITTING,
    state_substeps: int | None = None,
    n_seg_per_phase: int = DEFAULT_N_SEG_PER_PHASE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    extra_rows=no_extra_rows,
    initial_guess=None,
    fixed_duration_s: float | None = None,
) -> IafSolve:
    """Full CONSTRAINED solve from the scenario's OBSERVED start to the runway via one IAF path.

    Uses ``procedure.segments.build_constraint_segments`` (constraint geometry) and the **multiphase**
    optimiser. The start is the observed ``scenario.initial`` (so the result is comparable to the
    ADS-B track), NOT a synthetic IAF state: the optimiser flies a free transition from there to the
    procedure's first fix (pre-FAF legs are unpinned, altitude-only), then each procedure leg with
    its corridor / glidepath / floor. Raises on infeasibility. ``n_seg_per_phase`` sets the control
    segments PER leg (the multiphase mesh: n_seg_per_phase × legs). ``extra_rows`` and
    ``initial_guess`` go to the optimizer (the traffic loop's re-solves). ``fixed_duration_s`` makes
    it a fixed-time solve (``optimize_trajectory``): a flight to a controlled time of arrival (M2).
    """
    segments = build_constraint_segments(
        pc, target.latitude, target.longitude, target.altitude,
    )
    start_state = scenario.initial
    optimizer = CollocationOptimizer(
        aircraft, segments=segments,
        scheme=_scheme_for_fitting(fitting),
        n_seg_per_phase=n_seg_per_phase,
        min_speed_ms=min_speed_ms,
        state_substeps=state_substeps,
        max_iterations=max_iterations,
        verbose=verbose,
    )
    if fixed_duration_s is None:
        final_time, node_control, _ = optimizer.optimize_free_time(
            start_state, target, max_duration, initial_guess=initial_guess, extra_rows=extra_rows)
    else:
        final_time, node_control, _ = optimizer.optimize_trajectory(
            start_state, target, fixed_duration_s, initial_guess=initial_guess, extra_rows=extra_rows)
    return IafSolve(
        float(final_time), pc, start_state, node_control,
        optimizer.last_dense_states_geo, list(optimizer.last_dense_state_times_s),
        list(optimizer.segment_durations_s), optimizer.last_decision_vector,
    )


def iaf_result(
    best: IafSolve, scenario: FlightScenario, aircraft: Any,
    *, target: GeodeticState, candidates: int, rollout_dt_s: float, selection: str,
) -> ScenarioOptimization:
    """Assemble the chosen IAF solve into a :class:`ScenarioOptimization` (dense export + rollout).

    ``target`` is the scenario's own authoritative threshold state — the state the
    optimizer flew to AND the one ``evaluation`` grades against, which are the same point
    (see :func:`_require_procedure_threshold_agrees`).
    """
    initial_row = [best.initial.latitude, best.initial.longitude, best.initial.altitude,
                   best.initial.V, best.initial.psi, best.initial.gamma]
    dense_rows = [list(row) for row in best.dense_states]
    # The solver's own node times: multiphase durations are free and the substep count is
    # auto-selected PER PHASE, so an even spread over [0, T] time-warps the plan overlay.
    dense_times = [0.0] + [float(t) for t in best.dense_times]
    optimizer_states = node_states_to_samples(
        [initial_row] + dense_rows, dense_times, best.initial.m,
    )
    rollout = require_usable_rollout(rollout_controls(
        best.initial, best.controls, best.final_time, aircraft, dt=rollout_dt_s,
        segment_durations=best.segment_durations,
        min_altitude_m=rollout_guard_altitude_m(target.altitude),
    ))
    source = {
        **scenario.source,
        "chosenIaf": best.pc.waypoints[0].ident,
        "iafBranchId": best.pc.branch_id,
        "iafCandidates": candidates,
        "iafSelection": selection,
    }
    return ScenarioOptimization(
        source, best.final_time, optimizer_states,
        [StateSample.from_state(s.t, s.state) for s in rollout],
        evaluation=evaluation_record(
            best.initial, target, rollout, source, subject="optimized"
        ),
    )


def optimize_scenario_min_time_iaf(
    scenario: FlightScenario,
    *,
    procedure_root: str | Path = DEFAULT_PROCEDURE_ROOT,
    max_duration: float = DEFAULT_MAX_DURATION_S,
    rollout_dt_s: float = DEFAULT_ROLLOUT_DT_S,
    fitting: str = DEFAULT_FITTING,
    state_substeps: int | None = None,
    n_seg_per_phase: int = DEFAULT_N_SEG_PER_PHASE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    verbose: bool = False,
) -> ScenarioOptimization:
    """Constrained, fastest-IAF optimization for one scenario (one trajectory out).

    The EXACT (slow) IAF selection: solve a CONSTRAINED trajectory from EVERY IAF of the
    scenario's runway RNAV(GPS) procedure and return the single FASTEST (min ``final_time``).
    Infeasible IAFs are skipped; the scenario fails only if every IAF fails. For a cheap
    alternative that solves once, see :func:`optimize_scenario_shortest_iaf`.
    """
    target, paths, aircraft, min_speed_ms = iaf_setup(scenario, procedure_root)

    best: IafSolve | None = None
    attempts: list[tuple[str, str]] = []
    for pc in paths:
        try:
            candidate = solve_iaf(
                pc, scenario, target, aircraft, min_speed_ms,
                max_duration=max_duration, verbose=verbose,
                fitting=fitting, state_substeps=state_substeps,
                n_seg_per_phase=n_seg_per_phase, max_iterations=max_iterations,
            )
        except Exception as exc:  # noqa: BLE001 — try the next IAF; fail only if all IAFs fail
            attempts.append((pc.waypoints[0].ident, type(exc).__name__))
            continue
        if best is None or candidate.final_time < best.final_time:
            best = candidate

    if best is None:
        raise ValueError(
            f"all {len(paths)} IAF(s) infeasible for "
            f"{scenario.source.get('id')}: {attempts[:4]}"
        )
    return iaf_result(
        best, scenario, aircraft,
        target=target, candidates=len(paths), rollout_dt_s=rollout_dt_s, selection="minTime",
    )


def shortest_iaf_solve(
    scenario: FlightScenario, target: GeodeticState, paths: list, aircraft: Any, min_speed_ms: float,
    **solve_options: Any,
) -> IafSolve:
    """The solve of the shortest feasible IAF path (shortest horizontal polyline first; an
    infeasible one falls through to the next). Raises when every IAF fails."""
    attempts: list[tuple[str, str]] = []
    for pc in sorted(paths, key=path_length_m):    # shortest path first
        try:
            return solve_iaf(pc, scenario, target, aircraft, min_speed_ms, **solve_options)
        except Exception as exc:  # noqa: BLE001 — fall through to the next-shortest IAF
            attempts.append((pc.waypoints[0].ident, type(exc).__name__))
    raise ValueError(
        f"all {len(paths)} IAF(s) infeasible (shortest-first) for "
        f"{scenario.source.get('id')}: {attempts[:4]}"
    )


def optimize_scenario_shortest_iaf(
    scenario: FlightScenario,
    *,
    procedure_root: str | Path = DEFAULT_PROCEDURE_ROOT,
    max_duration: float = DEFAULT_MAX_DURATION_S,
    rollout_dt_s: float = DEFAULT_ROLLOUT_DT_S,
    fitting: str = DEFAULT_FITTING,
    state_substeps: int | None = None,
    n_seg_per_phase: int = DEFAULT_N_SEG_PER_PHASE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    verbose: bool = False,
) -> ScenarioOptimization:
    """Cheap, naive IAF selection: pick the IAF whose horizontal polyline path to the runway is
    SHORTEST, then run the full constrained optimization once for it.

    Avoids :func:`optimize_scenario_min_time_iaf`'s solve-every-IAF cost: the IAF is chosen by a
    pure-geometry path length (no NLP), so the common case is a single solve. It is greedy /
    robust — if the shortest IAF turns out infeasible it falls through to the next-shortest, so
    the scenario fails only if every IAF fails. The exact full-search remains available above.
    """
    target, paths, aircraft, min_speed_ms = iaf_setup(scenario, procedure_root)
    best = shortest_iaf_solve(
        scenario, target, paths, aircraft, min_speed_ms,
        max_duration=max_duration, verbose=verbose,
        fitting=fitting, state_substeps=state_substeps,
        n_seg_per_phase=n_seg_per_phase, max_iterations=max_iterations,
    )
    return iaf_result(
        best, scenario, aircraft,
        target=target, candidates=len(paths), rollout_dt_s=rollout_dt_s,
        selection="shortestPath",
    )


# Selection strategies for the per-scenario IAF optimization (picked by name in the batch/CLI).
_IAF_SELECTORS = {
    "minTime": optimize_scenario_min_time_iaf,    # exact: solve every IAF, keep the fastest
    "shortest": optimize_scenario_shortest_iaf,   # naive: shortest horizontal path, one solve
}


def _optimize_one_scenario_iaf(
    payload: tuple[int, FlightScenario, dict[str, Any]],
) -> tuple[int, str, dict[str, Any] | None, dict[str, Any] | None, str | None]:
    """Process-pool worker for the constrained-IAF batch; ``params['selection']`` chooses the
    per-scenario strategy (mirrors ``_optimize_one_scenario``)."""
    index, scenario, params = payload
    params = dict(params)
    selector = _IAF_SELECTORS[params.pop("selection")]
    flight_id = flight_key(scenario.source, index)
    try:
        result = selector(scenario, **params)
    except Exception as exc:  # noqa: BLE001 — skip + log per-scenario failures
        return (index, flight_id, None, None,
                f"{type(exc).__name__}: {str(exc).splitlines()[0][:90]}")
    return (index, flight_id, result.to_dict(), shipped_evaluation(result), None)


def optimize_scenarios_constrained_iaf(
    scenarios: list[FlightScenario],
    *,
    output_dir: str | Path,
    selection: str = "shortest",
    procedure_root: str | Path = DEFAULT_PROCEDURE_ROOT,
    max_duration: float = DEFAULT_MAX_DURATION_S,
    rollout_dt_s: float = DEFAULT_ROLLOUT_DT_S,
    fitting: str = DEFAULT_FITTING,
    state_substeps: int | None = None,
    n_seg_per_phase: int = DEFAULT_N_SEG_PER_PHASE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    jobs: int = 0,
    verbose: bool = False,
    scenarios_label: str | None = None,
    references_dir: str | None = None,
    resume: bool = False,
) -> list[Path]:
    """Batch constrained-IAF optimization — one trajectory per scenario, IAF chosen by ``selection``.

    ``selection``: ``"minTime"`` solves every IAF and keeps the fastest (exact, slow);
    ``"shortest"`` picks the shortest horizontal polyline path and solves once (naive, fast). Same
    output shape/IO as :func:`optimize_scenarios` — both are thin fronts over
    :func:`run_batch` — and each solved row/record reports the chosen IAF.
    """
    if selection not in _IAF_SELECTORS:
        raise ValueError(f"unknown selection {selection!r}; choose from {sorted(_IAF_SELECTORS)}")
    return run_batch(
        scenarios,
        output_dir=output_dir,
        worker=_optimize_one_scenario_iaf,
        params={
            "selection": selection,
            "procedure_root": str(procedure_root), "max_duration": max_duration,
            "rollout_dt_s": rollout_dt_s, "fitting": fitting,
            "state_substeps": state_substeps, "n_seg_per_phase": n_seg_per_phase,
            "max_iterations": max_iterations, "verbose": verbose,
        },
        optimization_config=build_optimization_config(
            constrained_iaf=True,
            fitting=fitting,
            n_segments=DEFAULT_N_SEGMENTS,   # a constrained recipe records only its per-phase mesh
            n_seg_per_phase=n_seg_per_phase,
            state_substeps=state_substeps,
            max_duration_s=max_duration,
            rollout_dt_s=rollout_dt_s,
            max_iterations=max_iterations,
            iaf_selection=selection,
        ),
        mode=f"constrainedIaf:{selection}",
        progress=f" [constrained IAF: {selection}]",
        jobs=jobs,
        scenarios_label=scenarios_label,
        references_dir=references_dir,
        resume=resume,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize flight scenarios -> state JSON files")
    parser.add_argument("--scenarios", required=True, help="Scenario JSON from flight_scenarios")
    parser.add_argument("--output-dir", required=True, help="Where to write the *_states.json files")
    parser.add_argument("--n-segments", type=int, default=DEFAULT_N_SEGMENTS,
                        help="unconstrained: control segments over the whole trajectory")
    parser.add_argument("--n-seg-per-phase", type=int, default=DEFAULT_N_SEG_PER_PHASE,
                        help="constrained-iaf: control segments PER procedure leg (the "
                             "multiphase mesh; unconstrained runs ignore it)")
    parser.add_argument("--max-duration", type=float, default=DEFAULT_MAX_DURATION_S)
    parser.add_argument("--rollout-dt", type=float, default=DEFAULT_ROLLOUT_DT_S)
    parser.add_argument(
        "--state-substeps", type=int, default=None,
        help="state-collocation subintervals per control segment (M; state nodes = N*M). "
             "Default: auto per phase — a ~3 s state step, capped at 16")
    parser.add_argument(
        "--fitting", choices=sorted(FITTING_SCHEMES), default=DEFAULT_FITTING,
        help="transcription fitting for the solves: 'hs' = Hermite-Simpson (4th order, "
             "default) or 'trapezoidal' (2nd order; its replays drift km-scale on "
             "aggressive min-time solves — kept for comparison runs)")
    parser.add_argument(
        "--jobs", type=int, default=0,
        help="parallel worker processes (0 = auto: half the CPU cores; 1 = serial)",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS,
        help=f"IPOPT iteration cap per solve (default {DEFAULT_MAX_ITERATIONS}). A scenario "
             "that will not converge pays this in full before being skipped — measured at "
             "~13x the cost of a successful solve, ~48%% of an unconstrained batch's CPU for "
             "6.7%% of its flights. Lowering it trades slow successes for failures",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="keep per-flight records already on disk for scenarios in THIS roster and "
             "solve only the rest (records for flights not in the roster are still swept). "
             "Without it every record is recomputed from scratch",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="show the full IPOPT solver log (per-iteration table); default is quiet "
             "(best paired with --jobs 1, since parallel logs interleave)",
    )
    parser.add_argument(
        "--reference-tracks", default=None,
        help="the harvest arrivals/manifest.json the scenarios came from; when given, "
             "reference eval records are written FIRST "
             "(observed tracks -> --references-dir) and every eval record "
             "points at its reference via reference_file",
    )
    parser.add_argument(
        "--references-dir",
        default=REFERENCES_DIR,
        help="reference directory relative to --output-dir; sibling paths such as "
             "../shared_references/runway let compatible categories share one canonical set",
    )
    parser.add_argument(
        "--constrained-iaf", action="store_true",
        help="constrained-IAF mode: per scenario, optimize from its runway's RNAV(GPS) procedure "
             "IAFs with path constraints and keep one trajectory (IAF chosen by --iaf-selection)",
    )
    parser.add_argument(
        "--iaf-selection", choices=sorted(_IAF_SELECTORS), default="shortest",
        help="how to pick the IAF (constrained-iaf mode): 'shortest' (default) picks the shortest "
             "horizontal path and solves once (fast); 'minTime' solves every IAF and keeps the fastest "
             "(exact, slow)",
    )
    parser.add_argument(
        "--procedure-root", default=str(DEFAULT_PROCEDURE_ROOT),
        help="root holding <ICAO>/procedure-details (constrained-iaf mode)",
    )
    args = parser.parse_args()

    if args.state_substeps is not None and args.state_substeps < 1:
        parser.error(f"--state-substeps must be >= 1, got {args.state_substeps}")
    if args.n_seg_per_phase < 1:
        parser.error(f"--n-seg-per-phase must be >= 1, got {args.n_seg_per_phase}")
    if args.max_iterations < 1:
        parser.error(f"--max-iterations must be >= 1, got {args.max_iterations}")
    scenarios = load_scenarios(args.scenarios)
    # Reference eval records come FIRST (the observed baseline exists whether or not a
    # solve succeeds); the batch then points every eval record at its reference.
    references_dir = None
    if args.reference_tracks:
        if args.references_dir != REFERENCES_DIR:
            remove_legacy_category_references(args.output_dir)
        source_signature = {
            "scenarios_sha256": _file_sha256(args.scenarios),
            "arrivals_manifest_sha256": _file_sha256(args.reference_tracks),
        }
        write_reference_records(
            scenarios,
            args.reference_tracks,
            output_dir=args.output_dir,
            references_dir=args.references_dir,
            source_signature=source_signature,
        )
        references_dir = args.references_dir
    if args.constrained_iaf:
        paths = optimize_scenarios_constrained_iaf(
            scenarios,
            output_dir=args.output_dir,
            selection=args.iaf_selection,
            procedure_root=args.procedure_root,
            max_duration=args.max_duration,
            rollout_dt_s=args.rollout_dt,
            fitting=args.fitting,
            state_substeps=args.state_substeps,
            n_seg_per_phase=args.n_seg_per_phase,
            max_iterations=args.max_iterations,
            jobs=args.jobs,
            verbose=args.verbose,
            scenarios_label=args.scenarios,
            references_dir=references_dir,
            resume=args.resume,
        )
    else:
        paths = optimize_scenarios(
            scenarios,
            output_dir=args.output_dir,
            n_segments=args.n_segments,
            max_duration=args.max_duration,
            rollout_dt_s=args.rollout_dt,
            fitting=args.fitting,
            state_substeps=args.state_substeps,
            max_iterations=args.max_iterations,
            jobs=args.jobs,
            verbose=args.verbose,
            scenarios_label=args.scenarios,
            references_dir=references_dir,
            resume=args.resume,
        )
    print(f"✓ wrote {len(paths)} state file(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
