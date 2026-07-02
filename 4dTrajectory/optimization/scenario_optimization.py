"""Teaching scaffold: optimize flight scenarios, dump optimizer-vs-simulator states.

For each :class:`FlightScenario` (from the ``flight_scenarios`` package) this:

  1. solves the trajectory optimization from the scenario's initial state to its target,
  2. rolls the optimizer's piecewise-constant controls forward through the **real**
     simulator (the same dynamics the live sim uses).

It writes **one JSON per scenario** with both state sequences:

    { source, final_time_s,
      optimizer_states: [ {t, lat, lon, alt, V, psi, gamma, m}, … ],   # the NLP's plan
      simulator_states: [ {t, lat, lon, alt, V, psi, gamma, m}, … ] }  # the real rollout

The two sequences are NOT identical — the optimizer's node states are the idealized plan,
the simulator states are what the dynamics actually do under those controls. The gap is the
point of the exercise (and what the CZML builder in ``aeroviz-4d/python`` visualizes).

This is a **teaching scaffold**: the loop / IO / serialization / CLI are wired; the two core
steps — running the optimizer (TODO ①) and the forward rollout (TODO ②) — are documented
TODOs. See ``flight_scenarios/README.md`` and the comments below.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_OPT_DIR = Path(__file__).resolve().parent
if str(_OPT_DIR) not in sys.path:
    sys.path.insert(0, str(_OPT_DIR))

from flight_scenarios import FlightScenario, load_scenarios  # noqa: E402
from aerodynamic_model.common import GeodeticState, LoadFactorControl  # noqa: E402
from aerodynamic_model.casadi_simulator import CasadiSimulator  # noqa: E402
from aerodynamic_model.rollout import rollout_piecewise_constant  # noqa: E402
from collocation import CollocationOptimizer  # noqa: E402

# Optimizer + rollout defaults (override on the CLI).
DEFAULT_N_SEGMENTS = 8
DEFAULT_DT = 1.0                 # optimizer dt (API parity; state mesh is auto-selected)
DEFAULT_MAX_DURATION_S = 2000.0  # free-time upper bound; the solver minimises T below this
DEFAULT_ROLLOUT_DT_S = 0.5       # forward-integration step for the simulator rollout

# Velocity floor = STALL_MARGIN x stall speed (at the scenario's landing mass), so the
# optimizer admits realistic touchdown-speed targets instead of forcing V >= Vref. Capped at
# Vref so it never raises the optimizer's default floor.
_STALL_MARGIN = 1.10
_RHO_SEA_LEVEL = 1.225
_GRAVITY = 9.81


def _stall_speed_ms(mass_kg: float, aero: Any) -> float:
    """Level-flight stall speed: V = sqrt(2 m g / (rho S Cl_max))."""
    return math.sqrt(2.0 * mass_kg * _GRAVITY / (_RHO_SEA_LEVEL * aero.S * aero.Cl_max))


# ── Output records (plumbing) ─────────────────────────────────────────────────

@dataclass
class StateSample:
    t: float
    lat: float
    lon: float
    alt: float
    V: float
    psi: float
    gamma: float
    m: float

    @classmethod
    def from_state(cls, t: float, state: GeodeticState) -> "StateSample":
        return cls(
            t=t,
            lat=state.latitude,
            lon=state.longitude,
            alt=state.altitude,
            V=state.V,
            psi=state.psi,
            gamma=state.gamma,
            m=state.m,
        )


@dataclass
class ScenarioOptimization:
    source: dict[str, Any]
    final_time_s: float
    optimizer_states: list[StateSample]
    simulator_states: list[StateSample]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "final_time_s": self.final_time_s,
            "optimizer_states": [asdict(s) for s in self.optimizer_states],
            "simulator_states": [asdict(s) for s in self.simulator_states],
        }


# ── The core: optimize one scenario into two state sequences ──────────────────

def optimize_scenario(
    scenario: FlightScenario,
    *,
    n_segments: int = DEFAULT_N_SEGMENTS,
    dt: float = DEFAULT_DT,
    max_duration: float = DEFAULT_MAX_DURATION_S,
    rollout_dt_s: float = DEFAULT_ROLLOUT_DT_S,
    verbose: bool = False,
) -> ScenarioOptimization:
    """Optimize ``scenario`` and return its optimizer + simulator state sequences."""
    initial = scenario.initial
    target = scenario.target
    aircraft = scenario.aircraft
    if target is None:
        raise ValueError(
            "scenario has no target state; build scenarios with flight_scenarios (its "
            "build_scenario populates target) before optimizing."
        )

    # Run the optimizer (initial -> target). Floor the velocity at a stall margin (not Vref)
    # so observed touchdown-speed targets are admissible; never above the aircraft's Vref.
    min_speed_ms = min(
        _STALL_MARGIN * _stall_speed_ms(initial.m, scenario.aero),
        aircraft.approach.reference_speed_ms,
    )
    optimizer = CollocationOptimizer(
        aircraft,
        scheme="trapezoidalNormalizedFullTransport",
        n_segments=n_segments,
        max_duration=max_duration,
        min_speed_ms=min_speed_ms,
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
    optimizer_states = _node_states_to_samples([initial_row] + dense_rows, final_time, initial.m)
    simulator_states = simulate_controls(initial, node_control, final_time, aircraft,
                                             dt=rollout_dt_s)
    return ScenarioOptimization(scenario.source, float(final_time),
                                   optimizer_states, simulator_states)


def _node_states_to_samples(
    node_state: Any, final_time: float, mass: float
) -> list[StateSample]:
    """Reshape the optimizer's node states into timed samples (provided plumbing).

    The boundary nodes are evenly spaced in time over ``[0, final_time]``.
    """
    states = list(node_state)
    count = len(states)
    samples: list[StateSample] = []
    for index, values in enumerate(states):
        lat, lon, alt, V, psi, gamma = (float(v) for v in values)
        t = (index / (count - 1)) * final_time if count > 1 else 0.0
        samples.append(StateSample(t=t, lat=lat, lon=lon, alt=alt, V=V, psi=psi, gamma=gamma, m=mass))
    return samples


def simulate_controls(
    initial_state: GeodeticState,
    node_control: Any,
    final_time: float,
    aircraft: Any,
    *,
    dt: float = DEFAULT_ROLLOUT_DT_S,
    segment_durations: Any = None,
) -> list[StateSample]:
    """Roll the piecewise-constant optimizer controls through the REAL simulator.

    This produces the "simulator real states": the optimizer's own controls integrated
    through the actual dynamics, which differ from the optimizer's node states (the plan).

    Thin adapter over ``aerodynamic_model.rollout_piecewise_constant`` — builds the
    load-factor controls + the simulator, runs the shared rollout (truncating silently if
    the replay leaves the envelope — this is a viz aid), and maps each neutral
    ``(t, state)`` sample onto a serializable :class:`StateSample`. ``segment_durations``
    (one per control) drives the multiphase non-uniform schedule; ``None`` = equal segments.
    """
    controls = [
        LoadFactorControl(thrust=float(row[0]), bank_rad=float(row[1]),
                          load_factor=float(row[2]))
        for row in node_control
    ]
    sim = CasadiSimulator(aircraft, dt)
    samples = rollout_piecewise_constant(
        sim, initial_state, controls, final_time,
        integrator_dt=dt,
        segment_durations=list(segment_durations) if segment_durations is not None else None,
        truncate_on_envelope_exit=True,
    )
    return [StateSample.from_state(s.t, s.state) for s in samples]

# ── Batch + IO (wired) ────────────────────────────────────────────────────────

def _optimize_one_scenario(
    payload: tuple[int, FlightScenario, dict[str, Any]],
) -> tuple[int, str, dict[str, Any] | None, str | None]:
    """Solve one scenario and return a picklable record (process-pool worker).

    Returns ``(index, flight_id, result_dict | None, error | None)``. Per-scenario
    failures are captured (not raised) so one infeasible scenario never kills the
    pool; the parent writes/logs from the returned record. The result is the plain
    ``to_dict()`` (pure JSON types) so it crosses the process boundary cheaply.
    """
    index, scenario, params = payload
    flight_id = scenario.source.get("id") or f"scenario{index}"
    try:
        result = optimize_scenario(scenario, **params)
    except Exception as exc:  # noqa: BLE001 — batch tool: skip + log per-scenario failures
        return (index, flight_id, None,
                f"{type(exc).__name__}: {str(exc).splitlines()[0][:90]}")
    return (index, flight_id, result.to_dict(), None)


def _resolve_jobs(jobs: int, n_tasks: int) -> int:
    """Resolve the requested worker count (``0`` ⇒ auto = half the CPU cores).

    Auto leaves cores free for other work; capped at the number of scenarios so we
    never spawn idle workers.
    """
    workers = jobs if jobs and jobs > 0 else max(1, (os.cpu_count() or 2) // 2)
    return max(1, min(workers, n_tasks)) if n_tasks else 1


def _limit_solver_threads() -> None:
    """Pin each worker's BLAS/OpenMP pools to one thread to avoid oversubscription.

    With ``spawn`` (the macOS default) child processes inherit ``os.environ``, so
    setting these BEFORE the pool is created makes every worker import numpy / casadi
    single-threaded. Without it, N worker processes would each spin up a full BLAS
    thread pool and fight over the cores — *slowing* the batch down instead of
    speeding it up. ``setdefault`` respects any value the caller already exported.
    """
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")


def optimize_scenarios(
    scenarios: list[FlightScenario],
    *,
    output_dir: str | Path,
    n_segments: int = DEFAULT_N_SEGMENTS,
    dt: float = DEFAULT_DT,
    max_duration: float = DEFAULT_MAX_DURATION_S,
    rollout_dt_s: float = DEFAULT_ROLLOUT_DT_S,
    jobs: int = 0,
    verbose: bool = False,
    scenarios_label: str | None = None,
) -> list[Path]:
    """Optimize each scenario and write one ``*_states.json`` per scenario.

    Each scenario is an independent NLP solve, so they run across a process pool
    (``jobs`` workers; ``0`` ⇒ half the CPU cores). Processes — not threads — because
    the IPOPT solve is CPU-bound C++; a pool sidesteps the GIL entirely. All file IO
    and logging stay in the parent (collected as workers finish), so the output is the
    same regardless of worker count, only the per-scenario order differs.

    Infeasible / failed scenarios are **skipped and logged** (a real landings file mixes
    feasible approaches with too-slow or noisy ones), so one bad scenario never aborts the
    batch. A summary of failures is printed at the end.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    params: dict[str, Any] = {
        "n_segments": n_segments, "dt": dt, "max_duration": max_duration,
        "rollout_dt_s": rollout_dt_s, "verbose": verbose,
    }
    payloads = [(index, scenario, params) for index, scenario in enumerate(scenarios)]
    workers = _resolve_jobs(jobs, len(scenarios))

    written: list[Path] = []
    failures: list[tuple[str, str]] = []
    records: dict[int, dict[str, Any]] = {}  # index -> summary record (parallel-safe ordering)

    def _handle(record: tuple[int, str, dict[str, Any] | None, str | None]) -> None:
        index, flight_id, result_dict, error = record
        if error is not None:
            failures.append((flight_id, error))
            records[index] = _summary_record(
                scenarios[index], status="failed", states_file=None, final_time_s=None, reason=error
            )
            print(f"✗ {flight_id}: skipped ({error.split(':', 1)[0]})")
            return
        name = _scenario_filename(scenarios[index], index)
        path = out / name
        path.write_text(json.dumps(result_dict, indent=2), encoding="utf-8")
        written.append(path)
        records[index] = _summary_record(
            scenarios[index], status="solved", states_file=name,
            final_time_s=float(result_dict["final_time_s"]), reason=None,
        )
        print(
            f"✓ {path.name}: optimizer {len(result_dict['optimizer_states'])} states, "
            f"simulator {len(result_dict['simulator_states'])} states, "
            f"T={result_dict['final_time_s']:.1f}s"
        )

    if workers == 1:
        for payload in payloads:
            _handle(_optimize_one_scenario(payload))
    else:
        _limit_solver_threads()
        print(f"… solving {len(scenarios)} scenario(s) across {workers} worker process(es)")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_optimize_one_scenario, payload) for payload in payloads]
            for future in as_completed(futures):
                _handle(future.result())

    # One summary per run: the failure rate + which flights solved/failed, with the ids
    # needed to find each flight's reference (e.g. for the comparison CZML).
    total = len(scenarios)
    summary = {
        "scenarios": scenarios_label,
        "total": total,
        "solved": len(written),
        "failed": len(failures),
        "failure_rate": (len(failures) / total) if total else 0.0,
        "results": [records[i] for i in sorted(records)],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if failures:
        print(f"\n⚠ {len(failures)}/{len(scenarios)} scenario(s) skipped:")
        for flight_id, reason in failures[:15]:
            print(f"    {flight_id}: {reason}")
        if len(failures) > 15:
            print(f"    … and {len(failures) - 15} more")
    print(f"✓ solved {len(written)}/{total} scenario(s) "
          f"(failure rate {summary['failure_rate']:.1%}) -> {out}")
    print(f"  summary -> {out / 'summary.json'}")
    return written


def _summary_record(
    scenario: FlightScenario,
    *,
    status: str,
    states_file: str | None,
    final_time_s: float | None,
    reason: str | None,
) -> dict[str, Any]:
    """One summary row: the flight's identity (so its reference can be found by id) + status."""
    src = scenario.source
    return {
        "id": src.get("id"),
        "callsign": src.get("callsign"),
        "icao24": src.get("icao24"),
        "arr_airport": src.get("arr_airport"),
        "runway": src.get("runway"),
        "target_source": src.get("target_source"),
        "status": status,
        "states_file": states_file,
        "final_time_s": final_time_s,
        "reason": reason,
    }


def _compact_time(iso: str | None) -> str | None:
    """``2026-06-18T21:37:36Z`` -> ``20260618T213736Z`` (a filename-safe stamp); None if absent."""
    if not iso:
        return None
    return re.sub(r"[^0-9TZ]", "", iso)


def _scenario_filename(scenario: FlightScenario, index: int) -> str:
    """A unique, stable filename for one scenario's states JSON.

    A callsign + runway is NOT unique: the same aircraft can land more than once, and a
    callsign can recur across days — so ``id_runway`` silently overwrote sibling scenarios.
    The name now keys on the full identity ``id_runway_icao24_landingTime``, which is unique
    per scenario in the dataset. Missing fields are skipped; ``index`` is the final fallback.
    """
    src = scenario.source
    parts = [str(src.get("id") or f"scenario{index}")]
    for value in (src.get("runway"), src.get("icao24"), _compact_time(src.get("landing_time_utc"))):
        if value:
            parts.append(str(value))
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", "_".join(parts))
    return f"{safe}_states.json"


# ── Constrained, min-time IAF optimization (NEW) ──────────────────────────────
#
# A wrapper around the per-IAF solve: for one scenario, optimize a CONSTRAINED trajectory
# from each of its runway RNAV(GPS) procedure's IAFs to the runway, and keep the single
# FASTEST (min final_time). Each scenario still yields exactly one trajectory.
#
# Everything heavy is reused: the canonical ``ProcedureConstraint`` + the backend's
# ``build_constraint_segments`` (constraint geometry), the **multiphase** optimiser (one phase per
# leg, fixes pinned, exact per-leg constraints), and this module's dense-export / rollout / batch-IO
# helpers. The functions above are untouched.

DEFAULT_PROCEDURE_ROOT = _REPO_ROOT / "aeroviz-4d" / "public" / "data" / "airports"
DEFAULT_AIRPORT = "KRDU"


@dataclass
class _IafSolve:
    """One feasible IAF candidate's solve, kept while searching for the fastest."""
    final_time: float
    pc: Any                 # the IAF→runway ProcedureConstraint (carries the chosen IAF)
    initial: GeodeticState  # the IAF initial state
    controls: Any           # node controls (for the rollout)
    dense_states: Any       # the optimizer's dense planned states
    segment_durations: Any  # per-control-segment durations (multiphase non-uniform)


def _resolve_procedure_path(procedure_root: str | Path, airport: str, runway: str) -> Path:
    """Path to the runway's RNAV(GPS) procedure detail document, via the airport's index.json."""
    root = Path(procedure_root) / airport.upper() / "procedure-details"
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    runway_ident = f"RW{runway.upper()}"
    for entry in index.get("runways", []):
        if entry.get("runwayIdent") != runway_ident:
            continue
        for proc in entry.get("procedures", []):
            if proc.get("procedureFamily") == "RNAV_GPS":
                return root / f"{proc['procedureUid']}.json"
    raise ValueError(f"no RNAV(GPS) procedure for {airport} {runway_ident}")


def _recompute_distances(waypoints: list) -> list:
    """Recompute each waypoint's along-track ``distance_from_start_m`` over a merged path."""
    from geokit import haversine_m
    out: list = []
    cumulative = 0.0
    previous = None
    for wp in waypoints:
        if previous is not None:
            cumulative += haversine_m(previous.lat_deg, previous.lon_deg, wp.lat_deg, wp.lon_deg)
        out.append(replace(wp, distance_from_start_m=round(cumulative, 1)))
        previous = wp
    return out


def _concat_to_runway(trans_pc, final_pc):
    """Join a transition (IAF→connecting fix) to the final (connecting fix→runway) into one
    IAF→runway ``ProcedureConstraint``, recomputing along-track distances. Returns ``None`` when
    the transition does not end at the final's first fix (so it doesn't feed this final)."""
    join = trans_pc.waypoints[-1]
    final_start = final_pc.waypoints[0]
    if join.fix_id != final_start.fix_id and join.ident != final_start.ident:
        return None
    merged = list(trans_pc.waypoints[:-1]) + list(final_pc.waypoints)
    return replace(
        final_pc,                              # glidepath / course / runway / nominal speed
        branch_id=trans_pc.branch_id,          # label the IAF by its transition branch
        waypoints=tuple(_recompute_distances(merged)),
    )


def _iaf_full_paths(document: dict[str, Any]) -> list:
    """All IAF→runway ``ProcedureConstraint``s for a procedure document.

    The final branch's own entry is one IAF; each transition branch is concatenated onto the
    final to form a complete IAF→runway path. Reuses ``ProcedureConstraint.from_detail_document``
    (single-branch) and concatenates here — the only glue this feature adds.
    """
    from aeroviz_backend.procedure_constraint import ProcedureConstraint
    branches = document.get("branches", [])
    final = next((b for b in branches if b.get("branchRole") == "final"), None)
    if final is None:
        return []
    final_pc = ProcedureConstraint.from_detail_document(document, final["branchId"])
    if final_pc is None or len(final_pc.waypoints) < 2:
        return []
    paths = [final_pc]
    for branch in branches:
        if branch.get("branchRole") != "transition":
            continue
        trans_pc = ProcedureConstraint.from_detail_document(document, branch["branchId"])
        if trans_pc is None or not trans_pc.waypoints:
            continue
        merged = _concat_to_runway(trans_pc, final_pc)
        if merged is not None:
            paths.append(merged)
    return paths


def _iaf_initial_state(pc, scenario: FlightScenario) -> GeodeticState:
    """The physics initial state at a path's IAF: coded altitude, the procedure's nominal speed,
    heading toward the next fix, level flight, the scenario's landing mass."""
    from geokit import FT_M, KT_MS, bearing_rad
    iaf, nxt = pc.waypoints[0], pc.waypoints[1]
    alt_ft = iaf.altitude_ref_ft if iaf.altitude_ref_ft is not None else iaf.geometry_alt_ft
    speed_ms = (
        pc.nominal_speed_kt * KT_MS
        if pc.nominal_speed_kt
        else scenario.aircraft.approach.reference_speed_ms
    )
    return GeodeticState(
        latitude=iaf.lat_deg,
        longitude=iaf.lon_deg,
        altitude=(alt_ft or 0.0) * FT_M,
        V=speed_ms,
        psi=bearing_rad(iaf.lat_deg, iaf.lon_deg, nxt.lat_deg, nxt.lon_deg),
        gamma=0.0,
        m=scenario.initial.m,
    )


def _lagrange_eval(xs, ys, query):
    """Evaluate the Lagrange polynomial through ``(xs, ys)`` at ``query`` (vectorized numpy)."""
    import numpy as np
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    query = np.asarray(query, dtype=float)
    out = np.zeros_like(query)
    for i in range(len(xs)):
        basis = np.ones_like(query)
        for j in range(len(xs)):
            if j != i:
                basis *= (query - xs[j]) / (xs[i] - xs[j])
        out += ys[i] * basis
    return out


def _path_curve_length_m(pc) -> float:
    """A cheap 3D path-length proxy for ranking IAFs (NO NLP solve).

    Fits a Lagrange curve through the IAF→runway waypoints in a runway-anchored metric frame
    ``(north, east, alt)`` and returns its arc length. Falls back to the straight 3D polyline
    length if the chord parameterisation is degenerate (coincident waypoints).
    """
    import numpy as np
    from geokit import FT_M, METRES_PER_DEG_LAT, metres_per_deg_lon
    wps = pc.waypoints
    if len(wps) < 2:
        return float("inf")
    lat0, lon0 = wps[-1].lat_deg, wps[-1].lon_deg
    m_per_lon = metres_per_deg_lon(lat0)
    points = np.array([
        [(w.lat_deg - lat0) * METRES_PER_DEG_LAT,
         (w.lon_deg - lon0) * m_per_lon,
         (w.altitude_ref_ft if w.altitude_ref_ft is not None else (w.geometry_alt_ft or 0.0)) * FT_M]
        for w in wps
    ])
    chord = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    if chord[-1] <= 0.0 or np.any(np.diff(chord) <= 0.0):
        return float(chord[-1])
    samples = np.linspace(chord[0], chord[-1], 200)
    curve = np.stack([_lagrange_eval(chord, points[:, dim], samples) for dim in range(3)], axis=1)
    return float(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum())


def _iaf_setup(scenario: FlightScenario, procedure_root: str | Path, airport: str | None):
    """Shared prologue for the IAF optimizers: resolve the runway's RNAV(GPS) procedure and return
    ``(target, iaf_paths, aircraft, min_speed_ms)``. Raises if the procedure / paths are missing."""
    target = scenario.target
    if target is None:
        raise ValueError("scenario has no target state; build it with flight_scenarios first.")
    runway = scenario.source.get("runway")
    if not runway:
        raise ValueError("scenario has no runway; cannot resolve its approach procedure.")
    apt = airport or scenario.source.get("arr_airport") or DEFAULT_AIRPORT
    document = json.loads(
        _resolve_procedure_path(procedure_root, apt, runway).read_text(encoding="utf-8")
    )
    paths = _iaf_full_paths(document)
    if not paths:
        raise ValueError(f"no IAF->runway paths in the procedure for {apt} {runway}")
    aircraft = scenario.aircraft
    min_speed_ms = min(
        _STALL_MARGIN * _stall_speed_ms(scenario.initial.m, scenario.aero),
        aircraft.approach.reference_speed_ms,
    )
    return target, paths, aircraft, min_speed_ms


def _solve_iaf(
    pc, scenario: FlightScenario, target: GeodeticState, aircraft: Any, min_speed_ms: float,
    *, n_segments: int, dt: float, max_duration: float, verbose: bool,
) -> _IafSolve:
    """Full CONSTRAINED solve from the scenario's OBSERVED start to the runway via one IAF path.

    Uses the backend's ``build_constraint_segments`` (constraint geometry) and the **multiphase**
    optimiser. The start is the observed ``scenario.initial`` (so the result is comparable to the
    ADS-B track), NOT a synthetic IAF state: the optimiser flies a free transition from there to the
    procedure's first fix (pre-FAF legs are unpinned, altitude-only), then each procedure leg with
    its corridor / glidepath / floor. Raises on infeasibility. ``n_segments``/``dt`` are accepted
    for CLI/API parity but unused by the multiphase optimiser (it sets its own per-phase mesh).
    """
    from aeroviz_backend.procedure_segments import build_constraint_segments
    segments, _spans = build_constraint_segments(
        pc, target.latitude, target.longitude, target.altitude,
    )
    if not segments:
        raise ValueError("no constraint segments")
    start_state = scenario.initial
    optimizer = CollocationOptimizer(
        aircraft, segments=segments,
        scheme="hermiteSimpsonNormalizedFullTransport",
        min_speed_ms=min_speed_ms,
        verbose=verbose,
    )
    final_time, node_control, _ = optimizer.optimize_free_time(start_state, target, max_duration)
    return _IafSolve(
        float(final_time), pc, start_state, node_control,
        optimizer.last_dense_states_geo, list(optimizer.segment_durations_s),
    )


def _iaf_result(
    best: _IafSolve, scenario: FlightScenario, aircraft: Any,
    *, candidates: int, rollout_dt_s: float, selection: str,
) -> ScenarioOptimization:
    """Assemble the chosen IAF solve into a :class:`ScenarioOptimization` (dense export + rollout)."""
    initial_row = [best.initial.latitude, best.initial.longitude, best.initial.altitude,
                   best.initial.V, best.initial.psi, best.initial.gamma]
    dense_rows = [list(row) for row in best.dense_states]
    optimizer_states = _node_states_to_samples(
        [initial_row] + dense_rows, best.final_time, best.initial.m,
    )
    simulator_states = simulate_controls(
        best.initial, best.controls, best.final_time, aircraft, dt=rollout_dt_s,
        segment_durations=best.segment_durations,
    )
    source = {
        **scenario.source,
        "chosenIaf": best.pc.waypoints[0].ident,
        "iafBranchId": best.pc.branch_id,
        "iafCandidates": candidates,
        "iafSelection": selection,
    }
    return ScenarioOptimization(source, best.final_time, optimizer_states, simulator_states)


def optimize_scenario_min_time_iaf(
    scenario: FlightScenario,
    *,
    procedure_root: str | Path = DEFAULT_PROCEDURE_ROOT,
    airport: str | None = None,
    n_segments: int = DEFAULT_N_SEGMENTS,
    dt: float = DEFAULT_DT,
    max_duration: float = DEFAULT_MAX_DURATION_S,
    rollout_dt_s: float = DEFAULT_ROLLOUT_DT_S,
    verbose: bool = False,
) -> ScenarioOptimization:
    """Constrained, fastest-IAF optimization for one scenario (one trajectory out).

    The EXACT (slow) IAF selection: solve a CONSTRAINED trajectory from EVERY IAF of the
    scenario's runway RNAV(GPS) procedure and return the single FASTEST (min ``final_time``).
    Infeasible IAFs are skipped; the scenario fails only if every IAF fails. For a cheap
    alternative that solves once, see :func:`optimize_scenario_shortest_iaf`.
    """
    target, paths, aircraft, min_speed_ms = _iaf_setup(scenario, procedure_root, airport)

    best: _IafSolve | None = None
    attempts: list[tuple[str, str]] = []
    for pc in paths:
        try:
            candidate = _solve_iaf(
                pc, scenario, target, aircraft, min_speed_ms,
                n_segments=n_segments, dt=dt, max_duration=max_duration, verbose=verbose,
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
    return _iaf_result(
        best, scenario, aircraft,
        candidates=len(paths), rollout_dt_s=rollout_dt_s, selection="minTime",
    )


def optimize_scenario_shortest_iaf(
    scenario: FlightScenario,
    *,
    procedure_root: str | Path = DEFAULT_PROCEDURE_ROOT,
    airport: str | None = None,
    n_segments: int = DEFAULT_N_SEGMENTS,
    dt: float = DEFAULT_DT,
    max_duration: float = DEFAULT_MAX_DURATION_S,
    rollout_dt_s: float = DEFAULT_ROLLOUT_DT_S,
    verbose: bool = False,
) -> ScenarioOptimization:
    """Cheap, naive IAF selection: pick the IAF whose 3D Lagrange-curve path to the runway is
    SHORTEST, then run the full constrained optimization once for it.

    Avoids :func:`optimize_scenario_min_time_iaf`'s solve-every-IAF cost: the IAF is chosen by a
    pure-geometry path length (no NLP), so the common case is a single solve. It is greedy /
    robust — if the shortest IAF turns out infeasible it falls through to the next-shortest, so
    the scenario fails only if every IAF fails. The exact full-search remains available above.
    """
    target, paths, aircraft, min_speed_ms = _iaf_setup(scenario, procedure_root, airport)

    attempts: list[tuple[str, str]] = []
    for pc in sorted(paths, key=_path_curve_length_m):   # shortest 3D path first
        try:
            best = _solve_iaf(
                pc, scenario, target, aircraft, min_speed_ms,
                n_segments=n_segments, dt=dt, max_duration=max_duration, verbose=verbose,
            )
        except Exception as exc:  # noqa: BLE001 — fall through to the next-shortest IAF
            attempts.append((pc.waypoints[0].ident, type(exc).__name__))
            continue
        return _iaf_result(
            best, scenario, aircraft,
            candidates=len(paths), rollout_dt_s=rollout_dt_s, selection="shortestPath",
        )

    raise ValueError(
        f"all {len(paths)} IAF(s) infeasible (shortest-first) for "
        f"{scenario.source.get('id')}: {attempts[:4]}"
    )


# Selection strategies for the per-scenario IAF optimization (picked by name in the batch/CLI).
_IAF_SELECTORS = {
    "minTime": optimize_scenario_min_time_iaf,    # exact: solve every IAF, keep the fastest
    "shortest": optimize_scenario_shortest_iaf,   # naive: shortest 3D path, one solve
}


def _optimize_one_scenario_iaf(
    payload: tuple[int, FlightScenario, dict[str, Any]],
) -> tuple[int, str, dict[str, Any] | None, str | None]:
    """Process-pool worker for the constrained-IAF batch; ``params['selection']`` chooses the
    per-scenario strategy (mirrors ``_optimize_one_scenario``)."""
    index, scenario, params = payload
    params = dict(params)
    selector = _IAF_SELECTORS[params.pop("selection", "shortest")]
    flight_id = scenario.source.get("id") or f"scenario{index}"
    try:
        result = selector(scenario, **params)
    except Exception as exc:  # noqa: BLE001 — skip + log per-scenario failures
        return (index, flight_id, None,
                f"{type(exc).__name__}: {str(exc).splitlines()[0][:90]}")
    return (index, flight_id, result.to_dict(), None)


def optimize_scenarios_constrained_iaf(
    scenarios: list[FlightScenario],
    *,
    output_dir: str | Path,
    selection: str = "shortest",
    procedure_root: str | Path = DEFAULT_PROCEDURE_ROOT,
    airport: str | None = None,
    n_segments: int = DEFAULT_N_SEGMENTS,
    dt: float = DEFAULT_DT,
    max_duration: float = DEFAULT_MAX_DURATION_S,
    rollout_dt_s: float = DEFAULT_ROLLOUT_DT_S,
    jobs: int = 0,
    verbose: bool = False,
    scenarios_label: str | None = None,
) -> list[Path]:
    """Batch constrained-IAF optimization — one trajectory per scenario, IAF chosen by ``selection``.

    ``selection``: ``"minTime"`` solves every IAF and keeps the fastest (exact, slow);
    ``"shortest"`` picks the shortest 3D path and solves once (naive, fast). Same output shape/IO
    as :func:`optimize_scenarios` (``*_states.json`` + ``summary.json``, parallel across
    scenarios), reusing its summary/filename/jobs helpers; each scenario reports the chosen IAF.
    """
    if selection not in _IAF_SELECTORS:
        raise ValueError(f"unknown selection {selection!r}; choose from {sorted(_IAF_SELECTORS)}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    params: dict[str, Any] = {
        "selection": selection,
        "procedure_root": str(procedure_root), "airport": airport,
        "n_segments": n_segments, "dt": dt, "max_duration": max_duration,
        "rollout_dt_s": rollout_dt_s, "verbose": verbose,
    }
    payloads = [(index, scenario, params) for index, scenario in enumerate(scenarios)]
    workers = _resolve_jobs(jobs, len(scenarios))

    written: list[Path] = []
    failures: list[tuple[str, str]] = []
    records: dict[int, dict[str, Any]] = {}

    def _handle(record: tuple[int, str, dict[str, Any] | None, str | None]) -> None:
        index, flight_id, result_dict, error = record
        if error is not None:
            failures.append((flight_id, error))
            records[index] = _summary_record(
                scenarios[index], status="failed", states_file=None, final_time_s=None, reason=error,
            )
            print(f"✗ {flight_id}: skipped ({error.split(':', 1)[0]})")
            return
        name = _scenario_filename(scenarios[index], index)
        (out / name).write_text(json.dumps(result_dict, indent=2), encoding="utf-8")
        written.append(out / name)
        record_row = _summary_record(
            scenarios[index], status="solved", states_file=name,
            final_time_s=float(result_dict["final_time_s"]), reason=None,
        )
        record_row["chosenIaf"] = result_dict["source"].get("chosenIaf")
        records[index] = record_row
        print(f"✓ {name}: IAF {result_dict['source'].get('chosenIaf')}, "
              f"T={result_dict['final_time_s']:.1f}s")

    if workers == 1:
        for payload in payloads:
            _handle(_optimize_one_scenario_iaf(payload))
    else:
        _limit_solver_threads()
        print(f"… solving {len(scenarios)} scenario(s) [constrained IAF: {selection}] "
              f"across {workers} worker process(es)")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_optimize_one_scenario_iaf, payload) for payload in payloads]
            for future in as_completed(futures):
                _handle(future.result())

    total = len(scenarios)
    summary = {
        "scenarios": scenarios_label,
        "mode": f"constrainedIaf:{selection}",
        "total": total,
        "solved": len(written),
        "failed": len(failures),
        "failure_rate": (len(failures) / total) if total else 0.0,
        "results": [records[i] for i in sorted(records)],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if failures:
        print(f"\n⚠ {len(failures)}/{total} scenario(s) skipped:")
        for flight_id, reason in failures[:15]:
            print(f"    {flight_id}: {reason}")
        if len(failures) > 15:
            print(f"    … and {len(failures) - 15} more")
    print(f"✓ solved {len(written)}/{total} scenario(s) [constrained IAF: {selection}] "
          f"(failure rate {summary['failure_rate']:.1%}) -> {out}")
    print(f"  summary -> {out / 'summary.json'}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize flight scenarios -> state JSON files")
    parser.add_argument("--scenarios", required=True, help="Scenario JSON from flight_scenarios")
    parser.add_argument("--output-dir", required=True, help="Where to write the *_states.json files")
    parser.add_argument("--n-segments", type=int, default=DEFAULT_N_SEGMENTS)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    parser.add_argument("--max-duration", type=float, default=DEFAULT_MAX_DURATION_S)
    parser.add_argument("--rollout-dt", type=float, default=DEFAULT_ROLLOUT_DT_S)
    parser.add_argument(
        "--jobs", type=int, default=0,
        help="parallel worker processes (0 = auto: half the CPU cores; 1 = serial)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="show the full IPOPT solver log (per-iteration table); default is quiet "
             "(best paired with --jobs 1, since parallel logs interleave)",
    )
    parser.add_argument(
        "--constrained-iaf", action="store_true",
        help="constrained-IAF mode: per scenario, optimize from its runway's RNAV(GPS) procedure "
             "IAFs with path constraints and keep one trajectory (IAF chosen by --iaf-selection)",
    )
    parser.add_argument(
        "--iaf-selection", choices=("minTime", "shortest"), default="shortest",
        help="how to pick the IAF (constrained-iaf mode): 'shortest' (default) picks the shortest "
             "3D path and solves once (fast); 'minTime' solves every IAF and keeps the fastest "
             "(exact, slow)",
    )
    parser.add_argument(
        "--procedure-root", default=str(DEFAULT_PROCEDURE_ROOT),
        help="root holding <ICAO>/procedure-details (constrained-iaf mode)",
    )
    parser.add_argument(
        "--airport", default=None,
        help="airport ICAO fallback when a scenario has no arr_airport (constrained-iaf mode)",
    )
    args = parser.parse_args()

    scenarios = load_scenarios(args.scenarios)
    if args.constrained_iaf:
        paths = optimize_scenarios_constrained_iaf(
            scenarios,
            output_dir=args.output_dir,
            selection=args.iaf_selection,
            procedure_root=args.procedure_root,
            airport=args.airport,
            n_segments=args.n_segments,
            dt=args.dt,
            max_duration=args.max_duration,
            rollout_dt_s=args.rollout_dt,
            jobs=args.jobs,
            verbose=args.verbose,
            scenarios_label=args.scenarios,
        )
    else:
        paths = optimize_scenarios(
            scenarios,
            output_dir=args.output_dir,
            n_segments=args.n_segments,
            dt=args.dt,
            max_duration=args.max_duration,
            rollout_dt_s=args.rollout_dt,
            jobs=args.jobs,
            verbose=args.verbose,
            scenarios_label=args.scenarios,
        )
    print(f"✓ wrote {len(paths)} state file(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
