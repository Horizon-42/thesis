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
from dataclasses import asdict, dataclass
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
from casadi_direct_collocation_optimizer import CasadiDirectCollocationOptimizer  # noqa: E402

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
    optimizer = CasadiDirectCollocationOptimizer(
        n_segments, dt, max_duration, aircraft,
        collocation_scheme="trapezoidalNormalizedFullTransport",
        min_speed_ms=min_speed_ms,
        verbose=verbose,
    )
    final_time, node_control, node_state = optimizer.optimize_free_time(
        initial, target, max_duration)
    #   • node_state  rows are [lat, lon, alt, V, psi, gamma]   — the optimizer's plan
    #   • node_control rows are [thrust_N, bank_rad, load_factor]
    #
    # Then build the two sequences (the helpers below do the work) and return:
    #
    optimizer_states = _node_states_to_samples(node_state, final_time, initial.m)
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
) -> list[StateSample]:
    """Roll the piecewise-constant optimizer controls through the REAL simulator.

    This produces the "simulator real states": the optimizer's own controls integrated
    through the actual dynamics, which differ from the optimizer's node states (the plan).

    Thin adapter over ``aerodynamic_model.rollout_piecewise_constant`` — builds the
    load-factor controls + the simulator, runs the shared rollout (truncating silently if
    the replay leaves the envelope — this is a viz aid), and maps each neutral
    ``(t, state)`` sample onto a serializable :class:`StateSample`.
    """
    controls = [
        LoadFactorControl(thrust=float(row[0]), bank_rad=float(row[1]),
                          load_factor=float(row[2]))
        for row in node_control
    ]
    sim = CasadiSimulator(aircraft, dt)
    samples = rollout_piecewise_constant(
        sim, initial_state, controls, final_time,
        integrator_dt=dt, truncate_on_envelope_exit=True,
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
    args = parser.parse_args()

    scenarios = load_scenarios(args.scenarios)
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
