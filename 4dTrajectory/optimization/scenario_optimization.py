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

Alongside each ``*_states.json`` the batch writes a ``*_eval.json`` — the neutral
evaluation-input record (initial/target state + the rollout states with a 1:1 aligned
control list; see ``evaluation_export.py`` and the root ``evaluation`` package). Failed
scenarios get one too, with EMPTY state/control lists, so the evaluation batch can compute
the solve rate from the file set alone.

This is a **teaching scaffold**: the loop / IO / serialization / CLI are wired; the two core
steps — running the optimizer (TODO ①) and the forward rollout (TODO ②) — are documented
TODOs. See ``flight_scenarios/README.md`` and the comments below.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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

from flight_scenarios import (  # noqa: E402
    FlightScenario,
    flight_key,
    load_model_arrivals,
    load_scenarios,
    state_samples_from_track,
)
from flight_scenarios.start_state import DEFAULT_WINDOW_S  # noqa: E402
from aircraft.aero_params import stall_speed_ms  # noqa: E402
from aerodynamic_model.common import GeodeticState, LoadFactorControl  # noqa: E402
from aerodynamic_model.casadi_simulator import CasadiSimulator  # noqa: E402
from aerodynamic_model.rollout import RolloutSample, rollout_piecewise_constant  # noqa: E402
from collocation import CollocationOptimizer  # noqa: E402
from collocation.components import (  # noqa: E402
    DEFAULT_MAX_ITERATIONS as _DEFAULT_MAX_ITERATIONS,
    altitude_floor_m,
)
# Single source for the control-mesh defaults (mirrored by the CLI + the pipeline) and for
# the target-anchored frame tolerance the constrained path validates its procedure against.
from collocation.optimizer import (  # noqa: E402
    _FRAME_ANCHOR_TOLERANCE_M,
    DEFAULT_N_SEGMENTS,
    DEFAULT_N_SEG_PER_PHASE,
)
from geokit import haversine_m  # noqa: E402
from evaluation_export import (  # noqa: E402
    STATE_DECIMALS,
    EVAL_SUFFIX as _EVAL_SUFFIX,
    OBSERVED_TRACKS_DIR,
    OBSERVED_TRACK_SUFFIX,
    REFERENCE_EVAL_SUFFIX as _REFERENCE_EVAL_SUFFIX,
    REFERENCES_DIR,
    STATES_SUFFIX as _STATES_SUFFIX,
    evaluation_record,
    failed_evaluation_record,
    observed_track_document,
    reference_evaluation_record,
    summary_row,
)
from optimization_run_config import (  # noqa: E402
    DEFAULT_MAX_DURATION_S,
    DEFAULT_ROLLOUT_DT_S,
    FITTING_SCHEMES,
    build_optimization_config,
)

# Optimizer + rollout defaults (override on the CLI). DEFAULT_N_SEGMENTS / the constrained
# DEFAULT_N_SEG_PER_PHASE are imported above from the optimizer (its own construction defaults).
DEFAULT_DT = 1.0                 # optimizer dt (API parity; state mesh is auto-selected)

# Fitting (transcription) selection for BOTH solve paths (unconstrained + constrained-IAF);
# each composes with the normalized full-transport dynamics. "hs" (Hermite-Simpson,
# 4th order) is the default — see the comment in optimize_scenario for why trapezoidal
# (2nd order) collapsed the batch success rate; it stays selectable for comparison runs.
# "rk4" is the 4th-order EXPLICIT shooting defect (same one-step map family as the replay
# integrator — playback-consistent by construction) at trapezoidal-like per-node cost.
DEFAULT_FITTING = "hs"

# IPOPT iteration cap for the batch. `components.DEFAULT_MAX_ITERATIONS` (3000) is the
# interactive/backend default: it is the right budget when a human is waiting on ONE hard
# solve, and the wrong one for a 70k-solve batch, where every unsolvable scenario pays the
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

# Velocity floor = STALL_MARGIN x stall speed (at the scenario's landing mass), so the
# optimizer admits realistic touchdown-speed targets instead of forcing V >= Vref. Capped at
# Vref so it never raises the optimizer's default floor.
_STALL_MARGIN = 1.10

# The replay ground guard sits this far BELOW the NLP's altitude floor. The guard exists
# to truncate DIVERGED replays (tens of metres to kilometres below the floor); but
# min-time plans deliberately RIDE the floor, and a faithful replay oscillates
# centimetres around it (measured: a 3.9 cm dip on a floor-riding HS solve whose
# unguarded replay landed 0.7 m from the target — a zero-margin guard cut that same
# replay 10 km short and failed 97% of an unconstrained batch). 5 m is two orders above
# that noise and far below any real divergence; the terminal evaluation is applied only
# to the final threshold event and is unaffected by where this mid-flight guard sits.
ROLLOUT_GUARD_MARGIN_M = 5.0


def rollout_guard_altitude_m(target_altitude_m: float) -> float:
    """The replay truncation altitude for a solve flying to ``target_altitude_m``:
    the NLP's own floor minus :data:`ROLLOUT_GUARD_MARGIN_M`."""
    return altitude_floor_m(target_altitude_m) - ROLLOUT_GUARD_MARGIN_M


def _stall_speed_ms(mass_kg: float, aero: Any) -> float:
    """The project stall model's 1-g stall speed (``aircraft.aero_params`` is the source)."""
    return stall_speed_ms(mass_kg, wing_area_m2=aero.S, cl_max=aero.Cl_max)


# ── Output records (plumbing) ─────────────────────────────────────────────────

def _quantized_sample(sample: "StateSample") -> dict[str, float]:
    """One ``StateSample`` as a JSON row at the record contract's serialized precision."""
    return {
        key: round(value, STATE_DECIMALS[key]) if key in STATE_DECIMALS else value
        for key, value in asdict(sample).items()
    }


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
    # The neutral evaluation-input record (evaluation_export.evaluation_record) —
    # written to its own *_eval.json by the batch, NOT part of the states file.
    evaluation: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        # Both arrays go out at the record contract's serialized precision (0.56 mm worst
        # case, 30% smaller — see evaluation_export.STATE_DECIMALS). `simulator_states` is
        # also what the eval record's `states_ref` resolves to, so the two views of the
        # rollout have to be written the same way or they would disagree in the last digits.
        return {
            "source": self.source,
            # `t` is deliberately NOT quantized (see evaluation_export.STATE_DECIMALS), so
            # this header and the eval record's `final_time_s` — both read off the same
            # rollout — agree exactly without either side rounding.
            "final_time_s": self.final_time_s,
            "optimizer_states": [_quantized_sample(s) for s in self.optimizer_states],
            "simulator_states": [_quantized_sample(s) for s in self.simulator_states],
        }


# ── The core: optimize one scenario into two state sequences ──────────────────

def optimize_scenario(
    scenario: FlightScenario,
    *,
    n_segments: int = DEFAULT_N_SEGMENTS,
    dt: float = DEFAULT_DT,
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
    optimizer_states = _node_states_to_samples([initial_row] + dense_rows, final_time, initial.m)
    rollout = _require_usable_rollout(rollout_controls(
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


class _GroundCheckedSimulator:
    """``CasadiSimulator`` plus a ground guard (the raw simulator has NO envelope checks).

    A replay stepping below ``min_altitude_m`` raises, so the shared rollout TRUNCATES
    (its envelope handling) instead of recording subterranean samples — a diverged
    replay used to record kilometres below sea level. Callers pass
    :func:`rollout_guard_altitude_m` (the NLP's floor minus a divergence margin), NOT
    the floor itself: plans ride the floor, and a zero-margin guard truncates faithful
    replays on centimetre-scale integration noise.
    """

    def __init__(self, simulator: CasadiSimulator, min_altitude_m: float) -> None:
        self._simulator = simulator
        self._min_altitude_m = float(min_altitude_m)

    def step(self, state: GeodeticState, control: Any, dt: float) -> GeodeticState:
        next_state = self._simulator.step(state, control, dt)
        if next_state.altitude < self._min_altitude_m:
            raise ValueError(
                f"altitude {next_state.altitude:.1f} m below the trajectory floor "
                f"{self._min_altitude_m:.1f} m"
            )
        return next_state


def rollout_controls(
    initial_state: GeodeticState,
    node_control: Any,
    final_time: float,
    aircraft: Any,
    *,
    dt: float = DEFAULT_ROLLOUT_DT_S,
    segment_durations: Any = None,
    min_altitude_m: float,
) -> list[RolloutSample]:
    """Roll the piecewise-constant optimizer controls through the REAL simulator.

    This produces the "simulator real states": the optimizer's own controls integrated
    through the actual dynamics, which differ from the optimizer's node states (the plan).

    Thin adapter over ``aerodynamic_model.rollout_piecewise_constant`` — builds the
    load-factor controls + the simulator and runs the shared rollout (truncating silently
    if the replay leaves the envelope — this is a viz aid). Returns the RAW rollout
    samples: each carries its state AND the control active at that time, which is what
    the evaluation export needs (aligned state/control lists). ``segment_durations``
    (one per control) drives the multiphase non-uniform schedule; ``None`` = equal segments.
    ``min_altitude_m`` truncates a replay that descends below it — REQUIRED: solve
    replays pass ``rollout_guard_altitude_m(target)``; a target-less replay states
    ``0.0`` (sea level) explicitly. It used to default to 0.0, which never fires for
    an elevated-airport target — a caller that forgot it silently recorded diverged
    replays kilometres below the field as valid rollouts.
    """
    controls = [
        LoadFactorControl(thrust=float(row[0]), bank_rad=float(row[1]),
                          load_factor=float(row[2]))
        for row in node_control
    ]
    sim = _GroundCheckedSimulator(CasadiSimulator(aircraft, dt), min_altitude_m)
    return rollout_piecewise_constant(
        sim, initial_state, controls, final_time,
        integrator_dt=dt,
        segment_durations=list(segment_durations) if segment_durations is not None else None,
        truncate_on_envelope_exit=True,
    )


def _require_usable_rollout(samples: list[RolloutSample]) -> list[RolloutSample]:
    """A rollout truncated before its first full step (envelope exit at t=0) has
    no usable trajectory — fail the scenario loudly instead of exporting a
    degenerate one-sample "solved" record (zero horizontal extent, which nothing
    downstream can arc-length match)."""
    if len(samples) < 2:
        raise ValueError(
            "control rollout exited the flight envelope at its first step — "
            "no usable trajectory"
        )
    return samples


def simulate_controls(
    initial_state: GeodeticState,
    node_control: Any,
    final_time: float,
    aircraft: Any,
    *,
    dt: float = DEFAULT_ROLLOUT_DT_S,
    segment_durations: Any = None,
    min_altitude_m: float,
) -> list[StateSample]:
    """:func:`rollout_controls` mapped onto serializable :class:`StateSample`s
    (``min_altitude_m`` REQUIRED — see there)."""
    samples = rollout_controls(
        initial_state, node_control, final_time, aircraft,
        dt=dt, segment_durations=segment_durations, min_altitude_m=min_altitude_m,
    )
    return [StateSample.from_state(s.t, s.state) for s in samples]

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
    flight_id = scenario.source.get("id") or f"scenario{index}"
    try:
        result = optimize_scenario(scenario, **params)
    except Exception as exc:  # noqa: BLE001 — batch tool: skip + log per-scenario failures
        return (index, flight_id, None, None,
                f"{type(exc).__name__}: {str(exc).splitlines()[0][:90]}")
    return (index, flight_id, result.to_dict(), result.evaluation, None)


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


# The record-filename suffixes are single-sourced in evaluation_export.py (imported above)
# — shared with ts_transformer/export.py, which writes the same directory shape.


def _clear_stale_records(out: Path, keep: set[str] | None = None) -> None:
    """Delete leftover per-trajectory records from a previous batch in ``out``.

    A fresh batch writes one ``*_states.json`` + ``*_eval.json`` per CURRENT scenario;
    records from an earlier run over a DIFFERENT scenario set survive by filename and
    pollute everything that scans the directory (``python -m evaluation`` once counted
    27 orphans into a KRDU report). The ``references/`` subdirectory is untouched —
    the CLI writes the reference records immediately before the batch.

    ``keep`` (the resume path) spares the named files. Orphan removal is what makes the
    directory a faithful image of the roster, so it happens either way — resume narrows
    WHICH files survive, it never turns the sweep off.
    """
    keep = keep or set()
    stale = [
        path
        for path in sorted(out.glob(f"*{_STATES_SUFFIX}")) + sorted(out.glob(f"*{_EVAL_SUFFIX}"))
        if path.name not in keep
    ]
    for path in stale:
        path.unlink()
    if stale:
        print(f"… cleared {len(stale)} record file(s) from a previous batch in {out}")


def _resumable_record(
    out: Path, scenario: FlightScenario, index: int
) -> tuple[str, dict[str, Any]] | None:
    """The summary row for one scenario's already-complete record pair, or ``None``.

    A 70k-solve batch runs for tens of hours and ``summary.json`` is only written at the
    end, so a crash at hour 25 used to discard every finished record with it. This reads
    one finished record back and rebuilds its roster row, so ``--resume`` re-runs only what
    is genuinely missing. It is deliberately strict — identity must match the scenario, a
    solved record must still have its states file — because a half-written record silently
    reused is worse than one re-solved.
    """
    name = _scenario_filename(scenario, index)
    eval_path = out / _eval_filename(name)
    if not eval_path.is_file():
        return None
    try:
        record = json.loads(eval_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    source = record.get("source")
    if not isinstance(source, dict) or any(
        source.get(key) != scenario.source.get(key)
        for key in ("id", "runway", "icao24", "landing_time_utc")
    ):
        return None
    final_time = record.get("final_time_s")
    if final_time is None:
        return (name, _summary_record(
            scenario, status="failed", states_file=None,
            eval_file=eval_path.name, final_time_s=None,
            reason=record.get("reason") or "unsolved (resumed)",
        ))
    if not (out / name).is_file():
        return None
    row = _summary_record(
        scenario, status="solved", states_file=name, eval_file=eval_path.name,
        final_time_s=float(final_time), reason=None,
    )
    chosen_iaf = source.get("chosenIaf")
    if chosen_iaf is not None:
        row["chosenIaf"] = chosen_iaf
    return (name, row)


def optimize_scenarios(
    scenarios: list[FlightScenario],
    *,
    output_dir: str | Path,
    n_segments: int = DEFAULT_N_SEGMENTS,
    dt: float = DEFAULT_DT,
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
    """Optimize each scenario and write one ``*_states.json`` per scenario.

    ``references_dir`` (a directory name under ``output_dir``, see
    :func:`write_reference_records`) makes every eval record — solved and failed —
    carry a ``reference_file`` pointer at its observed-track reference record.

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
    # Resume BEFORE the stale sweep: the rows we keep are exactly the files it must spare.
    records: dict[int, dict[str, Any]] = {}  # index -> summary record (parallel-safe ordering)
    resumed_files: set[str] = set()
    pending = list(range(len(scenarios)))
    if resume:
        pending = []
        for index, scenario in enumerate(scenarios):
            found = _resumable_record(out, scenario, index)
            if found is None:
                pending.append(index)
                continue
            name, row = found
            records[index] = row
            resumed_files.update({name, _eval_filename(name)})
        if records:
            print(f"… resuming: {len(records)} record(s) already complete, "
                  f"{len(pending)} to solve")
    _clear_stale_records(out, keep=resumed_files)
    params: dict[str, Any] = {
        "n_segments": n_segments, "dt": dt, "max_duration": max_duration,
        "rollout_dt_s": rollout_dt_s, "fitting": fitting,
        "state_substeps": state_substeps, "max_iterations": max_iterations,
        "verbose": verbose,
    }
    payloads = [(index, scenarios[index], params) for index in pending]
    workers = _resolve_jobs(jobs, len(payloads))

    written: list[Path] = []
    failures: list[tuple[str, str]] = []

    def _handle(
        record: tuple[int, str, dict[str, Any] | None, dict[str, Any] | None, str | None],
    ) -> None:
        index, flight_id, result_dict, eval_dict, error = record
        scenario = scenarios[index]
        name = _scenario_filename(scenario, index)
        eval_name = _eval_filename(name)
        reference_file = (
            f"{references_dir}/{_reference_filename(name)}" if references_dir else None
        )
        if error is not None:
            failures.append((flight_id, error))
            # Unsolved configurations still get an evaluation record (empty lists) —
            # that is how the evaluation batch computes the solve rate.
            failed_record = failed_evaluation_record(
                scenario.initial, scenario.target, scenario.source, error,
                subject="optimized",
            )
            if reference_file:
                failed_record["reference_file"] = reference_file
            (out / eval_name).write_text(
                json.dumps(failed_record, separators=(",", ":"), allow_nan=False), encoding="utf-8"
            )
            records[index] = _summary_record(
                scenario, status="failed", states_file=None, eval_file=eval_name,
                final_time_s=None, reason=error,
            )
            print(f"✗ {flight_id}: skipped ({error.split(':', 1)[0]})")
            return
        path = out / name
        path.write_text(json.dumps(result_dict, separators=(",", ":")), encoding="utf-8")
        eval_dict = dict(eval_dict)
        eval_dict["states_ref"] = {"file": name, "key": "simulator_states"}
        eval_dict["states"] = []
        if reference_file:
            eval_dict["reference_file"] = reference_file
        (out / eval_name).write_text(
            json.dumps(eval_dict, separators=(",", ":"), allow_nan=False), encoding="utf-8"
        )
        written.append(path)
        records[index] = _summary_record(
            scenario, status="solved", states_file=name, eval_file=eval_name,
            final_time_s=float(result_dict["final_time_s"]), reason=None,
        )
        print(
            f"✓ {path.name}: optimizer {len(result_dict['optimizer_states'])} states, "
            f"simulator {len(result_dict['simulator_states'])} states, "
            f"T={result_dict['final_time_s']:.1f}s"
        )

    if not payloads:
        pass
    elif workers == 1:
        for payload in payloads:
            _handle(_optimize_one_scenario(payload))
    else:
        _limit_solver_threads()
        print(f"… solving {len(payloads)} scenario(s) across {workers} worker process(es)")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_optimize_one_scenario, payload) for payload in payloads]
            for future in as_completed(futures):
                _handle(future.result())

    # One summary per run: the failure rate + which flights solved/failed, with the ids
    # needed to find each flight's reference (e.g. for the comparison CZML).
    total = len(scenarios)
    solved_rows = sum(1 for row in records.values() if row["status"] == "solved")
    summary = {
        "scenarios": scenarios_label,
        "optimization_config": build_optimization_config(
            constrained_iaf=False,
            fitting=fitting,
            n_segments=n_segments,
            n_seg_per_phase=DEFAULT_N_SEG_PER_PHASE,
            state_substeps=state_substeps,
            max_duration_s=max_duration,
            rollout_dt_s=rollout_dt_s,
            max_iterations=max_iterations,
        ),
        "total": total,
        "solved": solved_rows,
        "failed": total - solved_rows,
        "failure_rate": ((total - solved_rows) / total) if total else 0.0,
        "results": [records[i] for i in sorted(records)],
    }
    if len(records) != total:
        raise RuntimeError(
            f"roster is incomplete: {len(records)} row(s) for {total} scenario(s)"
        )
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if failures:
        print(f"\n⚠ {len(failures)}/{len(scenarios)} scenario(s) skipped:")
        for flight_id, reason in failures[:15]:
            print(f"    {flight_id}: {reason}")
        if len(failures) > 15:
            print(f"    … and {len(failures) - 15} more")
    print(f"✓ solved {solved_rows}/{total} scenario(s) "
          f"(failure rate {summary['failure_rate']:.1%}) -> {out}")
    print(f"  summary -> {out / 'summary.json'}")
    return written


def _summary_record(
    scenario: FlightScenario,
    *,
    status: str,
    states_file: str | None,
    eval_file: str | None,
    final_time_s: float | None,
    reason: str | None,
) -> dict[str, Any]:
    """One summary row: the flight's identity (so its reference can be found by id) + status.

    The row shape is single-sourced in ``evaluation_export.summary_row`` (shared with
    ts_transformer's batch writer).
    """
    return summary_row(
        scenario.source,
        status=status,
        states_file=states_file,
        eval_file=eval_file,
        final_time_s=final_time_s,
        reason=reason,
    )


def _eval_filename(states_name: str) -> str:
    """``<flight>_states.json`` → ``<flight>_eval.json`` (same identity key)."""
    return states_name.removesuffix(_STATES_SUFFIX) + _EVAL_SUFFIX


def _reference_filename(states_name: str) -> str:
    """``<flight>_states.json`` → ``<flight>_reference_eval.json`` (same identity key)."""
    return states_name.removesuffix(_STATES_SUFFIX) + _REFERENCE_EVAL_SUFFIX


def _observed_track_filename(states_name: str) -> str:
    """``<flight>_states.json`` → ``<flight>_track.json`` (same identity key)."""
    return states_name.removesuffix(_STATES_SUFFIX) + OBSERVED_TRACK_SUFFIX


# The reference cache contract. v3 replaced each record's inline state array with a
# `states_ref` into a shared observed-track store (see OBSERVED_TRACKS_DIR), so a v2
# manifest no longer describes what is on disk and must not be reused.
REFERENCE_CACHE_SCHEMA = "optimization-references-v3-shared-tracks"


def _observed_track_path(reference_path: Path) -> Path:
    """The shared track file a reference record quotes, from the record's own path."""
    stem = reference_path.name.removesuffix(_REFERENCE_EVAL_SUFFIX)
    return reference_path.parent.parent / OBSERVED_TRACKS_DIR / (
        stem + OBSERVED_TRACK_SUFFIX
    )


def _cached_track_matches(reference_path: Path, expected_sha256: str) -> bool:
    track = _observed_track_path(reference_path)
    return track.is_file() and _file_sha256(track) == expected_sha256


def _sweep_observed_tracks(tracks_dir: Path) -> None:
    """Drop track files no sibling reference directory quotes any more.

    The store is shared by every target dataset under the same anchor, so it must NOT be
    swept against one dataset's roster — that would delete the other's tracks. The keep set
    is the union over all sibling ``*_reference_eval.json`` names, which is exactly what a
    reader can still resolve.
    """
    if not tracks_dir.is_dir():
        return
    keep = {
        path.name.removesuffix(_REFERENCE_EVAL_SUFFIX) + OBSERVED_TRACK_SUFFIX
        for sibling in tracks_dir.parent.iterdir()
        if sibling.is_dir() and sibling != tracks_dir
        for path in sibling.glob(f"*{_REFERENCE_EVAL_SUFFIX}")
    }
    stale = [
        path for path in sorted(tracks_dir.glob(f"*{OBSERVED_TRACK_SUFFIX}"))
        if path.name not in keep
    ]
    for path in stale:
        path.unlink()
    if stale:
        print(f"… cleared {len(stale)} unreferenced observed track(s) from {tracks_dir}")


def write_reference_records(
    scenarios: list[FlightScenario],
    observed_tracks: str | Path | list[dict[str, Any]],
    *,
    output_dir: str | Path,
    references_dir: str = REFERENCES_DIR,
    source_signature: dict[str, Any] | None = None,
) -> list[Path]:
    """One reference eval record per scenario, from its OBSERVED track.

    ``observed_tracks`` is the harvest arrival manifest the scenarios came from.
    Each scenario's flight is looked up in it by its full
    identity ``(id, icao24, landing_time_utc)`` — the same key the output
    filenames disambiguate on. The track becomes a reference record in the
    evaluation contract (per-sample kinematics via
    ``flight_scenarios.state_samples_from_track``, EMPTY controls, the SAME target
    the optimizer flies to), written under ``<output_dir>/<references_dir>/`` and
    named by the scenario's identity — so the batch can point every eval record at
    its reference deterministically (``reference_file``). Missing flights raise:
    references and solves must come from the same dataset.
    """
    out = (Path(output_dir) / references_dir).resolve()
    expected = [
        out / _reference_filename(_scenario_filename(scenario, index))
        for index, scenario in enumerate(scenarios)
    ]
    expected_cache_rows = [
        {
            "file": path.name,
            "identity": {
                "flight_key": flight_key(scenario.source, index),
                **{
                    key: scenario.source.get(key)
                    for key in ("id", "runway", "icao24", "landing_time_utc")
                },
            },
        }
        for index, (path, scenario) in enumerate(zip(expected, scenarios))
    ]
    cache_manifest = out / "manifest.json"
    if source_signature is not None and cache_manifest.exists():
        try:
            cached = json.loads(cache_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = {}
        cached_rows = cached.get("records")
        cache_matches = (
            cached.get("schema_version") == REFERENCE_CACHE_SCHEMA
            and cached.get("source_signature") == source_signature
            and isinstance(cached_rows, list)
            and len(cached_rows) == len(expected_cache_rows)
        )
        if cache_matches:
            for path, expected_row, cached_row in zip(
                expected, expected_cache_rows, cached_rows
            ):
                if (
                    not isinstance(cached_row, dict)
                    or cached_row.get("file") != expected_row["file"]
                    or cached_row.get("identity") != expected_row["identity"]
                    or not isinstance(cached_row.get("sha256"), str)
                    or not isinstance(cached_row.get("track_sha256"), str)
                    or not path.is_file()
                    or _file_sha256(path) != cached_row["sha256"]
                    or not _cached_track_matches(path, cached_row["track_sha256"])
                ):
                    cache_matches = False
                    break
        if cache_matches:
            print(f"✓ reusing {len(expected)} canonical reference record(s) -> {out}")
            return expected

    # Through the SAME loader the scenarios came from: it converts the observed altitudes
    # from ellipsoidal (HAE) to MSL. Reading the file directly here would put the reference
    # record ~30 m below the scenario built from the identical track.
    flights = load_model_arrivals(observed_tracks)
    by_identity = {
        (f.get("id"), f.get("icao24"), f.get("landing_time_utc")): f for f in flights
    }
    out.mkdir(parents=True, exist_ok=True)
    # Fresh reference set = fresh directory: references from an earlier run over a
    # different flight set would otherwise accumulate (same stale-record class as
    # _clear_stale_records — dormant, but unbounded growth).
    stale = sorted(out.glob(f"*{_REFERENCE_EVAL_SUFFIX}"))
    for path in stale:
        path.unlink()
    if stale:
        print(f"… cleared {len(stale)} reference record(s) from a previous run in {out}")
    # The observed track store is a SIBLING of the per-target reference dirs, so the
    # fitted-ADS-B and runway datasets — which reference the same flights and differ only
    # in target_state — quote one copy of each track instead of two.
    tracks_dir = out.parent / OBSERVED_TRACKS_DIR
    tracks_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    track_paths: list[Path] = []
    for index, scenario in enumerate(scenarios):
        src = scenario.source
        if scenario.target is None:
            raise ValueError(
                f"scenario {src.get('id')!r} has no target state; build scenarios with "
                "flight_scenarios (its build_scenario populates target) first."
            )
        identity = (src.get("id"), src.get("icao24"), src.get("landing_time_utc"))
        flight = by_identity.get(identity)
        if flight is None:
            raise ValueError(
                f"no observed flight in the reference-tracks file for identity {identity}"
            )
        timed_states = state_samples_from_track(
            flight["waypoints"], mass_kg=scenario.initial.m,
            window_s=float(src.get("window_s") or DEFAULT_WINDOW_S),
        )
        states_name = _scenario_filename(scenario, index)
        track_path = tracks_dir / _observed_track_filename(states_name)
        track_path.write_text(
            json.dumps(observed_track_document(timed_states),
                       separators=(",", ":"), allow_nan=False),
            encoding="utf-8",
        )
        track_paths.append(track_path)
        record = reference_evaluation_record(
            scenario.initial, scenario.target, timed_states, src, subject="observed",
            track_ref=f"../{OBSERVED_TRACKS_DIR}/{track_path.name}",
        )
        path = out / _reference_filename(states_name)
        path.write_text(
            json.dumps(record, separators=(",", ":"), allow_nan=False), encoding="utf-8"
        )
        written.append(path)
    if source_signature is not None:
        records = [
            {**row, "sha256": _file_sha256(path), "track_sha256": _file_sha256(track)}
            for row, path, track in zip(expected_cache_rows, written, track_paths)
        ]
        manifest_payload = {
            "schema_version": REFERENCE_CACHE_SCHEMA,
            "source_signature": source_signature,
            "records": records,
        }
        temporary = cache_manifest.with_name(f".{cache_manifest.name}.tmp")
        temporary.write_text(
            json.dumps(manifest_payload, indent=2), encoding="utf-8",
        )
        temporary.replace(cache_manifest)
    _sweep_observed_tracks(tracks_dir)
    print(f"✓ wrote {len(written)} reference record(s) -> {out}")
    print(f"  observed tracks -> {tracks_dir} ({len(track_paths)} shared)")
    return written


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_legacy_category_references(output_dir: str | Path) -> None:
    """Drop only the old per-category reference copies after moving to a sibling anchor."""
    legacy = Path(output_dir) / REFERENCES_DIR
    if not legacy.is_dir():
        return
    removed = 0
    for path in legacy.glob(f"*{_REFERENCE_EVAL_SUFFIX}"):
        path.unlink()
        removed += 1
    (legacy / "manifest.json").unlink(missing_ok=True)
    try:
        legacy.rmdir()
    except OSError:
        return
    if removed:
        print(f"… removed {removed} superseded per-category reference copy/copies -> {legacy}")


def _scenario_filename(scenario: FlightScenario, index: int) -> str:
    """A unique, stable filename for one scenario's states JSON.

    A callsign + runway is NOT unique: the same aircraft can land more than once, and a
    callsign can recur across days — so ``id_runway`` silently overwrote sibling scenarios.
    The identity ``id_runway_icao24_landingTime`` is single-sourced in
    ``flight_scenarios.identity.flight_key`` — the SAME function keys ts_transformer's
    train/val/test split and record stems, so learned and optimized records for one flight
    always share a filename stem.
    """
    return f"{flight_key(scenario.source, index)}{_STATES_SUFFIX}"


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


def _iaf_setup(scenario: FlightScenario, procedure_root: str | Path, airport: str | None):
    """Shared prologue for the IAF optimizers: resolve the runway's RNAV(GPS) procedure and return
    ``(target, iaf_paths, aircraft, min_speed_ms)``. ``target`` is the scenario's own
    authoritative threshold state, unmodified — the procedure is validated against it (see
    :func:`_require_procedure_threshold_agrees`). Raises if the procedure / paths are missing."""
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
    _require_procedure_threshold_agrees(target, paths)
    aircraft = scenario.aircraft
    min_speed_ms = min(
        _STALL_MARGIN * _stall_speed_ms(scenario.initial.m, scenario.aero),
        aircraft.approach.reference_speed_ms,
    )
    return target, paths, aircraft, min_speed_ms


def _solve_iaf(
    pc, scenario: FlightScenario, target: GeodeticState, aircraft: Any, min_speed_ms: float,
    *, n_segments: int, dt: float, max_duration: float, verbose: bool,
    fitting: str = DEFAULT_FITTING,
    state_substeps: int | None = None,
    n_seg_per_phase: int = DEFAULT_N_SEG_PER_PHASE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> _IafSolve:
    """Full CONSTRAINED solve from the scenario's OBSERVED start to the runway via one IAF path.

    Uses the backend's ``build_constraint_segments`` (constraint geometry) and the **multiphase**
    optimiser. The start is the observed ``scenario.initial`` (so the result is comparable to the
    ADS-B track), NOT a synthetic IAF state: the optimiser flies a free transition from there to the
    procedure's first fix (pre-FAF legs are unpinned, altitude-only), then each procedure leg with
    its corridor / glidepath / floor. Raises on infeasibility. ``n_seg_per_phase`` sets the control
    segments PER leg (the multiphase mesh); ``n_segments``/``dt`` are accepted for CLI/API parity
    but unused here (the multiphase optimiser derives its total mesh from n_seg_per_phase × legs).
    """
    from aeroviz_backend.procedure_segments import build_constraint_segments
    segments = build_constraint_segments(
        pc, target.latitude, target.longitude, target.altitude,
    )
    if not segments:
        raise ValueError("no constraint segments")
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
    final_time, node_control, _ = optimizer.optimize_free_time(start_state, target, max_duration)
    return _IafSolve(
        float(final_time), pc, start_state, node_control,
        optimizer.last_dense_states_geo, list(optimizer.segment_durations_s),
    )


def _iaf_result(
    best: _IafSolve, scenario: FlightScenario, aircraft: Any,
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
    optimizer_states = _node_states_to_samples(
        [initial_row] + dense_rows, best.final_time, best.initial.m,
    )
    rollout = _require_usable_rollout(rollout_controls(
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
    airport: str | None = None,
    n_segments: int = DEFAULT_N_SEGMENTS,
    dt: float = DEFAULT_DT,
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
    target, paths, aircraft, min_speed_ms = _iaf_setup(scenario, procedure_root, airport)

    best: _IafSolve | None = None
    attempts: list[tuple[str, str]] = []
    for pc in paths:
        try:
            candidate = _solve_iaf(
                pc, scenario, target, aircraft, min_speed_ms,
                n_segments=n_segments, dt=dt, max_duration=max_duration, verbose=verbose,
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
    return _iaf_result(
        best, scenario, aircraft,
        target=target, candidates=len(paths), rollout_dt_s=rollout_dt_s, selection="minTime",
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
    fitting: str = DEFAULT_FITTING,
    state_substeps: int | None = None,
    n_seg_per_phase: int = DEFAULT_N_SEG_PER_PHASE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
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
                fitting=fitting, state_substeps=state_substeps,
                n_seg_per_phase=n_seg_per_phase, max_iterations=max_iterations,
            )
        except Exception as exc:  # noqa: BLE001 — fall through to the next-shortest IAF
            attempts.append((pc.waypoints[0].ident, type(exc).__name__))
            continue
        return _iaf_result(
            best, scenario, aircraft,
            target=target, candidates=len(paths), rollout_dt_s=rollout_dt_s,
            selection="shortestPath",
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
) -> tuple[int, str, dict[str, Any] | None, dict[str, Any] | None, str | None]:
    """Process-pool worker for the constrained-IAF batch; ``params['selection']`` chooses the
    per-scenario strategy (mirrors ``_optimize_one_scenario``)."""
    index, scenario, params = payload
    params = dict(params)
    selector = _IAF_SELECTORS[params.pop("selection", "shortest")]
    flight_id = scenario.source.get("id") or f"scenario{index}"
    try:
        result = selector(scenario, **params)
    except Exception as exc:  # noqa: BLE001 — skip + log per-scenario failures
        return (index, flight_id, None, None,
                f"{type(exc).__name__}: {str(exc).splitlines()[0][:90]}")
    return (index, flight_id, result.to_dict(), result.evaluation, None)


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
    ``"shortest"`` picks the shortest 3D path and solves once (naive, fast). Same output shape/IO
    as :func:`optimize_scenarios` (``*_states.json`` + ``summary.json``, parallel across
    scenarios), reusing its summary/filename/jobs helpers; each scenario reports the chosen IAF.
    """
    if selection not in _IAF_SELECTORS:
        raise ValueError(f"unknown selection {selection!r}; choose from {sorted(_IAF_SELECTORS)}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Resume BEFORE the stale sweep: the rows we keep are exactly the files it must spare.
    records: dict[int, dict[str, Any]] = {}  # index -> summary record (parallel-safe ordering)
    resumed_files: set[str] = set()
    pending = list(range(len(scenarios)))
    if resume:
        pending = []
        for index, scenario in enumerate(scenarios):
            found = _resumable_record(out, scenario, index)
            if found is None:
                pending.append(index)
                continue
            name, row = found
            records[index] = row
            resumed_files.update({name, _eval_filename(name)})
        if records:
            print(f"… resuming: {len(records)} record(s) already complete, "
                  f"{len(pending)} to solve")
    _clear_stale_records(out, keep=resumed_files)
    params: dict[str, Any] = {
        "selection": selection,
        "procedure_root": str(procedure_root), "airport": airport,
        "n_segments": n_segments, "dt": dt, "max_duration": max_duration,
        "rollout_dt_s": rollout_dt_s, "fitting": fitting,
        "state_substeps": state_substeps, "n_seg_per_phase": n_seg_per_phase,
        "max_iterations": max_iterations, "verbose": verbose,
    }
    payloads = [(index, scenarios[index], params) for index in pending]
    workers = _resolve_jobs(jobs, len(payloads))

    written: list[Path] = []
    failures: list[tuple[str, str]] = []

    def _handle(
        record: tuple[int, str, dict[str, Any] | None, dict[str, Any] | None, str | None],
    ) -> None:
        index, flight_id, result_dict, eval_dict, error = record
        scenario = scenarios[index]
        name = _scenario_filename(scenario, index)
        eval_name = _eval_filename(name)
        reference_file = (
            f"{references_dir}/{_reference_filename(name)}" if references_dir else None
        )
        if error is not None:
            failures.append((flight_id, error))
            failed_record = failed_evaluation_record(
                scenario.initial, scenario.target, scenario.source, error,
                subject="optimized",
            )
            if reference_file:
                failed_record["reference_file"] = reference_file
            (out / eval_name).write_text(
                json.dumps(failed_record, separators=(",", ":"), allow_nan=False), encoding="utf-8"
            )
            records[index] = _summary_record(
                scenario, status="failed", states_file=None, eval_file=eval_name,
                final_time_s=None, reason=error,
            )
            print(f"✗ {flight_id}: skipped ({error.split(':', 1)[0]})")
            return
        (out / name).write_text(
            json.dumps(result_dict, separators=(",", ":")), encoding="utf-8"
        )
        eval_dict = dict(eval_dict)
        eval_dict["states_ref"] = {"file": name, "key": "simulator_states"}
        eval_dict["states"] = []
        if reference_file:
            eval_dict["reference_file"] = reference_file
        (out / eval_name).write_text(
            json.dumps(eval_dict, separators=(",", ":"), allow_nan=False), encoding="utf-8"
        )
        written.append(out / name)
        record_row = _summary_record(
            scenario, status="solved", states_file=name, eval_file=eval_name,
            final_time_s=float(result_dict["final_time_s"]), reason=None,
        )
        record_row["chosenIaf"] = result_dict["source"].get("chosenIaf")
        records[index] = record_row
        print(f"✓ {name}: IAF {result_dict['source'].get('chosenIaf')}, "
              f"T={result_dict['final_time_s']:.1f}s")

    if not payloads:
        pass
    elif workers == 1:
        for payload in payloads:
            _handle(_optimize_one_scenario_iaf(payload))
    else:
        _limit_solver_threads()
        print(f"… solving {len(payloads)} scenario(s) [constrained IAF: {selection}] "
              f"across {workers} worker process(es)")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_optimize_one_scenario_iaf, payload) for payload in payloads]
            for future in as_completed(futures):
                _handle(future.result())

    total = len(scenarios)
    solved_rows = sum(1 for row in records.values() if row["status"] == "solved")
    summary = {
        "scenarios": scenarios_label,
        "mode": f"constrainedIaf:{selection}",
        "optimization_config": build_optimization_config(
            constrained_iaf=True,
            fitting=fitting,
            n_segments=n_segments,
            n_seg_per_phase=n_seg_per_phase,
            state_substeps=state_substeps,
            max_duration_s=max_duration,
            rollout_dt_s=rollout_dt_s,
            max_iterations=max_iterations,
            iaf_selection=selection,
        ),
        "total": total,
        "solved": solved_rows,
        "failed": total - solved_rows,
        "failure_rate": ((total - solved_rows) / total) if total else 0.0,
        "results": [records[i] for i in sorted(records)],
    }
    if len(records) != total:
        raise RuntimeError(
            f"roster is incomplete: {len(records)} row(s) for {total} scenario(s)"
        )
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if failures:
        print(f"\n⚠ {len(failures)}/{total} scenario(s) skipped:")
        for flight_id, reason in failures[:15]:
            print(f"    {flight_id}: {reason}")
        if len(failures) > 15:
            print(f"    … and {len(failures) - 15} more")
    print(f"✓ solved {solved_rows}/{total} scenario(s) [constrained IAF: {selection}] "
          f"(failure rate {summary['failure_rate']:.1%}) -> {out}")
    print(f"  summary -> {out / 'summary.json'}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize flight scenarios -> state JSON files")
    parser.add_argument("--scenarios", required=True, help="Scenario JSON from flight_scenarios")
    parser.add_argument("--output-dir", required=True, help="Where to write the *_states.json files")
    parser.add_argument("--n-segments", type=int, default=DEFAULT_N_SEGMENTS,
                        help="unconstrained: control segments over the whole trajectory")
    parser.add_argument("--n-seg-per-phase", type=int, default=DEFAULT_N_SEG_PER_PHASE,
                        help="constrained-iaf: control segments PER procedure leg (the "
                             "multiphase mesh; unconstrained runs ignore it)")
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
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
            _remove_legacy_category_references(args.output_dir)
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
            airport=args.airport,
            n_segments=args.n_segments,
            dt=args.dt,
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
            dt=args.dt,
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
