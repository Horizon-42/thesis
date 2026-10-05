"""Batch M1: each scenario flown in its airport's recorded traffic (multi-aircraft design §5.4, §9).

    python 4dTrajectory/optimization/traffic_optimization.py --airport KRDU --sample 50 --seed 1 \\
        --output-dir 4dTrajectory/outputs/KRDU/traffic_m1_runway_cons --jobs 8

The scenarios are a seeded sample of the airport's arrivals (stated in ``summary.json``), each built as
the constrained batch builds it (runway-threshold target, shortest IAF). Every solved scenario gets the
usual ``*_states.json`` + ``*_eval.json`` (the record of its last successful solve, so ``evaluation``
grades it unchanged) and a ``*_traffic.json`` sidecar (``traffic.loop.TRAFFIC_RECORD_SCHEMA``). The
recorded traffic is read once, in this process; each worker gets only its window's aircraft.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_OPT_DIR = Path(__file__).resolve().parent
if str(_OPT_DIR) not in sys.path:
    sys.path.insert(0, str(_OPT_DIR))

from flight_scenarios import FlightScenario  # noqa: E402
from flight_scenarios.build import build_scenario, load_model_arrivals  # noqa: E402
from flight_scenarios.scenario import NoAircraftDynamics  # noqa: E402
from collocation.optimizer import DEFAULT_N_SEGMENTS, DEFAULT_N_SEG_PER_PHASE  # noqa: E402
from optimization_run_config import DEFAULT_MAX_DURATION_S, DEFAULT_ROLLOUT_DT_S, build_optimization_config  # noqa: E402
from scenario_batch import run_batch  # noqa: E402
from scenario_optimization import (  # noqa: E402
    DEFAULT_FITTING,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_PROCEDURE_ROOT,
    shipped_evaluation,
)
from traffic.loop import TRAFFIC_RECORD_SCHEMA, LoopSettings, fly_in_traffic  # noqa: E402
from traffic.scene import Traffic, traffic_from_arrivals  # noqa: E402

HARVEST_ROOT = _REPO_ROOT / "trajectory_data_process" / "outputs" / "harvest"
TRAFFIC_SUFFIX = "_traffic.json"


@dataclass
class WindowScenario(FlightScenario):
    """A scenario with its window's recorded traffic (only the aircraft that can share it)."""

    traffic: Traffic = field(kw_only=True)


def window_scenarios(arrivals_manifest: Path, airport: str, sample: int, seed: int,
                     horizon_s: float) -> tuple[list[WindowScenario], dict[str, Any]]:
    """A seeded sample of ``sample`` buildable arrivals, each with its window's traffic, and the
    selection as stated in the summary."""
    flights = load_model_arrivals(arrivals_manifest)
    traffic = traffic_from_arrivals(flights)
    order = list(range(len(flights)))
    random.Random(seed).shuffle(order)
    scenarios, no_dynamics = [], 0
    for i in order:
        if len(scenarios) == sample:
            break
        try:
            scenario = build_scenario(flights[i], airport=airport, target_from_threshold=True)
        except NoAircraftDynamics:
            no_dynamics += 1
            continue
        own = traffic.flight(scenario.source["flight_key"])
        window = Traffic(traffic.airport,
                         (own, *traffic.airborne(own.start_utc_s, own.start_utc_s + horizon_s, exclude=own.flight_key)),
                         traffic.runway_targets)
        scenarios.append(WindowScenario(scenario.initial, scenario.aircraft, scenario.aero, scenario.source,
                                        scenario.target, traffic=window))
    selection = {"population": len(flights), "sample": len(scenarios), "seed": seed,
                 "skipped_no_dynamics": no_dynamics, "manifest": str(arrivals_manifest)}
    return scenarios, selection


def _fly_one_window(payload: tuple[int, WindowScenario, dict[str, Any]]):
    """Process-pool worker: one window flown in its traffic (``scenario_batch.run_batch`` contract)."""
    index, scenario, params = payload
    flight_id = scenario.source["flight_key"]
    try:
        result, sidecar = fly_in_traffic(
            scenario, scenario.traffic, procedure_root=params["procedure_root"],
            settings=LoopSettings(**params["settings"]), max_duration=params["max_duration"],
            rollout_dt_s=params["rollout_dt_s"], solve_options=params["solve_options"])
    except Exception as exc:  # noqa: BLE001 — batch tool: skip + log per-scenario failures
        return (index, flight_id, None, None, f"{type(exc).__name__}: {str(exc).splitlines()[0][:90]}")
    record = result.to_dict()
    record["sidecar"] = sidecar
    return (index, flight_id, record, shipped_evaluation(result), None)


def main() -> None:
    parser = argparse.ArgumentParser(description="M1: scenarios flown in recorded traffic")
    parser.add_argument("--airport", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample", type=int, required=True, help="arrivals to fly (a seeded sample)")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--harvest-root", default=str(HARVEST_ROOT),
                        help="the live harvest root (never harvest-heldout)")
    parser.add_argument("--procedure-root", default=str(DEFAULT_PROCEDURE_ROOT))
    parser.add_argument("--jobs", type=int, default=0, help="worker processes (0 = half the cores)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--max-duration", type=float, default=DEFAULT_MAX_DURATION_S)
    parser.add_argument("--rollout-dt", type=float, default=DEFAULT_ROLLOUT_DT_S)
    defaults = LoopSettings()
    parser.add_argument("--step-s", type=float, default=defaults.step_s)
    parser.add_argument("--row-window-s", type=float, default=defaults.row_window_s)
    parser.add_argument("--margin", type=float, default=defaults.margin)
    parser.add_argument("--max-rounds", type=int, default=defaults.max_rounds)
    args = parser.parse_args()
    if any("heldout" in part for part in Path(args.harvest_root).resolve().parts):
        parser.error("the held-out harvest is test data; the traffic batch reads the live root only")

    settings = LoopSettings(step_s=args.step_s, row_window_s=args.row_window_s, margin=args.margin,
                            max_rounds=args.max_rounds)
    manifest = Path(args.harvest_root) / args.airport / "arrivals" / "manifest.json"
    scenarios, selection = window_scenarios(manifest, args.airport, args.sample, args.seed, args.max_duration)
    config = build_optimization_config(
        constrained_iaf=True, fitting=DEFAULT_FITTING, n_segments=DEFAULT_N_SEGMENTS,
        n_seg_per_phase=DEFAULT_N_SEG_PER_PHASE, state_substeps=None, max_duration_s=args.max_duration,
        rollout_dt_s=args.rollout_dt, max_iterations=args.max_iterations, iaf_selection="shortest")
    config["traffic"] = {"schema": TRAFFIC_RECORD_SCHEMA, **asdict(settings), "selection": selection}
    run_batch(
        scenarios, output_dir=args.output_dir, worker=_fly_one_window,
        params={"procedure_root": args.procedure_root, "settings": asdict(settings),
                "max_duration": args.max_duration, "rollout_dt_s": args.rollout_dt,
                "solve_options": {"verbose": False, "max_iterations": args.max_iterations}},
        optimization_config=config, mode="traffic:m1", progress=" [traffic M1]", jobs=args.jobs,
        scenarios_label=f"{args.airport} sample {selection['sample']} seed {args.seed}",
        references_dir=None, resume=args.resume, sidecar_suffix=TRAFFIC_SUFFIX,
    )


if __name__ == "__main__":
    main()
