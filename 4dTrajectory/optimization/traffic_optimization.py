"""Multi-aircraft batches in an airport's recorded traffic (multi-aircraft design §5.4, §5.6, §9).

    python 4dTrajectory/optimization/traffic_optimization.py m1 --airport KRDU --sample 50 --seed 1 \\
        --output-dir 4dTrajectory/outputs/KRDU/traffic_m1_runway_cons --jobs 8
    python 4dTrajectory/optimization/traffic_optimization.py m2 --airport KRDU \\
        --block-start 2026-06-01T14:00:00Z --block-s 3600 --blocks 5 \\
        --output-dir 4dTrajectory/outputs/KRDU/traffic_m2_runway_cons --jobs 5

M1: a seeded sample of the airport's arrivals (stated in ``summary.json``), each flown in the recorded
traffic. M2: every arrival landing (in its record) inside each block, scheduled and flown in slot order
(``traffic.block``); one directory per block. Each scenario is built as the constrained batch builds it
(runway-threshold target, shortest IAF). Every solved one gets the usual ``*_states.json`` +
``*_eval.json`` (the record of its last successful solve, so ``evaluation`` grades it unchanged) and a
``*_traffic.json`` sidecar (``traffic.loop.TRAFFIC_RECORD_SCHEMA``). The recorded traffic is read once, in
this process; each worker gets only the aircraft that can share its window or block.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from scenario_batch import (  # noqa: E402
    _clear_stale_records,
    limit_solver_threads,
    resolve_jobs,
    run_batch,
    write_failed_record,
    write_solved_record,
)
from scenario_optimization import (  # noqa: E402
    DEFAULT_FITTING,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_PROCEDURE_ROOT,
    shipped_evaluation,
)
from traffic.block import BLOCK_RECORD_SCHEMA, fly_block  # noqa: E402
from traffic.loop import TRAFFIC_RECORD_SCHEMA, LoopSettings, fly_in_traffic  # noqa: E402
from trajectory_data_process.harvest.utc import parse_iso_utc_s  # noqa: E402
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
        return (index, flight_id, None, None, f"{type(exc).__name__}: {str(exc).partition(chr(10))[0][:90]}")
    record = result.to_dict()
    record["sidecar"] = sidecar
    return (index, flight_id, record, shipped_evaluation(result), None)


def block_scenarios(flights: list[dict], traffic: Traffic, airport: str, start_utc_s: float, end_utc_s: float,
                    horizon_s: float) -> tuple[list[FlightScenario], Traffic, int]:
    """The arrivals landing (in their records) in ``[start, end)``, built as scenarios, the traffic that can
    share the block, and the count of arrivals without dynamics (they stay records)."""
    scenarios, no_dynamics = [], 0
    for flight in flights:
        if not start_utc_s <= parse_iso_utc_s(flight["landing_time_utc"]) < end_utc_s:
            continue
        try:
            scenarios.append(build_scenario(flight, airport=airport, target_from_threshold=True))
        except NoAircraftDynamics:
            no_dynamics += 1
    keys = {s.source["flight_key"] for s in scenarios}
    starts = [traffic.flight(k).start_utc_s for k in keys]
    near = (traffic.airborne(min(starts), max(starts) + horizon_s, exclude="") if starts else [])
    return scenarios, Traffic(traffic.airport, tuple(near), traffic.runway_targets), no_dynamics


def _fly_one_block(payload: tuple[str, list[FlightScenario], Traffic, dict[str, Any]]):
    """Process-pool worker: one block (``traffic.block.fly_block``); results as picklable dicts. A block
    that raises is returned as its error, so one block never ends the run."""
    label, scenarios, traffic, params = payload
    try:
        flown, summary = fly_block(scenarios, traffic, procedure_root=params["procedure_root"],
                                   settings=LoopSettings(**params["settings"]), max_duration=params["max_duration"],
                                   rollout_dt_s=params["rollout_dt_s"], solve_options=params["solve_options"])
    except Exception as exc:  # noqa: BLE001 — batch tool: one failed block is recorded, the others go on
        return label, [], {"aircraft": len(scenarios), "scheduled": 0, "eta_failed": 0, "slot_failed": 0,
                           "error": f"{type(exc).__name__}: {str(exc).partition(chr(10))[0][:200]}"}
    out = []
    for f in flown:
        if f.result is None:
            out.append((f.scenario, None, None, f.error))
            continue
        record = f.result.to_dict()
        record["sidecar"] = f.sidecar
        out.append((f.scenario, record, shipped_evaluation(f.result), None))
    return label, out, summary


def _run_blocks(args, settings: LoopSettings, config: dict[str, Any]) -> None:
    manifest = Path(args.harvest_root) / args.airport / "arrivals" / "manifest.json"
    flights = load_model_arrivals(manifest)
    traffic = traffic_from_arrivals(flights)
    first = parse_iso_utc_s(args.block_start)
    payloads, skipped = [], {}
    for b in range(args.blocks):
        start = first + b * args.block_s
        scenarios, near, no_dynamics = block_scenarios(flights, traffic, args.airport, start, start + args.block_s,
                                                       args.max_duration)
        label = f"block_{b:02d}"
        skipped[label] = no_dynamics
        payloads.append((label, scenarios, near, {
            "procedure_root": args.procedure_root, "settings": asdict(settings), "max_duration": args.max_duration,
            "rollout_dt_s": args.rollout_dt,
            "solve_options": {"verbose": False, "max_iterations": args.max_iterations}}))
    del flights
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "blocks.json").unlink(missing_ok=True)          # the index is written last: no stale one
    for stale in out_root.glob("block_*"):                     # this run's directories are rewritten below
        if stale.name not in {label for label, *_ in payloads}:
            shutil.rmtree(stale)
    blocks = []
    limit_solver_threads()
    with ProcessPoolExecutor(max_workers=resolve_jobs(args.jobs, len(payloads))) as pool:
        for future in as_completed([pool.submit(_fly_one_block, payload) for payload in payloads]):
            label, flown, summary = future.result()
            out = out_root / label
            out.mkdir(parents=True, exist_ok=True)
            _clear_stale_records(out, sidecar_suffix=TRAFFIC_SUFFIX)
            rows = []
            for index, (scenario, record, evaluation, error) in enumerate(flown):
                if error is not None:
                    rows.append(write_failed_record(out, scenario, index, error, optimization_config=config,
                                                    references_dir=None))
                    continue
                rows.append(write_solved_record(out, scenario, index, record, evaluation, optimization_config=config,
                                                references_dir=None, sidecar_suffix=TRAFFIC_SUFFIX)[1])
            summary["skipped_no_dynamics"] = skipped[label]
            (out / "summary.json").write_text(json.dumps({
                "mode": "traffic:m2", "optimization_config": config, "block": summary,
                "total": len(rows), "solved": sum(r["status"] == "solved" for r in rows),
                "failed": sum(r["status"] != "solved" for r in rows), "results": rows}, indent=1), encoding="utf-8")
            blocks.append({"label": label, **{k: summary[k] for k in ("aircraft", "scheduled", "eta_failed",
                                                                         "slot_failed")},
                           **({"error": summary["error"]} if "error" in summary else {})})
            print(f"{'✗' if 'error' in summary else '✓'} {label}: {summary['scheduled']}/{summary['aircraft']} "
                  f"scheduled -> {out}")
    blocks.sort(key=lambda b: b["label"])
    (out_root / "blocks.json").write_text(json.dumps({"mode": "traffic:m2", "optimization_config": config,
                                                      "blocks": blocks}, indent=1), encoding="utf-8")


def main() -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--airport", required=True)
    common.add_argument("--output-dir", required=True)
    common.add_argument("--harvest-root", default=str(HARVEST_ROOT),
                        help="the live harvest root (never harvest-heldout)")
    common.add_argument("--procedure-root", default=str(DEFAULT_PROCEDURE_ROOT))
    common.add_argument("--jobs", type=int, default=0, help="worker processes (0 = half the cores)")
    common.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    common.add_argument("--max-duration", type=float, default=DEFAULT_MAX_DURATION_S)
    common.add_argument("--rollout-dt", type=float, default=DEFAULT_ROLLOUT_DT_S)
    defaults = LoopSettings()
    common.add_argument("--step-s", type=float, default=defaults.step_s)
    common.add_argument("--row-window-s", type=float, default=defaults.row_window_s)
    common.add_argument("--margin", type=float, default=defaults.margin)
    common.add_argument("--max-rounds", type=int, default=defaults.max_rounds)
    parser = argparse.ArgumentParser(description="Multi-aircraft optimization in recorded traffic")
    modes = parser.add_subparsers(dest="mode", required=True)
    m1 = modes.add_parser("m1", parents=[common], help="a seeded sample of windows, one optimized aircraft each")
    m1.add_argument("--sample", type=int, required=True, help="arrivals to fly (a seeded sample)")
    m1.add_argument("--seed", type=int, required=True)
    m1.add_argument("--resume", action="store_true")
    m2 = modes.add_parser("m2", parents=[common], help="blocks of arrivals: a schedule, then slot order")
    m2.add_argument("--block-start", required=True, help="UTC ISO time of the first block's start")
    m2.add_argument("--block-s", type=float, default=3600.0)
    m2.add_argument("--blocks", type=int, default=1)
    args = parser.parse_args()
    if any("heldout" in part for part in Path(args.harvest_root).resolve().parts):
        parser.error("the held-out harvest is test data; the traffic batch reads the live root only")

    settings = LoopSettings(step_s=args.step_s, row_window_s=args.row_window_s, margin=args.margin,
                            max_rounds=args.max_rounds)
    config = build_optimization_config(
        constrained_iaf=True, fitting=DEFAULT_FITTING, n_segments=DEFAULT_N_SEGMENTS,
        n_seg_per_phase=DEFAULT_N_SEG_PER_PHASE, state_substeps=None, max_duration_s=args.max_duration,
        rollout_dt_s=args.rollout_dt, max_iterations=args.max_iterations, iaf_selection="shortest")
    if args.mode == "m2":
        config["traffic"] = {"schema": BLOCK_RECORD_SCHEMA, **asdict(settings),
                             "blocks": {"start": args.block_start, "block_s": args.block_s, "count": args.blocks}}
        _run_blocks(args, settings, config)
        return
    manifest = Path(args.harvest_root) / args.airport / "arrivals" / "manifest.json"
    scenarios, selection = window_scenarios(manifest, args.airport, args.sample, args.seed, args.max_duration)
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
