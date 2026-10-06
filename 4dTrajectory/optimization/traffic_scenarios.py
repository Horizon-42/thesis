"""The scenario list of the Optimize task's multi-aircraft mode (design §10.6): a census of an airport's records.

    python 4dTrajectory/optimization/traffic_scenarios.py --airport KRDU --jobs 3

Each arrival that M1 can command (it has a dynamics model) is judged on its OWN RECORD as the flown track, in its
recorded traffic, with the M1 judge (``traffic.check.check``, reading VISUAL): the losses it answers for. The catalog
lists for M1 every arrival with at least one such loss, and for M2, per block length, every block in which an arrival
lands, with the losses its arrivals answer for. Written to ``traffic_job_files.catalog_path`` (schema
``CATALOG_SCHEMA``); the backend serves it. Reads the whole roster once (offline: run it at low priority).
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

_OPT_DIR = Path(__file__).resolve().parent
if str(_OPT_DIR) not in sys.path:
    sys.path.insert(0, str(_OPT_DIR))

from traffic_optimization import HARVEST_ROOT, WindowScenario, window_scenario  # noqa: E402  (sets the repo path)
from flight_scenarios.build import build_scenario, load_model_arrivals  # noqa: E402
from flight_scenarios.identity import flight_key  # noqa: E402
from flight_scenarios.scenario import NoAircraftDynamics  # noqa: E402
from optimization_run_config import DEFAULT_MAX_DURATION_S  # noqa: E402
from scenario_batch import limit_solver_threads, resolve_jobs  # noqa: E402
from scenario_optimization import DEFAULT_PROCEDURE_ROOT  # noqa: E402
from traffic import rules  # noqa: E402
from traffic.check import FlownTrack, check, make_window  # noqa: E402
from traffic.scene import traffic_from_arrivals  # noqa: E402
from traffic_job_files import BLOCK_LENGTHS_S, CATALOG_SCHEMA, OUTPUTS_ROOT, catalog_path, write_json_atomic  # noqa: E402
from trajectory_data_process.harvest.arrivals import SCHEMA_VERSION  # noqa: E402
from trajectory_data_process.harvest.utc import parse_iso_utc_s  # noqa: E402


def judge_record(payload: tuple[WindowScenario, str, float, float]) -> dict[str, Any]:
    """Process-pool worker: one arrival's record judged in its recorded traffic (the losses it answers for)."""
    scenario, procedure_root, horizon_s, step_s = payload
    window = make_window(scenario, scenario.traffic, procedure_root=procedure_root, horizon_s=horizon_s)
    own = scenario.traffic.flight(window.flight_key)
    found = check(window, FlownTrack.from_record(window, own, step_s), reading=rules.VISUAL, step_s=step_s)
    answered = [c for c in found.conflicts if c.responsible]
    return {"flightKey": window.flight_key, "lossInstants": len({c.t_s for c in answered}),
            "kinds": sorted({c.kind for c in answered}),
            "tightest": min((c.distance_m / c.required_m for c in answered), default=None),
            "recordedAircraft": len(window.recorded)}


def build_catalog(arrivals: list[dict[str, Any]], judged: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The M1 and M2 lists from every arrival of the roster (``flightKey, callsign, runway, type, entryUtc,
    landingUtc``) and the census rows of the judged ones (``judge_record``)."""
    m1 = sorted(({**a, **judged[a["flightKey"]]} for a in arrivals
                 if a["flightKey"] in judged and judged[a["flightKey"]]["lossInstants"] > 0),
                key=lambda r: (-r["lossInstants"], r["tightest"], r["flightKey"]))
    m2 = {}
    for length in BLOCK_LENGTHS_S:
        blocks: dict[float, dict[str, Any]] = {}
        for a in arrivals:
            start = parse_iso_utc_s(a["landingUtc"]) // length * length
            block = blocks.setdefault(start, {"startUtc": start, "arrivals": 0, "commandable": 0, "lossInstants": 0,
                                              "runways": set()})
            block["arrivals"] += 1
            block["runways"].add(a["runway"])
            if a["flightKey"] in judged:
                block["commandable"] += 1
                block["lossInstants"] += judged[a["flightKey"]]["lossInstants"]
        m2[str(length)] = sorted(({**b, "startUtc": _iso(b["startUtc"]), "runways": sorted(b["runways"])}
                                  for b in blocks.values()),
                                 key=lambda b: (-b["lossInstants"], -b["arrivals"], b["startUtc"]))
    return {"m1": m1, "m2": m2}


def _iso(t_utc_s: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_utc_s))


def _roster_row(flight: dict[str, Any], key: str, typecode: str | None) -> dict[str, Any]:
    return {"flightKey": key, "callsign": flight["callsign"], "runway": flight["runway"],
            "type": typecode, "entryUtc": flight["entry_time_utc"], "landingUtc": flight["landing_time_utc"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--airport", required=True)
    parser.add_argument("--harvest-root", default=str(HARVEST_ROOT), help="the live harvest root (never harvest-heldout)")
    parser.add_argument("--procedure-root", default=str(DEFAULT_PROCEDURE_ROOT))
    parser.add_argument("--outputs-root", default=str(OUTPUTS_ROOT))
    parser.add_argument("--step-s", type=float, default=1.0, help="the census check step (the M1 loop uses 1 s)")
    parser.add_argument("--jobs", type=int, default=3, help="worker processes (0 = half the cores)")
    parser.add_argument("--limit", type=int, default=0,
                        help="judge only the first N arrivals by landing time, in the whole roster's traffic (a timing "
                             "smoke; 0 = all; stated in the catalog)")
    args = parser.parse_args()

    manifest = Path(args.harvest_root) / args.airport / "arrivals" / "manifest.json"
    started = time.time()
    flights = sorted(load_model_arrivals(manifest), key=lambda f: f["landing_time_utc"])
    traffic = traffic_from_arrivals(flights)                    # the whole roster's traffic, also under --limit
    if args.limit:
        flights = flights[:args.limit]
    payloads, arrivals = [], []
    for flight in flights:
        key = flight_key(flight, 0)                 # the key the scenarios and the traffic carry (scene.recorded_flight)
        arrivals.append(_roster_row(flight, key, traffic.flight(key).typecode))
        try:
            scenario = build_scenario(flight, airport=args.airport, target_from_threshold=True)
        except NoAircraftDynamics:
            continue
        own = traffic.flight(key)
        # the window covers the whole record (one KRDU record runs 2295 s, longer than the M1 horizon)
        horizon = max(DEFAULT_MAX_DURATION_S, own.end_utc_s - own.start_utc_s)
        payloads.append((window_scenario(scenario, traffic, horizon), args.procedure_root, horizon, args.step_s))
    del flights
    print(f"… {len(arrivals)} arrivals, {len(payloads)} with a dynamics model; judging on {args.jobs} worker(s)",
          flush=True)
    limit_solver_threads()
    judged = {}
    with ProcessPoolExecutor(max_workers=resolve_jobs(args.jobs, len(payloads))) as pool:
        for n, row in enumerate(pool.map(judge_record, payloads, chunksize=16), start=1):
            judged[row["flightKey"]] = row
            if n % 1000 == 0:
                print(f"… {n}/{len(payloads)} judged ({time.time() - started:.0f} s)", flush=True)
    lists = build_catalog(arrivals, judged)
    out = catalog_path(Path(args.outputs_root), args.airport)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out, {
        "schema": CATALOG_SCHEMA, "airport": args.airport, "writtenUtc": _iso(time.time()),
        "config": {"reading": rules.VISUAL, "stepS": args.step_s,
                   # the window: max(minHorizonS, the record's duration) after the arrival's entry
                   "minHorizonS": DEFAULT_MAX_DURATION_S, "horizonCoversRecord": True,
                   "manifest": str(manifest.resolve()), "rosterSchema": SCHEMA_VERSION, "limit": args.limit or None},
        "counts": {"arrivals": len(arrivals), "judged": len(judged), "withLoss": len(lists["m1"])},
        **lists})
    print(f"✓ {len(lists['m1'])} of {len(judged)} judged arrivals have a loss they answer for "
          f"({time.time() - started:.0f} s) -> {out}")


if __name__ == "__main__":
    main()
