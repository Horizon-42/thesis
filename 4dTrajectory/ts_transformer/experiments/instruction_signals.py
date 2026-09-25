"""Instruction labeller, step 1: read the development flights into per-step signals, split BY OPERATING DAY.

The day split is the committed deal (`data.day_split.pinned_day_split`, prior design §3.3) of the
operating days of every harvested airport's assigned landings into test / val / select / train; a
harvest whose days differ from it is refused. A flight belongs to the day it lands on. The flights are the live harvest's eligible arrivals on the development days (a test day's
flights are never opened — only counted from the roster), built by the ts data plane (`build_series`
+ `usable_series` under the default `TSConfig`, so the population and preprocessing are the models'
own; its state output keeps every flight, `all-flights`, those whose type has no aircraft dynamics
included, since the labeller reads kinematics only) and projected into each airport's frame
(`instructions.signals`). The candidates are each manifest's published runway geometry; beside them
go every runway end the harvest builds, from the configuration and CIFP the harvest and the
evaluator read by default (`evaluation.cli.DEFAULT_CONFIG`, `DEFAULT_CIFP`). Every candidate must
publish a threshold crossing height there (the executor's crossing point for "descend to land",
`autopilot.runway_data.crossing_heights`): a runway without one is refused before anything is
written, never dropped quietly. Writes ``signals_{train,select,val}.npz``, ``signals.json`` (with the
day split) and ``candidates.json`` into a NEW directory.

    python run_ts.py instruction_signals --out 4dTrajectory/outputs/POOLED/instruction_language/<name>
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from aircraft.performance_index import performance_index_identity
from evaluation.cli import DEFAULT_CIFP, DEFAULT_CONFIG
from trajectory_data_process.harvest.airports import load_airport
from ts_transformer.autopilot.runway_data import crossing_heights
from ts_transformer.config import TSConfig
from ts_transformer.data.data_provenance import arrival_data_provenance
from ts_transformer.data.day_split import (
    DAY_SPLITS, PINNED_DAY_SPLIT, DaySplit, harvest_operating_days, landing_day, pinned_day_split,
)
from ts_transformer.data.lateral_eligibility import default_lateral_pass_roster_path
from ts_transformer.data.splits import SPLIT_ASSIGNMENT_METHOD, split_name_for_dataset_id
from ts_transformer.instructions.airport import AirportGeometry, airport_geometry
from ts_transformer.instructions.artefact import SPLITS, write_candidates, write_signals
from ts_transformer.io_utils import file_sha256, utc_now
from ts_transformer.repo_layout import REPO_ROOT, arrival_manifest_path, discover_k_airports, tracks_manifest_path

CHUNK = 500
CONFIG_FIELDS = ("dt_s", "seq_len", "aircraft_filter", "coordinate_frame")


def keys_by_day(provenance: dict[str, Any], manifests: dict[str, Path], days: DaySplit) -> dict[str, list[str]]:
    """The eligible arrivals' airport-qualified keys by the split of the day they land on — read from
    the arrival rosters' landing times (``manifests``: airport → its arrivals manifest), no track opened."""
    result: dict[str, list[str]] = {split: [] for split in DAY_SPLITS}
    for entry in provenance["manifests"]:
        landing = {row["flight_key"]: row["landing_time_utc"]
                   for row in json.loads(manifests[entry["airport"]].read_text(encoding="utf-8"))["records"]}
        for record in entry["source_records"]:
            day = landing_day(landing[record["flight_key"]])
            result[days.split_of(day)].append(f"{entry['airport']}:{record['flight_key']}")
    return result


def build_jobs(keys: dict[str, list[str]], manifests: dict[str, Path], limit: int) -> list[tuple[str, str, str, list[str]]]:
    """The chunks to build: ``(split, airport, manifest, keys)`` for the development splits only — a test day's
    key never reaches a worker. ``limit``: the first N flights per airport and split (0: all)."""
    jobs = []
    for split in SPLITS:
        for airport, manifest in manifests.items():
            mine = sorted(k for k in keys[split] if k.startswith(f"{airport}:"))
            if limit:
                mine = mine[:limit]
            for start in range(0, len(mine), CHUNK):
                jobs.append((split, airport, str(manifest), mine[start: start + CHUNK]))
    return jobs


def _build(geometry_data: dict[str, Any], manifest: str, keys: list[str]) -> tuple[list[Any], dict[str, Any], int, Counter, Counter]:
    """One chunk of one airport: flight dicts → series → usable series → signals, with the
    types and the flights without aircraft dynamics counted over the USABLE series (the ones
    that become signals; the build report's counts include flights `usable_series` drops)."""
    from ts_transformer.data.dataset import build_series, load_flight_dicts
    from ts_transformer.instructions.signals import signals_from_series
    from ts_transformer.training.train import usable_series

    config = TSConfig()
    geometry = AirportGeometry.from_dict(geometry_data)
    flights = load_flight_dicts([manifest], include_flight_keys=set(keys), verbose=False)
    built, report = build_series(flights, config)
    usable = usable_series(built, config, verbose=False)
    typecodes = Counter(str(item.scenario.source["resolved_typecode"] or "unresolved") for item in usable)
    without_dynamics = Counter(str(item.scenario.source["no_dynamics_reason"])
                               for item in usable if not item.scenario.has_dynamics)
    return ([signals_from_series(item, geometry) for item in usable], report.to_dict(), len(built) - len(usable),
            typecodes, without_dynamics)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    parser.add_argument("--airports", nargs="+", default=None, help="default: every harvested airport")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0,
                        help="SMOKE TEST ONLY: the first N flights per airport and split (recorded in signals.json)")
    args = parser.parse_args(argv)
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if out.exists():
        parser.error(f"{out} exists; an instruction artefact is never overwritten")
    started = time.perf_counter()
    config = TSConfig()
    airports = sorted(a.upper() for a in (args.airports or discover_k_airports()))
    manifests = {a: arrival_manifest_path(a) for a in airports}
    rosters = {a: default_lateral_pass_roster_path(m) for a, m in manifests.items()}
    provenance = arrival_data_provenance(list(manifests.values()), eligibility_rosters=list(rosters.values()))
    # one committed deal for all airports; the days of EVERY harvested airport must be its days, whichever are read
    days = pinned_day_split()
    harvest = set(harvest_operating_days([tracks_manifest_path(a) for a in discover_k_airports()]))
    if harvest != days.listed:
        parser.error(f"the harvest's operating days are not the pinned day split's ({PINNED_DAY_SPLIT.name}): "
                     f"{sorted(harvest - days.listed)} new, {sorted(days.listed - harvest)} gone — extending the "
                     f"split is a decision, never a re-deal")
    keys = keys_by_day(provenance, manifests, days)
    # the candidates: each manifest's published runway geometry (the modeling target's own source),
    # and every runway end the harvest builds (its landing rule's parallel runways)
    runways = {a: load_airport(a, config_file=DEFAULT_CONFIG, cifp_file=DEFAULT_CIFP).runways for a in airports}
    geometries = {a: airport_geometry(a, json.loads(m.read_text(encoding="utf-8"))["runway_targets"], runways[a])
                  for a, m in manifests.items()}
    # a candidate runway must publish a threshold crossing height: "descend to land" crosses it there (the executor's
    # crossing point, executor design §5.2) — refused before anything is written, never dropped quietly
    for airport, geometry in geometries.items():
        crossing_heights(geometry, runways[airport])

    jobs = build_jobs(keys, manifests, args.limit)
    geometry_data = {a: g.to_dict() for a, g in geometries.items()}
    print(f"{len(airports)} airports; days {', '.join(f'{s} {len(d)}' for s, d in days.days.items())}; flights requested "
          f"{', '.join(f'{s} {len(keys[s])}' for s in SPLITS)} ({len(keys['test'])} on test days stay closed); "
          f"{len(jobs)} chunks on {args.workers} workers", flush=True)

    collected: dict[str, list[Any]] = {split: [] for split in SPLITS}
    skipped: dict[str, Counter] = {split: Counter() for split in SPLITS}
    typecodes: Counter = Counter()
    without_dynamics: Counter = Counter()
    unusable: Counter = Counter()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
        futures = [(split, airport, pool.submit(_build, geometry_data[airport], manifest, chunk))
                   for split, airport, manifest, chunk in jobs]
        for done, (split, airport, future) in enumerate(futures, start=1):
            signals, report, short, chunk_typecodes, chunk_without_dynamics = future.result()
            collected[split] += signals
            skipped[split].update(report["skipped"])
            typecodes.update(chunk_typecodes)
            without_dynamics.update(chunk_without_dynamics)
            unusable[split] += short
            if done % 10 == 0 or done == len(futures):
                print(f"  {done}/{len(futures)} chunks, {sum(len(v) for v in collected.values())} flights, "
                      f"{time.perf_counter() - started:.0f}s", flush=True)
    for split in SPLITS:
        collected[split].sort(key=lambda item: item.dataset_id)

    out.mkdir(parents=True)
    record = {
        "written_utc": utc_now(),
        "config": {name: getattr(config, name) for name in CONFIG_FIELDS},
        "sources": [{
            "airport": entry["airport"],
            "arrival_manifest": str(manifests[entry["airport"]]), "arrival_manifest_sha256": entry["arrival_manifest_sha256"],
            "eligibility_roster": str(rosters[entry["airport"]]),
            "eligibility_roster_sha256": file_sha256(rosters[entry["airport"]]),
            "eligible_set_sha256": entry["eligibility"]["eligible_set_sha256"],
        } for entry in provenance["manifests"]],
        "counts": {split: {"requested": len(keys[split]), "built_usable": len(collected[split]),
                           "skipped": dict(skipped[split]), "too_short_for_one_window": unusable[split]}
                   for split in SPLITS},
        "runway_ends_from": {"config": str(DEFAULT_CONFIG), "config_sha256": file_sha256(DEFAULT_CONFIG),
                             "cifp": str(DEFAULT_CIFP), "cifp_sha256": file_sha256(DEFAULT_CIFP)},
        # the sealed days' flights, counted from the roster; the ones the per-flight split (`data.splits`, the
        # other ts models') also holds out are the flights no model on either line has seen
        "test_days": {"flights_not_opened": len(keys["test"]),
                      "of_which_in_the_flight_split_test": sum(split_name_for_dataset_id(k, config) == "test"
                                                               for k in keys["test"]),
                      "flight_split": {"method": SPLIT_ASSIGNMENT_METHOD, "seed": config.resolved_split_seed,
                                       "val_fraction": config.val_fraction, "test_fraction": config.test_fraction}},
        "limit_per_airport_and_split": args.limit or None,
        "typecodes": dict(typecodes.most_common()),
        # The labeller reads kinematics only, so a flight whose type has no dynamics is kept
        # (`all-flights`); how many, and why, against the index that decided it.
        "without_dynamics": dict(without_dynamics.most_common()),
        "performance_index": performance_index_identity(),
    }
    write_signals(out, collected, record, days)
    write_candidates(out, geometries)
    for split in SPLITS:
        c = record["counts"][split]
        print(f"  {split}: {c['built_usable']} of {c['requested']} flights; skipped {c['skipped']}, "
              f"too short {c['too_short_for_one_window']}")
    print(f"wrote {out} in {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
