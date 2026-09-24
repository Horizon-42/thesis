"""Instruction labeller, step 1: read the development flights (train + val) into per-step signals.

The flights are the live harvest's eligible arrivals, split by `data.splits` (the outer test
split's tracks are never opened), built by the ts data plane (`build_series` + `usable_series`
under the default `TSConfig`, so the population and preprocessing are the models' own; its
state output keeps every flight, `all-flights`, those whose type has no aircraft dynamics
included, since the labeller reads kinematics only) and
projected into each airport's frame (`instructions.signals`). The candidates are each manifest's
published runway geometry; beside them go every runway end the harvest builds, from the
configuration and CIFP the harvest and the evaluator read by default (`evaluation.cli.DEFAULT_CONFIG`,
`DEFAULT_CIFP`). Writes ``signals_{train,val}.npz``, ``signals.json`` and ``candidates.json`` into a
NEW directory.

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
from ts_transformer.config import TSConfig
from ts_transformer.data.data_provenance import arrival_data_provenance
from ts_transformer.data.lateral_eligibility import default_lateral_pass_roster_path
from ts_transformer.data.splits import flight_keys_by_split
from ts_transformer.instructions.airport import AirportGeometry, airport_geometry
from ts_transformer.instructions.artefact import SPLITS, write_candidates, write_signals
from ts_transformer.io_utils import file_sha256, utc_now
from ts_transformer.repo_layout import REPO_ROOT, arrival_manifest_path, discover_k_airports

CHUNK = 500
CONFIG_FIELDS = ("dt_s", "seq_len", "aircraft_filter", "coordinate_frame", "resolved_split_seed",
                 "val_fraction", "test_fraction")


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
    airports = [a.upper() for a in (args.airports or discover_k_airports())]
    manifests = [arrival_manifest_path(a) for a in airports]
    rosters = [default_lateral_pass_roster_path(m) for m in manifests]
    provenance = arrival_data_provenance(manifests, eligibility_rosters=rosters)
    keys = flight_keys_by_split(provenance, config)
    # the candidates: each manifest's published runway geometry (the modeling target's own source),
    # and every runway end the harvest builds (its landing rule's parallel runways)
    geometries = {a: airport_geometry(a, json.loads(m.read_text(encoding="utf-8"))["runway_targets"],
                                      load_airport(a, config_file=DEFAULT_CONFIG, cifp_file=DEFAULT_CIFP).runways)
                  for a, m in zip(airports, manifests)}

    jobs = []
    for split in SPLITS:
        for airport, manifest in zip(airports, manifests):
            mine = sorted(k for k in keys[split] if k.startswith(f"{airport}:"))
            if args.limit:
                mine = mine[: args.limit]
            for start in range(0, len(mine), CHUNK):
                jobs.append((split, airport, str(manifest), mine[start: start + CHUNK]))
    geometry_data = {a: g.to_dict() for a, g in geometries.items()}
    print(f"{len(airports)} airports; {len(keys['train'])} train / {len(keys['val'])} val flights requested "
          f"({len(keys['test'])} test flights stay closed); {len(jobs)} chunks on {args.workers} workers", flush=True)

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
            "arrival_manifest": str(manifest), "arrival_manifest_sha256": entry["arrival_manifest_sha256"],
            "eligibility_roster": str(roster), "eligibility_roster_sha256": file_sha256(roster),
            "eligible_set_sha256": entry["eligibility"]["eligible_set_sha256"],
        } for entry, manifest, roster in zip(provenance["manifests"], manifests, rosters)],
        "counts": {split: {"requested": len(keys[split]), "built_usable": len(collected[split]),
                           "skipped": dict(skipped[split]), "too_short_for_one_window": unusable[split]}
                   for split in SPLITS},
        "runway_ends_from": {"config": str(DEFAULT_CONFIG), "config_sha256": file_sha256(DEFAULT_CONFIG),
                             "cifp": str(DEFAULT_CIFP), "cifp_sha256": file_sha256(DEFAULT_CIFP)},
        "test_flights_not_opened": len(keys["test"]),
        "limit_per_airport_and_split": args.limit or None,
        "typecodes": dict(typecodes.most_common()),
        # The labeller reads kinematics only, so a flight whose type has no dynamics is kept
        # (`all-flights`); how many, and why, against the index that decided it.
        "without_dynamics": dict(without_dynamics.most_common()),
        "performance_index": performance_index_identity(),
    }
    write_signals(out, collected, record)
    write_candidates(out, geometries)
    for split in SPLITS:
        c = record["counts"][split]
        print(f"  {split}: {c['built_usable']} of {c['requested']} flights; skipped {c['skipped']}, "
              f"too short {c['too_short_for_one_window']}")
    print(f"wrote {out} in {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
