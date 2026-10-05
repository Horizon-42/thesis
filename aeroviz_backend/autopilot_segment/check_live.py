"""The live executor against the Training export (outline §6 item 6, vocabulary §12.1 A23): every word of every flight of
a published stage-A set, at every Δ, flown through the backend's own request path (`AutopilotSegmentBackend.fly`) and
compared with what the set's sample shows of the same flight —

- the flown states on the 2 s rows both have, against the sample's closed-loop states (written at 0.1 m, so within the
  rounding, `ROUNDING_M`) and against the artefact's stored states (`stored`, within the executor conformance's bound
  `STATE_BOUND_M`);
- a word flown to its outcome (its column's last): the outcome and the crossing with its decision-altitude check, equal
  to the sample's replay.

    python -m aeroviz_backend.autopilot_segment.check_live --set-id <set> --out <directory> [--airport KRDU …]
        [--flights-per-airport N --seed S]

``--flights-per-airport N`` flies every word of N flights of each airport, drawn in a permutation seeded by ``--seed``
(the user, 2026-10-05: a set is checked on a sample, not on every word; 0, the default: every flight); the sample is in
the summary. Writes ``check.json`` (every differing segment named) and prints the summary; exits 1 when any differs.
Read-only: nothing is written outside ``--out``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import aeroviz_backend.paths  # noqa: F401 — `ts_transformer` lives under 4dTrajectory/

from ts_transformer.autopilot.conformance import STATE_BOUND_M
from ts_transformer.instructions.words import COLUMNS
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.repo_layout import COMPARISON_AIRPORTS_ROOT

from aeroviz_backend.autopilot_segment.backend import AutopilotSegmentBackend

AIRPORTS = ("KMSY", "KRDU", "KSJC", "KSMF", "KSTL")
#: The sample's positions and heights are written at 0.1 m: a live state on the same row is within half of it, and
#: the live answer's own rounding (0.1 m horizontally) adds as much again.
ROUNDING_M = 0.1 + 1e-9


def compare(answer: dict[str, Any], closed: dict[str, Any], step_s: float, to_outcome: bool) -> list[str]:
    """What differs between a live answer and the sample's closed-loop sentence of the same flight and Δ (``step_s``:
    the sample's state rows; ``to_outcome``: the word is its column's last, flown to the outcome)."""
    differ = []
    states, track, base = closed["states"], answer["track"], closed["flownFromRow"]
    every = int(round(step_s / answer["executor"]["cycleS"]))
    for k, cycle in enumerate(track["cycle"]):
        row = base + cycle // every
        if cycle % every or row >= states["rows"]:
            continue
        apart = np.array([track["eM"][k] - states["eM"][row], track["nM"][k] - states["nM"][row],
                          track["altitudeMslM"][k] - states["heightMslM"][row]])
        if not np.abs(apart).max() <= ROUNDING_M:       # a NaN differs too
            differ.append(f"cycle {cycle}: {apart} m from the sample's row {row}")
            break
    if not (answer["stored"]["horizontalM"] <= STATE_BOUND_M and answer["stored"]["verticalM"] <= STATE_BOUND_M):
        differ.append(f"{answer['stored']} from the artefact's stored states")
    if to_outcome:
        replayed = closed["replay"]
        if (answer["segment"]["end"], answer["segment"]["endCycle"]) != (replayed["outcome"], replayed["endCycle"]):
            differ.append(f"ended {answer['segment']['end']} at cycle {answer['segment']['endCycle']}, the sample's "
                          f"replay {replayed['outcome']} at {replayed['endCycle']}")
        if answer["crossing"] != replayed["crossing"]:
            differ.append(f"crossing {answer['crossing']}, the sample's {replayed['crossing']}")
    return differ


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--set-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--airport", nargs="+", default=list(AIRPORTS))
    parser.add_argument("--root", type=Path, default=COMPARISON_AIRPORTS_ROOT)
    parser.add_argument("--flights-per-airport", type=int, default=0, help="a seeded sample of each airport's flights "
                                                                           "(0: every flight)")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)
    if args.flights_per_airport < 0:
        parser.error(f"--flights-per-airport {args.flights_per_airport}: a number of flights, 0 for every one")
    args.out.mkdir(parents=True, exist_ok=False)
    backend = AutopilotSegmentBackend(airports_root=args.root)
    started, seq, segments, differing, by_end, flown = time.perf_counter(), 0, 0, [], {}, {}
    for airport in args.airport:
        _, sample = backend.training_set(airport, args.set_id)
        flights = sample["flights"]
        if args.flights_per_airport:
            order = np.random.default_rng(args.seed).permutation(len(flights))[:args.flights_per_airport]
            flights = [flights[k] for k in sorted(order.tolist())]
        flown[airport] = [flight["flightKey"] for flight in flights]
        for flight in flights:
            for interval, closed in flight["closedLoop"].items():
                last = {}
                for event in closed["events"]:
                    last[event["column"]] = event["row"]
                for event in closed["events"]:
                    seq += 1
                    answer = backend.fly({"clientId": "check_live", "seq": seq, "airport": airport,
                                          "setId": args.set_id, "flightKey": flight["flightKey"],
                                          "rowIntervalS": float(interval), "column": COLUMNS[event["column"]],
                                          "row": event["row"]})
                    segments += 1
                    by_end[answer["segment"]["end"]] = by_end.get(answer["segment"]["end"], 0) + 1
                    differ = compare(answer, closed, sample["vocabulary"]["stepS"], last[event["column"]] == event["row"])
                    if differ:
                        differing.append({"airport": airport, "flightKey": flight["flightKey"], "rowIntervalS": interval,
                                          "column": COLUMNS[event["column"]], "row": event["row"], "differ": differ})
        print(f"{airport}: {segments} segments so far, {len(differing)} differing, "
              f"{time.perf_counter() - started:.0f} s", flush=True)
    summary = {"checkedUtc": utc_now(), "setId": args.set_id, "airports": args.airport,
               "flightsPerAirport": args.flights_per_airport or "every flight", "seed": args.seed,
               "flights": {airport: len(keys) for airport, keys in flown.items()}, "segments": segments,
               "differing": len(differing), "ends": by_end, "roundingM": ROUNDING_M, "stateBoundM": STATE_BOUND_M,
               "seconds": round(time.perf_counter() - started, 1)}
    write_json_atomic(args.out / "check.json", {**summary, "flightKeys": flown, "segmentsDiffering": differing})
    print(json.dumps(summary))
    return 1 if differing else 0


if __name__ == "__main__":
    sys.exit(main())
