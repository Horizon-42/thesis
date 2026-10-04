"""The live executor against the Training export (outline §6 item 6, vocabulary §12.1 A23): every word of every flight of
a published stage-A set, at every Δ, flown through the backend's own request path (`AutopilotSegmentBackend.fly`) and
compared with what the set's sample shows of the same flight —

- the flown states on the 2 s rows both have, against the sample's closed-loop states (written at 0.1 m, so within the
  rounding, `ROUNDING_M`) and against the artefact's stored states (`stored`, within the executor conformance's bound
  `STATE_BOUND_M`);
- a word flown to its outcome (its column's last): the outcome and the crossing with its decision-altitude check, equal
  to the sample's replay.

    python -m aeroviz_backend.autopilot_segment.check_live --set-id <set> --out <directory> [--airport KRDU …]

Writes ``check.json`` (every differing segment named) and prints the summary; exits 1 when any differs. Read-only:
nothing is written outside ``--out``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

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


def compare(answer: dict[str, Any], closed: dict[str, Any], last_word: bool) -> list[str]:
    """What differs between a live answer and the sample's closed-loop sentence of the same flight and Δ."""
    differ = []
    states, track, base = closed["states"], answer["track"], closed["flownFromRow"]
    for k, cycle in enumerate(track["cycle"]):
        row = base + cycle // 2
        if cycle % 2 or row >= states["rows"]:
            continue
        apart = max(abs(track["eM"][k] - states["eM"][row]), abs(track["nM"][k] - states["nM"][row]),
                    abs(track["altitudeMslM"][k] - states["heightMslM"][row]))
        if apart > ROUNDING_M:
            differ.append(f"cycle {cycle}: {apart:.3f} m from the sample's row {row}")
            break
    if max(answer["stored"]["horizontalM"], answer["stored"]["verticalM"]) > STATE_BOUND_M:
        differ.append(f"{answer['stored']} from the artefact's stored states")
    if last_word:
        replayed = closed["replay"]
        if answer["segment"]["end"] != replayed["outcome"]:
            differ.append(f"ended {answer['segment']['end']}, the sample's replay {replayed['outcome']}")
        if answer["crossing"] != replayed["crossing"]:
            differ.append(f"crossing {answer['crossing']}, the sample's {replayed['crossing']}")
    return differ


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--set-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--airport", nargs="+", default=list(AIRPORTS))
    parser.add_argument("--root", type=Path, default=COMPARISON_AIRPORTS_ROOT)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    backend = AutopilotSegmentBackend(airports_root=args.root)
    started, seq, segments, differing, by_end = time.perf_counter(), 0, 0, [], {}
    for airport in args.airport:
        _, sample = backend.training_set(airport, args.set_id)
        for flight in sample["flights"]:
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
                    differ = compare(answer, closed, last[event["column"]] == event["row"])
                    if differ:
                        differing.append({"airport": airport, "flightKey": flight["flightKey"], "rowIntervalS": interval,
                                          "column": COLUMNS[event["column"]], "row": event["row"], "differ": differ})
        print(f"{airport}: {segments} segments so far, {len(differing)} differing, "
              f"{time.perf_counter() - started:.0f} s", flush=True)
    summary = {"checkedUtc": utc_now(), "setId": args.set_id, "airports": args.airport, "segments": segments,
               "differing": len(differing), "ends": by_end, "roundingM": ROUNDING_M, "stateBoundM": STATE_BOUND_M,
               "seconds": round(time.perf_counter() - started, 1)}
    write_json_atomic(args.out / "check.json", {**summary, "segmentsDiffering": differing})
    print(json.dumps(summary))
    return 1 if differing else 0


if __name__ == "__main__":
    sys.exit(main())
