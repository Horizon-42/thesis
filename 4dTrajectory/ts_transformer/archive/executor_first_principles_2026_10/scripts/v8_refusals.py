#!/usr/bin/env python
"""instruction-v8 on every train flight of the local instruction artefact (`signals_train.npz`, the train split only):
how many flights the new rule refuses ("no final straight-in line", or a fitted angle outside the classes), and how
often the last angle word changes class. Each flight is read twice at leads 4/4/4 by the two-tier2 worktree's
labeller: as instruction-v8 reads it, and with the last piece read at its own angle (the reading before v8, by passing
the last piece's own angle as the line). Nothing is written outside --out.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import new_design_eval as nde   # noqa: E402  (puts the worktree on sys.path)

import numpy as np   # noqa: E402

from ts_transformer.instructions.artefact import load_candidates, load_signals   # noqa: E402
from ts_transformer.instructions.labeller import read as read_module             # noqa: E402
from ts_transformer.instructions.labeller.records import Refused                 # noqa: E402
from ts_transformer.instructions.labeller.vertical import vertical_pieces        # noqa: E402
from ts_transformer.instructions.words import ANGLE, Words                       # noqa: E402

SPEC = nde.spec_of(4.0, 4.0, 4.0)
WORDS = Words(SPEC)
GEOMETRIES = load_candidates(nde.INSTRUCTIONS)
V8_LINE = read_module.final_path_angle_deg


def own_line(relative, height_m, runway):
    """The reading before instruction-v8: the last piece's own angle stands in for the line (the caller's
    `read_heights` reads the same pieces)."""
    return OWN["angle"]


OWN = {"angle": None}


def last_angle(reading):
    return [i.value for i in reading.instructions if i.column == ANGLE][-1]


def read_both(flight):
    geometry = GEOMETRIES[flight.airport]
    out = {"airport": flight.airport}
    try:
        reading = read_module.read_flight(flight, geometry, SPEC, WORDS)
        out["v8"], out["v8_last"] = "read", last_angle(reading)
    except Refused as refusal:
        out["v8"] = refusal.reason
    try:
        admitted = read_module.admit(flight, geometry, SPEC)
        height = admitted.smoothed.altitude_m - geometry.elevation_m
        OWN["angle"] = vertical_pieces(admitted.smoothed.distance_m, height, SPEC, WORDS)[-1].angle_deg
        read_module.final_path_angle_deg = own_line
        reading = read_module.read_flight(flight, geometry, SPEC, WORDS)
        out["old"], out["old_last"] = "read", last_angle(reading)
    except Refused as refusal:
        out["old"] = refusal.reason
    finally:
        read_module.final_path_angle_deg = V8_LINE
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists()
    args.out.mkdir(parents=True)
    started = time.perf_counter()
    flights = load_signals(nde.INSTRUCTIONS, "train")
    with np.load(nde.INSTRUCTIONS / "sentences_train.npz") as data:
        labelled = {int(i) for i in data["signal_index"]}
    print(f"{len(flights)} train flights, {len(labelled)} labelled in the v7 artefact, {time.perf_counter() - started:.0f}s",
          flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(read_both, flights, chunksize=200))
    result = {}
    for name, members in (("all", range(len(rows))), ("labelled_v7", sorted(labelled))):
        part = [rows[i] for i in members]
        both = [r for r in part if r["v8"] == "read" and r["old"] == "read"]
        result[name] = {
            "flights": len(part),
            "read_old": sum(r["old"] == "read" for r in part), "read_v8": sum(r["v8"] == "read" for r in part),
            "refused_v8_only": dict(Counter(r["v8"] for r in part if r["old"] == "read" and r["v8"] != "read")),
            "read_v8_only": dict(Counter(r["old"] for r in part if r["v8"] == "read" and r["old"] != "read")),
            "refused_v8_only_by_airport": dict(Counter(r["airport"] for r in part
                                                       if r["old"] == "read" and r["v8"] != "read")),
            "last_angle_changed": sum(r["v8_last"] != r["old_last"] for r in both),
            "last_angle_old_to_v8": {f"{a}->{b}": n for (a, b), n in
                                     Counter((r["old_last"], r["v8_last"]) for r in both).most_common()},
        }
        print(name, json.dumps(result[name]), flush=True)
    (args.out / "results.json").write_text(json.dumps(result, indent=1))
    print(f"→ {args.out / 'results.json'}, {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
