"""Instruction labeller, the check by eye: draw a seeded sample of VAL flights with their
sentences (half straight-in, half vectored) into ``figures/`` of the artefact directory.

Each page: the plan view with the heading words and the capture, the altitude against distance
flown with the altitude and angle words and their tubes, the ground speed against time with the
speed words. An ``index.csv`` lists the pages for a verdict column.

    python run_ts.py instruction_figures --dir 4dTrajectory/outputs/POOLED/instruction_language/<name> --count 24
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from ts_transformer.instructions.artefact import load_candidates, load_signals, load_spec, require_current_labeller
from ts_transformer.instructions.figures import draw_flight
from ts_transformer.instructions.labeller.read import read_flight
from ts_transformer.instructions.labeller.records import Refused
from ts_transformer.instructions.readout import STRATA, flight_record
from ts_transformer.instructions.words import Words
from ts_transformer.repo_layout import REPO_ROOT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=24, help="pages, half per stratum")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)
    directory = args.dir if args.dir.is_absolute() else REPO_ROOT / args.dir
    out = directory / "figures"
    if out.exists():
        parser.error(f"{out} exists; an instruction artefact is never overwritten")
    if args.count % 2:
        parser.error("--count draws half per stratum: give an even number")
    require_current_labeller(directory)
    spec = load_spec(directory)
    words = Words(spec)
    geometries = load_candidates(directory)
    flights = load_signals(directory, "val")
    rng = np.random.default_rng(args.seed)
    wanted = {stratum: args.count // 2 for stratum in STRATA}
    chosen = []
    for index in rng.permutation(len(flights)):
        flight = flights[int(index)]
        try:
            reading = read_flight(flight, geometries[flight.airport], spec, words)
        except Refused:
            continue
        stratum = flight_record(reading)["stratum"]
        if wanted[stratum]:
            wanted[stratum] -= 1
            chosen.append((stratum, flight, reading))
        if not any(wanted.values()):
            break
    out.mkdir()
    with (out / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "dataset_id", "stratum", "verdict (对 / 漏读 / 误读)", "note"])
        for page, (stratum, flight, reading) in enumerate(chosen):
            name = f"{page:02d}_{stratum}_{flight.dataset_id.replace(':', '_')}.png"
            draw_flight(flight, reading, geometries[flight.airport], spec, out / name)
            writer.writerow([name, flight.dataset_id, stratum, "", ""])
    print(f"wrote {len(chosen)} pages to {out} (seed {args.seed}; short strata: "
          f"{ {s: n for s, n in wanted.items() if n} or 'none'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
