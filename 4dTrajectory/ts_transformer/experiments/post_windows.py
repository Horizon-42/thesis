"""Stage C's windows and their census (post-training §8 C1, C2; outline §5 rule 12, D55): for each airport and split
(train, select), the windows of recorded traffic at one Δ, the other aircraft at their first predicted step, the share
with a leader in the air, the windows that admit window D, those near the cut between two operating days and those
that open inside a loss of separation (at the first predicted step, on the record); and,
with the train split, the reference of the edge features (post-training §4 item 1), read again at once.

The user chooses the count of each kind of window in a round from this census (§2 item 4). It writes nothing under
`4dTrajectory/outputs/`: ``--out`` is a new directory (a scratch directory for the census of all train days).
``--sample N``: N windows of each airport and split, drawn at random with seed 1337 (D55) — a smoke.

Only the index of the closed-loop sentences is read (which flights have one at Δ), never a sentence: no closed-loop
check is needed (vocabulary §6 item 3). The validation days are never read (D85).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from flight_scenarios.procedure_final import DEFAULT_PROCEDURE_ROOT
from ts_transformer.instructions.artefact import load_candidates, load_day_split, load_spec
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.post.conformance import REFERENCE_SEED, reference_steps, require_conforming_edges, write_edge_reference
from ts_transformer.post.runways import airport_separation
from ts_transformer.post.scene import airport_scenes, census, inserted_window, leader_moved_window, real_windows
from ts_transformer.post.traffic import loss_at_first_step
from ts_transformer.prior.procedure import airport_finals
from ts_transformer.repo_layout import REPO_ROOT, git_state

CENSUS_SCHEMA = "post-windows-census-v1"
#: The splits a census reads (D85: never val; the test days are sealed, C32).
CENSUS_SPLITS = ("train", "select")
#: The seed of a smoke's sample (D55).
SAMPLE_SEED = 1337
#: The implementer's proposals for the shifts of windows A and D (post-training §8 C1), for the user: A, a flight landing
#: more than `A_APART_S` from the commanded flight's recorded landing, inserted to land within `A_LANDING_SHIFT_S` of it;
#: D, the aircraft ahead moved by a whole number of Δ within `D_SHIFT_S`, never 0.
A_LANDING_SHIFT_S = (-180.0, 180.0)
A_APART_S = 3_600.0
D_SHIFT_S = (-120.0, 120.0)


def first_step_losses(windows, separations, finals, step_s: float) -> dict:
    """For each airport: the windows whose commanded aircraft, on its record, loses separation that it answers for at
    its first predicted step (`post.traffic.loss_at_first_step`) — without a runway in force (as in the loop) and with
    its recorded runway in force — and the kinds of those losses (for the user's decision on windows that open inside a
    loss)."""
    out: dict[str, dict] = {}
    for window in windows:
        code = window.scene.geometry.code
        counted = out.setdefault(code, {"without_runway": 0, "with_recorded_runway": 0,
                                        "kinds": {"without_runway": Counter(), "with_recorded_runway": Counter()}})
        for name, recorded in (("without_runway", False), ("with_recorded_runway", True)):
            loss = loss_at_first_step(window, separations[code], finals[code], step_s, recorded_runway=recorded)
            if loss is not None:
                counted[name] += 1
                counted["kinds"][name][loss.kind] += 1
    return {code: {**counted, "kinds": {name: dict(kinds) for name, kinds in counted["kinds"].items()}}
            for code, counted in out.items()}


def draw_checks(windows, separations, seed: int) -> dict:
    """For each airport: how many windows a draw of A and of D gives (the drawing run once over every window)."""
    rng = np.random.default_rng(seed)
    out: dict[str, dict] = {}
    for window in windows:
        code = window.scene.geometry.code
        counted = out.setdefault(code, {"A": 0, "D": 0})
        counted["A"] += inserted_window(window, rng, landing_shift_s=A_LANDING_SHIFT_S, apart_s=A_APART_S) is not None
        counted["D"] += leader_moved_window(window, separations[code], rng, shift_s=D_SHIFT_S) is not None
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--instructions", type=Path, required=True, help="the sentence artefact")
    parser.add_argument("--interval-s", type=float, required=True, help="Δ (vocabulary D11: the user chose 4 s)")
    parser.add_argument("--out", type=Path, required=True, help="a new directory (not under 4dTrajectory/outputs/)")
    parser.add_argument("--splits", nargs="+", default=list(CENSUS_SPLITS), choices=CENSUS_SPLITS)
    parser.add_argument("--sample", type=int, default=None, help="SMOKE: N windows of each airport and split (D55)")
    parser.add_argument("--procedure-root", type=Path, default=DEFAULT_PROCEDURE_ROOT,
                        help="the CIFP procedures of the finals (prior §7 item 6)")
    args = parser.parse_args(argv)
    instructions, out, procedure_root = (p if p.is_absolute() else REPO_ROOT / p
                                         for p in (args.instructions, args.out, args.procedure_root))
    if out.exists():
        parser.error(f"{out} exists; a census is written to a new directory")
    if out.resolve().is_relative_to((REPO_ROOT / "4dTrajectory" / "outputs").resolve()):
        parser.error("a census writes nothing under 4dTrajectory/outputs/ (outline §5 rules 7 and 12)")
    spec, geometries, days = load_spec(instructions), load_candidates(instructions), load_day_split(instructions)
    separations = {code: airport_separation(g) for code, g in geometries.items()}
    finals = {code: airport_finals(g, root=procedure_root) for code, g in geometries.items()}
    rng = np.random.default_rng(SAMPLE_SEED)
    record: dict = {"schema": CENSUS_SCHEMA, "written_utc": utc_now(), "instructions": str(instructions),
                    "spec_sha256": spec.sha256, "row_interval_s": args.interval_s, "sample": args.sample,
                    "sample_seed": SAMPLE_SEED if args.sample else None, "git": git_state(),
                    "proposals": {"A_landing_shift_s": A_LANDING_SHIFT_S, "A_apart_s": A_APART_S,
                                  "D_shift_s": D_SHIFT_S}, "splits": {}}
    out.mkdir(parents=True)
    for split in args.splits:
        scenes, signals = airport_scenes(instructions, split, spec, args.interval_s, geometries)
        windows = real_windows(instructions, split, spec, args.interval_s, scenes, signals)
        if args.sample is not None:
            chosen = []
            for code in sorted(geometries):
                mine = [w for w in windows if w.scene.geometry.code == code]
                chosen += [mine[int(k)] for k in sorted(rng.choice(len(mine), min(args.sample, len(mine)), replace=False))]
            windows = chosen
        record["splits"][split] = {"airports": census(windows, separations, days),
                                   "augmented": draw_checks(windows, separations, SAMPLE_SEED),
                                   "lost_at_first_step": first_step_losses(windows, separations, finals, spec.step_s)}
        if split == "train":
            path = out / "conformance" / "edges.npz"
            write_edge_reference(path, reference_steps(windows, np.random.default_rng(REFERENCE_SEED)), geometries,
                                 spec.step_s)
            checked = require_conforming_edges(path)
            record["edges_reference"] = {"path": "conformance/edges.npz", "steps": checked.steps,
                                         "tokens": checked.tokens, "max_difference": checked.max_difference}
    write_json_atomic(out / "census.json", record)
    print(json.dumps({"out": str(out), "splits": {s: {c: v["windows"] for c, v in r["airports"].items()}
                                                  for s, r in record["splits"].items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
