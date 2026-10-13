"""MC1: the census of stage D's windows (multi-aircraft control §11 MC1, §3.1, D146, O16; `multi/`) — for the user to
choose the span L of a window, its kinds and c_min.

For each split (train, select; never val, D85; the test days are sealed, C32), each span L and each kind (the real
windows and, for each c_min, the compressed ones): every flight with a closed-loop sentence at Δ as an anchor
(`multi.windows.Anchors`), its window, and the census of `multi.census` — the commanded aircraft of a window, the
recorded aircraft at each first predicted step, the windows left out (D146) and by which aircraft, the losses of
separation on the records by pair (D145), and the baseline of §5 item 4: the same windows with the commanded aircraft at
their stored closed-loop states (the executor's own losses; the closed loop's checks run first, vocabulary D69, D73).

A census is a measurement, not a formal build (outline §5 rules 7, 12; D55): it writes ``census.json`` into a new
directory outside `4dTrajectory/outputs/` (a scratch directory). ``--sample N``: N anchors of each airport and split,
drawn at random with seed 1337, the sample stated in the record (never silent). A compressed window draws its c from a
generator of the seed, the span, c_min and the anchor's place.

    python run_ts.py multi_windows --instructions <A34's artefact> --executor <its spec> --interval-s 4 \\
        --out <scratch>/multi_windows_census
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from flight_scenarios.procedure_final import DEFAULT_PROCEDURE_ROOT
from ts_transformer.autopilot.closed_loop import require_conforming_closed_loop
from ts_transformer.instructions.artefact import closed_loop_sentences, load_candidates, load_spec
from ts_transformer.instructions.words import Words
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.multi.census import closed_loop_positions, summary, window_count
from ts_transformer.multi.windows import REAL_KIND, Anchors, Drawn, compressed
from ts_transformer.post.runways import airport_separation
from ts_transformer.post.scene import airport_scenes, real_windows
from ts_transformer.prior.procedure import airport_finals
from ts_transformer.repo_layout import OPT_OUTPUTS_ROOT as OUTPUTS
from ts_transformer.repo_layout import REPO_ROOT, git_state

CENSUS_SCHEMA = "multi-windows-census-v1"
#: The splits a census reads (D85: never val).
CENSUS_SPLITS = ("train", "select")
#: The spans of MC1 (min) and the c_min of its compressed windows (multi-aircraft control §11 MC1).
SPANS_MIN = (0.0, 5.0, 10.0, 20.0)
C_MINS = (0.6, 0.8)
#: The seed of a sample and of the compressed windows' c (D55).
SEED = 1337


def chosen_anchors(windows, sample: int | None, rng: np.random.Generator):
    """Every real window as an anchor, or ``sample`` of each airport drawn at random (D55)."""
    if sample is None:
        return list(windows)
    out = []
    for code in sorted({w.scene.geometry.code for w in windows}):
        mine = [w for w in windows if w.scene.geometry.code == code]
        out += [mine[int(k)] for k in sorted(rng.choice(len(mine), min(sample, len(mine)), replace=False))]
    return out


def split_census(anchors: Anchors, chosen, sentences, words: Words, separations, finals, spans_min, c_mins,
                 step_s: float) -> dict:
    """One split's census (module docstring): for each span and kind, the summary on the records and the baseline's."""
    out: dict = {}
    for span in spans_min:
        kinds: dict[str, list[Drawn]] = {REAL_KIND: []}
        for place, anchor in enumerate(chosen):
            window = anchors.window_of(anchor, span * 60.0)
            kinds[REAL_KIND].append(Drawn(window, REAL_KIND))
            for k, c_min in enumerate(c_mins):
                if window.joined:
                    drawn = compressed(window, np.random.default_rng([SEED, int(span * 60), k, place]), c_min)
                    kinds.setdefault(f"compressed_{c_min:g}", []).append(drawn)
        by_kind = {}
        for kind, drawn in kinds.items():
            records, baseline = [], []
            for item in drawn:
                code = item.window.scene.geometry.code
                records.append(window_count(item, separations[code], finals[code], step_s))
                baseline.append(window_count(item, separations[code], finals[code], step_s,
                                             positions=closed_loop_positions(item.window, sentences, words)))
            by_kind[kind] = {"windows": len(drawn), "records": summary(records), "baseline": summary(baseline)}
        out[f"{span:g}min"] = by_kind
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--instructions", type=Path, required=True, help="the sentence artefact")
    parser.add_argument("--executor", type=Path, required=True, help="the directory of the artefact's executor spec")
    parser.add_argument("--interval-s", type=float, required=True, help="Δ (vocabulary D11: the user chose 4 s)")
    parser.add_argument("--out", type=Path, required=True, help="a new directory outside 4dTrajectory/outputs/")
    parser.add_argument("--splits", nargs="+", default=list(CENSUS_SPLITS), choices=CENSUS_SPLITS)
    parser.add_argument("--spans-min", nargs="+", type=float, default=list(SPANS_MIN))
    parser.add_argument("--c-min", nargs="+", type=float, default=list(C_MINS))
    parser.add_argument("--sample", type=int, default=None, help="N anchors of each airport and split (D55)")
    parser.add_argument("--procedure-root", type=Path, default=DEFAULT_PROCEDURE_ROOT,
                        help="the CIFP procedures of the finals (prior §7 item 6)")
    args = parser.parse_args(argv)
    instructions, executor, out, procedure_root = (p if p.is_absolute() else REPO_ROOT / p for p in (
        args.instructions, args.executor, args.out, args.procedure_root))
    if out.exists():
        parser.error(f"{out} exists; a census is written to a new directory")
    if out.resolve().is_relative_to(OUTPUTS.resolve()):
        parser.error("a census of MC1 is a measurement: it writes outside 4dTrajectory/outputs/ (outline §5 rules 7, 12)")
    if any(span < 0 for span in args.spans_min) or any(not 0.0 < c < 1.0 for c in args.c_min):
        parser.error("a span is not negative, a c_min is in (0, 1)")
    _, checks, words = require_conforming_closed_loop(instructions, executor)        # D69, D73: before any sentence
    spec, geometries = load_spec(instructions), load_candidates(instructions)
    separations = {code: airport_separation(g) for code, g in geometries.items()}
    finals = {code: airport_finals(g, root=procedure_root) for code, g in geometries.items()}
    rng = np.random.default_rng(SEED)
    record: dict = {"schema": CENSUS_SCHEMA, "written_utc": utc_now(), "instructions": str(instructions),
                    "spec_sha256": spec.sha256, "row_interval_s": args.interval_s, "spans_min": args.spans_min,
                    "c_min": args.c_min, "sample": args.sample, "sample_seed": SEED if args.sample else None,
                    "git": git_state(), "checks": checks, "splits": {}}
    out.mkdir(parents=True)
    for split in args.splits:
        scenes, signals = airport_scenes(instructions, split, spec, args.interval_s, geometries)
        windows = real_windows(instructions, split, spec, args.interval_s, scenes, signals)
        sentences = closed_loop_sentences(instructions, split, args.interval_s, spec)
        chosen = chosen_anchors(windows, args.sample, rng)
        record["splits"][split] = {"anchors": len(chosen), "real_windows": len(windows),
                                   "spans": split_census(Anchors(windows), chosen, sentences, words, separations,
                                                         finals, args.spans_min, args.c_min, spec.step_s)}
        write_json_atomic(out / "census.json", {**record, "complete": False})     # each split kept as it is done
    record["complete"] = True
    write_json_atomic(out / "census.json", record)
    print(json.dumps({"out": str(out), "anchors": {s: r["anchors"] for s, r in record["splits"].items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
