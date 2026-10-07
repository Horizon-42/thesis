"""P48: the ceiling readout — how many of a campaign's select windows a model lands at all when it speaks each of them
many times (stage C's requests P48; the user, 2026-10-07). Select days only, never val (outline D85).

THE READ: each of a campaign's models (``--models``: ``start``, the base with zero-output traffic modules as the
campaign starts (D29), or a round, its checkpoint by `post_train.round_model`) reads the selection readout's windows
(`post_train.selection_windows`) ``--draws`` times, their first pass with no branch. Draw 0 uses the readout's own
numbers (`readout_numbers`), so a round's draw 0 is that round's selection readout: it is checked against the round's
``round.json`` before the other draws and refused by name where it differs. Draw d uses numbers of its own
(`readout_numbers(seed, place, d)`). For each window it keeps every draw's outcome and reward. For each model,
``model_<model>.json`` (written when its draws end): in all and by airport, the share of windows landed in at least one
of the first n draws (n = 1, 2, 4, …, N), the mean of the best reward among the first n draws, the outcomes counted,
the windows never landed and those lost to separation in every draw. ``summary.json`` at the end: each model's curve
and how the windows never landed overlap between the models. The flattening of a curve is the most that sampling the
model reaches: a lower bound of the setting's ceiling (another policy might say what sampling never draws). No
criterion is applied.

THE CHECKS run first: the closed loop's (with the labeller's and the executor's, D69) and the edge features' reference
of the census (D104); the campaign's recorded paths are read as this checkout reads them (`inputs_here`). A clean tree
unless ``--smoke``; into a new directory. ``--speak-workers`` reads the batches in worker processes (`Speakers.read`:
the same ends as one process).

    python run_ts.py post_ceiling --campaign 4dTrajectory/outputs/POOLED/post/<campaign id> --models start 6 13 \\
        --draws 32 --out <a new directory> [--speak-workers 4]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ts_transformer.autopilot.closed_loop import require_conforming_closed_loop
from ts_transformer.experiments.post_train import (
    CAMPAIGN_SCHEMA, Speakers, batches, counted_ends, done_rounds, inputs_here, open_context, read_batch, round_model,
    selection_windows, settings_of, window_record,
)
from ts_transformer.experiments.post_window_loop import LOST_SEPARATION, WindowResult, checked_edges
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.post.reward import LANDED
from ts_transformer.repo_layout import REPO_ROOT, git_state

#: The format of the readout's files (``config.json``, ``model_<model>.json``, ``summary.json``).
CEILING_SCHEMA = "ts-post-ceiling-v1"
#: The model at the campaign's start (D29).
START = "start"
#: The split read: the select days (D85: never val).
SPLIT = "select"


def firsts(draws: int) -> list[int]:
    """The counts of first draws the curve is read at: 1, 2, 4, … and ``draws`` itself."""
    out, n = [], 1
    while n < draws:
        out.append(n)
        n *= 2
    return out + [draws]


def ceiling_of(windows: Sequence[Any], outcomes: Sequence[Sequence[str]], rewards: Sequence[Sequence[float]]
               ) -> dict[str, Any]:
    """A model's reading from its windows' draws (``outcomes[w][d]``, ``rewards[w][d]``): in all and by airport."""
    landed = np.array([[o == LANDED for o in row] for row in outcomes], dtype=bool)
    lost = np.array([[o == LOST_SEPARATION for o in row] for row in outcomes], dtype=bool)
    reward = np.array(rewards, dtype=np.float64)
    codes = np.array([w.scene.geometry.code for w in windows])

    def part(mine: np.ndarray) -> dict[str, Any]:
        return {"windows": int(mine.sum()),
                "landed_in_first": {str(n): float(landed[mine, :n].any(axis=1).mean()) for n in firsts(landed.shape[1])},
                "best_reward_of_first": {str(n): float(reward[mine, :n].max(axis=1).mean())
                                         for n in firsts(landed.shape[1])},
                "outcomes": dict(Counter(o for w in np.flatnonzero(mine) for o in outcomes[w])),
                "never_landed": int((~landed[mine].any(axis=1)).sum()),
                "lost_every_draw": int(lost[mine].all(axis=1).sum())}

    return {"all": part(np.ones(len(windows), dtype=bool)),
            "airports": {code: part(codes == code) for code in sorted(set(codes))}}


def in_window_order(places: Sequence[Sequence[int]], read: Sequence[Sequence[WindowResult]], windows: int
                    ) -> list[WindowResult]:
    """The batches' ends (``read``, in batch order) put in the order of the windows."""
    ends: list[WindowResult | None] = [None] * windows
    for batch, got in zip(places, read, strict=True):
        for place, end in zip(batch, got, strict=True):
            ends[place] = end
    assert all(e is not None for e in ends), "a window not read"
    return ends  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--campaign", type=Path, required=True, help="a post_train campaign's directory")
    parser.add_argument("--models", nargs="+", required=True, help=f"'{START}' or a round's number, each once")
    parser.add_argument("--draws", type=int, required=True, help="the draws of each window (draw 0: the readout's)")
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--speak-workers", type=int, default=1, help="processes that read the batches (Speakers)")
    parser.add_argument("--smoke", action="store_true", help="SMOKE: a tree with changes too; no result")
    args = parser.parse_args(argv)
    campaign, out = (p if p.is_absolute() else REPO_ROOT / p for p in (args.campaign, args.out))
    record = json.loads((campaign / "campaign.json").read_text(encoding="utf-8"))
    if record["schema"] != CAMPAIGN_SCHEMA:
        parser.error(f"{campaign} is a {record['schema']} campaign, not {CAMPAIGN_SCHEMA}")
    if record["inputs"]["smoke"] and not args.smoke:
        parser.error(f"{campaign} is a smoke campaign: only a --smoke readout reads it")
    done, settings = done_rounds(campaign), settings_of(record)
    if len(set(args.models)) != len(args.models):
        parser.error(f"--models {args.models}: each model once")
    if any(m != START and not m.isdigit() for m in args.models):
        parser.error(f"--models {args.models}: each '{START}' or a round's number")
    rounds = [None if m == START else int(m) for m in args.models]
    if any(r is not None and not 0 <= r < done for r in rounds):
        parser.error(f"{campaign} holds the checkpoints of rounds 0–{done - 1}: --models {args.models}")
    if args.draws < 1 or args.speak_workers < 1:
        parser.error("--draws and --speak-workers are at least 1")
    if out.exists():
        parser.error(f"{out} exists: the readout writes into a new directory")
    git = git_state()
    if git["dirty"] and not args.smoke:
        parser.error("a readout that is not a smoke needs a clean tree")
    inputs = inputs_here(record["inputs"])         # the paths as this checkout reads them (D157)
    instructions, executor = Path(inputs["instructions"]), Path(inputs["executor"])
    _, opened, _ = require_conforming_closed_loop(instructions, executor)          # D69: the checks run here (D73)
    edges_reference = Path(inputs["windows"]) / "conformance" / "edges.npz"
    checked_edges(edges_reference)                                                 # D104
    device = torch.device(args.device)
    # with workers, the context is opened on the CPU and they are forked before this process uses the GPU
    context = open_context(Path(inputs["prior"]), instructions, executor, edges_reference,
                           torch.device("cpu") if args.speak_workers > 1 else device, Path(inputs["procedure_root"]),
                           formal=not args.smoke, data=False, splits=(SPLIT,))   # D132
    speakers = Speakers(context, settings, args.speak_workers, device) if args.speak_workers > 1 else None
    context = replace(context, device=device, base=context.base.to(device).eval())
    windows = selection_windows(context, settings, SPLIT)
    places = batches(windows, settings.batch_windows)
    out.mkdir(parents=True)
    write_json_atomic(out / "config.json", {
        "schema": CEILING_SCHEMA, "written_utc": utc_now(), "campaign": str(campaign), "models": args.models,
        "draws": args.draws, "settings": inputs["settings"], "instructions": str(instructions),
        "executor": str(executor), "checks": opened["checks"], "git": git, "smoke": args.smoke, "device": str(device),
        "speak_workers": args.speak_workers, "windows": [window_record(w) for w in windows]})
    never: dict[str, list[int]] = {}
    curves: dict[str, Any] = {}
    try:
        for name, round_ in zip(args.models, rounds):
            model = round_model(context, settings, campaign, round_)
            drawn: list[list[WindowResult]] = []
            for draw in range(args.draws):
                read = (speakers.read(model, windows, places, SPLIT, draw) if speakers is not None else
                        [read_batch(model, context, windows, p, settings, SPLIT, draw) for p in places])
                if draw == 0 and round_ is not None:
                    held = json.loads((campaign / f"round_{round_}" / "round.json").read_text(encoding="utf-8"))
                    if json.loads(json.dumps(counted_ends(windows, places, read))) != held["selection_readout"]:
                        raise SystemExit(f"round {round_}: draw 0 is not the round's selection readout (round.json); "
                                         f"the readout is not the campaign's — stopped")
                drawn.append(in_window_order(places, read, len(windows)))
                print(json.dumps({"model": name, "draw": draw}), flush=True)
            outcomes = [[drawn[d][w].outcome for d in range(args.draws)] for w in range(len(windows))]
            rewards = [[drawn[d][w].reward for d in range(args.draws)] for w in range(len(windows))]
            reading = ceiling_of(windows, outcomes, rewards)
            write_json_atomic(out / f"model_{name}.json", {
                "schema": CEILING_SCHEMA, "model": name, "round": round_, "draws": args.draws,
                "draw0_checked": round_ is not None, **reading, "outcomes": outcomes, "rewards": rewards})
            never[name] = [w for w, row in enumerate(outcomes) if LANDED not in row]
            curves[name] = reading["all"]
            print(json.dumps({"model": name, "landed_in_first": reading["all"]["landed_in_first"]}), flush=True)
    finally:
        if speakers is not None:
            speakers.close()
    sets = {name: set(places_) for name, places_ in never.items()}
    write_json_atomic(out / "summary.json", {
        "schema": CEILING_SCHEMA, "written_utc": utc_now(), "draws": args.draws, "curves": curves,
        "never_landed": {name: len(s) for name, s in sets.items()},
        "never_landed_by_every_model": len(set.intersection(*sets.values())),
        "never_landed_pairs": {f"{a}&{b}": len(sets[a] & sets[b]) for i, a in enumerate(sets) for b in list(sets)[i + 1:]}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
