"""MC4: the validation readout of stage D's chosen round (multi-aircraft control §5 item 3; outline D85; post-training
D132; prior D119, D128) — the round's model on stage D's windows of the validation days, read one time, with the
readouts of the selection readout.

THE READ: the chosen round's checkpoint of a campaign of stage D (`multi_train.round_model`, refused for another start,
base, masks, token part, settings or round), its first pass (no branch) on at most the campaign's ``select_per_airport``
real windows of span L of each airport of the val days, drawn as the selection readout draws the select days'
(`multi_train.selection_windows`: the seed, D146's rule) with the same random numbers; by airport and kind: W per
aircraft and per window, the outcomes, the go-arounds, the silent aircraft, the rows of the speed-word mask, the faulty
points, the losses of separation by pair, the time the aircraft take (`multi_train.selection_readout`); and each
airport's anchors and windows read (the coverage stated). No criterion is applied (D7). Only the val days are opened.

ONCE (D85, as stage C's `post_validation`): the campaign's directory holds the claim of the val read (its reader
`multi_train.MULTI_CLAIM_READER`, prior D119): the read's lock, the runner's own options checked (the campaign, the
round's checkpoint), the claim naming its output and options (the round, the device: D128), before the val days are
opened, their selection counts checked against the base's identity (`require_selection_of`). The read is spent when
``readout.json`` (`CLAIM_SPENT_BY`) is in the output; a second read is refused by name; a run that stopped before its
readout may be run again to the same output, its directory moved aside first. From a clean checkout (a run worktree,
outline D163), once the campaign has done every round; a smoke campaign has no formal read. ``--smoke`` reads the
select days instead and claims nothing. THE CHECKS run first: the closed loop's (D69) and the edge features' (D104).

    python run_ts.py multi_validation --campaign 4dTrajectory/outputs/POOLED/multi/<campaign id> --round <r> \\
        --out <a new directory>
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import torch

from ts_transformer.autopilot.closed_loop import require_conforming_closed_loop
from ts_transformer.experiments.multi_train import (
    MULTI_CLAIM_READER, multi_settings_of, round_model, selection_readout, selection_windows, stage_d, window_record,
)
from ts_transformer.experiments.post_train import done_rounds, inputs_here, open_context, split_data
from ts_transformer.experiments.post_window_loop import checked_edges
from ts_transformer.io_utils import file_sha256, utc_now, write_json_atomic
from ts_transformer.prior.checkpoint import (
    CLAIM_SPENT_BY, claim_validation_read, lock_val_read, readable_identity, settle_written_claim,
    spend_validation_claim,
)
from ts_transformer.prior.source import require_selection_of
from ts_transformer.repo_layout import REPO_ROOT, git_state

#: The format of the readout's files (``config.json``, ``readout.json``).
MULTI_VALIDATION_SCHEMA = "ts-multi-validation-v1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--campaign", type=Path, required=True, help="a multi_train campaign's directory")
    parser.add_argument("--round", type=int, required=True, help="the chosen round (the user's, D7)")
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true", help="SMOKE: a tree with changes; reads the select days")
    args = parser.parse_args(argv)
    split = "select" if args.smoke else "val"
    campaign, out = (p if p.is_absolute() else REPO_ROOT / p for p in (args.campaign, args.out))
    record = json.loads((campaign / "campaign.json").read_text(encoding="utf-8"))
    try:
        settings = multi_settings_of(record)
    except ValueError as refused:
        parser.error(f"{campaign}: {refused}")
    if record["inputs"]["smoke"] and not args.smoke:
        parser.error(f"{campaign} is a smoke campaign: it has no formal read of the val days")
    done = done_rounds(campaign)
    if not 0 <= args.round < done:
        parser.error(f"{campaign} holds the checkpoints of rounds 0–{done - 1}, not round {args.round}")
    if done < settings.rounds and not args.smoke:
        parser.error(f"{campaign} has done {done} of its {settings.rounds} rounds: the val days are read once the round "
                     f"is chosen on every round's selection readout (D7, D85)")
    # the val read's lock, before the output is looked at, kept to the end (D128)
    held = lock_val_read(campaign, MULTI_CLAIM_READER) if split == "val" else None  # noqa: F841
    if out.exists() and split == "val":
        settle_written_claim(campaign, MULTI_CLAIM_READER, out)              # a kill after its readout (D128)
    if out.exists() and not (out / CLAIM_SPENT_BY).exists():
        parser.error(f"{out} exists without its {CLAIM_SPENT_BY}: a run that stopped; move it aside "
                     f"({out.name}.aborted-<UTC>, outline E8) and run again to the same output")
    if out.exists():
        parser.error(f"{out} exists; the validation days are read once (D85)")
    git = git_state()
    if git["dirty"] and not args.smoke:
        parser.error("a readout that is not a smoke needs a clean tree")
    inputs = inputs_here(record["inputs"])         # the paths as this checkout reads them (D157)
    instructions, executor = Path(inputs["instructions"]), Path(inputs["executor"])
    _, opened, _ = require_conforming_closed_loop(instructions, executor)          # D69: the checks run here (D73)
    edges_reference = Path(inputs["windows"]) / "conformance" / "edges.npz"
    checked_edges(edges_reference)                                                 # D104
    device = torch.device(args.device)
    context = open_context(Path(inputs["prior"]), instructions, executor, edges_reference, device,
                           Path(inputs["procedure_root"]), formal=not args.smoke, data=False, splits=())   # D132
    model = round_model(context, settings, campaign, args.round)                   # the read's own options, checked
    if split == "val":
        claim_validation_read(campaign, MULTI_CLAIM_READER, out, {"round": args.round, "device": str(device)})
        require_selection_of(instructions, context.interval_s, context.base_identity, split)
    context = replace(context, splits={split: split_data(instructions, split, context.words, context.interval_s,
                                                         context.geometries, executor)})
    windows = selection_windows(context, settings, split)
    readout = selection_readout(model, context, windows, settings, None, stage=stage_d(), split=split)
    anchors = Counter(w.scene.geometry.code for w in context.splits[split]["windows"])
    read = Counter(w.scene.geometry.code for w in windows)
    coverage = {code: {"anchors": anchors[code], "read": read[code]} for code in sorted(anchors)}
    checkpoint = campaign / f"round_{args.round}" / "checkpoint.pt"
    out.mkdir(parents=True)
    write_json_atomic(out / "config.json", {
        "schema": MULTI_VALIDATION_SCHEMA, "written_utc": utc_now(), "campaign": str(campaign), "round": args.round,
        "checkpoint_sha256": file_sha256(checkpoint), "base": readable_identity(context.base_identity),
        "settings": inputs["settings"], "instructions": str(instructions), "executor": str(executor),
        "checks": opened["checks"], "edges_reference": str(edges_reference), "git": git, "smoke": args.smoke,
        "device": str(device)})
    write_json_atomic(out / CLAIM_SPENT_BY, {
        "schema": MULTI_VALIDATION_SCHEMA, "split": split, "round": args.round, "coverage": coverage,
        "windows": [window_record(w) for w in windows], "readout": readout})
    if split == "val":
        spend_validation_claim(campaign, MULTI_CLAIM_READER, out)               # its readout written: the read is spent
    print(json.dumps({"out": str(out), "reward_mean": {code: r["reward_mean"] for code, r in readout.items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
