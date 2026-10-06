"""C10: the validation readout of the chosen round (post-training §8 C10; outline D85; prior D119, D128) — the round's
model on real windows of the validation days, read one time, with the readouts of the selection readout.

THE READ: the chosen round's checkpoint of a campaign (`post_train.round_model`, refused for another base, masks,
traffic shape, seed or round), its first pass (no branch) on at most the campaign's ``select_per_airport`` real windows
of each airport of the val days, drawn as the selection readout draws the select days' (`post_train.selection_windows`:
the seed, D113 applied) with the same random numbers (`readout_numbers`); by airport: the rewards, the outcomes (a loss
of separation among them), the rows the speed-word mask acted (D101), the steps reading a faulty point and the losses
near one (D114) (`post_train.selection_readout`); and each airport's count of real windows, of those left out for
opening inside a loss (D113) and of those read (the coverage stated: val is not read again to learn it). No criterion
is applied (D7). Only the val days are opened: the train and select splits are never read here.

ONCE (D85): the campaign's directory holds the claim of the val read (``val_read_post_validation.json``, prior D119):
the runner takes the read's lock, checks its own options (the campaign, the round's checkpoint) and claims the read,
naming its output and its options (the round, the device: D128, a rerun is the same read), before the val days are
opened; their selection counts are then checked against the base's identity (`require_selection_of`, the val readers'
recount). The read is spent when ``readout.json`` (`CLAIM_SPENT_BY`, the last file written) is in the output; a second
read — another round, another output, or the same one again — is refused by name. A run that stopped before its readout
may be run again to the same output with the same options, its directory moved aside first if it made one (outline
E8). The campaign's recorded paths are read as this checkout reads them (`post_train.inputs_here`, D157: a campaign
recorded in a worktree, read from another checkout). THE CHECKS (§4 item 2) run first: the closed loop's (with the labeller's and the executor's, D69) and the edge
features' reference of the census (D104). From a clean checkout, once the campaign has done every round (the round is
chosen on every round's selection readout, D7); a campaign that is a smoke has no formal read.
``--smoke`` (a tree with changes, a smoke campaign too) reads the select days instead and claims nothing, so the chain is
checked before the one read.

    python run_ts.py post_validation --campaign 4dTrajectory/outputs/POOLED/post/<campaign id> --round <r> \\
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
from ts_transformer.experiments.post_train import (
    CAMPAIGN_SCHEMA, CLAIM_READER, done_rounds, inputs_here, open_context, readout_pool, round_model, selection_readout,
    selection_windows, settings_of, split_data, window_record,
)
from ts_transformer.experiments.post_window_loop import checked_edges
from ts_transformer.io_utils import file_sha256, utc_now, write_json_atomic
from ts_transformer.prior.checkpoint import (CLAIM_SPENT_BY, claim_validation_read, lock_val_read, readable_identity,
                                             settle_written_claim, spend_validation_claim)
from ts_transformer.prior.source import require_selection_of
from ts_transformer.repo_layout import REPO_ROOT, git_state

#: The format of the readout's files (``config.json``, ``readout.json``).
VALIDATION_SCHEMA = "ts-post-validation-v1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--campaign", type=Path, required=True, help="a post_train campaign's directory")
    parser.add_argument("--round", type=int, required=True, help="the chosen round (the user's, D7)")
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true", help="SMOKE: a tree with changes; reads the select days")
    args = parser.parse_args(argv)
    split = "select" if args.smoke else "val"
    campaign, out = (p if p.is_absolute() else REPO_ROOT / p for p in (args.campaign, args.out))
    record = json.loads((campaign / "campaign.json").read_text(encoding="utf-8"))
    if record["schema"] != CAMPAIGN_SCHEMA:
        parser.error(f"{campaign} is a {record['schema']} campaign, not {CAMPAIGN_SCHEMA}")
    if record["inputs"]["smoke"] and not args.smoke:
        parser.error(f"{campaign} is a smoke campaign: it has no formal read of the val days")
    done, settings = done_rounds(campaign), settings_of(record)
    if not 0 <= args.round < done:
        parser.error(f"{campaign} holds the checkpoints of rounds 0–{done - 1}, not round {args.round}")
    if done < settings.rounds and not args.smoke:
        parser.error(f"{campaign} has done {done} of its {settings.rounds} rounds: the val days are read once the round "
                     f"is chosen on every round's selection readout (D7, D85)")
    # the val read's lock, before the output is looked at, kept to the end (D128)
    held = lock_val_read(campaign, CLAIM_READER) if split == "val" else None  # noqa: F841
    if out.exists() and split == "val":
        settle_written_claim(campaign, CLAIM_READER, out)                   # a kill after its readout (D128)
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
        claim_validation_read(campaign, CLAIM_READER, out, {"round": args.round, "device": str(device)})
        require_selection_of(instructions, context.interval_s, context.base_identity, split)
    context = replace(context, splits={split: split_data(instructions, split, context.words, context.interval_s,
                                                         context.geometries, executor)})
    windows = selection_windows(context, settings, split)
    readout = selection_readout(model, context, windows, settings, split)
    real = Counter(w.scene.geometry.code for w in context.splits[split]["windows"])
    read = Counter(w.scene.geometry.code for w in windows)
    coverage = {code: {"real": real[code], "left_out_inside_loss": real[code] - len(pool), "read": read[code]}
                for code, pool in readout_pool(context, split).items()}
    checkpoint = campaign / f"round_{args.round}" / "checkpoint.pt"
    out.mkdir(parents=True)
    write_json_atomic(out / "config.json", {
        "schema": VALIDATION_SCHEMA, "written_utc": utc_now(), "campaign": str(campaign), "round": args.round,
        "checkpoint_sha256": file_sha256(checkpoint), "base": readable_identity(context.base_identity),
        "settings": inputs["settings"], "instructions": str(instructions), "executor": str(executor),
        "checks": opened["checks"], "edges_reference": str(edges_reference), "git": git, "smoke": args.smoke,
        "device": str(device)})
    write_json_atomic(out / CLAIM_SPENT_BY, {
        "schema": VALIDATION_SCHEMA, "split": split, "round": args.round,
        "coverage": coverage, "windows": [window_record(w) for w in windows], "readout": readout})
    if split == "val":
        spend_validation_claim(campaign, CLAIM_READER, out)                     # its readout written: the read is spent
    print(json.dumps({"out": str(out), "reward_mean": {code: r["reward_mean"] for code, r in readout.items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
