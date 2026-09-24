"""The prior's step 1: the choice among its variants and the one val readout (prior design §7, §9; the rule in the
readouts doc §4, written before the runs).

Reads every run under ``--campaign`` (each subdirectory a `prior_train` output: `PRIOR_CHECKPOINT_SCHEMA`, a
selection readout). They must be one comparison — the same artefact (spec, labeller, day split), the same tracks
rosters, the same training settings but the seed, the same smoke limit, one commit on a clean tree (unless every run
is a smoke run) — and this code must be that commit.

The rule:

1. the leader: the variant (`prior.data.VARIANTS`) with the lowest NLL per predicted step on the selection split at
   ``--seed``;
2. the seed line: ``full`` at ``--seed`` minus ``full`` at ``--replicate-seed``, in absolute value;
3. the choice: among the variants at ``--seed`` within the seed line of the leader, the first in `PREFERENCE` —
   ``no-context`` (the fewest inputs), then ``full``, then ``unordered`` (the design's heads are ordered: the words said
   at one step fit each other when the prior speaks freely, §5).

Then reads val once on the chosen run, and writes ``choice.json`` into the campaign and the chosen run's
``readout.json`` — val, against the two baselines counted from that run's own training flights, the airport's own
runway frequency and the runway rules; refused if either exists.

    python run_ts.py prior_select --campaign 4dTrajectory/outputs/POOLED/prior/<campaign>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from ts_transformer.experiments.prior_train import (
    PRIOR_CHECKPOINT_SCHEMA, load_prior, print_readout, roster_record, rosters, splits,
)
from ts_transformer.instructions.artefact import load_spec
from ts_transformer.instructions.words import COLUMNS, Words
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.prior.readout import Baselines, full_readout, runway_rules
from ts_transformer.prior.train import TrainConfig
from ts_transformer.repo_layout import REPO_ROOT, git_state

CHOICE_SCHEMA = "ts-prior-choice-v2"
#: The order a variant is chosen in among those within the seed line (rule step 3).
PREFERENCE = ("no-context", "full", "unordered")
#: The variant whose two seeds draw the seed line.
REPLICATED = "full"


def read_runs(campaign: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Every run of the campaign by ``(variant, seed)``, checked to be one comparison."""
    runs: dict[tuple[str, int], dict[str, Any]] = {}
    for directory in sorted(path for path in campaign.iterdir() if path.is_dir()):
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if config["schema"] != PRIOR_CHECKPOINT_SCHEMA:
            raise SystemExit(f"{directory} is not a {PRIOR_CHECKPOINT_SCHEMA} run")
        key = (config["model"]["variant"], config["train"]["seed"])
        if key in runs:
            raise SystemExit(f"{directory} repeats {key}, already {runs[key]['directory']}")
        readout = json.loads((directory / "selection_readout.json").read_text(encoding="utf-8"))
        runs[key] = {"directory": directory, "config": config, "readout": readout}
    if not runs:
        raise SystemExit(f"{campaign} holds no run")
    reference = next(iter(runs.values()))["config"]

    def identity(config: dict[str, Any]) -> str:
        train = {name: value for name, value in config["train"].items() if name != "seed"}
        return json.dumps([config["instructions"], config["tracks_rosters"], train, config["limit"], config["git"]],
                          sort_keys=True)

    for run in runs.values():
        if identity(run["config"]) != identity(reference):
            raise SystemExit(f"{run['directory']} is not the same comparison as the others (artefact, rosters, "
                             f"training settings, smoke limit or commit)")
    if not reference["smoke"] and reference["git"]["dirty"]:
        raise SystemExit("the runs were made on a dirty tree")
    return runs


def choose(runs: dict[tuple[str, int], dict[str, Any]], seed: int, replicate_seed: int) -> dict[str, Any]:
    """The rule's three steps, and every number it read."""
    if seed == replicate_seed:
        raise SystemExit(f"the replicate seed is the seed ({seed}): no seed line")
    missing = [key for key in [*((name, seed) for name in PREFERENCE), (REPLICATED, replicate_seed)] if key not in runs]
    if missing:
        raise SystemExit(f"no run of {missing}")
    nll = {name: runs[(name, seed)]["readout"]["model"]["nll_per_step"] for name in PREFERENCE}
    leader = min(PREFERENCE, key=lambda name: nll[name])
    seed_line = abs(nll[REPLICATED] - runs[(REPLICATED, replicate_seed)]["readout"]["model"]["nll_per_step"])
    within = [name for name in PREFERENCE if nll[name] <= nll[leader] + seed_line]
    return {"leader": leader, "seed_line": seed_line, "within_seed_line": within, "chosen": within[0],
            "nll_per_step": nll}


def run_row(run: dict[str, Any]) -> dict[str, Any]:
    readout, config = run["readout"], run["config"]
    model = readout["model"]
    return {"variant": config["model"]["variant"], "seed": config["train"]["seed"], "directory": run["directory"].name,
            "nll_per_step": model["nll_per_step"], "nll_after_first_per_step": model["nll_after_first_per_step"],
            "first_step_runway": {key: model["first_step_runway"][key]
                                  for key in ("top1", "top2", "direction", "side_given_direction", "by_establishment")},
            "per_column": {name: {key: model["per_column"][name][key]
                                  for key in ("nll_after_first_per_step", "first_step_top1",
                                              "mean_change_probability_where_changed", "top1_given_change",
                                              "false_change_share_where_kept")}
                           for name in COLUMNS},
            "best_epoch": readout["best_epoch"], "epochs": readout["epochs"], "elapsed_s": readout["elapsed_s"],
            "parameters": config["parameters"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--campaign", type=Path, required=True, help="the directory holding the variants' runs")
    parser.add_argument("--seed", type=int, default=TrainConfig().seed)
    parser.add_argument("--replicate-seed", type=int, default=2024)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    campaign = args.campaign if args.campaign.is_absolute() else REPO_ROOT / args.campaign
    runs = read_runs(campaign)
    reference = next(iter(runs.values()))["config"]
    git = git_state()
    if not reference["smoke"] and git != reference["git"]:
        parser.error(f"this code ({git}) is not the runs' commit ({reference['git']})")
    started = time.perf_counter()
    rule = choose(runs, args.seed, args.replicate_seed)
    chosen = runs[(rule["chosen"], args.seed)]
    if (campaign / "choice.json").exists() or (chosen["directory"] / "readout.json").exists():
        parser.error(f"{campaign / 'choice.json'} or the chosen run's readout.json exists; neither is overwritten")
    print(f"leader {rule['leader']}, seed line {rule['seed_line']:.4f}, within it {rule['within_seed_line']} → "
          f"{rule['chosen']}", flush=True)

    config = chosen["config"]
    instructions = Path(config["instructions"]["directory"])
    model, payload, _ = load_prior(chosen["directory"], instructions)
    tracks = rosters(instructions)
    if roster_record(tracks) != config["tracks_rosters"]:
        raise SystemExit("the tracks rosters changed since the runs were trained")
    device = torch.device(args.device)
    model.to(device)
    spec = load_spec(instructions)
    data = splits(instructions, spec, Words(spec), model.config.variant, tracks, config["limit"])
    if data["train"].airports != model.config.airports:
        raise SystemExit("the artefact's airports are not the prior's")
    readout = {"split": "val", **full_readout(model, data["val"], Baselines.count(data["train"]),
                                               runway_rules(instructions, tracks, data),
                                               TrainConfig(**payload["train_config"]), device),
               "best_epoch": chosen["readout"]["best_epoch"], "chosen_by": str(campaign / "choice.json"),
               "elapsed_s": time.perf_counter() - started}
    # both files after the readout: a run that stops before it leaves neither, and can simply be run again
    write_json_atomic(campaign / "choice.json", {
        "schema": CHOICE_SCHEMA, "written_utc": utc_now(), "seed": args.seed, "replicate_seed": args.replicate_seed,
        "preference": list(PREFERENCE), **rule, "chosen_directory": chosen["directory"].name,
        "runs": [run_row(run) for _, run in sorted(runs.items())], "git": git})
    write_json_atomic(chosen["directory"] / "readout.json", readout)
    print_readout(readout)
    print(f"→ {chosen['directory'] / 'readout.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
