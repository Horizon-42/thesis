"""The prior's second version: the choice among its variants and the one val readout (prior design §8.3).

Reads every run under ``--campaign`` (each subdirectory a `prior_train` output: `PRIOR_CHECKPOINT_SCHEMA`, a
selection readout). They must be one comparison — the same artefact (spec and labeller), the same selection set (its
sha256), the same training settings but the seed, the same smoke limit, one commit on a clean tree (unless every run
is a smoke run) — and this code must be that commit.

The rule (written into §8.3 before the runs):

1. the leader: the choosable variant (`prior.data.CHOOSABLE`) with the lowest NLL per step on the selection set at
   ``--seed`` (step 0 counts only the runway);
2. the seed line: the leader's NLL at ``--seed`` minus its NLL at ``--replicate-seed``, in absolute value;
3. the choice: among the choosable variants at ``--seed`` whose NLL is within the seed line of the leader's, the one
   adding the fewest inputs (`InputSet.added`); a tie by the lower NLL.

Variant 0 (the first version's fitted inputs) takes no part: it measures the future those inputs carry — its gap to
the others in the probability of a change on the steps where one is said is written beside the choice.

``--leader`` prints the leader's name and stops (the queue trains its replicate next). Otherwise reads val once on the
chosen run, then writes ``choice.json`` into the campaign and the chosen run's ``readout.json`` — val, the model and the
two baselines counted from that run's own training flights; refused if either exists.

    python run_ts.py prior_select --campaign 4dTrajectory/outputs/POOLED/prior/<campaign> [--leader]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from ts_transformer.experiments.prior_train import (
    PRIOR_CHECKPOINT_SCHEMA, load_prior, print_readout, selection_record, training_flights,
)
from ts_transformer.instructions.artefact import load_spec
from ts_transformer.instructions.words import COLUMNS, Words
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.prior.data import CHOOSABLE, INPUT_SETS, load_split
from ts_transformer.prior.readout import Baselines, model_readout
from ts_transformer.prior.train import TrainConfig
from ts_transformer.repo_layout import REPO_ROOT, git_state

CHOICE_SCHEMA = "ts-prior-choice-v1"
FUTURE_VARIANT = "V0"


def read_runs(campaign: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Every run of the campaign by ``(inputs, seed)``, checked to be one comparison."""
    runs: dict[tuple[str, int], dict[str, Any]] = {}
    for directory in sorted(path for path in campaign.iterdir() if path.is_dir()):
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if config["schema"] != PRIOR_CHECKPOINT_SCHEMA:
            raise SystemExit(f"{directory} is not a {PRIOR_CHECKPOINT_SCHEMA} run")
        key = (config["model"]["inputs"], config["train"]["seed"])
        if key in runs:
            raise SystemExit(f"{directory} repeats {key}, already {runs[key]['directory']}")
        readout = json.loads((directory / "selection_readout.json").read_text(encoding="utf-8"))
        runs[key] = {"directory": directory, "config": config, "readout": readout}
    if not runs:
        raise SystemExit(f"{campaign} holds no run")
    reference = next(iter(runs.values()))["config"]

    def identity(config: dict[str, Any]) -> tuple:
        train = {name: value for name, value in config["train"].items() if name != "seed"}
        return (config["instructions"]["spec_sha256"], config["instructions"]["labeller_source_sha256"],
                config["instructions"]["directory"], config["selection"]["ids_sha256"], json.dumps(train, sort_keys=True),
                config["limit"], json.dumps(config["git"], sort_keys=True))

    for key, run in runs.items():
        if identity(run["config"]) != identity(reference):
            raise SystemExit(f"{run['directory']} is not the same comparison as the others (artefact, selection set, "
                             f"training settings, smoke limit or commit)")
    if not reference["smoke"] and reference["git"]["dirty"]:
        raise SystemExit("the runs were made on a dirty tree")
    return runs


def leader(runs: dict[tuple[str, int], dict[str, Any]], seed: int) -> str:
    missing = [name for name in CHOOSABLE if (name, seed) not in runs]
    if missing:
        raise SystemExit(f"no run of {missing} at seed {seed}")
    return min(CHOOSABLE, key=lambda name: runs[(name, seed)]["readout"]["model"]["nll_per_step"])


def choose(runs: dict[tuple[str, int], dict[str, Any]], seed: int, replicate_seed: int) -> dict[str, Any]:
    """The rule's three steps, and every number it read."""
    first = leader(runs, seed)
    if (first, replicate_seed) not in runs:
        raise SystemExit(f"no replicate of the leader {first} at seed {replicate_seed}")
    nll = {name: runs[(name, seed)]["readout"]["model"]["nll_per_step"] for name in CHOOSABLE}
    seed_line = abs(nll[first] - runs[(first, replicate_seed)]["readout"]["model"]["nll_per_step"])
    within = [name for name in CHOOSABLE if nll[name] <= nll[first] + seed_line]
    chosen = min(within, key=lambda name: (INPUT_SETS[name].added, nll[name]))
    return {"leader": first, "seed_line": seed_line, "within_seed_line": within, "chosen": chosen,
            "nll_per_step": nll}


def run_row(run: dict[str, Any]) -> dict[str, Any]:
    model = run["readout"]["model"]
    config = run["config"]
    return {"inputs": config["model"]["inputs"], "seed": config["train"]["seed"],
            "directory": run["directory"].name, "nll_per_step": model["nll_per_step"],
            "nll_from_step1_per_step": model["nll_from_step1_per_step"],
            "step0_runway_top1": model["step0_runway"]["top1"], "step0_runway_top2": model["step0_runway"]["top2"],
            "step0_runway_nll_per_flight": model["step0_runway"]["nll_per_flight"],
            "step0_runway_by_handover_approach": model["step0_runway"]["by_handover_approach"],
            "per_column": {name: {key: model["per_column"][name][key]
                                  for key in ("nll_from_step1_per_step", "mean_change_probability_where_changed",
                                              "top1_given_change", "false_change_share_where_kept")}
                           for name in COLUMNS},
            "best_epoch": run["readout"]["best_epoch"], "epochs": run["readout"]["epochs"],
            "elapsed_s": run["readout"]["elapsed_s"], "parameters": config["parameters"]}


def future_gap(runs: dict[tuple[str, int], dict[str, Any]], seed: int) -> dict[str, Any]:
    """Variant 0 minus each choosable variant, per column: the probability of a change and the value's top-1 on the
    steps where one is said."""
    future = runs[(FUTURE_VARIANT, seed)]["readout"]["model"]["per_column"]
    out = {}
    for name in CHOOSABLE:
        other = runs[(name, seed)]["readout"]["model"]["per_column"]
        out[name] = {column: {key: future[column][key] - other[column][key]
                              for key in ("mean_change_probability_where_changed", "top1_given_change")}
                     for column in COLUMNS}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--campaign", type=Path, required=True, help="the directory holding the variants' runs")
    parser.add_argument("--seed", type=int, default=TrainConfig().seed)
    parser.add_argument("--replicate-seed", type=int, default=2024)
    parser.add_argument("--leader", action="store_true", help="print the leader's name and stop")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    campaign = args.campaign if args.campaign.is_absolute() else REPO_ROOT / args.campaign
    runs = read_runs(campaign)
    if args.leader:
        print(leader(runs, args.seed))
        return 0
    reference = next(iter(runs.values()))["config"]
    git = git_state()
    if not reference["smoke"] and git != reference["git"]:
        parser.error(f"this code ({git}) is not the runs' commit ({reference['git']})")
    if (FUTURE_VARIANT, args.seed) not in runs:
        parser.error(f"no run of {FUTURE_VARIANT} at seed {args.seed}")
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
    device = torch.device(args.device)
    model.to(device)
    spec = load_spec(instructions)
    words = Words(spec)
    inputs = model.config.inputs
    selection = config["selection"]
    train_split, _, ids = training_flights(instructions, spec, words, inputs, selection["share"], selection["seed"],
                                           config["limit"])
    if selection_record(ids, selection["share"], selection["seed"]) != selection:
        raise SystemExit("the selection set drawn now is not the one the run was trained without")
    val_split = load_split(instructions, "val", spec, words, inputs, airports=model.config.airports,
                           limit=config["limit"])
    baselines = Baselines.count(train_split)
    train_config = TrainConfig(**payload["train_config"])
    readout = {"split": "val", "model": model_readout(model, val_split, train_config, device),
               "baselines": baselines.nll_per_step(val_split),
               "airport_runway_reference": baselines.airport_runway_reference(val_split),
               "best_epoch": chosen["readout"]["best_epoch"], "chosen_by": str(campaign / "choice.json"),
               "elapsed_s": time.perf_counter() - started}
    # both files after the readout: a run that stops before it leaves neither, and can simply be run again
    write_json_atomic(campaign / "choice.json", {
        "schema": CHOICE_SCHEMA, "written_utc": utc_now(), "seed": args.seed, "replicate_seed": args.replicate_seed,
        **rule, "chosen_directory": chosen["directory"].name,
        "runs": [run_row(run) for _, run in sorted(runs.items())],
        "future_information": {"variant": FUTURE_VARIANT, "minus": future_gap(runs, args.seed)}, "git": git})
    write_json_atomic(chosen["directory"] / "readout.json", readout)
    print_readout(readout)
    print(f"→ {chosen['directory'] / 'readout.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
