"""The prior's training (stage 5, prior design §4–§5): teacher forcing on the sentence artefact's train split, early
stopping on val, then the readout against the two baselines.

Writes into ``--out`` (a new directory, never over an existing one; a clean tree unless ``--limit``, a SMOKE option
that reads only the first flights of each split and says so in ``config.json``): ``checkpoint.pt`` (the best state,
the model config, the train config), ``config.json`` (the artefact, its spec and labeller, the git state),
``history.json`` (every epoch) and ``readout.json`` (val: the model and the baselines).

    python run_ts.py prior_train \\
        --instructions 4dTrajectory/outputs/POOLED/instruction_language/v2_20260924 \\
        --out 4dTrajectory/outputs/POOLED/prior/<name>
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import torch

from ts_transformer.instructions.artefact import labeller_source_sha256, load_spec, spec_labeller_source
from ts_transformer.instructions.words import Words
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.prior.data import load_split
from ts_transformer.prior.model import Prior, PriorConfig
from ts_transformer.prior.readout import Baselines, model_readout
from ts_transformer.prior.train import TrainConfig, train
from ts_transformer.repo_layout import REPO_ROOT, git_state

PRIOR_CHECKPOINT_SCHEMA = "ts-prior-checkpoint-v1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None, help="SMOKE: the first N flights of each split")
    for field, default in asdict(TrainConfig()).items():
        parser.add_argument(f"--{field.replace('_', '-')}", type=type(default), default=default)
    args = parser.parse_args(argv)
    instructions = args.instructions if args.instructions.is_absolute() else REPO_ROOT / args.instructions
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if out.exists():
        parser.error(f"{out} exists; a training run is never overwritten")
    git = git_state()
    if args.limit is None and git["dirty"]:
        parser.error("the tree has uncommitted changes; a training run is made at a commit")
    started = time.perf_counter()
    spec = load_spec(instructions)
    words = Words(spec)
    device = torch.device(args.device)
    config = TrainConfig(**{field: getattr(args, field) for field in asdict(TrainConfig())})

    train_split = load_split(instructions, "train", spec, words, limit=args.limit)
    val_split = load_split(instructions, "val", spec, words, airports=train_split.airports, limit=args.limit)
    print(f"train {len(train_split.flights)} flights, {sum(f.rows for f in train_split.flights)} steps; val "
          f"{len(val_split.flights)} flights; {time.perf_counter() - started:.0f}s", flush=True)
    baselines = Baselines.count(train_split)

    model_config = PriorConfig(classes=train_split.classes, airports=train_split.airports,
                               candidate_slots=train_split.candidates.shape[1])
    model = Prior(model_config, torch.as_tensor(train_split.candidates)).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"model {parameters} parameters", flush=True)
    best_state, history = train(model, train_split, val_split, config, device, lambda line: print(line, flush=True))
    model.load_state_dict(best_state)

    out.mkdir(parents=True)
    torch.save({"schema": PRIOR_CHECKPOINT_SCHEMA, "model_config": model_config.to_dict(), "train_config": asdict(config),
                "state": best_state, "spec_sha256": spec.sha256}, out / "checkpoint.pt")
    write_json_atomic(out / "config.json", {
        "schema": PRIOR_CHECKPOINT_SCHEMA, "written_utc": utc_now(), "model": model_config.to_dict(),
        "train": asdict(config), "parameters": parameters,
        "instructions": {"directory": str(instructions), "spec_sha256": spec.sha256,
                         "labeller_source_sha256": spec_labeller_source(instructions),
                         "labeller_now": labeller_source_sha256()},
        "limit": args.limit, "smoke": args.limit is not None, "git": git})
    write_json_atomic(out / "history.json", {"epochs": history})
    readout = {"split": "val", "model": model_readout(model, val_split, config, device),
               "baselines": baselines.nll_per_step(val_split),
               "best_epoch": min(history, key=lambda row: row["val_nll_per_step"])["epoch"],
               "elapsed_s": time.perf_counter() - started}
    write_json_atomic(out / "readout.json", readout)
    print(f"val NLL per step: model {readout['model']['nll_per_step']:.4f}, repeat "
          f"{readout['baselines']['repeat']['all']:.4f}, previous word {readout['baselines']['previous word']['all']:.4f}")
    for name, column in readout["model"]["per_column"].items():
        print(f"  {name:9s} nll {column['nll_per_step']:.4f}  changes {column['change_steps']:6d}  "
              f"P(change) {column['mean_change_probability_where_changed']:.3f}  top1 {column['top1_given_change']:.3f}  "
              f"top5 {column['top5_given_change']:.3f}  false change {column['false_change_share_where_kept']:.4f}")
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
