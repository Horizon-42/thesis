"""The prior's training, one variant (stage 5; prior design §4, the second version §8): teacher forcing on the
sentence artefact's train split less the train-internal selection set, early stopping on val, then the readout on
the selection set against the two baselines counted from the training flights.

The selection set (§8.3) is ``--selection-share`` of each airport's train flights, drawn with ``--selection-seed``
(`prior.data.selection_ids`); every variant of a comparison draws the same one. The input set ``--inputs`` is one of
`prior.data.INPUT_SETS`. Val is read only per epoch, for early stopping; its one readout is `prior_select`'s, on the
chosen variant.

Writes into ``--out`` (a new directory, never over an existing one; a clean tree unless ``--limit``, a SMOKE option
that reads only the first flights of each split and says so in ``config.json``): ``checkpoint.pt`` (the best state,
the model config, the train config, schema `PRIOR_CHECKPOINT_SCHEMA`), ``config.json`` (the artefact, its spec and
labeller, the selection set's size and sha256, the git state), ``selection.json`` (the selection set's dataset ids),
``history.json`` (every epoch) and ``selection_readout.json`` (the model and the baselines on the selection set).

    python run_ts.py prior_train --inputs V2d --seed 1337 \\
        --instructions 4dTrajectory/outputs/POOLED/instruction_language/v3_20260924 \\
        --out 4dTrajectory/outputs/POOLED/prior/<campaign>/V2d_s1337
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ts_transformer.instructions.artefact import (
    labeller_source_sha256, load_candidates, load_spec, spec_labeller_source,
)
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.instructions.words import Words
from ts_transformer.io_utils import sha256_bytes, utc_now, write_json_atomic
from ts_transformer.prior.data import INPUT_SETS, Split, candidate_table, load_split, partition, selection_ids
from ts_transformer.prior.model import Prior, PriorConfig
from ts_transformer.prior.readout import Baselines, model_readout
from ts_transformer.prior.train import TrainConfig, train
from ts_transformer.repo_layout import REPO_ROOT, git_state

#: v2 (2026-09-24, prior design §8): positions-only state blocks by input set, candidate-runway tokens and a scoring
#: runway head, the hand-over's columns given at step 0, the selection set. v1 is not opened.
PRIOR_CHECKPOINT_SCHEMA = "ts-prior-checkpoint-v2"
SELECTION_SCHEMA = "ts-prior-selection-v1"
DEFAULT_SELECTION_SHARE = 0.1
DEFAULT_SELECTION_SEED = 1337


def selection_record(ids: tuple[str, ...], share: float, seed: int) -> dict[str, Any]:
    return {"share": share, "seed": seed, "flights": len(ids),
            "ids_sha256": sha256_bytes("\n".join(ids).encode("utf-8"))}


def load_prior(directory: Path, instructions: Path) -> tuple[Prior, dict[str, Any], dict[str, Any]]:
    """The prior at ``directory`` on CPU, in eval mode, with its checkpoint payload and ``config.json`` — refused unless
    it is a `PRIOR_CHECKPOINT_SCHEMA` checkpoint of ``instructions``' spec and labeller whose candidate table is the
    artefact's and whose state loads whole."""
    payload = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
    config_file = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    spec = load_spec(instructions)
    if payload["schema"] != PRIOR_CHECKPOINT_SCHEMA or config_file["schema"] != PRIOR_CHECKPOINT_SCHEMA:
        raise SystemExit(f"{directory} is not a {PRIOR_CHECKPOINT_SCHEMA} prior")
    if payload["spec_sha256"] != spec.sha256:
        raise SystemExit(f"the prior was trained on spec {payload['spec_sha256'][:12]}, {instructions} holds "
                         f"{spec.sha256[:12]}")
    if config_file["instructions"]["labeller_source_sha256"] != spec_labeller_source(instructions):
        raise SystemExit(f"the prior was trained on sentences of another labeller than {instructions}'s")
    config = PriorConfig.from_dict(payload["model_config"])
    table = candidate_table(load_candidates(instructions), config.airports, config.candidate_slots)
    if not np.array_equal(table, payload["state"]["candidates"].numpy()):
        raise SystemExit("the prior's candidate runways are not the artefact's")
    model = Prior(config, torch.as_tensor(table))
    model.load_state_dict(payload["state"], strict=True)
    model.eval()
    return model, payload, config_file


def training_flights(instructions: Path, spec: VocabularySpec, words: Words, inputs: str, share: float, seed: int,
                     limit: int | None) -> tuple[Split, Split, tuple[str, ...]]:
    """``(the training flights, the selection set, its ids)`` of the artefact's train split."""
    whole = load_split(instructions, "train", spec, words, inputs, limit=limit)
    ids = selection_ids(whole, share, seed)
    rest, selected = partition(whole, ids)
    return rest, selected, ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    parser.add_argument("--inputs", required=True, choices=sorted(INPUT_SETS))
    parser.add_argument("--selection-share", type=float, default=DEFAULT_SELECTION_SHARE)
    parser.add_argument("--selection-seed", type=int, default=DEFAULT_SELECTION_SEED)
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

    train_split, selection_split, ids = training_flights(instructions, spec, words, args.inputs, args.selection_share,
                                                         args.selection_seed, args.limit)
    val_split = load_split(instructions, "val", spec, words, args.inputs, airports=train_split.airports,
                           limit=args.limit)
    print(f"inputs {args.inputs}: train {len(train_split.flights)} flights, {sum(f.rows for f in train_split.flights)} "
          f"steps; selection {len(selection_split.flights)}; val {len(val_split.flights)}; "
          f"{time.perf_counter() - started:.0f}s", flush=True)
    baselines = Baselines.count(train_split)

    model_config = PriorConfig(classes=train_split.classes, airports=train_split.airports,
                               candidate_slots=train_split.candidates.shape[1], inputs=args.inputs)
    model = Prior(model_config, torch.as_tensor(train_split.candidates)).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"model {parameters} parameters", flush=True)
    best_state, history = train(model, train_split, val_split, config, device, lambda line: print(line, flush=True))
    model.load_state_dict(best_state)

    selection = selection_record(ids, args.selection_share, args.selection_seed)
    out.mkdir(parents=True)
    torch.save({"schema": PRIOR_CHECKPOINT_SCHEMA, "model_config": model_config.to_dict(), "train_config": asdict(config),
                "state": best_state, "spec_sha256": spec.sha256, "selection": selection}, out / "checkpoint.pt")
    write_json_atomic(out / "config.json", {
        "schema": PRIOR_CHECKPOINT_SCHEMA, "written_utc": utc_now(), "model": model_config.to_dict(),
        "train": asdict(config), "parameters": parameters,
        "instructions": {"directory": str(instructions), "spec_sha256": spec.sha256,
                         "labeller_source_sha256": spec_labeller_source(instructions),
                         "labeller_now": labeller_source_sha256()},
        "selection": selection,
        "flights": {"train": len(train_split.flights), "selection": len(selection_split.flights),
                    "val": len(val_split.flights)},
        "limit": args.limit, "smoke": args.limit is not None, "git": git})
    write_json_atomic(out / "selection.json", {"schema": SELECTION_SCHEMA, **selection, "ids": list(ids)})
    write_json_atomic(out / "history.json", {"epochs": history})
    best_epoch = min(history, key=lambda row: row["val_nll_per_step"])["epoch"]
    readout = {"split": "train-selection", "model": model_readout(model, selection_split, config, device),
               "baselines": baselines.nll_per_step(selection_split),
               "airport_runway_reference": baselines.airport_runway_reference(selection_split),
               "best_epoch": best_epoch, "epochs": len(history), "elapsed_s": time.perf_counter() - started}
    write_json_atomic(out / "selection_readout.json", readout)
    print_readout(readout)
    print(f"→ {out}")
    return 0


def print_readout(readout: dict[str, Any]) -> None:
    model, baselines = readout["model"], readout["baselines"]
    print(f"{readout['split']} NLL per step: model {model['nll_per_step']:.4f} (from step 1 "
          f"{model['nll_from_step1_per_step']:.4f}), repeat {baselines['repeat']['all']:.4f}, previous word "
          f"{baselines['previous word']['all']:.4f}")
    step0 = model["step0_runway"]
    print(f"  step 0 runway: top1 {step0['top1']:.3f}  top2 {step0['top2']:.3f}  NLL/flight {step0['nll_per_flight']:.3f}"
          f"  (each airport's own frequency: top1 {readout['airport_runway_reference']['top1']:.3f})")
    for name, column in model["per_column"].items():
        print(f"  {name:9s} nll {column['nll_per_step']:.4f}  from step 1 {column['nll_from_step1_per_step']:.4f}  "
              f"changes {column['change_steps']:7d}  P(change) {column['mean_change_probability_where_changed']:.3f}  "
              f"top1 {column['top1_given_change']:.3f}  top5 {column['top5_given_change']:.3f}  "
              f"false change {column['false_change_share_where_kept']:.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
