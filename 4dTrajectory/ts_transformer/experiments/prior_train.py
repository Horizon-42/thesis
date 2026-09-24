"""The prior's training, one variant (prior design §7, §9 step 1: single-aircraft scenes): teacher forcing on the
sentence artefact's train split, early stopping on val, then the readout on the internal selection split against the
two baselines, the airport's own runway frequency and the causal runway rules.

The artefact is split by operating day (`data.day_split`): train, select (the internal selection set, its own days)
and val; the sealed test days are in none of them. ``--variant`` is one of `prior.data.VARIANTS`. Val is read only
per epoch, for early stopping; its one readout is `prior_select`'s, on the chosen variant. The landing context and the
runway rules come from each airport's tracks roster (`repo_layout.tracks_manifest_path`) less the sealed test days;
the rosters' sha256 are recorded.

Writes into ``--out`` (a new directory, never over an existing one; a clean tree unless ``--limit``, a SMOKE option
that reads only the first flights of each split and says so in ``config.json``): ``checkpoint.pt`` (the best state,
the model config, the train config, schema `PRIOR_CHECKPOINT_SCHEMA`), ``config.json`` (the artefact, its spec,
labeller and day split, the rosters, the git state), ``history.json`` (every epoch) and ``selection_readout.json``.

    python run_ts.py prior_train --variant full --seed 1337 \\
        --instructions 4dTrajectory/outputs/POOLED/instruction_language/v4_20260924 \\
        --out 4dTrajectory/outputs/POOLED/prior/<campaign>/full_s1337
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ts_transformer.instructions.artefact import (
    labeller_source_sha256, load_candidates, load_day_split, load_spec, spec_labeller_source,
)
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.instructions.words import Words
from ts_transformer.io_utils import file_sha256, utc_now, write_json_atomic
from ts_transformer.prior.data import VARIANTS, Split, airport_landings, candidate_table, load_split
from ts_transformer.prior.model import Prior, PriorConfig
from ts_transformer.prior.readout import TOP_K, Baselines, full_readout, runway_rules
from ts_transformer.prior.scene import CONTEXT_WINDOW_S, N_LOOK
from ts_transformer.prior.train import TrainConfig, train
from ts_transformer.repo_layout import REPO_ROOT, git_state, tracks_manifest_path

#: v3 (2026-09-24, prior design §4–§7): single-aircraft scenes on the day split; the first predicted step at row
#: N_LOOK says every column; the words said so far shifted by a step; the landing context; ordered heads; the aircraft
#: axis and its attention. v2 is not opened.
PRIOR_CHECKPOINT_SCHEMA = "ts-prior-checkpoint-v3"


def rosters(instructions: Path) -> dict[str, Path]:
    """Each of the artefact's airports' tracks roster."""
    return {code: tracks_manifest_path(code) for code in load_candidates(instructions)}


def roster_record(tracks: Mapping[str, Path]) -> dict[str, Any]:
    return {code: {"path": str(path), "sha256": file_sha256(path)} for code, path in sorted(tracks.items())}


def load_prior(directory: Path, instructions: Path) -> tuple[Prior, dict[str, Any], dict[str, Any]]:
    """The prior at ``directory`` on CPU, in eval mode, with its checkpoint payload and ``config.json`` — refused unless
    it is a `PRIOR_CHECKPOINT_SCHEMA` checkpoint of ``instructions``' spec, labeller and day split whose candidate table
    is the artefact's and whose state loads whole."""
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
    if config_file["instructions"]["day_split"] != load_day_split(instructions).to_dict():
        raise SystemExit(f"the prior was trained on another day split than {instructions}'s")
    config = PriorConfig.from_dict(payload["model_config"])
    table = candidate_table(load_candidates(instructions), config.airports, config.candidate_slots)
    if not np.array_equal(table, payload["state"]["candidates"].numpy()):
        raise SystemExit("the prior's candidate runways are not the artefact's")
    model = Prior(config, torch.as_tensor(table))
    model.load_state_dict(payload["state"], strict=True)
    model.eval()
    return model, payload, config_file


def splits(instructions: Path, spec: VocabularySpec, words: Words, variant: str, tracks: Mapping[str, Path],
           limit: int | None) -> dict[str, Split]:
    """The artefact's train, select and val flights under ``variant``."""
    landings = airport_landings(instructions, tracks)
    first = load_split(instructions, "train", spec, words, variant, landings=landings, limit=limit)
    return {"train": first, **{name: load_split(instructions, name, spec, words, variant, landings=landings,
                                                airports=first.airports, limit=limit) for name in ("select", "val")}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
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
    tracks = rosters(instructions)

    data = splits(instructions, spec, words, args.variant, tracks, args.limit)
    print(f"variant {args.variant}: " + "; ".join(
        f"{name} {len(s.flights)} flights, {sum(f.rows - N_LOOK for f in s.flights)} predicted steps"
        for name, s in data.items()) + f"; {time.perf_counter() - started:.0f}s", flush=True)
    baselines = Baselines.count(data["train"])
    rules = runway_rules(instructions, tracks, data)

    model_config = PriorConfig(classes=data["train"].classes, airports=data["train"].airports,
                               candidate_slots=data["train"].candidates.shape[1], variant=args.variant)
    torch.manual_seed(config.seed)                                              # the initial weights are the seed's
    model = Prior(model_config, torch.as_tensor(data["train"].candidates)).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"model {parameters} parameters", flush=True)
    best_state, history = train(model, data["train"], data["val"], config, device, lambda line: print(line, flush=True))
    model.load_state_dict(best_state)

    out.mkdir(parents=True)
    torch.save({"schema": PRIOR_CHECKPOINT_SCHEMA, "model_config": model_config.to_dict(), "train_config": asdict(config),
                "state": best_state, "spec_sha256": spec.sha256}, out / "checkpoint.pt")
    write_json_atomic(out / "config.json", {
        "schema": PRIOR_CHECKPOINT_SCHEMA, "written_utc": utc_now(), "model": model_config.to_dict(),
        "train": asdict(config), "parameters": parameters, "n_look": N_LOOK, "context_window_s": CONTEXT_WINDOW_S,
        "instructions": {"directory": str(instructions), "spec_sha256": spec.sha256,
                         "labeller_source_sha256": spec_labeller_source(instructions),
                         "labeller_now": labeller_source_sha256(),
                         "day_split": load_day_split(instructions).to_dict()},
        "tracks_rosters": roster_record(tracks),
        "flights": {name: len(s.flights) for name, s in data.items()},
        "limit": args.limit, "smoke": args.limit is not None, "git": git})
    write_json_atomic(out / "history.json", {"epochs": history})
    best_epoch = min(history, key=lambda row: row["val_nll_per_step"])["epoch"]
    readout = {"split": "select", **full_readout(model, data["select"], baselines, rules, config, device),
               "best_epoch": best_epoch, "epochs": len(history), "elapsed_s": time.perf_counter() - started}
    write_json_atomic(out / "selection_readout.json", readout)
    print_readout(readout)
    print(f"→ {out}")
    return 0


def print_readout(readout: dict[str, Any]) -> None:
    model, baselines = readout["model"], readout["baselines"]
    print(f"{readout['split']} NLL per predicted step: model {model['nll_per_step']:.4f} (after the first "
          f"{model['nll_after_first_per_step']:.4f}), repeat {baselines['repeat']['all']:.4f}, previous word "
          f"{baselines['previous word']['all']:.4f}")
    first = model["first_step_runway"]

    def line(name: str, part: dict[str, Any]) -> str:
        side = part["side_given_direction"]
        return (f"    {name:22s} top1 {part['top1']:.3f}  direction {part['direction']:.3f}  side "
                f"{'—' if side is None else f'{side:.3f}'}  "
                + "  ".join(f"{group} {v['top1']:.3f}" for group, v in part["by_establishment"].items()))

    print(f"  first-step runway: top2 {first['top2']:.3f}  NLL/flight {first['nll_per_flight']:.3f}")
    print(line("model", first))
    print(line("airport frequency", readout["airport_runway_frequency"]))
    for rule, part in readout["rules"].items():
        print(line(rule, part) + f"  fallback {part['fallback_share']:.3f}")
    def share(value: float | None) -> str:
        return "  —  " if value is None else f"{value:.3f}"

    for name, column in model["per_column"].items():
        print(f"  {name:9s} nll {column['nll_per_step']:.4f}  after first {column['nll_after_first_per_step']:.4f}  "
              f"first top1 {column['first_step_top1']:.3f}  changes {column['change_steps']:7d}  "
              f"P(change) {share(column['mean_change_probability_where_changed'])}  "
              f"top1 {share(column['top1_given_change'])}  top{TOP_K} {share(column[f'top{TOP_K}_given_change'])}  "
              f"false change {column['false_change_share_where_kept']:.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
