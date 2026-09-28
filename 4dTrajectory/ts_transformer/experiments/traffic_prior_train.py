"""Multi-aircraft M2 (design §6.1, §6.5 step 4): the scene prior "scene" — base's recipe (prior design §7) on scene samples.

The same network and training configuration as base (`prior_train`, whose defaults base was trained with), teacher
forcing on the training days' scene samples (`traffic_scene_data.build_split`, design §2.3) with every aircraft's edge
features (`inference.scene_edges.EDGE_FEATURES`, design §2.5; one variant, one seed — §9 item 14), early stopping on the
val days' samples as base stopped (its readout is M2's, on the select days). A batch holds samples of similar size, at
most ``--tokens-per-batch`` padded aircraft-steps (a larger sample is a batch of its own); each sample's edge features are
computed as its batch is formed. The loss is the columns' cross entropy summed over the cells asked — a flight with a
sentence, from its own first predicted step, on the sample's loss steps — per aircraft-step asked.

The same token budget is a smaller batch than base's: base's tokens are the asked flights' own steps, a scene sample's
also hold background aircraft, context before the cut and the aircraft axis' padding — 1,976 batches per epoch on the
training days against base's 531. So the gradient of ``--accumulate`` (`ACCUMULATE`) consecutive batches is summed
before each update, the loss per aircraft-step asked over all of them: 494 updates per epoch of about 16,700 asked steps
each, against base's 531 of about 15,600 (design §6.5 step 6, §9 item 21; the first formal run, `scene_s1337`, updated
after every batch). The warm-up counts updates, as base's. ``history.json`` records each epoch's batches, updates and
asked steps.

Writes into ``--out`` (a new directory; a clean tree unless ``--limit``, a SMOKE option that reads only the first samples
of each split and says so; a sample of background aircraft alone asks nothing and is left out, counted):
``checkpoint.pt`` (`prior_train.SCENE_CHECKPOINT_SCHEMA`: base's payload plus the edge
features and `traffic_scene_data.edge_source_sha256`), ``config.json`` (the artefact, its spec, labeller and day split, the rosters, what the samples hold, the batches per update, the git
state), ``procedure_masks.json`` (none: teacher forcing speaks under no procedure's masks) and ``history.json``.

    python run_ts.py traffic_prior_train --variant full --seed 1337 \\
        --instructions 4dTrajectory/outputs/POOLED/instruction_language/v5_20260926 \\
        --out 4dTrajectory/outputs/POOLED/prior/<campaign>/scene_s1337
"""

from __future__ import annotations

import argparse
import copy
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np
import torch
from torch import nn

from ts_transformer.experiments.prior_train import SCENE_CHECKPOINT_SCHEMA, roster_record, rosters
from ts_transformer.experiments.traffic_scene_data import Built, build_split, edge_source_sha256, edges
from ts_transformer.inference.scene_edges import EDGE_FEATURES
from ts_transformer.instructions.artefact import (
    labeller_source_sha256, load_candidates, load_day_split, load_spec, spec_labeller_source,
)
from ts_transformer.instructions.words import Words
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.prior.data import VARIANTS, airport_landings, candidate_table, column_classes
from ts_transformer.prior.masks import ProcedureMasks, write_masks
from ts_transformer.prior.model import Prior, PriorConfig
from ts_transformer.prior.scene import CONTEXT_WINDOW_S, N_LOOK
from ts_transformer.prior.scene_data import batch_tensors
from ts_transformer.prior.train import TrainConfig, batch_logits, column_nll
from ts_transformer.repo_layout import REPO_ROOT, git_state

RUNNER = "ts_transformer.experiments.traffic_prior_train"
#: Batches per update: the integer nearest 1,976 / 531, the training days' batches per epoch here and base's (module
#: docstring).
ACCUMULATE = 4


def scene_batches(built: Sequence[Built], tokens: int, rng: np.random.Generator | None) -> Iterator[list[int]]:
    """Indices of samples in batches of similar size, each at most ``tokens`` padded aircraft-steps (a larger sample is a
    batch of its own); shuffled when ``rng`` is given, else in size order."""
    def size(i: int) -> tuple[int, int]:
        return len(built[i].sample.keys), built[i].sample.steps

    order = sorted(range(len(built)), key=lambda i: (size(i)[0] * size(i)[1], i))
    groups, current, aircraft, steps = [], [], 0, 0
    for i in order:
        a, t = size(i)
        if current and max(aircraft, a) * max(steps, t) * (len(current) + 1) > tokens:
            groups.append(current)
            current, aircraft, steps = [], 0, 0
        current.append(i)
        aircraft, steps = max(aircraft, a), max(steps, t)
    if current:
        groups.append(current)
    if rng is not None:
        rng.shuffle(groups)
    yield from groups


def scene_batch(built: Sequence[Built], indices: Sequence[int], slots: int, device: torch.device) -> dict[str, torch.Tensor]:
    return batch_tensors([built[i].sample for i in indices], [edges(built[i]) for i in indices], slots, device)


def scene_nll(model: Prior, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, int]:
    """``(summed NLL per column [6], the aircraft-steps asked)`` of one batch of scene samples."""
    nll = column_nll(batch_logits(model, batch), batch["targets"], batch["present"], batch["asked"])
    return nll, int(batch["asked"].any(dim=-1).sum())


def asked_steps(built: Built) -> int:
    """The aircraft-steps its sample asks (the steps any column is asked at)."""
    return sum(int(built.sample.asked(node).any(axis=1).sum()) for node in built.sample.nodes)


def accumulate(model: Prior, batches: Iterable[dict[str, torch.Tensor]], asked: int, where: str) -> float:
    """Backpropagate ``batches`` one at a time (their gradients add up), each batch's summed NLL divided by ``asked``,
    the aircraft-steps all of them ask — the gradient of their loss per aircraft-step asked, taken as one batch; returns
    that loss."""
    total = 0.0
    for batch in batches:
        nll, _ = scene_nll(model, batch)
        part = nll.sum() / asked
        if not torch.isfinite(part):
            raise FloatingPointError(f"{where}: the loss is {float(part)}")
        part.backward()
        total += float(part.detach())
    return total


@torch.no_grad()
def scene_evaluate(model: Prior, built: Sequence[Built], config: TrainConfig, slots: int, device: torch.device
                   ) -> dict[str, Any]:
    model.eval()
    total, steps = torch.zeros(6, dtype=torch.float64, device=device), 0
    for indices in scene_batches(built, config.tokens_per_batch, None):
        nll, asked = scene_nll(model, scene_batch(built, indices, slots, device))
        total += nll.double()
        steps += asked
    per_column = (total / steps).cpu().numpy()
    return {"nll_per_step": float(per_column.sum()), "per_column": per_column.tolist(), "steps": steps}


def scene_train(model: Prior, train_built: Sequence[Built], val_built: Sequence[Built], config: TrainConfig, slots: int,
                accumulate_batches: int, device: torch.device, log: Callable[[str], None]
                ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """`prior.train.train` on scene samples: the same optimiser, schedule, clipping and early stopping, an update after
    every ``accumulate_batches`` batches."""
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda step: min(1.0, (step + 1) / config.warmup_steps))
    best, best_state, stale, history = float("inf"), copy.deepcopy(model.state_dict()), 0, []
    asked_by_sample = [asked_steps(b) for b in train_built]
    for epoch in range(1, config.max_epochs + 1):
        started = time.perf_counter()
        model.train()
        total, steps, batches, updates = 0.0, 0, 0, 0
        groups = list(scene_batches(train_built, config.tokens_per_batch, rng))
        for first in range(0, len(groups), accumulate_batches):
            update = groups[first: first + accumulate_batches]
            asked = sum(asked_by_sample[i] for indices in update for i in indices)
            optimiser.zero_grad(set_to_none=True)
            loss = accumulate(model, (scene_batch(train_built, indices, slots, device) for indices in update), asked,
                              f"epoch {epoch}")
            nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm)
            optimiser.step()
            schedule.step()
            total += loss * asked
            steps += asked
            batches += len(update)
            updates += 1
        val = scene_evaluate(model, val_built, config, slots, device)
        if not np.isfinite(val["nll_per_step"]):
            raise FloatingPointError(f"epoch {epoch}: the val NLL is {val['nll_per_step']}")
        row = {"epoch": epoch, "train_nll_per_step": total / steps, "val_nll_per_step": val["nll_per_step"],
               "val_per_column": val["per_column"], "batches": batches, "updates": updates, "asked_steps": steps,
               "seconds": time.perf_counter() - started}
        history.append(row)
        improved = val["nll_per_step"] < best
        if improved:
            best, best_state, stale = val["nll_per_step"], copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        log(f"epoch {epoch:2d}  train {row['train_nll_per_step']:.4f}  val {row['val_nll_per_step']:.4f}"
            f"{'  (best)' if improved else ''}  {row['seconds']:.0f}s")
        if stale >= config.patience:
            break
    return best_state, history


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None, help="SMOKE: the first N samples of each split")
    parser.add_argument("--accumulate", type=int, default=ACCUMULATE, help="batches per update (module docstring)")
    for field, default in asdict(TrainConfig()).items():
        parser.add_argument(f"--{field.replace('_', '-')}", type=type(default), default=default)
    args = parser.parse_args(argv)
    if args.accumulate < 1:
        parser.error("--accumulate is at least 1")
    instructions = args.instructions if args.instructions.is_absolute() else REPO_ROOT / args.instructions
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if out.exists():
        parser.error(f"{out} exists; a training run is never overwritten")
    git = git_state()
    if args.limit is None and git["dirty"]:
        parser.error("the tree has uncommitted changes; a training run is made at a commit")
    started = time.perf_counter()
    edge_source = edge_source_sha256()
    spec = load_spec(instructions)
    device = torch.device(args.device)
    config = TrainConfig(**{field: getattr(args, field) for field in asdict(TrainConfig())})
    tracks = rosters(instructions)
    geometries = load_candidates(instructions)
    airports = tuple(sorted(geometries))
    slots = max(len(g.candidates) for g in geometries.values())
    model_config = PriorConfig(classes=column_classes(Words(spec), slots), airports=airports, candidate_slots=slots,
                               variant=args.variant)
    landings = airport_landings(instructions, tracks) if VARIANTS[args.variant].landing_context else None
    data, counts = {}, {}
    for name in ("train", "val"):
        built, counts[name] = build_split(instructions, name, spec, airports, landings, model_config.max_rows)
        # a sample of background aircraft alone asks nothing: nothing to learn or to score
        counts[name]["samples_asking_nothing"] = sum(not b.sample.asks for b in built)
        data[name] = [b for b in built if b.sample.asks][: args.limit]
    print("; ".join(f"{name} {len(b)} samples ({counts[name]})" for name, b in data.items())
          + f"; {time.perf_counter() - started:.0f}s", flush=True)

    torch.manual_seed(config.seed)                                              # the initial weights are the seed's
    model = Prior(model_config, torch.as_tensor(candidate_table(geometries, airports, slots)), EDGE_FEATURES).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"model {parameters} parameters, {len(EDGE_FEATURES)} edge features", flush=True)
    best_state, history = scene_train(model, data["train"], data["val"], config, slots, args.accumulate, device,
                                      lambda line: print(line, flush=True))
    out.mkdir(parents=True)
    torch.save({"schema": SCENE_CHECKPOINT_SCHEMA, "model_config": model_config.to_dict(), "train_config": asdict(config),
                "state": best_state, "spec_sha256": spec.sha256, "edge_features": list(EDGE_FEATURES),
                "edge_source_sha256": edge_source},
               out / "checkpoint.pt")
    write_json_atomic(out / "config.json", {
        "schema": SCENE_CHECKPOINT_SCHEMA, "written_utc": utc_now(), "model": model_config.to_dict(),
        "edge_features": list(EDGE_FEATURES), "edge_source_sha256": edge_source, "train": asdict(config),
        "accumulate": args.accumulate, "parameters": parameters, "n_look": N_LOOK,
        "context_window_s": CONTEXT_WINDOW_S,
        "instructions": {"directory": str(instructions), "spec_sha256": spec.sha256,
                         "labeller_source_sha256": spec_labeller_source(instructions),
                         "labeller_now": labeller_source_sha256(),
                         "day_split": load_day_split(instructions).to_dict()},
        "tracks_rosters": roster_record(tracks), "samples": counts,
        "limit": args.limit, "smoke": args.limit is not None, "git": git, "elapsed_s": time.perf_counter() - started})
    write_masks(out, ProcedureMasks.none(), writer=RUNNER, git=git)
    write_json_atomic(out / "history.json", {"epochs": history})
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
