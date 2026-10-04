"""Train one prior (prior design §5, §8, §12 B3): a run on the train days of every airport, or one fold of the
cross-validation (D39) with ``--held-out``, stopped on the select days of its training airports (D31).

Reads the sentence artefact through the vocabulary's public interface at one row interval (``--row-interval-s``), only
after the closed-loop check with the artefact's executor spec (``--executor``) passes in this process
(`autopilot.closed_loop.require_conforming_closed_loop`, D69, D73), and each
airport's landings from its tracks roster (D63). Never reads the validation days. Before training it checks the memory
at the formal size: the largest batches, one forward and backward pass each on the device, and refuses
a run that would not fit (``memory.json``).

``--memory-check-only`` stops after the check (it writes ``config.json`` and ``memory.json``): the check at the formal
size before a formal run, from any tree.

Writes into ``--out`` (a new directory, never over an existing one; a clean tree unless ``--sample``, a SMOKE option
that trains on a random sample of N sentences of each airport and split, seed 1337, D55, and says so in
``config.json``): ``checkpoint.pt`` (`prior.checkpoint`, the identity of the data, §8 item 1), ``config.json``,
``memory.json``, ``history.json`` (every epoch), ``procedure_masks.json`` (§8 item 2: the set it speaks under, the
checkpoint's sha256, the CIFP procedure documents' digests) and, for a fold, ``held_out.json`` (the per-step loss on
the held-out airport's select days: the fold's score, §5).

    python run_ts.py prior_train --instructions 4dTrajectory/outputs/POOLED/instruction_language/<artefact> \\
        --row-interval-s 2 --executor 4dTrajectory/outputs/POOLED/executor/<spec> \\
        --variant full --selection landed --out <a new directory> [--held-out KSJC] [--sample 200]
"""

from __future__ import annotations

import argparse
import json
import resource
from dataclasses import fields
from pathlib import Path

import torch

from flight_scenarios.procedure_final import DEFAULT_PROCEDURE_ROOT
from ts_transformer.autopilot.closed_loop import require_conforming_closed_loop
from ts_transformer.instructions.artefact import load_candidates, load_day_split
from ts_transformer.io_utils import file_sha256, utc_now, write_json_atomic
from ts_transformer.prior.batch import VARIANTS
from ts_transformer.prior.checkpoint import CHECKPOINT_SCHEMA, save_checkpoint
from ts_transformer.prior.model import Prior, PriorConfig
from ts_transformer.prior.procedure import PROCEDURE_MASKS, procedure_digests
from ts_transformer.prior.runs import SAMPLE_SEED, Run, held_out_sentences, run_data, sampled_run_data
from ts_transformer.prior.selection import RULES, selection_totals
from ts_transformer.prior.source import ArtefactSource, airport_landings, artefact_identity
from ts_transformer.prior.train import TrainConfig, evaluate, largest_batch_memory, train
from ts_transformer.repo_layout import REPO_ROOT, git_state

#: The GPU must hold the largest batches' reserved peak and the training state this many times over: the batches of a
#: run differ in shape, so the allocator's cache grows past one batch's peak (Claude's choice).
GPU_MARGIN = 1.25
#: The model's shape flags: configuration A (D40) is the default.
SHAPE = ("d_model", "layers", "heads", "feedforward", "dropout")


def host_available_bytes() -> int:
    """The host's available memory (`/proc/meminfo` MemAvailable)."""
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise ValueError("/proc/meminfo has no MemAvailable")


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--row-interval-s", type=float, required=True)
    parser.add_argument("--executor", type=Path, required=True,
                        help="the directory of the artefact's executor spec: the closed-loop check flies with it (D69, D73)")
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--selection", required=True, choices=RULES,
                        help="the sentences trained and stopped on (D75): every run of B5 reads `landed`")
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    parser.add_argument("--held-out", default=None, help="the fold's held-out airport (D39); none: every airport")
    parser.add_argument("--sample", type=int, default=None,
                        help=f"SMOKE: a random sample of N sentences of each airport and split, seed {SAMPLE_SEED}")
    parser.add_argument("--memory-check-only", action="store_true",
                        help="the memory check at this run's size, written to --out (memory.json, config.json); no "
                             "training (§12 B3: the check at the formal size before a formal run)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--procedure-root", type=Path, default=DEFAULT_PROCEDURE_ROOT)
    shape = {item.name: item.default for item in fields(PriorConfig) if item.name in SHAPE}
    for name, default in shape.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=type(default), default=default)
    for item in fields(TrainConfig):
        parser.add_argument(f"--{item.name.replace('_', '-')}", type=type(item.default), default=item.default)
    args = parser.parse_args(argv)
    instructions, executor, out = _absolute(args.instructions), _absolute(args.executor), _absolute(args.out)
    if out.exists():
        parser.error(f"{out} exists; a training run is never overwritten")
    git = git_state()
    if git["dirty"] and args.sample is None and not args.memory_check_only:
        parser.error("a run that is not a smoke run needs a clean tree")
    _, opened, _ = require_conforming_closed_loop(instructions, executor)   # D69: the checks run here (D73)
    device = torch.device(args.device)

    geometries = load_candidates(instructions)
    landings = airport_landings(geometries, load_day_split(instructions))
    source = ArtefactSource(instructions, args.row_interval_s, args.variant, landings, args.selection)
    run = Run(tuple(sorted(geometries)), args.held_out)
    data = run_data(source, run) if args.sample is None else sampled_run_data(source, run, args.sample)
    identity = artefact_identity(instructions, args.row_interval_s, landings, args.selection)
    # the artefact's sentences that the rule keeps and leaves out, each split, every airport (the run's own: "sentences")
    print(json.dumps({"artefact_selection": selection_totals(identity["selection"])}), flush=True)
    config = TrainConfig(**{item.name: getattr(args, item.name) for item in fields(TrainConfig)})
    torch.manual_seed(config.seed)
    model = Prior(PriorConfig.from_words(source.words, args.variant,
                                         **{name: getattr(args, name) for name in SHAPE})).to(device)

    # the memory at the formal size, before anything is written (§12 B3)
    free = int(torch.cuda.mem_get_info(device)[0]) if device.type == "cuda" else None
    memory = largest_batch_memory(model, data.train, config, device)
    memory.update(gpu_free_before_bytes=free, gpu_margin=GPU_MARGIN,
                  host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  host_available_bytes=host_available_bytes())
    if free is not None and (memory["gpu_peak_reserved_bytes"] + memory["training_state_bytes"]) * GPU_MARGIN > free:
        raise SystemExit(f"the largest batches reserve {memory['gpu_peak_reserved_bytes'] / 2**30:.2f} GiB and the "
                         f"training state takes {memory['training_state_bytes'] / 2**30:.2f} GiB (× {GPU_MARGIN}); "
                         f"{free / 2**30:.2f} GiB of the GPU is free")
    sample = None if args.sample is None else {"per_airport_and_split": args.sample, "seed": SAMPLE_SEED}
    out.mkdir(parents=True)
    write_json_atomic(out / "memory.json", memory)
    write_json_atomic(out / "config.json", {
        "schema": CHECKPOINT_SCHEMA, "written_utc": utc_now(), "instructions": str(instructions), "executor": str(executor),
        "checks": opened["checks"],
        "identity": identity, "run": run.to_dict(), "model_config": model.config.to_dict(),
        "train_config": config.to_dict(), "git": git, "device": str(device),
        "sample": sample,
        "sentences": {"train": len(data.train), "select": len(data.select)},
        "parameters": sum(p.numel() for p in model.parameters())})

    if args.memory_check_only:
        print(json.dumps({"out": str(out), "memory": memory}))
        return 0
    result = train(model, data, config, device, print)
    write_json_atomic(out / "history.json", {"best_epoch": result.best_epoch, "epochs": result.history})
    # the run record says whether it is a smoke run: the checkpoint alone must tell (its identity is the artefact's)
    save_checkpoint(out / "checkpoint.pt", model, result.state, identity=identity,
                    run={**run.to_dict(), "sample": sample}, train_config=config.to_dict())
    write_json_atomic(out / "procedure_masks.json", {
        "set": PROCEDURE_MASKS, "checkpoint_sha256": file_sha256(out / "checkpoint.pt"),
        "procedure_data": procedure_digests(geometries, root=args.procedure_root)})
    if run.held_out is not None:
        model.load_state_dict(result.state)
        held_out = evaluate(model, held_out_sentences(source, run), config.tokens_per_batch, device)
        write_json_atomic(out / "held_out.json", {"airport": run.held_out, **held_out})
        print(f"held out {run.held_out}: {held_out['loss_per_step']:.4f} per step")
    print(json.dumps({"out": str(out), "best_epoch": result.best_epoch}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
