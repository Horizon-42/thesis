"""Train the instruction prior on a cohort's sentences and read it (two-tier v3 stage B, B3′ pretraining; plan §5.2.2).

    python run_ts.py instruction_prior --vocabulary <instruction_vocabulary.json> --executor <ckpt> \\
        --cohort <development_cohort.json> --out <dir> [--epochs 200] [--patience 20] [--batch-size 128] \\
        [--learning-rate 3e-4] [--seed 1337] [--d-model 128] [--n-layers 4] [--n-heads 4] [--d-ff 256] \\
        [--dropout 0.1] [--landed-loss-weight 1.0] [--limit N] [--device auto]

The cohort's train and val flights are rebuilt through the executor checkpoint's data
provenance (`support.rebuild_cohort`, C25 — the checkpoint is the door to the data), read into
sentences under the vocabulary artefact (`instructions.read_instructions`) and turned into
sequences (`instruction_sequences.flight_sequence`: the words in force, the truth's state at
each position, the next position's words). The prior (`instruction_prior.InstructionPrior`,
D46's starting shape) is fitted with teacher forcing, the epoch kept on the val next-word
term. Written under ``--out`` (refused if it exists):

    prior.pt          the prior (config, weights, the aircraft-type vocabulary, the instruction
                      vocabulary's sha and the artefact's runway classes — the runway head's
                      size is the cohort's, not the spec's — the split keys, the executor's sha)
    history.json      every epoch's train / val terms
    readings.json     `evaluate` on val (and on train, as a reference) and `hold_baseline`
    summary.txt       the readings, one screen
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

import torch

from ts_transformer.backbone.adapters import resolve_device
from ts_transformer.config import TSConfig
from ts_transformer.data.data_provenance import provenance_eligible_set_digests
from ts_transformer.data.development_cohorts import development_cohort_audit, load_development_cohort
from ts_transformer.experiments.support import REPO_ROOT, cohort_splits, rebuild_cohort
from ts_transformer.io_utils import file_sha256, utc_now, write_json_atomic
from ts_transformer.manoeuvre.context import TypeVocabulary
from ts_transformer.manoeuvre.instruction_prior import (
    PRIOR_SCHEMA, InstructionPrior, PriorConfig, evaluate, fit, hold_baseline,
)
from ts_transformer.manoeuvre.instruction_sequences import InstructionSequence, flight_sequence
from ts_transformer.manoeuvre.instructions import INSTRUCTION_KINDS, runway_sha256, load_vocabulary, read_instructions, word_counts
from ts_transformer.training.train import load_checkpoint_payload

PRIOR_FILE = "prior.pt"


def _f(value: float | None) -> str:
    return "     —" if value is None else f"{value:6.3f}"


def render(readings: dict[str, Any], baseline: dict[str, Any], *, limit: int | None = None) -> str:
    """The val readings on one screen. Denominators: NLL / top-k / change over the positions with
    a next word, flip over the positions where the truth holds the word, miss / recall where it
    changes; a dash is a reading with nothing to count (or a top-k at or past the kind's class
    count, 1 by construction)."""
    lines = ["instruction prior · val readings (positions with a next word: "
             f"{readings['positions_with_next']}" + (f"; a PREFIX of {limit} flights per split" if limit else "") + ")"]
    lines.append(f"  next-word NLL (nats): total {_f(readings['next'])} vs bigram "
                 f"{_f(baseline['bigram_nll']['next'])} · terminal (continue / landed / go-around) "
                 f"top-1 {_f(readings['top1']['terminal'])}, NLL {_f(readings['terminal'])}; "
                 f"go-around is NEVER in the truth here (results §13.5) — the kind measures "
                 f"continue-vs-landed only")
    lines.append("  kind       NLL    bigram   top-1   hold    top-2   top-4   top-8   flip    miss    recall  change")
    lines.append("             ·· over positions with a next ··          ·· held ·· ·· changed ··  share")
    for kind in INSTRUCTION_KINDS:
        lines.append(
            f"  {kind:<9} {_f(readings[kind])}  {_f(baseline['bigram_nll'][kind])}  {_f(readings['top1'][kind])}  "
            f"{_f(baseline['hold_accuracy'][kind])}  {_f(readings['top_k'][kind]['2'])}  {_f(readings['top_k'][kind]['4'])}  "
            f"{_f(readings['top_k'][kind]['8'])}  {_f(readings['flip_rate'][kind])}  {_f(readings['miss_rate'][kind])}  "
            f"{_f(readings['change_recall'][kind])}  {_f(readings['change_share'][kind])}"
        )
    lines.append("  joint top-K coverage (truth 5-tuple among the K best product candidates): "
                 + ", ".join(f"K={k} {_f(v)}" for k, v in readings["joint_top_k"].items()))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--vocabulary", type=Path, required=True, help="the instruction_vocabulary.json the sentences are read under")
    parser.add_argument("--executor", type=Path, required=True, help="a checkpoint.pt: the door to the cohort's data")
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=0, help="a PREFIX of each split (a smoke test)")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if out.exists():
        parser.error(f"{out} exists; a prior is never overwritten")
    started = time.perf_counter()
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)

    vocabulary_path = args.vocabulary if args.vocabulary.is_absolute() else REPO_ROOT / args.vocabulary
    vocabulary, runway_vocabulary, _payload = load_vocabulary(vocabulary_path)
    executor = args.executor if args.executor.is_absolute() else REPO_ROOT / args.executor
    payload = load_checkpoint_payload(executor)
    config = TSConfig.from_dict(payload["config"])
    cohort_path = args.cohort if args.cohort.is_absolute() else REPO_ROOT / args.cohort
    cohort = load_development_cohort(cohort_path)
    splits = cohort_splits(payload, cohort, args.limit)
    series = rebuild_cohort(payload, config, [*splits["train"], *splits["val"]])
    by_split = {"train": series[: len(splits["train"])], "val": series[len(splits["train"]) :]}
    sequences: dict[str, list[InstructionSequence]] = {
        name: [flight_sequence(item, read_instructions(item, vocabulary, runway_vocabulary)) for item in items]
        for name, items in by_split.items()
    }
    types = TypeVocabulary.from_typecodes(item.typecode for item in sequences["train"])
    # the prior holds exactly the positions this cohort has: a longer sentence is refused at
    # `collate`, never read through an untrained position row
    longest = max(item.length for items in sequences.values() for item in items)
    prior_config = PriorConfig(
        words=word_counts(vocabulary, runway_vocabulary), type_count=types.size, vocabulary_sha256=vocabulary.sha256, d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers, d_ff=args.d_ff, dropout=args.dropout,
        max_positions=longest,
    )
    print(f"  {len(series)} flights rebuilt: train {len(sequences['train'])}, val {len(sequences['val'])}; "
          f"longest sentence {longest} events; {types.size - 1} aircraft types; "
          f"{len(runway_vocabulary)} runways ({', '.join(runway_vocabulary.idents)}); "
          f"vocabulary {vocabulary.sha256[:12]}…", flush=True)

    model = InstructionPrior(prior_config)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"  prior: {parameters} parameters, d {args.d_model} × {args.n_layers} layers, device {device}", flush=True)

    def log(row: dict[str, Any]) -> None:
        val = row["val"]
        print(f"  epoch {row['epoch']:3d}: train next {row['train']['next']:.3f} · val next {val['next']:.3f} "
              f"(h {val['heading']:.3f} a {val['altitude']:.3f} s {val['speed']:.3f} "
              f"r {val['runway']:.3f} d {val['duration']:.3f} t {val['terminal']:.3f}) "
              f"top-1 h {val['top1']['heading']:.3f} a {val['top1']['altitude']:.3f} "
              f"s {val['top1']['speed']:.3f} t {val['top1']['terminal']:.3f}", flush=True)

    result = fit(model, sequences["train"], sequences["val"], types, epochs=args.epochs, patience=args.patience,
                 batch_size=args.batch_size, learning_rate=args.learning_rate, seed=args.seed, device=device, log=log)
    readings = {name: evaluate(model, items, types, batch_size=args.batch_size, device=device) for name, items in sequences.items()}
    baseline = hold_baseline(sequences["train"], sequences["val"], prior_config)

    out.mkdir(parents=True)
    settings = {"epochs": args.epochs, "patience": args.patience, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
                "seed": args.seed, "device": str(device), "limit": args.limit or None}
    torch.save({
        "schema": PRIOR_SCHEMA, "prior_config": prior_config.to_dict(), "state_dict": result.state_dict,
        "types": types.to_dict(), "vocabulary_sha256": vocabulary.sha256, "vocabulary_path": str(vocabulary_path),
        "vocabulary_spec": vocabulary.to_dict(), "runway_idents": list(runway_vocabulary.idents),
        "runway_sha256": runway_sha256(runway_vocabulary),
        "split": {name: [item.dataset_id for item in items] for name, items in by_split.items()},
        "executor": str(executor), "executor_sha256": file_sha256(executor),
        "best_epoch": result.best_epoch, "best_val_next": result.best_val_next, "stopped_early": result.stopped_early,
        "settings": settings,
    }, out / PRIOR_FILE)
    write_json_atomic(out / "history.json", {"schema": PRIOR_SCHEMA, "history": result.history})
    write_json_atomic(out / "readings.json", {
        "schema": PRIOR_SCHEMA, "written_utc": utc_now(), "prior_sha256": file_sha256(out / PRIOR_FILE),
        "vocabulary_sha256": vocabulary.sha256, "runway_idents": list(runway_vocabulary.idents),
        "runway_sha256": runway_sha256(runway_vocabulary), "prior_config": prior_config.to_dict(),
        "cohort_identity": {**development_cohort_audit(cohort_path, cohort), "path": str(cohort_path),
                            "eligible_set_sha256": provenance_eligible_set_digests(payload["data_provenance"])},
        "flights": {name: len(items) for name, items in sequences.items()}, "settings": settings,
        "best_epoch": result.best_epoch, "epochs_run": len(result.history), "stopped_early": result.stopped_early,
        "readings": readings, "hold_baseline": baseline, "elapsed_s": time.perf_counter() - started,
    })
    table = render(readings["val"], baseline, limit=args.limit or None)
    (out / "summary.txt").write_text(table, encoding="utf-8")
    print(table)
    print(f"  best epoch {result.best_epoch} of {len(result.history)}; written to {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
