"""Train the prior (plan §2.5, P2.1) on a frozen codebook over the executor's own cohort.

    python run_ts.py manoeuvre_prior --codebook 4dTrajectory/outputs/codebooks/<name> \\
        --executor <campaign>/<arm>/checkpoint.pt --out <dir> [--continuous] \\
        [--epochs 200 --patience 20 --batch-size 256 --learning-rate 3e-4 --seed 1337] \\
        [--d-model 128 --n-layers 4 --n-heads 4 --d-ff 256 --dropout 0.1] [--limit N] [--device auto] \\
        [--campaign-id <id> --experiment-id <id>]

The executor checkpoint supplies the COHORT (its train / val split — the development cohort
every arm of the campaign trained on — and its data provenance, verified against today's
manifests), the anchor (its fixed anchor) and the horizon, which must be the codebook's
segment. The prior never sees a val flight (§6.3): the lockstep pairs the executor's val
flights, so the split is the executor's, not an operating-day split (that is P4's, for the
runway token). Sequences are the truth's codes through the codebook (`manoeuvre/sequences.py`);
``--continuous`` trains gate P's control instead (the unrounded coordinates, one MSE).

Writes ``prior.pt`` (weights, the prior config, the type vocabulary, the codebook sha, the
executor sha, the split ids' sha, the selection), ``history.json`` (every epoch), and
``prior_metadata.json`` with the bigram baseline (gate T(ii)'s number) beside the val NLL.
``--campaign-id`` / ``--experiment-id`` open an experiment manifest (the clean-worktree guard).
``--out`` must not exist.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import torch

from ts_transformer.backbone.adapters import resolve_device
from ts_transformer.config import TSConfig, default_anchor
from ts_transformer.data.data_provenance import checkpoint_data_provenance, require_matching_data_provenance
from ts_transformer.data.dataset import build_series, load_flight_dicts, truth_duration_s
from ts_transformer.experiments.support import REPO_ROOT
from ts_transformer.io_utils import file_sha256, sha256_bytes, utc_now, write_json_atomic
from ts_transformer.manoeuvre.context import TypeVocabulary
from ts_transformer.manoeuvre.lockstep import required_positions
from ts_transformer.manoeuvre.prior import PRIOR_SCHEMA, ManoeuvrePrior, PriorConfig, bigram_nll, evaluate, fit
from ts_transformer.manoeuvre.sequences import continuous_targets, flight_sequences
from ts_transformer.manoeuvre.tokenizer import load_codebook
from ts_transformer.repo_layout import arrival_manifest_path
from ts_transformer.training.experiment_index import begin_run, finish_run
from ts_transformer.training.train import load_checkpoint_payload, usable_series

PRIOR_FILE = "prior.pt"


def _split_sha(keys: list[str]) -> str:
    return sha256_bytes("\n".join(keys).encode())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--codebook", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True, help="a checkpoint.pt of the campaign (its split, provenance, anchor)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--continuous", action="store_true", help="gate P's control: regress the unrounded coordinates")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--landed-loss-weight", type=float, default=1.0)
    parser.add_argument("--landed-fraction-loss-weight", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0, help="a PREFIX of each split (a smoke test)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--campaign-id", default=None)
    parser.add_argument("--experiment-id", default=None)
    args = parser.parse_args(argv)
    if (args.campaign_id is None) != (args.experiment_id is None):
        parser.error("--campaign-id and --experiment-id go together")
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if out.exists():
        parser.error(f"{out} exists; a prior is never overwritten")
    started = time.perf_counter()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    codebook = load_codebook(args.codebook if args.codebook.is_absolute() else REPO_ROOT / args.codebook)
    executor = args.executor if args.executor.is_absolute() else REPO_ROOT / args.executor
    payload = load_checkpoint_payload(executor)
    config = TSConfig.from_dict(payload["config"])
    if config.control_horizon_s != codebook.segment_s or config.dt_s != codebook.dt_s:
        parser.error(f"the executor's horizon {config.control_horizon_s:g} s / {config.dt_s:g} s is not the codebook's "
                     f"{codebook.segment_s:g} s / {codebook.dt_s:g} s")
    if args.continuous and codebook.kind != "learned":
        parser.error("the continuous prior regresses a learned tokenizer's coordinates; the command vocabulary has none")
    anchor = default_anchor(config)
    airports = tuple(entry["airport"] for entry in payload["data_provenance"]["manifests"])
    manifests = [arrival_manifest_path(item) for item in airports]
    require_matching_data_provenance(payload, checkpoint_data_provenance(payload, manifests))
    split = {name: list(payload["split"][name]) for name in ("train", "val")}
    if args.limit:
        split = {name: keys[: args.limit] for name, keys in split.items()}
    wanted = set(split["train"]) | set(split["val"])
    built, report = build_series(load_flight_dicts(manifests, include_flight_keys=wanted, verbose=False), config,
                                 aircraft_type=config.aircraft_type)
    print(f"  {report.format()}", flush=True)
    by_id = {item.dataset_id: item for item in usable_series(built, config, verbose=False)}
    missing = [key for key in wanted if key not in by_id]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(wanted)} cohort flights could not be rebuilt (first: {missing[0]!r})")
    train_series = [by_id[key] for key in split["train"]]
    val_series = [by_id[key] for key in split["val"]]

    manifest_path = None
    if args.campaign_id:
        manifest_path = begin_run(
            out, repo_root=REPO_ROOT, campaign_id=args.campaign_id, run_id=args.experiment_id,
            config={"prior": vars(args) | {"codebook": str(args.codebook), "executor": str(executor), "out": str(out)}},
            data_selection={"executor_split_sha256": {name: _split_sha(keys) for name, keys in split.items()},
                            "limit": args.limit or None},
            command=sys.argv,
        )
    try:
        train_sequences = flight_sequences(train_series, codebook, anchor)
        val_sequences = flight_sequences(val_series, codebook, anchor)
        train_targets = val_targets = None
        if args.continuous:
            train_targets = continuous_targets(train_series, train_sequences, codebook)
            val_targets = continuous_targets(val_series, val_sequences, codebook)
        vocabulary = TypeVocabulary.from_typecodes(item.typecode for item in train_sequences)
        longest = max(item.length for item in (*train_sequences, *val_sequences))
        # sized for the LOCKSTEP's budget (the rounds a flight may be asked for under protocol
        # A), not for the truth alone (`lockstep.required_positions`)
        longest_truth_s = max(truth_duration_s(item, anchor) for item in (*train_series, *val_series))
        prior_config = PriorConfig(
            code_count=codebook.code_count, z_dim=codebook.z_dim, type_count=vocabulary.size, continuous=args.continuous,
            d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers, d_ff=args.d_ff, dropout=args.dropout,
            max_positions=max(required_positions(longest_truth_s, codebook.segment_s, longest), 8),
            landed_loss_weight=args.landed_loss_weight, landed_fraction_loss_weight=args.landed_fraction_loss_weight,
        )
        model = ManoeuvrePrior(prior_config)
        print(f"  prior: {sum(p.numel() for p in model.parameters())} parameters, K={codebook.code_count}, "
              f"{'continuous' if args.continuous else 'discrete'}, {len(train_sequences)} train / {len(val_sequences)} val flights, "
              f"longest {longest} segments, {vocabulary.size - 1} types", flush=True)
        baseline = bigram_nll(train_sequences, val_sequences, codebook.code_count)
        print(f"  bigram baseline: val NLL {baseline['nll_per_code']:.4f} nats/code ({baseline['nll_per_token']:.4f} per token)", flush=True)

        def log(row):
            val = row["val"]
            extra = "" if args.continuous else f"  acc {val['next_code_accuracy']:.3f}"
            print(f"  epoch {row['epoch']:>4}  train {row['train']['total']:.4f}  val next {val['next']:.4f}{extra}"
                  f"  landed-acc {val['landed_accuracy']:.3f}", flush=True)

        result = fit(model, train_sequences, val_sequences, vocabulary, train_targets=train_targets, val_targets=val_targets,
                     epochs=args.epochs, patience=args.patience, batch_size=args.batch_size, learning_rate=args.learning_rate,
                     seed=args.seed, device=device, log=log)
        final = evaluate(model, val_sequences, val_targets, vocabulary, batch_size=args.batch_size, device=device)
        out.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": PRIOR_SCHEMA, "prior_config": prior_config.to_dict(), "state_dict": result.state_dict,
            "vocabulary": vocabulary.to_dict(), "codebook_sha256": codebook.sha256, "codebook": str(codebook.path),
            "executor": str(executor), "executor_sha256": file_sha256(executor), "anchor": anchor,
            "segment_s": codebook.segment_s, "split": split, "best_epoch": result.best_epoch,
            "best_val_next": result.best_val_next, "seed": args.seed,
        }, out / PRIOR_FILE)
        write_json_atomic(out / "history.json", {"schema": PRIOR_SCHEMA, "history": result.history})
        metadata = {
            "schema": PRIOR_SCHEMA, "written_utc": utc_now(), "prior_sha256": file_sha256(out / PRIOR_FILE),
            "prior_config": prior_config.to_dict(), "codebook_sha256": codebook.sha256, "codebook": str(codebook.path),
            "executor": str(executor), "executor_sha256": file_sha256(executor), "anchor": anchor,
            "segment_s": codebook.segment_s, "flights": {"train": len(train_sequences), "val": len(val_sequences)},
            "split_sha256": {name: _split_sha(keys) for name, keys in split.items()}, "limit": args.limit or None,
            "best_epoch": result.best_epoch, "epochs_run": len(result.history), "stopped_early": result.stopped_early,
            "val": final, "bigram_baseline": baseline,
            "gate_t_ii": None if args.continuous else bool(final["next"] < baseline["nll_per_code"]),
            "elapsed_s": time.perf_counter() - started,
        }
        write_json_atomic(out / "prior_metadata.json", metadata)
        print(f"  kept epoch {result.best_epoch}: val next {result.best_val_next:.4f}"
              + ("" if args.continuous else f" vs bigram {baseline['nll_per_code']:.4f} → T(ii) {'PASS' if metadata['gate_t_ii'] else 'FAIL'}"), flush=True)
    except BaseException as exc:
        if manifest_path is not None:
            finish_run(manifest_path, failure=repr(exc))
        raise
    if manifest_path is not None:
        finish_run(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
