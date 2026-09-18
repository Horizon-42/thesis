"""The prior's own readouts (plan §3.2 "先验（开环）" row, the parts that need no executor): val
NLL / accuracy against the bigram baseline (gate T(ii)), the landed decision's accuracy, and
the flip rate between adjacent asks on the truth history (plan §2.5's stability reading).

    python run_ts.py manoeuvre_prior_readout --prior <dir>/prior.pt --codebook <dir> --out <dir> [--limit N]

The cohort is the prior's own recorded val split (the executor's), rebuilt; the codebook must
be the one the prior trained on (sha). The open-loop displacement and the discrete-vs-continuous
control are lockstep readings (`run_ts.py manoeuvre_lockstep --protocol A-truth / A`).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch

from ts_transformer.backbone.adapters import resolve_device
from ts_transformer.config import TSConfig
from ts_transformer.data.data_provenance import checkpoint_data_provenance, require_matching_data_provenance
from ts_transformer.data.dataset import build_series, load_flight_dicts
from ts_transformer.experiments.manoeuvre_lockstep import load_prior
from ts_transformer.experiments.support import REPO_ROOT
from ts_transformer.io_utils import file_sha256, utc_now, write_json_atomic
from ts_transformer.manoeuvre.prior import bigram_nll, evaluate, flip_rate
from ts_transformer.manoeuvre.sequences import continuous_targets, flight_sequences
from ts_transformer.manoeuvre.tokenizer import load_codebook
from ts_transformer.repo_layout import arrival_manifest_path
from ts_transformer.training.train import load_checkpoint_payload, usable_series


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--codebook", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="a PREFIX of each split (a smoke test)")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if out.exists():
        parser.error(f"{out} exists; a readout is never overwritten")
    device = resolve_device(args.device)
    started = time.perf_counter()
    prior_path = args.prior if args.prior.is_absolute() else REPO_ROOT / args.prior
    prior, prior_payload = load_prior(prior_path, device)
    codebook = load_codebook(args.codebook if args.codebook.is_absolute() else REPO_ROOT / args.codebook)
    if prior_payload["codebook_sha256"] != codebook.sha256:
        parser.error(f"{prior_path} was trained on codebook {prior_payload['codebook_sha256'][:12]}…, not {codebook.sha256[:12]}…")
    executor = Path(prior_payload["executor"])
    payload = load_checkpoint_payload(executor)
    config = TSConfig.from_dict(payload["config"])
    airports = tuple(entry["airport"] for entry in payload["data_provenance"]["manifests"])
    manifests = [arrival_manifest_path(item) for item in airports]
    require_matching_data_provenance(payload, checkpoint_data_provenance(payload, manifests))
    split = {name: list(keys) for name, keys in prior_payload["split"].items()}
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
    anchor = int(prior_payload["anchor"])
    train_sequences = flight_sequences([by_id[key] for key in split["train"]], codebook, anchor)
    val_series = [by_id[key] for key in split["val"]]
    val_sequences = flight_sequences(val_series, codebook, anchor)
    continuous = prior.model.config.continuous
    val_targets = continuous_targets(val_series, val_sequences, codebook) if continuous else None
    validation = evaluate(prior.model, val_sequences, val_targets, prior.vocabulary, batch_size=args.batch_size, device=device)
    baseline = bigram_nll(train_sequences, val_sequences, codebook.code_count)
    flips = None if continuous else flip_rate(prior.model, val_sequences, prior.vocabulary, batch_size=args.batch_size, device=device)
    result = {
        "schema": "ts-manoeuvre-prior-readout-v1", "written_utc": utc_now(),
        "prior": str(prior_path), "prior_sha256": file_sha256(prior_path), "codebook_sha256": codebook.sha256,
        "executor": str(executor), "continuous": continuous, "segment_s": codebook.segment_s, "anchor": anchor,
        "flights": {"train": len(train_sequences), "val": len(val_sequences)}, "limit": args.limit or None,
        "val": validation, "bigram_baseline": baseline,
        "gate_t_ii": None if continuous else bool(validation["next"] < baseline["nll_per_code"]),
        "flip_rate": flips,
        "elapsed_s": time.perf_counter() - started,
    }
    out.mkdir(parents=True)
    write_json_atomic(out / "manoeuvre_prior_readout.json", result)
    lines = [
        f"prior readout · {prior_path.parent.name} · {'continuous' if continuous else 'discrete'} · K={codebook.code_count} · "
        f"segment {codebook.segment_s:g} s · val {len(val_sequences)} flights",
        f"  val next {validation['next']:.4f}" + ("" if continuous else f" nats/code vs bigram {baseline['nll_per_code']:.4f} → T(ii) {'PASS' if result['gate_t_ii'] else 'FAIL'}")
        + ("" if continuous else f"  next-code accuracy {validation['next_code_accuracy']:.3f}"),
        f"  landed accuracy {validation['landed_accuracy']:.3f}  landed-fraction L1 {validation['landed_fraction']:.3f}",
    ]
    if flips is not None:
        lines.append(f"  flip rate {flips['rate']:.3f} ({flips['flips']} of {flips['pairs']} adjacent-ask pairs)")
    text = "\n".join(lines) + "\n"
    (out / "manoeuvre_prior_readout.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
