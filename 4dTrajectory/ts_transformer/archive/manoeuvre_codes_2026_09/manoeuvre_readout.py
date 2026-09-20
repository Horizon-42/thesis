"""The tokenizer readout of one P1.4 segment length: gate T (i) and (iii) over the campaign's arms.

    python run_ts.py manoeuvre_readout --campaign 4dTrajectory/outputs/KRDU/experiments/manoeuvre_tok_20260918 \\
        --segment-s 60 --out <campaign>/readout_s60 [--batch-size 64] [--limit N] [--device auto]

Reads every TRAINED arm of the segment (`S<segment>_*` directories holding `history.json`; the
rest are listed as pending), rebuilds the checkpoints' shared val cohort once (every arm of a
campaign trains on one development cohort, so their val splits are one list — asserted), flies
every flight from the fixed anchor under protocol C (`manoeuvre/readout.py`), pairs each token
arm with its own seed's no-token twin, and writes ``manoeuvre_readout.json`` (per-flight rows
included) and ``manoeuvre_readout.txt`` under ``--out`` — refused if it exists. ``--limit N``
narrows the cohort DELIBERATELY (a smoke test) and the artifact says so. ``--write-records``
also writes every arm's Δ-long protocol-C forecasts as a predict-shaped record directory under
``<out>/records/<arm>/`` (the shape the publisher takes), each with a ``manoeuvre_readout``
summary block naming the arm, the protocol, the anchor and the horizon.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import time

from ts_transformer.backbone.adapters import resolve_device
from ts_transformer.experiments.frame_ablation import TRAIN_COMPLETE_ARTIFACT
from ts_transformer.experiments.support import REPO_ROOT, rebuild_cohort
from ts_transformer.inference.export import write_batch
from ts_transformer.io_utils import file_sha256, utc_now, write_json_atomic
from ts_transformer.manoeuvre.readout import (
    READOUT_SCHEMA, RECORDS_BLOCK, arm_reading, fixed_anchor_readings, gate_t, records_block, render,
)
from ts_transformer.run_naming import run_display_name
from ts_transformer.training.train import load_checkpoint

def discover_arms(campaign: Path, segment_s: float) -> tuple[list[Path], list[str]]:
    """``(trained arm directories, pending arm keys)`` of one segment length, in name order.

    An arm directory is named ``S<segment>_<K|cv|nt>_s<seed>`` (a smoke campaign may prefix it);
    the P2/P3 chains write ``p23_S60_K32`` / ``p33_S60_cv`` beside the arms, and those are not arms
    (they read as six "pending" arms on 2026-09-18 when the pattern only asked for ``S60_``)."""
    pattern = re.compile(rf"(^|_)S{int(segment_s)}_(K\d+|cv|nt)_s\d+$")
    trained, pending = [], []
    for path in sorted(campaign.iterdir()):
        if not path.is_dir() or not pattern.search(path.name) or path.name.endswith("_pred_val"):
            continue
        if (path / TRAIN_COMPLETE_ARTIFACT).is_file():
            trained.append(path)
        else:
            pending.append(path.name)
    return trained, pending


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--segment-s", type=float, required=True,
                        help="the arms' horizon (control_horizon_s; the 09-18 arms' token span IS their horizon)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="a PREFIX of the val cohort (a smoke test)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--write-records", action="store_true",
                        help="write every arm's protocol-C forecasts as records under <out>/records/<arm>/")
    args = parser.parse_args(argv)
    campaign = args.campaign if args.campaign.is_absolute() else REPO_ROOT / args.campaign
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if out.exists():
        parser.error(f"{out} exists; a readout is never overwritten (choose another --out)")
    trained, pending = discover_arms(campaign, args.segment_s)
    if not trained:
        parser.error(f"no trained S{int(args.segment_s)}_* arm under {campaign}")
    device = resolve_device(args.device)
    started = time.perf_counter()

    loaded = []
    for path in trained:
        model, config, normalizer, payload = load_checkpoint(path / "checkpoint.pt")
        if config.control_horizon_s != args.segment_s:
            parser.error(f"{path.name}: control_horizon_s={config.control_horizon_s:g} is not the segment {args.segment_s:g}")
        loaded.append((path, model.to(device).eval(), config, normalizer, payload))
        print(f"  {path.name:<18} {run_display_name(config.to_dict())}", flush=True)

    # one cohort: every arm's val split is the same list, in the same order
    val_ids = loaded[0][4]["split"]["val"]
    for path, _m, _c, _n, payload in loaded[1:]:
        if payload["split"]["val"] != val_ids:
            parser.error(f"{path.name}: its val split differs from {loaded[0][0].name}'s — not one campaign cohort")
    wanted = val_ids[: args.limit] if args.limit else val_ids
    series = rebuild_cohort(loaded[0][4], loaded[0][2], wanted)

    readings = []
    record_dirs: dict[str, str] = {}
    for path, model, config, normalizer, payload in loaded:
        pairs = [] if args.write_records else None
        rows = fixed_anchor_readings(model, config, normalizer, series, device, batch_size=args.batch_size, records=pairs)
        reading = arm_reading(path.name, config, rows)
        readings.append(reading)
        print(f"  {path.name:<18} read {len(rows)} flights", flush=True)
        if pairs:
            directory = out / "records" / path.name
            write_batch(
                [record for _, record, _ in pairs], output_dir=directory, config_dict=config.to_dict(),
                flight_metrics=[metrics for _, _, metrics in pairs], checkpoint=str(path / "checkpoint.pt"), split="val",
                extra_summary={RECORDS_BLOCK: records_block(
                    reading, config, campaign=campaign.name, flights=len(series), records=len(pairs),
                    limit=args.limit or None,
                )},
            )
            record_dirs[path.name] = str(directory.relative_to(out))
    verdict = gate_t(readings)
    arms_payload = {}
    for reading in readings:
        entry = {
            "kind": reading.kind, "k": reading.k, "seed": reading.seed,
            "checkpoint_sha256": file_sha256(campaign / reading.key / "checkpoint.pt"),
            "summary": reading.summary, "usage": reading.usage, "rows": reading.rows,
        }
        for seeds in verdict["arms"].values():
            for seed_entry in seeds.values():
                if seed_entry["key"] == reading.key:
                    entry["gain"] = seed_entry["gain"]
                    entry["twin"] = seed_entry.get("twin")
        arms_payload[reading.key] = entry
    payload = {
        "schema": READOUT_SCHEMA,
        "written_utc": utc_now(),
        "campaign": campaign.name,
        "segment_s": args.segment_s,
        "anchor": readings[0].rows[next(iter(readings[0].rows))]["anchor"] if readings[0].rows else None,
        "split": "val",
        "limit": args.limit or None,
        "flights": len(series),
        "truth_shorter_than_horizon": readings[0].summary["truth_shorter_than_horizon"],
        "pending_arms": pending,
        "record_dirs": record_dirs,
        "arms": arms_payload,
        "gate_t": verdict,
        "elapsed_s": time.perf_counter() - started,
    }
    out.mkdir(parents=True, exist_ok=True)   # the record directories may already sit under it
    write_json_atomic(out / "manoeuvre_readout.json", payload)
    table = render(payload)
    (out / "manoeuvre_readout.txt").write_text(table, encoding="utf-8")
    print(table)
    if pending:
        print(f"pending (not yet trained): {', '.join(pending)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
