"""Runway-intent R2b: a plan expert's development cohort restricted to a day partition's TRAINING days.

Plan `docs/2026-09-13_runway_intent_plan.zh.md` §16.2. The expert R2a flies trained on the
per-flight split, so it saw other flights of the days R2a evaluates on. R2b retrains it on its own
cohort minus every flight whose operating day is not a training day of the partition (its
validation days and the test days): train flights stay train and validation flights stay
validation, so the development-cohort mechanism takes it as it is, and the retrained expert has
seen no flight of the days it is then evaluated on.

    python run_ts.py runway_intent_r2b_cohort --airport KSMF --partition day_a \\
        --cohort 4dTrajectory/outputs/KRDU/experiments/plan_guidance_20260910/step5b_KSMF_cohort/development_cohort.json \\
        --output-dir 4dTrajectory/outputs/POOLED/experiments/runway_intent_r2b_20260913/KSMF_day_a_cohort
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ts_transformer.config import TSConfig
from ts_transformer.data.development_cohorts import load_development_cohort, write_development_cohort
from ts_transformer.data.runway_context import operational_day, parse_utc
from ts_transformer.experiments.runway_hypotheses import HARVEST_ROOT
from ts_transformer.experiments.runway_intent_r1 import day_folds

COHORT_FILE = "development_cohort.json"
PARTITION_FOLD = {"day_a": "a", "day_b": "b"}


def training_day_flights(flight_ids: tuple[str, ...], landing_utc: dict[str, str], partition: str,
                         config: TSConfig) -> list[str]:
    """The flights whose operating day is a training day of ``partition``."""
    fold = PARTITION_FOLD[partition]
    return [fid for fid in flight_ids
            if day_folds(operational_day(parse_utc(landing_utc[fid])), config)[fold] == "train"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--airport", required=True)
    parser.add_argument("--partition", choices=tuple(PARTITION_FOLD), default="day_a")
    parser.add_argument("--cohort", required=True, help="the expert's own development cohort")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    airport = args.airport.upper()
    config = TSConfig()   # the locked split contract R1 folds days with
    manifest = json.loads((HARVEST_ROOT / airport / "arrivals" / "manifest.json").read_text(encoding="utf-8"))
    landing_utc = {f"{airport}:{row['flight_key']}": row["landing_time_utc"] for row in manifest["records"]}
    source = load_development_cohort(args.cohort)
    train = training_day_flights(source.train_flight_ids, landing_utc, args.partition, config)
    val = training_day_flights(source.val_flight_ids, landing_utc, args.partition, config)
    selection = {
        **source.selection,
        "runway_intent_r2b": {
            "rule": f"the source cohort's flights whose operating day is a {args.partition} TRAINING day "
                    "(runway_intent_r1.day_folds over runway_context.operational_day); train stays train, "
                    "validation stays validation",
            "source_cohort": str(args.cohort),
            "source_name": source.name,
            "partition": args.partition,
            "kept": {"train": len(train), "val": len(val)},
            "dropped": {"train": len(source.train_flight_ids) - len(train),
                        "val": len(source.val_flight_ids) - len(val)},
        },
    }
    out = Path(args.output_dir)
    cohort = write_development_cohort(
        out / COHORT_FILE, name=f"{source.name}-{args.partition}-training-days",
        train_flight_ids=train, val_flight_ids=val, selection=selection,
    )
    print(f"{airport}: {cohort.name}: train {len(train)} of {len(source.train_flight_ids)}, "
          f"val {len(val)} of {len(source.val_flight_ids)} -> {out / COHORT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
