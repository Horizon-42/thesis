#!/usr/bin/env python
"""B2: split-conformal calibration of a quantile checkpoint's arrival-time interval.

Anytime / calibrated-ETA design
(`4dTrajectory/ts_transformer/docs/2026-09-07_anytime_prediction_and_calibrated_eta_design.zh.md`)
§三 3.2. The quantile duration head (B1) publishes five levels of ``p(T | history)``; this
runner turns them into an interval with a MEASURED coverage, per difficulty stratum, and
stores the result beside the checkpoint so `predict` can stamp it on every record::

    conda activate aeroviz
    python run_ts_eta_calibration.py \\
        --checkpoint 4dTrajectory/outputs/KRDU/experiments/<campaign>/<arm>/checkpoint.pt \\
        --out 4dTrajectory/outputs/KRDU/experiments/<campaign>/<arm>_calibration

What it does, and why each part is not negotiable:

- **It reads the DURATION HEAD only.** One forward pass per flight, no rollout and no CTA
  (`forecast.duration_quantile_predictions`), which is why it runs in seconds on a CPU and
  why it can calibrate a ``cta_conditioning=given`` checkpoint — B3's arm — without handing
  the network an arrival time first. An ``intent_conditioning=truth-…`` checkpoint is still
  refused: that oracle is inside the history the head reads.
- **The calibration set is the VALIDATION split** (design §六 4 and the repo's experiment
  principles). ``--split test`` and ``--split train`` are offered only so the refusal can
  say why: the sealed outer test would be spent on a number that has no gate, and the
  training split's quantiles are fitted to their own targets, so its δ would be ~0.
- **Two halves, swapped.** The val flights are halved deterministically by the checkpoint's
  own ``split_seed``; δ is fitted on one half and its coverage measured on the OTHER, then
  the halves swap. Both coverages are published beside the mean δ — the artifact's own
  honesty check (design gate 3.4-2 asks for 80 % coverage in [0.76, 0.84]).
- **Per stratum** (`approach_difficulty.strata_masks`), because the vectored |Δt| p80 is
  3.6–5.5× the straight-in one (B0). A stratum with fewer than
  `calibration.MIN_CALIBRATION_FLIGHTS` in a half is REFUSED, printed and recorded.

The table lands in the checkpoint's ``checkpoint_metadata.json`` under ``conformal`` — a
SIDECAR, deliberately never in ``data_provenance``, which `evaluate-fit` and `freeze-test`
compare for equality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent
TS_DIR = REPO_ROOT / "4dTrajectory" / "ts_transformer"
for path in (TS_DIR, REPO_ROOT / "geokit" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from approach_difficulty import approach_difficulty  # noqa: E402
from calibration import (  # noqa: E402
    CalibrationSample,
    calibrate,
    render,
    write_conformal_table,
)
from config import DURATION_HEAD_QUANTILE  # noqa: E402
from dataset import truth_duration_s  # noqa: E402
from forecast import default_anchor, duration_quantile_predictions  # noqa: E402
from io_utils import file_sha256  # noqa: E402
from models import resolve_device  # noqa: E402
from run_ts_anytime_curve import Grid, cohort_series, load_arm  # noqa: E402

RESULT_SCHEMA = "ts-eta-calibration-b2-v1"
CALIBRATION_INSTRUMENT = "the ETA calibration"
#: The only split a calibration set may be. The other two are offered so the refusal names
#: the reason instead of argparse printing "invalid choice".
CALIBRATION_SPLIT = "val"
SPLITS = (CALIBRATION_SPLIT, "test", "train")
JSON_NAME, TEXT_NAME = "eta_calibration.json", "eta_calibration.txt"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="a checkpoint trained with duration_head=quantile")
    parser.add_argument("--out", type=Path, required=True,
                        help="directory for the readout (the table itself goes into the "
                             "checkpoint's own checkpoint_metadata.json)")
    parser.add_argument("--split", choices=SPLITS, default=CALIBRATION_SPLIT,
                        help=f"which split to calibrate on (only {CALIBRATION_SPLIT!r} is "
                             "admissible; the others are refused with the reason)")
    parser.add_argument("--limit", type=int, default=0, metavar="N",
                        help="calibrate on the first N flights of the split — a SMOKE TEST, "
                             "marked as one in the artifact")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    if args.split != CALIBRATION_SPLIT:
        parser.error(
            f"--split {args.split!r} cannot be a calibration set: the outer test split is "
            "sealed (a delta fitted on it spends the one-shot ledger on a number with no "
            "gate) and the training split's quantiles are fitted to their own targets, so "
            f"its delta would be near zero. Calibrate on {CALIBRATION_SPLIT!r}"
        )
    device = resolve_device(args.device)
    grid = Grid(split=args.split, bins_m=(), min_future_s=0.0, batch_size=None, limit=args.limit)
    arm = load_arm(
        "calibration", args.checkpoint, grid, device,
        instrument=CALIBRATION_INSTRUMENT, refuse_cta_given=False,
    )
    if arm.config.duration_head != DURATION_HEAD_QUANTILE:
        raise SystemExit(
            f"{args.checkpoint}: duration_head={arm.config.duration_head!r} emits one number, "
            "and there is no interval to calibrate. Train the arm with "
            f"duration_head={DURATION_HEAD_QUANTILE!r} (B1)"
        )
    metadata_path = args.checkpoint.parent / "checkpoint_metadata.json"
    if not metadata_path.is_file():
        raise SystemExit(
            f"{metadata_path} does not exist; the conformal table is a sidecar of the "
            "checkpoint's own metadata and there is nothing to attach it to"
        )

    series = cohort_series(arm, grid)
    anchor = default_anchor(arm.config)
    quantiles = duration_quantile_predictions(
        arm.model, series, arm.config, arm.normalizer, anchor=anchor, device=device
    )
    samples = [
        CalibrationSample(
            key=item.dataset_id,
            quantiles_s=quantiles[row],
            truth_final_time_s=float(truth_duration_s(item, anchor)),
            covariates=approach_difficulty(item, anchor).to_dict(),
        )
        for row, item in enumerate(series)
    ]
    table = calibrate(
        samples,
        split_seed=arm.config.resolved_split_seed,
        split=args.split,
        checkpoint_sha256=file_sha256(args.checkpoint),
    )
    write_conformal_table(metadata_path, table)

    text = render(table)
    if args.limit:
        text = (
            f"SMOKE TEST: --limit {args.limit} calibrated on the first {len(samples)} "
            f"flights of the {args.split} split, not the split\n" + text
        )
    print(text, end="")
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / JSON_NAME).write_text(json.dumps({
        "schema": RESULT_SCHEMA,
        "checkpoint": str(args.checkpoint),
        "limit": args.limit,
        "smoke_test": bool(args.limit),
        "airports": list(arm.airports),
        "conformal": table,
    }, indent=2))
    (out / TEXT_NAME).write_text(text)
    print(f"wrote {metadata_path} [conformal] and {out / JSON_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
