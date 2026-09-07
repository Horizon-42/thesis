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
- **Two halves, one deployed.** The val flights are halved deterministically by the
  checkpoint's own ``split_seed``. **Half A fits the δ that is deployed**; half B, which it
  was not fitted on, measures what it covered — that measurement is the published number.
  The mirror (fit on B, score on A) is printed beside it as a STABILITY check and is never
  averaged in: a mean of the two would be fitted on every flight it is then scored against.
- **The cut itself can be probed, and only read.** When the deployed coverage and its mirror
  disagree by more than binomial noise, one fixed seed cannot say whether the cohort splits
  that way or the one half rule ever tried does. ``--half-seed N`` re-cuts the halves through
  the same `calibration.calibration_halves`; it is REFUSED without ``--readout-only``, which
  writes the readout to ``--out`` and does not touch the checkpoint's sidecar. The table
  records the seed it was cut with and says, in its own bytes and at the top of the text, when
  that is not the deployed rule — and `calibration.write_conformal_table` refuses such a table
  even if a caller reaches it directly. A deployed δ comes from the documented rule alone.
- **The DEPLOYED coverage is the gate's number.** Per-stratum δ are measured on their own
  stratum's members, but a flight is handed the first stratum in
  `calibration.INTERVAL_STRATUM_PRECEDENCE` that it is in AND that has a δ — so a flight
  whose stratum refused takes the pooled δ, and the pooled row says nothing about whether it
  covers THOSE flights. The ``deployed`` block assigns every held-out flight the δ it would
  actually get and measures that, pooled and broken out by fall-through group. Design gate
  3.4-2 (80 % coverage in [0.76, 0.84], 50 % in [0.45, 0.55]) reads this block.
- **It is a MEASUREMENT, not a guarantee.** The calibration split is also the split the
  checkpoint was selected on (LR schedule, best epoch, early stopping), so the finite-sample
  split-conformal guarantee does not hold as constructed. The coupling is through a
  trajectory metric (common-grid ADE), not the duration residual these δ are quantiles of.
  The guarantee-bearing read is pre-registered as a single held-out test-split measurement at
  the `freeze-test` ledger stage — not now.
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
from config import DURATION_HEADS_WITH_QUANTILES  # noqa: E402
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="a checkpoint whose duration head emits quantiles "
                             "(duration_head=quantile or two-head)")
    parser.add_argument("--out", type=Path, required=True,
                        help="directory for the readout (the table itself goes into the "
                             "checkpoint's own checkpoint_metadata.json)")
    parser.add_argument("--split", choices=SPLITS, default=CALIBRATION_SPLIT,
                        help=f"which split to calibrate on (only {CALIBRATION_SPLIT!r} is "
                             "admissible; the others are refused with the reason)")
    parser.add_argument("--limit", type=int, default=0, metavar="N",
                        help="calibrate on the first N flights of the split — a SMOKE TEST, "
                             "marked as one in the table and REFUSED at the sidecar unless "
                             "--allow-smoke-table")
    parser.add_argument("--allow-smoke-table", action="store_true",
                        help="write a --limit table into the checkpoint's sidecar anyway; "
                             "`predict` would then deploy a delta fitted on a prefix of the "
                             "split. The table says so wherever it travels")
    parser.add_argument("--half-seed", type=int, default=None, metavar="INT",
                        help="cut the two halves with this seed instead of the checkpoint's "
                             "split_seed — a PROBE of the half rule itself, refused unless "
                             "--readout-only is given with it (default: the split_seed, "
                             "which is the deployed rule)")
    parser.add_argument("--readout-only", action="store_true",
                        help="write the readout to --out and REFUSE to touch the "
                             "checkpoint's checkpoint_metadata.json: nothing is deployed")
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.split != CALIBRATION_SPLIT:
        parser.error(
            f"--split {args.split!r} cannot be a calibration set: the outer test split is "
            "sealed (a delta fitted on it spends the one-shot ledger on a number with no "
            "gate) and the training split's quantiles are fitted to their own targets, so "
            f"its delta would be near zero. Calibrate on {CALIBRATION_SPLIT!r}"
        )
    if args.half_seed is not None and not args.readout_only:
        parser.error(
            f"--half-seed {args.half_seed} re-cuts the two halves, and the DEPLOYED delta "
            "comes from the documented half rule — the checkpoint's own split_seed — and "
            "from nothing else. Pass --readout-only with --half-seed: the probe table is "
            "then written to --out and the checkpoint's sidecar is left untouched"
        )
    device = resolve_device(args.device)
    grid = Grid(split=args.split, bins_m=(), min_future_s=0.0, batch_size=None, limit=args.limit)
    arm = load_arm(
        "calibration", args.checkpoint, grid, device,
        instrument=CALIBRATION_INSTRUMENT, refuse_cta_given=False,
    )
    if arm.config.duration_head not in DURATION_HEADS_WITH_QUANTILES:
        raise SystemExit(
            f"{args.checkpoint}: duration_head={arm.config.duration_head!r} emits one number, "
            "and there is no interval to calibrate. Train the arm with duration_head in "
            f"{DURATION_HEADS_WITH_QUANTILES} (B1 / B1.b)"
        )
    metadata_path = args.checkpoint.parent / "checkpoint_metadata.json"
    if not args.readout_only and not metadata_path.is_file():
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
        airports=arm.airports,
        limit=args.limit,
        half_seed=args.half_seed,
    )
    # The readout is written and printed BEFORE the sidecar: a --limit run whose table the
    # sidecar refuses is still a smoke test someone wanted to read.
    # `render` opens with the smoke-test banner when the table carries one.
    text = render(table)
    print(text, end="")
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / JSON_NAME).write_text(json.dumps({
        "schema": RESULT_SCHEMA,
        "checkpoint": str(args.checkpoint),
        "conformal": table,
    }, indent=2))
    (out / TEXT_NAME).write_text(text)
    if args.readout_only:
        print(f"wrote {out / JSON_NAME} — READOUT ONLY, {metadata_path} [conformal] untouched")
        return 0
    write_conformal_table(metadata_path, table, allow_smoke=args.allow_smoke_table)
    print(f"wrote {out / JSON_NAME} and {metadata_path} [conformal]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
