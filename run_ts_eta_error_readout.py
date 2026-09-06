#!/usr/bin/env python
"""B0: what the arrival-time error actually looks like, per difficulty stratum.

Anytime / calibrated-ETA design
(`4dTrajectory/ts_transformer/docs/2026-09-07_anytime_prediction_and_calibrated_eta_design.zh.md`)
§三 3.1: the duration head is a point estimate and there is no interval anywhere in the
package. Before a quantile head or a conformal calibrator is worth building, the SIZE of
the thing they would have to cover has to be known — and it is already on disk, one
`final_time_error_s` per flight in every prediction directory's `summary.json`.

This is a pure readout: it opens summaries, it never predicts, and it writes nothing unless
asked for `--json`::

    python run_ts_eta_error_readout.py \\
        L1_native32=4dTrajectory/outputs/KRDU/experiments/l1_lowdim_20260907/L1_native32_pred_val

Per stratum (`approach_difficulty.strata_masks` — the same cut every other readout uses) it
reports the |final-time error| p50 / p80 / p90 and the SIGNED p10 / p50 / p90, and the same
quantiles of `fde_m`. Both are needed: the absolute quantiles are the width an interval
would have to have, and the signed ones say whether the head is late or early (a symmetric
interval around a biased point estimate is the wrong shape). ``fde_m`` is non-negative, so
its signed and absolute quantiles are the same numbers by construction — it is here because
the arrival-TIME error and the arrival-PLACE error are the two halves of one 4D miss and
reading one without the other has already produced a wrong conclusion in this package.

Coverage is stated per arm: rows whose metrics are null (an unscored flight) are counted
and dropped, never silently averaged over.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
TS_DIR = REPO_ROOT / "4dTrajectory" / "ts_transformer"
for path in (TS_DIR, REPO_ROOT / "geokit" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from approach_difficulty import strata_masks  # noqa: E402
from flight_scenarios.identity import summary_row_key  # noqa: E402

RESULT_SCHEMA = "ts-eta-error-readout-b0-v1"

# The width an interval would need (absolute) and the shape it would need (signed).
ABSOLUTE_QUANTILES = (50, 80, 90)
SIGNED_QUANTILES = (10, 50, 90)
# The metrics read out of the summary rows, and what each is called in the result.
METRICS = {"final_time_error_s": "abs_final_time_error_s", "fde_m": "fde_m"}


def scored_rows(summary: dict) -> tuple[dict[str, dict], dict[str, int]]:
    """The arm's rows keyed by flight, plus what was dropped for want of a metric."""
    results = summary["results"]
    rows = {
        summary_row_key(row): row
        for row in results
        if all(row.get(name) is not None for name in METRICS)
        and row.get("route_tortuosity") is not None
    }
    coverage = {
        "summary_rows": len(results),
        "scored_rows": len(rows),
        "dropped_unscored_rows": len(results) - len(rows),
    }
    return rows, coverage


def quantile_block(values: np.ndarray) -> dict[str, float]:
    """One metric's shape: the absolute quantiles and the signed ones, one definition."""
    absolute = np.abs(values)
    return {
        **{f"abs_p{q}": float(np.percentile(absolute, q)) for q in ABSOLUTE_QUANTILES},
        **{f"signed_p{q}": float(np.percentile(values, q)) for q in SIGNED_QUANTILES},
        "mean_signed": float(values.mean()),
    }


def readout(label: str, pred_dir: Path) -> dict:
    summary = json.loads((pred_dir / "summary.json").read_text())
    rows, coverage = scored_rows(summary)
    keys = sorted(rows)
    if not keys:
        raise SystemExit(f"{pred_dir} has no scored rows carrying {tuple(METRICS)}")
    masks = strata_masks(rows, keys)
    values = {
        name: np.array([float(rows[key][name]) for key in keys], dtype=np.float64)
        for name in METRICS
    }
    strata = {}
    for stratum, mask in masks.items():
        selected = np.flatnonzero(mask)
        if not len(selected):
            continue
        strata[stratum] = {
            "n": int(len(selected)),
            **{
                published: quantile_block(values[name][selected])
                for name, published in METRICS.items()
            },
        }
    return {
        "label": label,
        "arm": str(pred_dir),
        "split": summary.get("split"),
        "predictor": summary.get("config", {}).get("prediction_output"),
        "flights": len(keys),
        "coverage": coverage,
        "strata": strata,
    }


def render(payload: dict) -> str:
    lines = [
        "B0 arrival-time error distribution — the width a calibrated interval would need",
        "|dt| = |final_time_error_s| (seconds); signed dt < 0 = predicted EARLY. FDE is "
        "non-negative, so its signed and absolute quantiles coincide.",
    ]
    for arm in payload["arms"]:
        coverage = arm["coverage"]
        lines += [
            "",
            f"── {arm['label']} — {arm['arm']} (split {arm['split']}, {arm['flights']} flights)",
            f"   coverage: {coverage['scored_rows']} of {coverage['summary_rows']} summary rows "
            f"({coverage['dropped_unscored_rows']} unscored, dropped)",
            f"   {'stratum':>46s} {'n':>5s} {'|dt|p50':>8s} {'|dt|p80':>8s} {'|dt|p90':>8s} "
            f"{'dt p10':>8s} {'dt p50':>8s} {'dt p90':>8s} {'FDE p50':>9s} {'FDE p80':>9s} "
            f"{'FDE p90':>9s}",
        ]
        for stratum, block in arm["strata"].items():
            time_block, fde = block["abs_final_time_error_s"], block["fde_m"]
            lines.append(
                f"   {stratum:>46s} {block['n']:>5d} "
                f"{time_block['abs_p50']:>8.1f} {time_block['abs_p80']:>8.1f} "
                f"{time_block['abs_p90']:>8.1f} {time_block['signed_p10']:>8.1f} "
                f"{time_block['signed_p50']:>8.1f} {time_block['signed_p90']:>8.1f} "
                f"{fde['abs_p50']:>9.1f} {fde['abs_p80']:>9.1f} {fde['abs_p90']:>9.1f}"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("arms", nargs="+", metavar="LABEL=PRED_DIR",
                        help="a scored prediction directory and the name it is reported "
                             "under; repeatable")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    arms: dict[str, Path] = {}
    for entry in args.arms:
        label, separator, raw = entry.partition("=")
        if not separator or not label.strip() or not raw.strip():
            parser.error(f"each arm is LABEL=PRED_DIR, got {entry!r}")
        if label.strip() in arms:
            parser.error(f"arm label {label.strip()!r} is used twice")
        path = Path(raw.strip())
        arms[label.strip()] = path if path.is_absolute() else REPO_ROOT / path

    payload = {
        "schema": RESULT_SCHEMA,
        "absolute_quantiles": list(ABSOLUTE_QUANTILES),
        "signed_quantiles": list(SIGNED_QUANTILES),
        "arms": [readout(label, path) for label, path in arms.items()],
    }
    text = render(payload)
    print(text, end="")
    if args.json is not None:
        out = args.json if args.json.is_absolute() else REPO_ROOT / args.json
        out.write_text(json.dumps(payload, indent=2))
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
