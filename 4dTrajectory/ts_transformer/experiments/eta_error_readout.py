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

    python run_ts.py eta_error_readout \\
        L1_native32=4dTrajectory/outputs/KRDU/experiments/l1_lowdim_20260907/L1_native32_pred_val

Per stratum (`approach_difficulty.strata_masks` — the same cut every other readout uses) it
reports the |final-time error| p50 / p80 / p90 and the SIGNED p10 / p50 / p90, and the same
quantiles of `fde_m`. Both are needed: the absolute quantiles are the width an interval
would have to have, and the signed ones say whether the head is late or early (a symmetric
interval around a biased point estimate is the wrong shape). ``fde_m`` is non-negative, so
its signed and absolute quantiles are the same numbers by construction — it is here because
the arrival-TIME error and the arrival-PLACE error are the two halves of one 4D miss and
reading one without the other has already produced a wrong conclusion in this package.

Coverage is stated per arm: a row is used only if it carries every metric AND every
covariate the strata are cut on, and what that drops is counted, never silently averaged
over. `established_at_anchor` is on that list for a Python reason — a present-but-null flag
reads as False and would move the flight into the vectored stratum unnoticed.

`final_time_error_s` is the error of the duration the ROLLOUT flew. On a `two-head` arm
(B1.b) that is the POINT head's, while the published interval is the quantile head's — two
different numbers — so an arm whose rows carry `duration_quantiles_s` also gets a second
block, `duration_q50_error_s`, derived as ``q50 − true_final_time_s`` from the same rows.
Gate 1 of design §三 3.1b reads its MAE (that is the number `B1_quantile`'s 23.9 / 10.3 s
were measured as), and the ADE clause of the same gate reads the point head. Under
`duration_head=quantile` the two blocks are the same measurement twice, because q50 IS the
duration the rollout flew.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ts_transformer.experiments.support import REPO_ROOT

from ts_transformer.approach_difficulty import STRATA_COVARIATES, strata_masks  # noqa: E402
from ts_transformer.config import DURATION_HEAD_QUANTILE, DURATION_MEDIAN_INDEX  # noqa: E402
from flight_scenarios.identity import summary_row_key  # noqa: E402

RESULT_SCHEMA = "ts-eta-error-readout-b0-v3"

# The width an interval would need (absolute) and the shape it would need (signed).
ABSOLUTE_QUANTILES = (50, 80, 90)
SIGNED_QUANTILES = (10, 50, 90)
# The metrics read out of the summary rows. The published block keeps the row's own name:
# it carries the signed quantiles as well, so calling it "abs_..." would misname half of it.
METRICS = ("final_time_error_s", "fde_m")
#: B1.b: the QUANTILE head's median error, derived here rather than published per record —
#: `duration_quantiles_s` and `true_final_time_s` are both already in the summary row, and a
#: second stored copy of their difference is a second thing that can disagree. Gate 1 of
#: §三 3.1b is read on its MAE.
Q50_METRIC = "duration_q50_error_s"
#: The fields it is derived from; absent on every point-head arm, which simply has no block.
Q50_SOURCE_FIELDS = ("duration_quantiles_s", "true_final_time_s")
# A row is usable only if it carries every metric AND every covariate the strata are cut
# on — `established_at_anchor` included, because a present-but-null flag reads as False and
# would move that flight into the vectored stratum unnoticed.
REQUIRED_FIELDS = METRICS + STRATA_COVARIATES


def scored_rows(summary: dict) -> tuple[dict[str, dict], dict[str, int]]:
    """The arm's rows keyed by flight, plus what was dropped for want of a field."""
    results = summary["results"]
    rows = {
        summary_row_key(row): row
        for row in results
        if all(row.get(name) is not None for name in REQUIRED_FIELDS)
    }
    coverage = {
        "summary_rows": len(results),
        "scored_rows": len(rows),
        "dropped_unscored_rows": len(results) - len(rows),
    }
    return rows, coverage


def quantile_block(values: np.ndarray) -> dict[str, float]:
    """One metric's shape: the absolute quantiles and the signed ones, one definition.

    ``mae`` rides along because the B gates are written in MAE (§三 3.1b gate 1) while the
    interval width they justify is a quantile; deriving it anywhere else would be a second
    definition of the same mean.
    """
    absolute = np.abs(values)
    return {
        "mae": float(absolute.mean()),
        **{f"abs_p{q}": float(np.percentile(absolute, q)) for q in ABSOLUTE_QUANTILES},
        **{f"signed_p{q}": float(np.percentile(values, q)) for q in SIGNED_QUANTILES},
        "mean_signed": float(values.mean()),
    }


def q50_errors(rows: dict[str, dict], keys: list[str]) -> np.ndarray | None:
    """``q50 − truth`` per flight, or None when this arm has no quantile head.

    One arm is one checkpoint and one config, so either every scored row carries the five
    quantiles or none does; a partial set means the directory holds two decodes and is
    refused rather than averaged over.
    """
    carried = [key for key in keys if rows[key].get("duration_quantiles_s") is not None]
    if not carried:
        return None
    if len(carried) != len(keys):
        raise SystemExit(
            f"{len(carried)} of {len(keys)} scored rows carry duration_quantiles_s: one arm "
            "is one duration head, so this directory is not one arm"
        )
    # `true_final_time_s` is written on every scored row (`export.py`), so a missing one is
    # a broken directory, not a point-head arm — and it must say which field it is.
    missing = [key for key in keys if rows[key].get("true_final_time_s") is None]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(keys)} scored rows carry duration_quantiles_s but no "
            "true_final_time_s; the q50 error cannot be derived from this directory"
        )
    return np.array(
        [rows[key]["duration_quantiles_s"][DURATION_MEDIAN_INDEX]
         - float(rows[key]["true_final_time_s"]) for key in keys],
        dtype=np.float64,
    )


def readout(label: str, pred_dir: Path) -> dict:
    summary = json.loads((pred_dir / "summary.json").read_text())
    rows, coverage = scored_rows(summary)
    keys = sorted(rows)
    if not keys:
        raise SystemExit(f"{pred_dir} has no rows carrying all of {REQUIRED_FIELDS}")
    masks = strata_masks(rows, keys)
    values = {
        name: np.array([float(rows[key][name]) for key in keys], dtype=np.float64)
        for name in METRICS
    }
    q50 = q50_errors(rows, keys)
    if q50 is not None:
        values[Q50_METRIC] = q50
    metrics = tuple(values)
    strata = {}
    for stratum, mask in masks.items():
        selected = np.flatnonzero(mask)
        if not len(selected):
            continue
        strata[stratum] = {
            "n": int(len(selected)),
            **{name: quantile_block(values[name][selected]) for name in metrics},
        }
    return {
        "label": label,
        "arm": str(pred_dir),
        "split": summary.get("split"),
        "predictor": summary.get("config", {}).get("prediction_output"),
        # Which duration head wrote these rows: it decides whether the q50 block exists and
        # whether it is a second measurement or the same one twice.
        "duration_head": summary.get("config", {}).get("duration_head"),
        "flights": len(keys),
        "coverage": coverage,
        "metrics": list(metrics),
        "strata": strata,
    }


def render(payload: dict) -> str:
    # v3 added `mae` to every block and `metrics` to every arm, and this reads both. A v2
    # artifact is re-rendered by re-running the readout, not by half-filling the table.
    if payload.get("schema") != RESULT_SCHEMA:
        raise SystemExit(
            f"schema {payload.get('schema')!r} is not {RESULT_SCHEMA}; re-run the readout "
            "on the prediction directory rather than re-rendering the stored block"
        )
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
            f"   {'stratum':>46s} {'n':>5s} {'MAE':>7s} {'|dt|p50':>8s} {'|dt|p80':>8s} "
            f"{'|dt|p90':>8s} {'dt p10':>8s} {'dt p50':>8s} {'dt p90':>8s} {'FDE p50':>9s} "
            f"{'FDE p80':>9s} {'FDE p90':>9s}",
        ]
        for stratum, block in arm["strata"].items():
            time_block, fde = block["final_time_error_s"], block["fde_m"]
            lines.append(
                f"   {stratum:>46s} {block['n']:>5d} {time_block['mae']:>7.1f} "
                f"{time_block['abs_p50']:>8.1f} {time_block['abs_p80']:>8.1f} "
                f"{time_block['abs_p90']:>8.1f} {time_block['signed_p10']:>8.1f} "
                f"{time_block['signed_p50']:>8.1f} {time_block['signed_p90']:>8.1f} "
                f"{fde['abs_p50']:>9.1f} {fde['abs_p80']:>9.1f} {fde['abs_p90']:>9.1f}"
            )
        lines += _q50_lines(arm)
    return "\n".join(lines) + "\n"


def _q50_lines(arm: dict) -> list[str]:
    """The quantile head's own duration error, when this arm has one.

    Printed apart from the table above rather than as more columns in it: the table's
    `final_time_error_s` is the error of the duration the ROLLOUT flew, and on a `two-head`
    arm that is a different head's answer. Gate 1 of §三 3.1b reads the MAE here.
    """
    if Q50_METRIC not in arm["metrics"]:
        return []
    same = arm["duration_head"] == DURATION_HEAD_QUANTILE
    lines = [
        "   quantile head q50 vs truth — GATE 1 of §三 3.1b reads this MAE"
        + ("  (identical to the rows above: q50 IS the duration the rollout flew)" if same
           else "  (the rows above are the POINT head, which drove the rollout)"),
        f"   {'stratum':>46s} {'n':>5s} {'MAE':>7s} {'|dt|p50':>8s} {'|dt|p80':>8s} "
        f"{'|dt|p90':>8s} {'dt p10':>8s} {'dt p50':>8s} {'dt p90':>8s}",
    ]
    for stratum, block in arm["strata"].items():
        q50 = block[Q50_METRIC]
        lines.append(
            f"   {stratum:>46s} {block['n']:>5d} {q50['mae']:>7.1f} "
            f"{q50['abs_p50']:>8.1f} {q50['abs_p80']:>8.1f} {q50['abs_p90']:>8.1f} "
            f"{q50['signed_p10']:>8.1f} {q50['signed_p50']:>8.1f} {q50['signed_p90']:>8.1f}"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("arms", nargs="+", metavar="LABEL=PRED_DIR",
                        help="a scored prediction directory and the name it is reported "
                             "under; repeatable. A relative PRED_DIR resolves against the "
                             "repository root")
    parser.add_argument("--json", type=Path, default=None,
                        help="write the block here as well; must not exist (an immutable "
                             "artifact). A relative path resolves against the repository root")
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
        if out.exists():
            raise FileExistsError(f"{out} exists; a readout is an immutable artifact")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
