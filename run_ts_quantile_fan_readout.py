#!/usr/bin/env python
"""B3: does the quantile fan actually contain the flight it is a fan of?

Anytime / calibrated-ETA design
(`4dTrajectory/ts_transformer/docs/2026-09-07_anytime_prediction_and_calibrated_eta_design.zh.md`)
§三 3.3 and gate 3.4-3. `predict --cta-from-quantiles` decodes one trajectory per duration
quantile — a `cta=self-q` arm, the first CTA arm that reads no future — and this is its
readout::

    conda activate aeroviz
    python run_ts_quantile_fan_readout.py --arm <pred_dir> --json <out>/quantile_fan.json

``<pred_dir>`` is the top-1 directory (its own records ARE the q50 decode); the five
trajectories live under ``<pred_dir>/quantiles/qNN/``. **Only those five are read** — the
calibrated interval-endpoint directories (`predict --interval-endpoints`, off by default)
are an extra arm to look at, not part of any gate here, so a fan predicted without them is
complete as far as this readout is concerned.

Three readings, per `approach_difficulty.strata_masks` stratum:

1. **The duration fan**: the share of flights whose truth ``T`` falls in ``[q10, q90]``, and
   in the CALIBRATED interval when the arm was predicted with a conformal table, beside the
   median widths. §六 6 — these are quantiles of the DURATION. **The calibrated-hit column is
   IN-SAMPLE whenever this arm's split is the split the δ was fitted on** (the usual case:
   both are `val`); it is marked as such, and the gate number is the DEPLOYED coverage in
   `run_ts_eta_calibration.py`'s readout, measured on the half the δ was not fitted on.
2. **The width**: the median ``q90 − q10`` and the median calibrated width per α. The design's
   veto is on this number: a vectored median width above 120 s has no scheduling meaning.
3. **The geometric coverage (gate 3.4-3)**: the truth path's chamfer to the NEAREST of the
   five decoded trajectories against its chamfer to q50. The gate reads the subset whose
   truth duration is inside the fan — asking whether a path drawn to the right arrival time
   is a better path — and the whole cohort is printed beside it, never instead of it. **It
   is a readout, not a coverage guarantee** (§六 6): five trajectories are not a
   distribution over trajectories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
TS_DIR = REPO_ROOT / "4dTrajectory" / "ts_transformer"
for path in (TS_DIR.parent, REPO_ROOT / "geokit" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ts_transformer.geometric_metrics as gm  # noqa: E402
from ts_transformer.approach_difficulty import STRATA_COVARIATES, strata_masks  # noqa: E402
from ts_transformer.calibration import (  # noqa: E402
    CONFORMAL_ALPHAS,
    QUANTILE_DIR_NAME,
    quantile_directory_name,
    quantile_pair_indices,
)
from ts_transformer.config import DURATION_MEDIAN_INDEX, DURATION_QUANTILES  # noqa: E402
from flight_scenarios.identity import summary_row_key  # noqa: E402

RESULT_SCHEMA = "ts-quantile-fan-readout-b3-v1"

# What a row must carry to be scored by a fan readout: the truth duration the geometry is
# closed onto, and the covariates every stratum is cut by. A readout that needs more (the
# latent fan cross-checks ADE) passes its own superset.
FAN_REQUIRED_FIELDS = ("true_final_time_s", *STRATA_COVARIATES)
# This readout's fan is read against q50, which IS one of the five leaves.
QUANTILE_REFERENCE_KEY = "chamfer_q50_p50_m"


def leaf_rows(
    pred_dir: Path, *, required: tuple[str, ...] = FAN_REQUIRED_FIELDS
) -> tuple[dict[str, dict], dict[str, int], str]:
    """One prediction directory's scored rows by flight key, what a missing field dropped,
    and the split. Every fan readout reads a leaf through this, so "scored" means the same
    thing in the top-1 directory and in each leaf of its fan."""
    summary = json.loads((pred_dir / "summary.json").read_text())
    results = summary["results"]
    rows = {
        summary_row_key(row): row
        for row in results
        if all(row.get(name) is not None for name in required)
    }
    return rows, {
        "summary_rows": len(results),
        "scored_rows": len(rows),
        "dropped_unscored_rows": len(results) - len(rows),
    }, str(summary.get("split"))


def _interval_entry(row: dict, alpha: float) -> dict:
    """That row's calibrated interval for ``alpha``, found by its own key."""
    entry = next(
        (item for item in row["duration_interval_s"] if item["alpha"] == alpha), None
    )
    if entry is None:
        raise SystemExit(
            f"a record carries no calibrated interval for alpha={alpha:g}; it was predicted "
            "under a different set of conformal levels"
        )
    return entry


def _flight_geometry(pred_dir: Path, row: dict, geometry_truth: str) -> tuple[float, np.ndarray]:
    """One flight's time-free distance between its decoded path and the observed truth, and
    where that decode ENDS in the flight's threshold-anchored chart — both off one read of
    the record, because a fan readout wants the spread of the leaves' endpoints as well as
    each leaf's distance to the truth."""
    eval_record = json.loads((pred_dir / row["eval_file"]).read_text())
    states = json.loads((pred_dir / row["states_file"]).read_text())
    metrics = gm.record_geometry(eval_record, states, row, geometry_truth=geometry_truth)
    endpoint = gm.chart_rows(states["predicted_states"][-1:], eval_record["target_state"])[0, :2]
    return float(metrics["chamfer_m"]), endpoint


def _quantile_arms(arm: Path) -> dict[float, Path]:
    """The five per-quantile prediction directories, refusing a partial fan."""
    directories = {
        tau: arm / QUANTILE_DIR_NAME / quantile_directory_name(tau)
        for tau in DURATION_QUANTILES
    }
    missing = [str(path) for path in directories.values() if not (path / "summary.json").is_file()]
    if missing:
        raise SystemExit(
            f"{arm} has no complete quantile fan; missing {len(missing)} of "
            f"{len(directories)} directories, first {missing[0]}. Predict it with "
            "`predict --cta-from-quantiles`"
        )
    return directories


def share(values: np.ndarray) -> float | None:
    return float(np.mean(values)) if len(values) else None


def median(values: np.ndarray) -> float | None:
    return float(np.median(values)) if len(values) else None


def readout(arm: Path, *, geometry_truth: str) -> dict:
    rows, coverage, summary_split = leaf_rows(arm)
    keys = sorted(rows)
    if not keys:
        raise SystemExit(f"{arm} has no rows carrying a truth duration and the strata covariates")
    quantiles = np.array([rows[key]["duration_quantiles_s"] for key in keys], dtype=np.float64)
    if quantiles.shape[1] != len(DURATION_QUANTILES):
        raise SystemExit(
            f"{arm} carries {quantiles.shape[1]} duration quantiles, not "
            f"{len(DURATION_QUANTILES)}; it was not predicted by this build"
        )
    truth = np.array([rows[key]["true_final_time_s"] for key in keys], dtype=np.float64)
    low_index, high_index = quantile_pair_indices(min(CONFORMAL_ALPHAS))
    in_fan = (truth >= quantiles[:, low_index]) & (truth <= quantiles[:, high_index])
    fan_width = quantiles[:, high_index] - quantiles[:, low_index]

    calibrated = all(rows[key].get("duration_interval_s") for key in keys)
    # MEDIUM-6: when the arm's split IS the split the delta was fitted on, the interval-hit
    # column is IN-SAMPLE — the flights scored here are the ones the calibration saw. The
    # held-out number lives in the calibration readout, and the header says so.
    cohort = rows[keys[0]].get("duration_interval_cohort") if calibrated else None
    in_sample = bool(cohort) and cohort.get("split") == summary_split
    interval_hit: dict[float, np.ndarray] = {}
    interval_width: dict[float, np.ndarray] = {}
    if calibrated:
        for alpha in CONFORMAL_ALPHAS:
            # By the entry's OWN alpha, never by position: the record is a list and a future
            # alpha inserted anywhere but the end would silently re-label every column.
            bounds = np.array(
                [[_interval_entry(rows[key], alpha)[end] for end in ("lo", "hi")]
                 for key in keys],
                dtype=np.float64,
            )
            interval_hit[alpha] = (truth >= bounds[:, 0]) & (truth <= bounds[:, 1])
            interval_width[alpha] = bounds[:, 1] - bounds[:, 0]

    # The geometry: chamfer to each of the five, and to q50 (which IS the top-1 directory).
    directories = _quantile_arms(arm)
    fan_chamfer = np.stack([
        leaf_geometry(directories[tau], keys, geometry_truth=geometry_truth)["chamfer_m"]
        for tau in DURATION_QUANTILES
    ])                                                            # [Q, N]
    nearest = fan_chamfer.min(axis=0)
    median_chamfer = fan_chamfer[DURATION_MEDIAN_INDEX]

    masks = strata_masks(rows, keys)
    strata = {}
    for stratum, mask in masks.items():
        selected = np.flatnonzero(mask)
        if not len(selected):
            continue
        gated = np.flatnonzero(mask & in_fan)
        strata[stratum] = {
            "n": int(len(selected)),
            "truth_in_fan_share": share(in_fan[selected]),
            "fan_width_p50_s": median(fan_width[selected]),
            "calibrated": calibrated,
            "interval": {
                f"{alpha:g}": {
                    "truth_in_interval_share": share(interval_hit[alpha][selected]),
                    "width_p50_s": median(interval_width[alpha][selected]),
                }
                for alpha in CONFORMAL_ALPHAS
            } if calibrated else {},
            # Gate 3.4-3 reads the in-fan subset; the whole stratum is beside it.
            "geometry": {
                "all": geometry_cell(median_chamfer[selected], nearest[selected],
                                     reference_key=QUANTILE_REFERENCE_KEY),
                "in_fan": geometry_cell(median_chamfer[gated], nearest[gated],
                                        reference_key=QUANTILE_REFERENCE_KEY),
                "in_fan_flights": int(len(gated)),
            },
        }
    return {
        "arm": str(arm),
        "geometry_truth": geometry_truth,
        "quantiles": list(DURATION_QUANTILES),
        "fan_alpha": min(CONFORMAL_ALPHAS),
        "flights": len(keys),
        "coverage": coverage,
        "calibrated": calibrated,
        "calibration_cohort": cohort,
        # The interval-hit columns are in-sample when this arm's split is the calibration's.
        "interval_hit_in_sample": in_sample,
        "strata": strata,
    }


def leaf_geometry(
    pred_dir: Path,
    keys: list[str],
    *,
    geometry_truth: str,
    required: tuple[str, ...] = FAN_REQUIRED_FIELDS,
) -> dict:
    """What one leaf contributes to a fan readout, all of it in ``keys`` order: its scored
    ``rows``, its ``chamfer_m`` (``[F]``) to the truth, and its decoded ``endpoint_en``
    (``[F, 2]``) in each flight's threshold-anchored chart.

    The directory's summary is parsed ONCE — each leaf writes its own, and a caller that
    also wants the leaf's ADE reads it out of ``rows`` rather than parsing it again — and
    the cohort must match the top-1's exactly, or the fan is not a fan of these flights.
    """
    rows, _coverage, _split = leaf_rows(pred_dir, required=required)
    missing = [key for key in keys if key not in rows]
    if missing:
        raise SystemExit(
            f"{pred_dir} is missing {len(missing)} of {len(keys)} flights (first "
            f"{missing[0]!r}); the fan is not the same cohort as the top-1"
        )
    ordered = [rows[key] for key in keys]
    pairs = [_flight_geometry(pred_dir, row, geometry_truth) for row in ordered]
    return {
        "rows": ordered,
        "chamfer_m": np.array([chamfer for chamfer, _end in pairs], dtype=np.float64),
        "endpoint_en": np.array([end for _chamfer, end in pairs], dtype=np.float64),
    }


def geometry_cell(reference: np.ndarray, nearest: np.ndarray, *, reference_key: str) -> dict:
    """The chamfer-to-nearest-leaf reading: the reference decode's chamfer, the nearest
    leaf's, and the share of flights the fan improves on. ``reference_key`` names what the
    fan is read against — q50 in the B line, the top-1 decode in the latent line — so the
    JSON never says "q50" about a number that is not one."""
    return {
        reference_key: median(reference),
        "chamfer_nearest_p50_m": median(nearest),
        "nearest_better_share": share(nearest < reference) if len(nearest) else None,
    }


def render(payload: dict) -> str:
    fan = payload["fan_alpha"]
    low, high = quantile_pair_indices(fan)
    in_sample = payload.get("interval_hit_in_sample")
    hit_header = "cal.hit*" if in_sample else "cal.hit"
    lines = [
        f"B3 quantile fan — {payload['arm']} ({payload['flights']} flights, "
        f"{'CALIBRATED' if payload['calibrated'] else 'uncalibrated'})",
        f"the fan is [q{DURATION_QUANTILES[low]:g}, q{DURATION_QUANTILES[high]:g}] "
        f"(nominal {(1 - fan) * 100:.0f}% of the DURATION); the geometry columns are a "
        "readout, not a coverage guarantee",
    ]
    if in_sample:
        lines.append(
            f"   * cal.hit is IN-SAMPLE: this arm's split ({payload['calibration_cohort']['split']}) "
            "IS the split the conformal delta was fitted on. The gate number is the "
            "DEPLOYED coverage in run_ts_eta_calibration.py's readout, measured on the "
            "held-out half."
        )
    if payload.get("calibration_cohort", {}) and payload["calibration_cohort"].get("smokeTest"):
        lines.append("   * the conformal table is a SMOKE table (--limit): its delta was "
                     "fitted on a prefix of the split")
    lines.append(
        f"   {'stratum':>46s} {'n':>5s} {'in fan':>7s} {'width s':>8s} "
        f"{hit_header:>8s} {'cal.w s':>8s} {'cham q50':>9s} {'cham min':>9s} "
        f"{'min<q50':>8s} {'n in fan':>9s}"
    )
    for stratum, block in payload["strata"].items():
        geometry = block["geometry"]["in_fan"]
        interval = block["interval"].get(f"{fan:g}", {})
        lines.append(
            f"   {stratum:>46s} {block['n']:>5d} "
            f"{block['truth_in_fan_share']:>7.3f} {block['fan_width_p50_s']:>8.1f} "
            f"{cell(interval.get('truth_in_interval_share'), '.3f'):>8s} "
            f"{cell(interval.get('width_p50_s'), '.1f'):>8s} "
            f"{cell(geometry[QUANTILE_REFERENCE_KEY], '.0f'):>9s} "
            f"{cell(geometry['chamfer_nearest_p50_m'], '.0f'):>9s} "
            f"{cell(geometry['nearest_better_share'], '.3f'):>8s} "
            f"{block['geometry']['in_fan_flights']:>9d}"
        )
    return "\n".join(lines) + "\n"


def cell(value: float | None, spec: str) -> str:
    return "n/a" if value is None else f"{value:{spec}}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--arm", type=Path, required=True,
                        help="the top-1 prediction directory of a --cta-from-quantiles run "
                             "(its quantiles/ subdirectories hold the fan)")
    parser.add_argument("--geometry-truth", choices=gm.GEOMETRY_TRUTHS,
                        default=gm.GEOMETRY_TRUTH_CLOSED,
                        help="truth the time-free metrics are read against "
                             "(see geometric_metrics)")
    parser.add_argument("--json", type=Path, default=None,
                        help="write the block here as well; must not exist (an immutable "
                             "artifact)")
    args = parser.parse_args(argv)
    arm = args.arm if args.arm.is_absolute() else REPO_ROOT / args.arm
    payload = {"schema": RESULT_SCHEMA, **readout(arm, geometry_truth=args.geometry_truth)}
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
