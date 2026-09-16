#!/usr/bin/env python
"""Displacement error against LEAD TIME from the anchor, per stratum, off stored records.

Two-tier feasibility (`docs/2026-09-16_two_tier_transformer_feasibility.zh.md` §3): the
premise "a short-horizon model has a much lower ADE" is a statement about how a
prediction's error GROWS with the lead time, and that curve is already on disk — every
`predict` directory stores the predicted states and the observed states of each flight on
one clock from the anchor. Reading it off is what this runner does; it never predicts and
writes nothing unless asked for `--json`::

    python run_ts.py lead_time_error \\
        native32=4dTrajectory/outputs/KRDU/experiments/l1_lowdim_20260907/L1_native32_pred_val

Per arm and stratum (`approach_difficulty.strata_masks`, the cut every readout uses) it
reports, at each lead h of `--leads` (positive whole multiples of the 1 s grid):

* the 3D displacement between the predicted and the observed position at h
  (mean / p50 / p90) over the flights whose prediction AND truth both reach h — a flight
  that ends earlier is ABSENT from that lead, never scored 0 and never held at its last
  node. So every lead holds its OWN cohort and the printed `n` is part of the number: a
  column read downward past the point where flights start ending is composition, not
  growth (the 300 → 420 s medians FALL on the KRDU arms for exactly that reason);
* the ADE over [0, h] for the same flights — the "short-horizon ADE" a re-asked model
  would be graded on if it were graded only to its own horizon;
* the growth exponent: the log-log slope of the displacement MEDIAN between 10 and 60 s
  and between 60 and 120 s, PAIRED over the flights present at both leads of the pair
  (the package's rule for any reading across cohorts that differ, `anytime_curve`), with
  that n (1 = a heading error integrated once, 2 = a turn-rate or acceleration error
  integrated twice);
* the whole-record ADE of the stratum from the summary rows, for reference ONLY: it is a
  different accounting in three ways — scored against the observed track WITH its fitted
  tail to the threshold (`true_final_time_s`, a median ~6 s past the last observed row this
  curve stops at), on 64 fractions of that duration excluding t = 0, and with the
  prediction held at its last node when it ends first. It is not the h → ∞ limit of the
  `ADE[0,h]` column.

Positions are the records' own geodetic states in the package's one equirectangular
chart (`geometry.geometric_metrics.chart_rows`, the chart every geometry readout uses),
about each flight's anchor, both series interpolated onto the grid. A record with fewer
than two rows from the anchor in either series refuses the whole arm (a record, not a
row, is broken there); a summary row missing a field is dropped and counted.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from ts_transformer.experiments.support import REPO_ROOT

from ts_transformer.data.approach_difficulty import STRATA_COVARIATES, strata_masks  # noqa: E402
import ts_transformer.geometry.geometric_metrics as gm  # noqa: E402
from flight_scenarios.identity import summary_row_key  # noqa: E402

RESULT_SCHEMA = "ts-lead-time-error-v1"

DEFAULT_LEADS_S = (10.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0, 150.0, 180.0, 240.0, 300.0,
                   360.0, 420.0)
#: The two lead pairs the growth exponent is read between: inside the tier-1 horizon the
#: two-tier design asks about (≤ 60 s) and across it (60 → 120 s).
SLOPE_PAIRS_S = ((10.0, 60.0), (60.0, 120.0))
GRID_DT_S = 1.0
# A row is usable only if it names its states file, carries the whole-record ADE this
# curve is read against, and every covariate the strata are cut on.
REQUIRED_FIELDS = ("states_file", "ade_m") + STRATA_COVARIATES


def scored_rows(summary: dict) -> tuple[dict[str, dict], dict[str, int]]:
    """The arm's rows keyed by flight, plus what was dropped for want of a field."""
    results = summary["results"]
    rows = {
        summary_row_key(row): row
        for row in results
        if all(row.get(name) is not None for name in REQUIRED_FIELDS)
    }
    return rows, {
        "summary_rows": len(results),
        "scored_rows": len(rows),
        "dropped_unscored_rows": len(results) - len(rows),
    }


def validated_leads(leads: Sequence[float]) -> tuple[float, ...]:
    """Each lead finite, positive, a whole multiple of the grid, and given once — a
    fractional lead would be rounded onto a neighbour's key, a negative one would index
    the curve from its END."""
    out: list[float] = []
    for h in leads:
        steps = h / GRID_DT_S
        if not math.isfinite(h) or h <= 0.0 or abs(steps - round(steps)) > 1e-9:
            raise SystemExit(f"lead {h!r} s is not a positive whole multiple of the "
                             f"{GRID_DT_S:g} s grid")
        if h in out:
            raise SystemExit(f"lead {h:g} s is given twice")
        out.append(float(h))
    return tuple(out)


def lead_key(h: float) -> str:
    return str(int(round(h / GRID_DT_S)) * int(GRID_DT_S)) if GRID_DT_S.is_integer() else f"{h:g}"


def _validated_clock(t: np.ndarray, label: str) -> None:
    if not np.isfinite(t).all() or t[0] != 0.0 or np.any(np.diff(t) <= 0.0):
        raise SystemExit(f"{label}: clock must be finite, start at the anchor (t = 0) and "
                         "increase strictly")


def displacement_curve(states: dict, label: str) -> tuple[np.ndarray, np.ndarray]:
    """The grid from the anchor to the EARLIER of the two records' ends, and the 3D
    displacement between the predicted and the observed position on it."""
    predicted = states["predicted_states"]
    observed = [row for row in states["observed_states"] if row["t"] >= 0.0]
    if len(predicted) < 2 or len(observed) < 2:
        raise SystemExit(f"{label}: fewer than two rows from the anchor in one of the records")
    anchor = predicted[0]
    pred = gm.chart_rows(predicted, anchor)
    truth = gm.chart_rows(observed, anchor)
    _validated_clock(pred[:, 3], f"{label} predicted_states")
    _validated_clock(truth[:, 3], f"{label} observed_states")
    t_end = min(pred[-1, 3], truth[-1, 3])
    grid = np.arange(0.0, t_end + 1e-9, GRID_DT_S)
    sample = lambda rows: np.column_stack([np.interp(grid, rows[:, 3], rows[:, k]) for k in range(3)])
    return grid, np.linalg.norm(sample(pred) - sample(truth), axis=1)


def log_log_slope(h1: float, e1: float, h2: float, e2: float) -> float:
    if min(e1, e2) <= 0.0:
        raise ValueError(f"a log-log slope needs positive displacements, got {e1}, {e2}")
    return math.log(e2 / e1) / math.log(h2 / h1)


def readout(label: str, pred_dir: Path, *, leads: Sequence[float] = DEFAULT_LEADS_S) -> dict:
    leads = validated_leads(leads)
    summary = json.loads((pred_dir / "summary.json").read_text())
    rows, coverage = scored_rows(summary)
    keys = sorted(rows)
    if not keys:
        raise SystemExit(f"{pred_dir} has no rows carrying all of {REQUIRED_FIELDS}")
    curves = {}
    for key in keys:
        states = json.loads((pred_dir / rows[key]["states_file"]).read_text())
        curves[key] = displacement_curve(states, f"{label} {rows[key]['states_file']}")
    masks = strata_masks(rows, keys)
    strata = {}
    for stratum, mask in masks.items():
        selected = [key for key, keep in zip(keys, mask) if keep]
        if not selected:
            continue
        whole = np.array([float(rows[key]["ade_m"]) for key in selected])
        reaching = {h: [key for key in selected if curves[key][0][-1] >= h] for h in leads}
        at_lead, ade_to = {}, {}
        for h in leads:
            if not reaching[h]:
                continue
            index = int(round(h / GRID_DT_S))
            disp = np.array([curves[key][1][index] for key in reaching[h]])
            cumulative = np.array([curves[key][1][: index + 1].mean() for key in reaching[h]])
            at_lead[lead_key(h)] = {
                "n": int(len(disp)), "mean": float(disp.mean()),
                "p50": float(np.median(disp)), "p90": float(np.percentile(disp, 90)),
            }
            ade_to[lead_key(h)] = {
                "n": int(len(cumulative)), "mean": float(cumulative.mean()),
                "p50": float(np.median(cumulative)),
            }
        slopes = {}
        for h1, h2 in SLOPE_PAIRS_S:
            if h1 not in leads or h2 not in leads:
                continue
            both = [key for key in reaching[h1] if curves[key][0][-1] >= h2]
            if not both:
                continue
            i1, i2 = int(round(h1 / GRID_DT_S)), int(round(h2 / GRID_DT_S))
            p50_short = float(np.median([curves[key][1][i1] for key in both]))
            p50_long = float(np.median([curves[key][1][i2] for key in both]))
            slopes[f"{lead_key(h1)}-{lead_key(h2)}"] = {
                "n": len(both), "p50_short": p50_short, "p50_long": p50_long,
                "slope": log_log_slope(h1, p50_short, h2, p50_long),
            }
        strata[stratum] = {
            "n": len(selected),
            "whole_record_ade": {"mean": float(whole.mean()), "p50": float(np.median(whole))},
            "at_lead": at_lead,
            "ade_to": ade_to,
            "paired_p50_log_log_slope": slopes,
        }
    return {
        "label": label,
        "arm": str(pred_dir),
        "split": summary.get("split"),
        "predictor": summary.get("config", {}).get("prediction_output"),
        "flights": len(keys),
        "coverage": coverage,
        "leads_s": list(leads),
        "strata": strata,
    }


def render(payload: dict) -> str:
    if payload.get("schema") != RESULT_SCHEMA:
        raise SystemExit(
            f"schema {payload.get('schema')!r} is not {RESULT_SCHEMA}; re-run the readout "
            "on the prediction directory rather than re-rendering the stored block"
        )
    dt = payload["grid_dt_s"]
    lines = [
        "Displacement error against lead time from the anchor (metres), per stratum",
        f"at each lead: the flights whose prediction AND truth both reach it (absent, never 0 "
        f"or held) — n changes with the lead, so read a column with its n; ADE[0,h] = mean "
        f"displacement on the {dt:g} s grid up to h over the same flights; the slope is the "
        f"log-log slope of the p50 PAIRED over the flights present at both leads (n given).",
        "whole = the summary rows' ade_m, a different accounting: scored against the observed "
        "track plus its fitted tail to the threshold, on 64 fractions of true_final_time_s "
        "excluding t = 0, the prediction held at its last node — not the limit of ADE[0,h].",
    ]
    for arm in payload["arms"]:
        coverage = arm["coverage"]
        lines += [
            "",
            f"── {arm['label']} — {arm['arm']} ({arm['predictor']} output, split "
            f"{arm['split']}, {arm['flights']} flights; {coverage['dropped_unscored_rows']} "
            f"of {coverage['summary_rows']} rows unscored, dropped)",
        ]
        for stratum, block in arm["strata"].items():
            whole = block["whole_record_ade"]
            slopes = ", ".join(
                f"{pair} s: {v['slope']:.2f} (n={v['n']})"
                for pair, v in block["paired_p50_log_log_slope"].items()
            )
            lines += [
                f"   {stratum}  [n={block['n']}]  whole-record ADE mean {whole['mean']:.0f} "
                f"/ p50 {whole['p50']:.0f};  paired p50 log-log slope {slopes or 'n/a'}",
                f"   {'lead s':>7} {'n':>5} {'disp mean':>10} {'p50':>7} {'p90':>7} | "
                f"{'ADE[0,h] mean':>14} {'p50':>7}",
            ]
            for h in arm["leads_s"]:
                key = lead_key(h)
                if key not in block["at_lead"]:
                    continue
                a, c = block["at_lead"][key], block["ade_to"][key]
                lines.append(
                    f"   {h:>7.0f} {a['n']:>5d} {a['mean']:>10.0f} {a['p50']:>7.0f} "
                    f"{a['p90']:>7.0f} | {c['mean']:>14.0f} {c['p50']:>7.0f}"
                )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("arms", nargs="+", metavar="LABEL=PRED_DIR",
                        help="a scored prediction directory and the name it is reported "
                             "under; repeatable. A relative PRED_DIR resolves against the "
                             "repository root")
    parser.add_argument("--leads", default=",".join(f"{h:g}" for h in DEFAULT_LEADS_S),
                        help="comma-separated lead times in seconds: positive whole "
                             f"multiples of the {GRID_DT_S:g} s grid")
    parser.add_argument("--json", type=Path, default=None,
                        help="write the block here (.json) and its rendering beside it "
                             "(.txt); neither may exist (immutable artifacts). A relative "
                             "path resolves against the repository root")
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
    leads = validated_leads(tuple(float(x) for x in args.leads.split(",")))

    out = text_out = None
    if args.json is not None:
        out = args.json if args.json.is_absolute() else REPO_ROOT / args.json
        if out.suffix == ".txt":
            parser.error(f"--json {args.json} names the rendering's own suffix; give the .json")
        text_out = out.with_suffix(".txt")
        for path in (out, text_out):
            if path.exists():
                raise FileExistsError(f"{path} exists; a readout is an immutable artifact")

    payload = {
        "schema": RESULT_SCHEMA,
        "grid_dt_s": GRID_DT_S,
        "slope_pairs_s": [list(pair) for pair in SLOPE_PAIRS_S],
        "arms": [readout(label, path, leads=leads) for label, path in arms.items()],
    }
    text = render(payload)
    print(text, end="")
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        text_out.write_text(text)
        print(f"wrote {out} and {text_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
