#!/usr/bin/env python
"""Readout for the final-approach constraint campaign (2026-09-04).

Every arm is a prediction directory (``summary.json`` + per-flight records) on the same
validation split; the first is the reference (arm A). Beyond the paired accuracy and
endpoint numbers ``compare_frame_arms`` already prints, this readout scores each
prediction against the optimizer's two final-leg constraint rows on the rows where the
OBSERVED track is established (``final_approach_geometry.truth_final_gate``: inside the
k=0.5 LPV cone from there to the threshold, beyond the last 300 m), so the gate is the
same for every arm and comes from the truth, never from the model:

* corridor: metres outside ``k·halfwidth(d)`` on gated rows — violation rate (rows),
  flights with any violation, mean/p95 of the per-flight worst excess;
* glidepath window: the same for ``[−60, +120] m`` about the coded glidepath;
* the unweighted hinge² terms at the runway scale (100 m / 30 m) — what the penalty
  arm's λ multiplies, i.e. the calibration numbers;
* the observed floor: the truth's own window violation on those rows (its corridor
  violation is zero by construction of the gate);
* what the prediction CLAIMS to be on the final (the ``on-final`` membership read from
  its own rows) and how much of that is inside the corridor.

Rows are matched by time: the truth row at ``t`` (post-anchor observed track) and the
predicted row at the same ``t``; truth rows beyond a truncated forecast's end are
"uncovered" and counted, not scored.

    python compare_constraint_arms.py A=<A_pred_val> proj=<A_project_on_final_pred_val> ... [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TS_DIR = HERE.parent
REPO_ROOT = TS_DIR.parents[1]
for path in (HERE, TS_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import compare_frame_arms as cfa  # noqa: E402
import final_approach_geometry as fag  # noqa: E402
from config import TSConfig  # noqa: E402

# The hinge scales the penalty arm trains with (one source: the config defaults).
LATERAL_SCALE_M = TSConfig().procedure_loss_lateral_scale_m
VERTICAL_SCALE_M = TSConfig().procedure_loss_vertical_scale_m


def _chart(rows: list[dict], lat0: float, lon0: float, alt0: float) -> np.ndarray:
    lat = np.array([r["lat"] for r in rows]); lon = np.array([r["lon"] for r in rows])
    e = (lon - lon0) * cfa.metres_per_deg_lon(lat0)
    n = (lat - lat0) * cfa.METRES_PER_DEG_LAT
    u = np.array([r["alt"] for r in rows]) - alt0
    return np.stack([e, n, u], axis=1)


def _velocity(rows: list[dict]) -> np.ndarray:
    v = np.array([r["V"] for r in rows]); psi = np.array([r["psi"] for r in rows])
    gamma = np.array([r["gamma"] for r in rows])
    return np.stack([v * np.cos(gamma) * np.cos(psi), v * np.cos(gamma) * np.sin(psi)], axis=1)


def corridor_metrics(pred_dir: Path, row: dict) -> dict:
    eval_record = json.loads((pred_dir / row["eval_file"]).read_text())
    states = json.loads((pred_dir / row["states_file"]).read_text())
    target = eval_record["target_state"]
    lat0, lon0, alt0 = float(target["lat"]), float(target["lon"]), float(target["alt"])
    psi = torch.tensor([float(target["psi"])], dtype=torch.float64)
    tan_gpa = torch.tensor([math.tan(-float(target["gamma"]))], dtype=torch.float64)

    truth_rows = [r for r in states["observed_states"] if r["t"] >= 0.0]
    truth = _chart(truth_rows, lat0, lon0, alt0)
    d_t, xt_t = fag.runway_axes(
        torch.tensor(truth[:, 0])[None], torch.tensor(truth[:, 1])[None], psi
    )
    gate = fag.truth_final_gate(d_t, xt_t, torch.ones_like(d_t, dtype=torch.bool))[0].numpy()
    truth_lat, truth_vert = fag.corridor_violations(d_t, xt_t, torch.tensor(truth[:, 2])[None], tan_gpa)
    truth_vert = truth_vert[0].numpy()

    pred_rows = states["predicted_states"]
    pred_by_time = {round(float(r["t"]), 3): i for i, r in enumerate(pred_rows)}
    pred = _chart(pred_rows, lat0, lon0, alt0)
    d_p, xt_p = fag.runway_axes(
        torch.tensor(pred[:, 0])[None], torch.tensor(pred[:, 1])[None], psi
    )
    lat_v, vert_v = fag.corridor_violations(d_p, xt_p, torch.tensor(pred[:, 2])[None], tan_gpa)
    lat_v, vert_v = lat_v[0].numpy(), vert_v[0].numpy()
    # "Claimed" = what the exported record says about itself: its own (position-derived,
    # exported) velocity heading inside the membership cone. A diagnostic column, NOT the
    # gate that bound the output (that one reads position differences at prediction time).
    velocity = _velocity(pred_rows)
    cos_align = fag.alignment_cosine(
        torch.tensor(velocity[:, 0])[None], torch.tensor(velocity[:, 1])[None], psi
    )
    claimed = fag.hard_on_final(d_p, xt_p, cos_align)[0].numpy()

    # Recovery (the hook campaign's R-vs-F question): among flights whose FIRST claimed
    # on-final row (before the threshold) is outside the k-corridor, the share whose LAST
    # such row is inside. Read on the prediction's own gate, since that is where a hook acts.
    claimed_approach = np.flatnonzero(claimed & (d_p[0].numpy() > 0.0))
    outside_at_gate_start = bool(lat_v[claimed_approach[0]] > 0) if len(claimed_approach) else None
    recovered = (
        bool(lat_v[claimed_approach[-1]] == 0) if outside_at_gate_start else None
    )

    gated_index = np.flatnonzero(gate)
    matched = [pred_by_time.get(round(float(truth_rows[i]["t"]), 3)) for i in gated_index]
    covered = np.array([m for m in matched if m is not None], dtype=int)
    lat_g = lat_v[covered] if len(covered) else np.zeros(0)
    vert_g = vert_v[covered] if len(covered) else np.zeros(0)
    return {
        "gated_rows": int(len(gated_index)),
        "covered_rows": int(len(covered)),
        "gate_start_d_m": float(d_t[0, gated_index[0]]) if len(gated_index) else math.nan,
        "lateral_violation_rows": int((lat_g > 0).sum()),
        "vertical_violation_rows": int((vert_g > 0).sum()),
        "lateral_excess_max_m": float(lat_g.max()) if len(lat_g) else 0.0,
        "vertical_excess_max_m": float(vert_g.max()) if len(vert_g) else 0.0,
        "lateral_hinge": float(((lat_g / LATERAL_SCALE_M) ** 2).mean()) if len(lat_g) else math.nan,
        "vertical_hinge": float(((vert_g / VERTICAL_SCALE_M) ** 2).mean()) if len(vert_g) else math.nan,
        # The observed floor, over the SAME rows the prediction is scored on.
        "truth_vertical_violation_rows": int(
            (truth_vert[[gated_index[j] for j, m in enumerate(matched) if m is not None]] > 0).sum()
        ),
        "claimed_rows": int(claimed.sum()),
        "claimed_lateral_violation_rows": int(((lat_v > 0) & claimed).sum()),
        "claimed_vertical_violation_rows": int(((vert_v > 0) & claimed).sum()),
        "outside_at_gate_start": outside_at_gate_start,
        "recovered": recovered,
        "projected": states["source"].get("projectedOntoFinal"),
        "command_hook": states["source"].get("commandHook"),
    }


def load_arm(pred_dir: Path) -> dict[str, dict]:
    rows = cfa.load_arm(pred_dir)
    for row in rows.values():
        row.update(corridor_metrics(pred_dir, row))
    return rows


def _rate(numer: np.ndarray, denom: np.ndarray) -> float:
    total = float(denom.sum())
    return float(numer.sum()) / total if total else math.nan


def _stratum_block(rows: list[dict]) -> dict[str, float]:
    gated = np.array([r["gated_rows"] for r in rows], dtype=float)
    covered = np.array([r["covered_rows"] for r in rows], dtype=float)
    lat_rows = np.array([r["lateral_violation_rows"] for r in rows], dtype=float)
    vert_rows = np.array([r["vertical_violation_rows"] for r in rows], dtype=float)
    lat_max = np.array([r["lateral_excess_max_m"] for r in rows])
    vert_max = np.array([r["vertical_excess_max_m"] for r in rows])
    with_gate = covered > 0
    return {
        "flights": len(rows),
        "flights_with_gated_rows": int(with_gate.sum()),
        "coverage": _rate(covered, gated),
        "lateral_violation_rate": _rate(lat_rows, covered),
        "vertical_violation_rate": _rate(vert_rows, covered),
        "truth_vertical_violation_rate": _rate(
            np.array([r["truth_vertical_violation_rows"] for r in rows], dtype=float), covered
        ),
        "flights_any_lateral": float((lat_max[with_gate] > 0).mean()) if with_gate.any() else math.nan,
        "flights_any_vertical": float((vert_max[with_gate] > 0).mean()) if with_gate.any() else math.nan,
        "lateral_excess_max_mean_m": float(lat_max[with_gate].mean()) if with_gate.any() else math.nan,
        "lateral_excess_max_p95_m": float(np.percentile(lat_max[with_gate], 95)) if with_gate.any() else math.nan,
        "vertical_excess_max_mean_m": float(vert_max[with_gate].mean()) if with_gate.any() else math.nan,
        "lateral_hinge_mean": float(np.nanmean([r["lateral_hinge"] for r in rows])),
        "vertical_hinge_mean": float(np.nanmean([r["vertical_hinge"] for r in rows])),
        "claimed_rows": int(sum(r["claimed_rows"] for r in rows)),
        "claimed_lateral_violation_rate": _rate(
            np.array([r["claimed_lateral_violation_rows"] for r in rows], dtype=float),
            np.array([r["claimed_rows"] for r in rows], dtype=float),
        ),
        "outside_at_gate_start_share": (
            float(np.mean([r["outside_at_gate_start"] for r in rows if r["outside_at_gate_start"] is not None]))
            if any(r["outside_at_gate_start"] is not None for r in rows) else math.nan
        ),
        "recovery_candidates": int(sum(bool(r["outside_at_gate_start"]) for r in rows)),
        "recovery_rate": (
            float(np.mean([r["recovered"] for r in rows if r["recovered"] is not None]))
            if any(r["recovered"] is not None for r in rows) else math.nan
        ),
        "ade_mean_m": float(np.mean([r["ade_m"] for r in rows])),
        "fde_mean_m": float(np.mean([r["fde_m"] for r in rows])),
        "fde_median_m": float(np.median([r["fde_m"] for r in rows])),
        "endpoint_cross_track_median_m": float(np.median([r["endpoint_cross_track_m"] for r in rows])),
        "endpoint_cross_track_abs_p95_m": float(np.percentile(np.abs([r["endpoint_cross_track_m"] for r in rows]), 95)),
        "first_step_offset_median_m": float(np.median([r["first_step_offset_m"] for r in rows])),
        "closer_to_sibling": float(np.mean([bool(r["closer_to_sibling"]) for r in rows if r["closer_to_sibling"] is not None])) if any(r["closer_to_sibling"] is not None for r in rows) else math.nan,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("arms", nargs="+", help="label=prediction_dir; the first is the reference")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    arms: dict[str, dict[str, dict]] = {}
    for spec in args.arms:
        label, _, path = spec.partition("=")
        if not path:
            parser.error(f"{spec}: expected label=prediction_dir")
        pred_dir = Path(path)
        if not pred_dir.is_absolute():
            pred_dir = REPO_ROOT / pred_dir
        arms[label] = load_arm(pred_dir)
        print(f"loaded {label}: {len(arms[label])} flights from {pred_dir}")
    reference_label = next(iter(arms))
    keys = sorted(set.intersection(*(set(rows) for rows in arms.values())))
    print(f"paired flights: {len(keys)}")
    strata = cfa.strata_masks(arms[reference_label], keys)

    output: dict = {"reference": reference_label, "paired_flights": len(keys), "strata": {}}
    for stratum, mask in strata.items():
        selected = [k for k, m in zip(keys, mask) if m]
        if not selected:
            continue
        blocks = {label: _stratum_block([rows[k] for k in selected]) for label, rows in arms.items()}
        output["strata"][stratum] = blocks
        ref = blocks[reference_label]
        print(f"\n== {stratum} (n={len(selected)}; {ref['flights_with_gated_rows']} with gated rows; "
              f"observed vertical-window violation on the covered rows "
              f"{ref['truth_vertical_violation_rate']:.1%}) ==")
        header = ["arm", "ADE", "FDE mean", "FDE p50", "xt@thr p50", "|xt| p95", "1st-step",
                  "coverage", "lat viol rows", "flights any lat", "lat excess mean/p95",
                  "vert viol rows", "flights any vert", "hinge lat/vert", "claimed rows", "claimed lat viol",
                  "outside@gate", "recovered (n)"]
        table = []
        for label, b in blocks.items():
            table.append([
                label, cfa._fmt(b["ade_mean_m"]), cfa._fmt(b["fde_mean_m"]), cfa._fmt(b["fde_median_m"]),
                cfa._fmt(b["endpoint_cross_track_median_m"]), cfa._fmt(b["endpoint_cross_track_abs_p95_m"]),
                cfa._fmt(b["first_step_offset_median_m"]),
                f"{b['coverage']:.1%}", f"{b['lateral_violation_rate']:.1%}", f"{b['flights_any_lateral']:.1%}",
                f"{cfa._fmt(b['lateral_excess_max_mean_m'])}/{cfa._fmt(b['lateral_excess_max_p95_m'])}",
                f"{b['vertical_violation_rate']:.1%}", f"{b['flights_any_vertical']:.1%}",
                f"{b['lateral_hinge_mean']:.3f}/{b['vertical_hinge_mean']:.3f}",
                str(b["claimed_rows"]), f"{b['claimed_lateral_violation_rate']:.1%}",
                f"{b['outside_at_gate_start_share']:.1%}", f"{b['recovery_rate']:.1%} ({b['recovery_candidates']})",
            ])
        cfa.print_table("", header, table)
        # Paired deltas against the reference on the same flights.
        for label, rows in arms.items():
            if label == reference_label:
                continue
            ref_rows = arms[reference_label]
            fde = np.array([rows[k]["fde_m"] - ref_rows[k]["fde_m"] for k in selected])
            ade = np.array([rows[k]["ade_m"] - ref_rows[k]["ade_m"] for k in selected])
            print(f"   {label} vs {reference_label}: ADE better on {np.mean(ade < 0):.1%} "
                  f"(median Δ {np.median(ade):+.0f} m), FDE better on {np.mean(fde < 0):.1%} "
                  f"(median Δ {np.median(fde):+.0f} m)")
            blocks[label]["paired"] = {
                "ade_better_share": float(np.mean(ade < 0)), "ade_delta_median_m": float(np.median(ade)),
                "fde_better_share": float(np.mean(fde < 0)), "fde_delta_median_m": float(np.median(fde)),
            }
    if args.json is not None:
        args.json.write_text(json.dumps(output, indent=1))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
