#!/usr/bin/env python
"""Read out the airport-center frame ablation: stratified, paired, and at the runway.

Plan: ``docs/2026-09-03_airport_frame_ablation_plan.md`` (Phase 5). The three questions
were fixed before any number was looked at, and each one has its own table here:

  H1  target conditioning — FDE / endpoint error by arm, per stratum, plus the endpoint's
      signed CROSS-TRACK against the ASSIGNED runway centreline and the share of
      endpoints that sit closer to a sibling parallel runway (KSJC 12L/12R, 30L/30R;
      KRDU 05L/05R, 23L/23R). An airport frame that cannot tell the runways apart lands
      between them; that shows up here and nowhere else.
  H2  route stability — ADE on the VECTORED stratum (tortuosity >= 1.05, not established
      at the anchor), where the threshold-anchored chart puts the same airspace at
      different coordinates per runway.
  H3  symbolic conditioning — arm C against arm A on everything above.

Arms are paired flight-by-flight (same split seed), so every comparison is a paired
difference on the shared flight set, never two independent means. Seed replicates
(``*_s2024``) are read as a within-arm noise floor: a between-arm margin smaller than the
seed-to-seed margin of the same arm is noise, whatever a p-value says.

Every per-stratum table carries BOTH metric families (scene design doc §一): the
time-aligned ADE/FDE/time MAE the package scores, and the time-free geometry from
``geometric_metrics`` — chamfer, discrete Fréchet, arc-aligned ADE — with the along-path
lag that carries the rest of the ADE. ``--geometry-truth`` picks the truth those are read
against (default ``closed``: the observed rows closed to the threshold at
``true_final_time_s``; ``observed`` reproduces the Phase 0 diagnostics' convention).

    python 4dTrajectory/ts_transformer/docs/compare_frame_arms.py <campaign-dir> [...]
        [--json out.json] [--geometry-truth closed|observed]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[3]
TS_DIR = Path(__file__).resolve().parents[1]
for path in (REPO, REPO / "geokit" / "src", TS_DIR.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import ts_transformer.geometry.geometric_metrics as gm  # noqa: E402
from ts_transformer.data.approach_difficulty import strata_masks  # noqa: E402
from ts_transformer.inference.arm_readout import (  # noqa: E402
    fmt, load_arm, print_table,
)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaigns", nargs="+")
    parser.add_argument("--json", type=Path, default=None, help="write every number here")
    parser.add_argument("--reference", default=None,
                        help="arm every paired difference is taken against (default: first A_*)")
    parser.add_argument("--only", default=None,
                        help="comma-separated arm names to read (their _s2024 replicates included)")
    parser.add_argument("--geometry-truth", choices=gm.GEOMETRY_TRUTHS, default=gm.GEOMETRY_TRUTH_CLOSED,
                        help="truth the time-free metrics are read against (see geometric_metrics)")
    args = parser.parse_args(argv)

    arms: dict[str, dict[str, dict]] = {}
    for campaign in args.campaigns:
        root = Path(campaign)
        if not root.is_absolute():
            root = REPO / campaign
        for pred_dir in sorted(root.glob("*_pred_*")):
            if (pred_dir / "summary.json").is_file():
                rows = load_arm(pred_dir, geometry_truth=args.geometry_truth)
                if rows:
                    arms[pred_dir.name.split("_pred_")[0]] = rows
    if args.only:
        wanted = {name.strip() for name in args.only.split(",") if name.strip()}
        arms = {
            name: rows for name, rows in arms.items()
            if name in wanted or name.removesuffix("_s2024") in wanted
        }
    if not arms:
        print("no arms with difficulty covariates found")
        return 1
    names = sorted(arms)
    reference_name = args.reference or next(n for n in names if n.startswith("A_"))
    shared = sorted(set.intersection(*(set(r) for r in arms.values())))
    reference = arms[reference_name]
    masks = strata_masks(reference, shared)
    airport = reference[shared[0]].get("arr_airport")
    print(f"# {airport}: {len(shared)} validation flights predicted by every arm; "
          f"paired differences against {reference_name}")
    print(gm.geometry_truth_notice(
        args.geometry_truth, gm.summarize([reference[k] for k in shared]), len(shared)))
    out: dict = {"airport": airport, "flights": len(shared), "reference": reference_name,
                 "geometry_truth": args.geometry_truth, "arms": {}, "strata": {}}

    def metric(name: str, key: str, mask: np.ndarray) -> np.ndarray:
        return np.array([arms[name][k][key] for k in shared])[mask]

    # ── pooled + per-stratum accuracy, paired against the reference ────────────
    for stratum, mask in masks.items():
        n = int(mask.sum())
        if n == 0:
            continue
        rows = []
        block = {"n": n, "arms": {}}
        for name in names:
            ade, fde = metric(name, "ade_m", mask), metric(name, "fde_m", mask)
            endpoint = metric(name, "arrival_endpoint_error_m", mask)
            capped = metric(name, "horizon_capped", mask).astype(bool)
            time_err = np.abs(metric(name, "final_time_error_s", mask))
            d_ade = ade - metric(reference_name, "ade_m", mask)
            d_fde = fde - metric(reference_name, "fde_m", mask)
            d_chamfer = metric(name, "chamfer_m", mask) - metric(reference_name, "chamfer_m", mask)
            geometry = gm.summarize([arms[name][k] for k, m in zip(shared, mask) if m])
            block["arms"][name] = {
                "ade_mean": float(ade.mean()), "ade_median": float(np.median(ade)),
                "fde_mean": float(fde.mean()), "fde_median": float(np.median(fde)),
                "endpoint_mean": float(endpoint.mean()), "endpoint_median": float(np.median(endpoint)),
                "time_mae_s": float(time_err.mean()),
                "horizon_capped": int(capped.sum()),
                "paired_ade_median_delta": float(np.median(d_ade)),
                "paired_fde_median_delta": float(np.median(d_fde)),
                "ade_better_share": float(np.mean(d_ade < 0)),
                "fde_better_share": float(np.mean(d_fde < 0)),
                "paired_chamfer_median_delta": float(np.median(d_chamfer)),
                "chamfer_better_share": float(np.mean(d_chamfer < 0)),
                **geometry,
            }
            a = block["arms"][name]
            rows.append([name, fmt(a["ade_mean"]), fmt(a["ade_median"]), fmt(a["fde_mean"]),
                         fmt(a["fde_median"]), fmt(a["endpoint_median"]), fmt(a["time_mae_s"], 1),
                         str(a["horizon_capped"]),
                         *gm.geometry_table_cells(a),
                         f"{a['paired_ade_median_delta']:+.0f} ({a['ade_better_share'] * 100:.0f}%)",
                         f"{a['paired_fde_median_delta']:+.0f} ({a['fde_better_share'] * 100:.0f}%)",
                         f"{a['paired_chamfer_median_delta']:+.0f} ({a['chamfer_better_share'] * 100:.0f}%)"])
        out["strata"][stratum] = block
        print_table(
            f"{stratum} — n = {n}",
            ["arm", "ADE mean", "ADE med", "FDE mean", "FDE med", "endpoint med", "time MAE s",
             "capped", *gm.GEOMETRY_TABLE_HEADER,
             "ΔADE med (better %)", "ΔFDE med (better %)", "Δchamfer med (better %)"],
            rows,
        )

    # ── H1 at the runway: endpoint cross-track vs the assigned centreline ─────
    rows = []
    for name in names:
        cross = metric(name, "endpoint_cross_track_m", masks["all"])
        along = metric(name, "endpoint_along_track_m", masks["all"])
        sibling = np.array([arms[name][k]["closer_to_sibling"] for k in shared], dtype=object)
        has_sibling = np.array([s is not None for s in sibling])
        closer = np.array([bool(s) for s in sibling[has_sibling]]) if has_sibling.any() else np.array([])
        first_lateral = metric(name, "first_step_lateral_m", masks["all"])
        first_offset = metric(name, "first_step_offset_m", masks["all"])
        block = {
            "first_step_lateral_median": float(np.median(first_lateral)),
            "first_step_offset_median": float(np.median(first_offset)),
            "cross_track_median": float(np.median(cross)), "cross_track_mean_abs": float(np.abs(cross).mean()),
            "cross_track_p25": float(np.percentile(cross, 25)), "cross_track_p75": float(np.percentile(cross, 75)),
            "cross_track_p95_abs": float(np.percentile(np.abs(cross), 95)),
            "along_track_median": float(np.median(along)),
            "flights_with_sibling": int(has_sibling.sum()),
            "closer_to_sibling_share": float(closer.mean()) if len(closer) else None,
        }
        out["arms"][name] = block
        rows.append([name, f"{block['first_step_lateral_median']:+.0f} / {block['first_step_offset_median']:.0f}",
                     fmt(block["cross_track_median"]), fmt(block["cross_track_p25"]),
                     fmt(block["cross_track_p75"]), fmt(block["cross_track_mean_abs"]),
                     fmt(block["cross_track_p95_abs"]), fmt(block["along_track_median"]),
                     f"{block['closer_to_sibling_share'] * 100:.1f}% of {block['flights_with_sibling']}"
                     if block["closer_to_sibling_share"] is not None else "n/a"])
    print_table(
        "H1 smoking gun — predicted endpoint vs the ASSIGNED runway centreline (m; + = right of inbound course)",
        ["arm", "1st-step lateral / |offset| med", "cross med", "p25", "p75", "mean |cross|",
         "p95 |cross|", "along med (+ past thr)", "endpoint closer to sibling runway"],
        rows,
    )
    # per runway, for the parallel pairs
    runways = sorted({reference[k]["runway"] for k in shared})
    rows = []
    for runway in runways:
        mask = np.array([reference[k]["runway"] == runway for k in shared])
        if mask.sum() < 5:
            continue
        cells = [f"{runway} (n={int(mask.sum())})"]
        for name in names:
            cross = metric(name, "endpoint_cross_track_m", mask)
            fde = metric(name, "fde_m", mask)
            cells.append(f"{np.median(cross):+.0f} / {np.median(fde):.0f}")
        rows.append(cells)
        out.setdefault("by_runway", {})[runway] = {
            name: {"cross_track_median": float(np.median(metric(name, "endpoint_cross_track_m", mask))),
                   "fde_median": float(np.median(metric(name, "fde_m", mask))), "n": int(mask.sum())}
            for name in names
        }
    print_table("Per runway — endpoint cross-track median / FDE median (m)", ["runway", *names], rows)

    # ── seed noise floor: same arm, two seeds ─────────────────────────────────
    pairs = [(n, f"{n}_s2024") for n in names if f"{n}_s2024" in arms]
    if pairs:
        rows = []
        for base, replicate in pairs:
            d_ade = metric(replicate, "ade_m", masks["all"]) - metric(base, "ade_m", masks["all"])
            d_fde = metric(replicate, "fde_m", masks["all"]) - metric(base, "fde_m", masks["all"])
            out.setdefault("seed_noise", {})[base] = {
                "ade_mean_delta": float(d_ade.mean()), "ade_median_delta": float(np.median(d_ade)),
                "fde_mean_delta": float(d_fde.mean()), "fde_median_delta": float(np.median(d_fde)),
                "ade_better_share": float(np.mean(d_ade < 0)),
            }
            rows.append([base, f"{d_ade.mean():+.0f}", f"{np.median(d_ade):+.0f}",
                         f"{d_fde.mean():+.0f}", f"{np.median(d_fde):+.0f}",
                         f"{np.mean(d_ade < 0) * 100:.0f}%"])
        print_table("Seed noise floor — seed 2024 minus seed 1337, same arm (m)",
                    ["arm", "ΔADE mean", "ΔADE med", "ΔFDE mean", "ΔFDE med", "ADE better %"], rows)

    if args.json is not None:
        args.json.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
