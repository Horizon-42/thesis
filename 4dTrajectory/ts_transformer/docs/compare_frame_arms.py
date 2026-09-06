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
import math
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[3]
TS_DIR = Path(__file__).resolve().parents[1]
for path in (REPO, REPO / "geokit" / "src", TS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from flight_scenarios.runway_target import find_threshold  # noqa: E402
import geometric_metrics as gm  # noqa: E402
from approach_difficulty import STRAIGHT_TORTUOSITY, strata_masks  # noqa: E402

# approach_difficulty's own boundary for "the easy one".
# Parallel siblings whose separation a runway-blind frame would average across.
SIBLINGS = {
    "KSJC": {"12L": "12R", "12R": "12L", "30L": "30R", "30R": "30L"},
    "KRDU": {"05L": "05R", "05R": "05L", "23L": "23R", "23R": "23L"},
}


def flight_key(row: dict) -> str:
    return "_".join(str(row.get(f, "")) for f in ("id", "runway", "icao24", "landing_time_utc"))


def _threshold_frame(target: dict) -> tuple[float, float, float, float]:
    return float(target["lat"]), float(target["lon"]), float(target["psi"]), 0.0


def _world_en(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    east, north = gm.chart_en(lat, lon, lat0, lon0)
    return float(east), float(north)


def endpoint_geometry(eval_record: dict, states: dict, row: dict) -> dict:
    """The predicted endpoint against the ASSIGNED threshold and its parallel sibling."""
    target = eval_record["target_state"]
    last = states["predicted_states"][-1]
    lat0, lon0, psi = float(target["lat"]), float(target["lon"]), float(target["psi"])
    east, north = _world_en(float(last["lat"]), float(last["lon"]), lat0, lon0)
    cosine, sine = math.cos(psi), math.sin(psi)
    # The first predicted step against the kinematic extrapolation of the anchor state:
    # a model that is not anchored to the aircraft shows up HERE, as a jump at t = dt
    # (the KRDU NW translation, docs/2026-09-03_krdu_nw_endpoint_bias.md).
    anchor = eval_record["initial_state"]
    first = states["predicted_states"][1]
    a_e, a_n = _world_en(float(anchor["lat"]), float(anchor["lon"]), lat0, lon0)
    p_e, p_n = _world_en(float(first["lat"]), float(first["lon"]), lat0, lon0)
    ground_speed = float(anchor["V"]) * math.cos(float(anchor["gamma"]))
    dt = float(first["t"])
    off_e = p_e - (a_e + ground_speed * dt * math.cos(float(anchor["psi"])))
    off_n = p_n - (a_n + ground_speed * dt * math.sin(float(anchor["psi"])))
    result = {
        "first_step_along_m": off_e * cosine + off_n * sine,
        "first_step_lateral_m": off_e * sine - off_n * cosine,
        "first_step_offset_m": math.hypot(off_e, off_n),
        # Same convention as approach_difficulty: positive to the RIGHT of the inbound
        # course; along positive PAST the threshold.
        "endpoint_cross_track_m": east * sine - north * cosine,
        "endpoint_along_track_m": east * cosine + north * sine,
        "endpoint_height_m": float(last["alt"]) - float(target["alt"]),
        "closer_to_sibling": None,
    }
    airport, runway = row.get("arr_airport"), row.get("runway")
    sibling = SIBLINGS.get(airport, {}).get(runway)
    if sibling is not None:
        other = find_threshold(airport, sibling)
        if other is not None:
            e2, n2 = _world_en(
                float(last["lat"]), float(last["lon"]), float(other["lat"]), float(other["lon"])
            )
            # Distance to each centreline (cross-track magnitude), not to the point.
            own = abs(result["endpoint_cross_track_m"])
            theirs = abs(e2 * sine - n2 * cosine)
            result["closer_to_sibling"] = bool(theirs < own)
            result["sibling_cross_track_m"] = e2 * sine - n2 * cosine
    return result


def load_arm(pred_dir: Path, *, geometry_truth: str = gm.GEOMETRY_TRUTH_CLOSED) -> dict[str, dict]:
    """Every scored flight of one prediction directory, keyed by flight_key, with the
    endpoint geometry and the time-free path metrics attached."""
    summary = json.loads((pred_dir / "summary.json").read_text())
    rows: dict[str, dict] = {}
    for row in summary.get("results", []):
        if row.get("ade_m") is None or row.get("route_tortuosity") is None:
            continue
        row = dict(row)
        eval_record = json.loads((pred_dir / row["eval_file"]).read_text())
        states = json.loads((pred_dir / row["states_file"]).read_text())
        row.update(endpoint_geometry(eval_record, states, row))
        row.update(gm.record_geometry(eval_record, states, row, geometry_truth=geometry_truth))
        rows[flight_key(row)] = row
    return rows


def _fmt(value: float, digits: int = 0) -> str:
    return f"{value:.{digits}f}"


def print_table(title: str, header: list[str], rows: list[list[str]]) -> None:
    print(f"\n### {title}\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---:" if i else "---" for i in range(len(header))) + "|")
    for row in rows:
        print("| " + " | ".join(row) + " |")


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
            rows.append([name, _fmt(a["ade_mean"]), _fmt(a["ade_median"]), _fmt(a["fde_mean"]),
                         _fmt(a["fde_median"]), _fmt(a["endpoint_median"]), _fmt(a["time_mae_s"], 1),
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
                     _fmt(block["cross_track_median"]), _fmt(block["cross_track_p25"]),
                     _fmt(block["cross_track_p75"]), _fmt(block["cross_track_mean_abs"]),
                     _fmt(block["cross_track_p95_abs"]), _fmt(block["along_track_median"]),
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
