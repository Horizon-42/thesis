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

    python 4dTrajectory/ts_transformer/docs/compare_frame_arms.py <campaign-dir> [...]
        [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "geokit" / "src"))
from geokit import METRES_PER_DEG_LAT, metres_per_deg_lon  # noqa: E402
from flight_scenarios.runway_target import find_threshold  # noqa: E402

# approach_difficulty's own boundary for "the easy one".
STRAIGHT_TORTUOSITY = 1.05
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
    return (lon - lon0) * metres_per_deg_lon(lat0), (lat - lat0) * METRES_PER_DEG_LAT


def endpoint_geometry(pred_dir: Path, row: dict) -> dict:
    """The predicted endpoint against the ASSIGNED threshold and its parallel sibling."""
    eval_record = json.loads((pred_dir / row["eval_file"]).read_text())
    states = json.loads((pred_dir / row["states_file"]).read_text())
    target = eval_record["target_state"]
    last = states["predicted_states"][-1]
    lat0, lon0, psi = float(target["lat"]), float(target["lon"]), float(target["psi"])
    east, north = _world_en(float(last["lat"]), float(last["lon"]), lat0, lon0)
    cosine, sine = math.cos(psi), math.sin(psi)
    result = {
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


def load_arm(pred_dir: Path) -> dict[str, dict]:
    summary = json.loads((pred_dir / "summary.json").read_text())
    rows: dict[str, dict] = {}
    for row in summary.get("results", []):
        if row.get("ade_m") is None or row.get("route_tortuosity") is None:
            continue
        row = dict(row)
        row.update(endpoint_geometry(pred_dir, row))
        rows[flight_key(row)] = row
    return rows


def strata_masks(reference: dict[str, dict], keys: list[str]) -> dict[str, np.ndarray]:
    tort = np.array([reference[k]["route_tortuosity"] for k in keys])
    established = np.array([bool(reference[k]["established_at_anchor"]) for k in keys])
    remaining = np.array([reference[k]["remaining_path_m"] for k in keys])
    return {
        "all": np.ones(len(keys), dtype=bool),
        f"straight-in (tortuosity < {STRAIGHT_TORTUOSITY})": tort < STRAIGHT_TORTUOSITY,
        "vectored (tortuosity >= 1.05, not established)": (tort >= STRAIGHT_TORTUOSITY) & ~established,
        "established at anchor": established,
        "remaining path < 13 km": remaining < 13_000.0,
        "remaining path >= 13 km": remaining >= 13_000.0,
    }


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
    args = parser.parse_args(argv)

    arms: dict[str, dict[str, dict]] = {}
    for campaign in args.campaigns:
        root = Path(campaign)
        if not root.is_absolute():
            root = REPO / campaign
        for pred_dir in sorted(root.glob("*_pred_*")):
            if (pred_dir / "summary.json").is_file():
                rows = load_arm(pred_dir)
                if rows:
                    arms[pred_dir.name.split("_pred_")[0]] = rows
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
    out: dict = {"airport": airport, "flights": len(shared), "reference": reference_name,
                 "arms": {}, "strata": {}}

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
            }
            a = block["arms"][name]
            rows.append([name, _fmt(a["ade_mean"]), _fmt(a["ade_median"]), _fmt(a["fde_mean"]),
                         _fmt(a["fde_median"]), _fmt(a["endpoint_median"]), _fmt(a["time_mae_s"], 1),
                         str(a["horizon_capped"]),
                         f"{a['paired_ade_median_delta']:+.0f} ({a['ade_better_share'] * 100:.0f}%)",
                         f"{a['paired_fde_median_delta']:+.0f} ({a['fde_better_share'] * 100:.0f}%)"])
        out["strata"][stratum] = block
        print_table(
            f"{stratum} — n = {n}",
            ["arm", "ADE mean", "ADE med", "FDE mean", "FDE med", "endpoint med", "time MAE s",
             "capped", "ΔADE med (better %)", "ΔFDE med (better %)"],
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
        block = {
            "cross_track_median": float(np.median(cross)), "cross_track_mean_abs": float(np.abs(cross).mean()),
            "cross_track_p25": float(np.percentile(cross, 25)), "cross_track_p75": float(np.percentile(cross, 75)),
            "cross_track_p95_abs": float(np.percentile(np.abs(cross), 95)),
            "along_track_median": float(np.median(along)),
            "flights_with_sibling": int(has_sibling.sum()),
            "closer_to_sibling_share": float(closer.mean()) if len(closer) else None,
        }
        out["arms"][name] = block
        rows.append([name, _fmt(block["cross_track_median"]), _fmt(block["cross_track_p25"]),
                     _fmt(block["cross_track_p75"]), _fmt(block["cross_track_mean_abs"]),
                     _fmt(block["cross_track_p95_abs"]), _fmt(block["along_track_median"]),
                     f"{block['closer_to_sibling_share'] * 100:.1f}% of {block['flights_with_sibling']}"
                     if block["closer_to_sibling_share"] is not None else "n/a"])
    print_table(
        "H1 smoking gun — predicted endpoint vs the ASSIGNED runway centreline (m; + = right of inbound course)",
        ["arm", "cross med", "p25", "p75", "mean |cross|", "p95 |cross|", "along med (+ past thr)",
         "endpoint closer to sibling runway"],
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
