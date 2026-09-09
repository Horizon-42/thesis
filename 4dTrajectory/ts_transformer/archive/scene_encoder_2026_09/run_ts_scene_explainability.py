#!/usr/bin/env python
"""L4 acceptance (latent-intent design §六 L4, the scene design's P2.d): does the traffic
scene explain the join decision better than Phase 0's coarse context did?

Same population and protocol as Phase 0's `context` / `timing` readings
(`intent_explainability`: one row per rostered arrival, 5-fold gradient boosting, R² and
median |error| against a constant baseline), with the scene data plane's entity-level
features attached to every ego at its anchor time t₀ — the neighbours airborne in the
window before t₀ (position, ETA, lead over the ego, established, …, from their samples up
to t₀ only) and the runway-use scalars. Four feature sets, nested:

    ego                      the anchor state alone            (Phase 0: d_join R² 0.34)
    ego + coarse context     Phase 0's roster counts           (Phase 0: 0.38; the baseline)
    ego + scene scalars      the data plane's runway-use scalars
    ego + scene entities     + the N nearest-by-ETA neighbours' static rows

Targets: d_join on ALL flights and on the joins AFTER the anchor (the population Phase 0
quoted), and the remaining raw duration on the latter. Pre-registered gate: on joins after
the anchor, d_join R² ≥ 0.55 and remaining-duration median |error| < 28 s with the scene
(Phase 0: 0.38 and 35.8 s). Falling short says the context's value lies elsewhere
(runway use, departures) — measure before modelling, never the reverse.

The truth lead ETA column Phase 0 also scored reads the lead's landing time and is kept
ONLY as the upper-bound reference row it was (0.47).

    python run_ts_scene_explainability.py --airport KRDU \\
        --out 4dTrajectory/outputs/KRDU/experiments/l4_scene_explainability_20260907 [--limit N] [--neighbours 4]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
TS_DIR = REPO_ROOT / "4dTrajectory" / "ts_transformer"
for path in (TS_DIR.parent, REPO_ROOT / "geokit" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from flight_scenarios.scene_context import scene_context  # noqa: E402
from ts_transformer.intent_explainability import CONTEXT_NAMES, cv_r2, population  # noqa: E402
from ts_transformer.scene.features import SCALAR_NAMES, STATIC_NAMES, scene_arrays  # noqa: E402
from trajectory_data_process.harvest.store import HarvestPaths  # noqa: E402
from trajectory_data_process.scene_index import load_scene_index  # noqa: E402

HARVEST_ROOT = REPO_ROOT / "trajectory_data_process" / "outputs" / "harvest"
RESULT_SCHEMA = "l4-scene-explainability-v1"
GATE_D_JOIN_R2 = 0.55
GATE_DURATION_MEDIAN_S = 28.0
PHASE0 = {"d_join_r2_coarse": 0.38, "duration_median_ego_s": 35.8}


def scene_features(paths: HarvestPaths, index, row: dict, targets: dict, *, neighbours: int) -> tuple[np.ndarray, np.ndarray]:
    """(scalars, the first ``neighbours`` static rows flattened, zero where absent) for one ego."""
    scene = scene_context(
        paths, index,
        ego_flight_key=row["flight_key"], ego_runway=row["runway"], ego_target=targets[row["runway"]],
        t0_utc_s=row["t0_utc_s"], ego_lat=row["anchor_lat"], ego_lon=row["anchor_lon"],
        ego_alt_hae_m=row["anchor_alt_hae_m"], ego_ground_speed_mps=row["anchor_ground_speed_mps"],
    )
    arrays = scene_arrays(scene)
    static = arrays.neighbour_static[:neighbours]
    valid = arrays.neighbour_valid[:neighbours].astype(np.float64)[:, None]
    return arrays.scalars.astype(np.float64), np.concatenate([static * valid, valid], axis=1).ravel()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--airport", default="KRDU")
    parser.add_argument("--out", type=Path, required=True, help="must not exist (immutable artifact)")
    parser.add_argument("--limit", type=int, default=0, help="0 = the whole population (stated in the output)")
    parser.add_argument("--neighbours", type=int, default=4, help="static rows of the first N neighbours")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.neighbours < 1:
        parser.error("--neighbours must be positive")
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)

    rows = population(HARVEST_ROOT, args.airport)
    total = len(rows)
    if args.limit and args.limit < len(rows):
        rng = np.random.default_rng(args.seed)
        rows = [rows[i] for i in sorted(rng.choice(len(rows), size=args.limit, replace=False))]
        print(f"limited to {len(rows)} of {total} arrivals (seed {args.seed})")
    manifest = json.loads((HARVEST_ROOT / args.airport / "arrivals" / "manifest.json").read_text())
    targets = manifest["runway_targets"]
    paths = HarvestPaths(root=HARVEST_ROOT, code=args.airport)
    index = load_scene_index(paths)

    started = time.time()
    scalars, entities = [], []
    for k, row in enumerate(rows):
        s, e = scene_features(paths, index, row, targets, neighbours=args.neighbours)
        scalars.append(s)
        entities.append(e)
        if k % 500 == 499:
            print(f"  scene features: {k + 1}/{len(rows)} ({time.time() - started:.0f} s)", flush=True)
    scalars = np.array(scalars)
    entities = np.array(entities)
    ego = np.array([r["ego"] for r in rows])
    coarse = np.array([r["context"] for r in rows])
    lead = np.array([r["lead"] for r in rows])
    runway_column = ego.shape[1] + CONTEXT_NAMES.index("runway")
    d_join = np.array([r["d_join"] for r in rows])
    duration = np.array([r["raw_duration_s"] for r in rows])
    after = ~np.array([r["join_before_anchor"] for r in rows])

    feature_sets = [
        ("ego", ego, []),
        ("ego + coarse context (Phase 0)", np.hstack([ego, coarse]), [runway_column]),
        ("ego + scene scalars", np.hstack([ego, scalars]), []),
        (f"ego + scene scalars + {args.neighbours} neighbours", np.hstack([ego, scalars, entities]), []),
        ("ego + coarse + TRUTH lead ETA (upper bound)", np.hstack([ego, coarse, lead]), [runway_column]),
    ]
    result = {"schema": RESULT_SCHEMA, "airport": args.airport, "population": total, "measured": len(rows),
              "neighbours": args.neighbours, "scalar_names": list(SCALAR_NAMES), "static_names": list(STATIC_NAMES),
              "phase0": PHASE0, "readings": []}
    lines = [f"L4 scene explainability — {args.airport}, {len(rows)} of {total} arrivals, "
             f"{args.neighbours} neighbours per ego, 5-fold boosting", ""]
    for target_name, y, mask, unit, scale in (
        ("d_join, ALL flights", d_join, np.ones(len(rows), bool), "km", 1e3),
        ("d_join, joins AFTER the anchor", d_join, after, "km", 1e3),
        ("remaining duration, joins AFTER the anchor", duration, after, "s", 1.0),
    ):
        baseline = float(np.median(np.abs(y[mask] - np.median(y[mask]))))
        lines.append(f"{target_name} (n={int(mask.sum())}; constant baseline median |err| {baseline / scale:.2f} {unit})")
        for label, X, categorical in feature_sets:
            r2, err = cv_r2(X[mask], y[mask], categorical)
            result["readings"].append({"target": target_name, "features": label, "n": int(mask.sum()),
                                       "r2": float(r2), "median_abs_error": float(err)})
            lines.append(f"  {label:46s} R2 {r2:5.2f} | median |err| {err / scale:7.2f} {unit}")
        lines.append("")
    after_d = [r for r in result["readings"] if r["target"].startswith("d_join, joins AFTER") and "scene" in r["features"]]
    after_t = [r for r in result["readings"] if r["target"].startswith("remaining") and "scene" in r["features"]]
    best_r2 = max(r["r2"] for r in after_d)
    best_t = min(r["median_abs_error"] for r in after_t)
    result["verdict"] = {
        "gate": f"joins after the anchor: d_join R2 >= {GATE_D_JOIN_R2} and duration median |err| < {GATE_DURATION_MEDIAN_S} s",
        "best_scene_d_join_r2": best_r2, "best_scene_duration_median_s": best_t,
        "status": "pass" if best_r2 >= GATE_D_JOIN_R2 and best_t < GATE_DURATION_MEDIAN_S else "fail",
    }
    lines.append(f"gate: {result['verdict']['gate']} -> {result['verdict']['status']} "
                 f"(best scene d_join R2 {best_r2:.2f}, duration {best_t:.1f} s; "
                 f"Phase 0 coarse {PHASE0['d_join_r2_coarse']}, ego duration {PHASE0['duration_median_ego_s']} s)")
    text = "\n".join(lines) + "\n"
    (out / "explainability.txt").write_text(text)
    (out / "explainability.json").write_text(json.dumps(result, indent=2))
    print(text, end="")
    print(f"wrote {out / 'explainability.txt'} and {out / 'explainability.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
