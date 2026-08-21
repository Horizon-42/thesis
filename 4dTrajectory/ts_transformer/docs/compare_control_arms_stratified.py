#!/usr/bin/env python
"""Compare control arms inside a matched route-mix stratum, not across it.

`approach_difficulty` measured that an airport's ADE mostly reports which approaches it
poses: 78 % of KSJC's validation flights are already established straight-in at the anchor
against 41-61 % elsewhere, and reweighting KSJC to the pooled mix moves it from best of five
to worst. Within a matched stratum the five airports land at 412-509 m.

The same confound operates WITHIN one airport whenever two arms are compared on an aggregate:
they share a flight set, so the mix is identical and the aggregate is fair — but it hides
which kind of approach an arm won or lost on. A dose that helps only the straight-in majority
and hurts every vectored arrival reads as a small net gain.

    python .../compare_control_arms_stratified.py <campaign-dir> [...]
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "4dTrajectory" / "ts_transformer"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_control_arms import _imitation_dose  # noqa: E402

# The stratum boundary approach_difficulty itself uses for "the easy one".
STRAIGHT_TORTUOSITY = 1.05


def arm_rows(pred_dir: Path) -> dict[str, dict]:
    """Per-flight rows, keyed on flight identity.

    `id` is the callsign and is NOT unique (see the flight-identity invariant), so the key
    is the same four-part identity the rest of the pipeline uses.
    """
    summary = json.loads((pred_dir / "summary.json").read_text())
    rows = {}
    for row in summary.get("results", []):
        if row.get("ade_m") is None or row.get("route_tortuosity") is None:
            continue
        key = "_".join(str(row.get(field, "")) for field in
                       ("id", "runway", "icao24", "landing_time_utc"))
        rows[key] = row
    return rows


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    arms: dict[str, dict] = {}
    for campaign in argv:
        root = Path(campaign)
        if not root.is_absolute():
            root = REPO / campaign
        for pred_dir in sorted(root.glob("*_pred_*")):
            if (pred_dir / "summary.json").is_file():
                rows = arm_rows(pred_dir)
                if rows:
                    arms[pred_dir.name.split("_pred_")[0]] = rows
    if not arms:
        print("no arms with difficulty covariates found — were they predicted before "
              "approach_difficulty existed?")
        return 1

    shared = set.intersection(*(set(r) for r in arms.values()))
    names = sorted(arms, key=lambda n: _dose_of(argv, n))
    print(f"{len(shared)} flights predicted by every arm\n")

    reference = arms[names[0]]
    tort = np.array([reference[k]["route_tortuosity"] for k in sorted(shared)])
    established = np.array(
        [bool(reference[k].get("established_at_anchor")) for k in sorted(shared)]
    )
    strata = {
        "straight-in (tortuosity < 1.05)": tort < STRAIGHT_TORTUOSITY,
        "vectored (>= 1.05)": tort >= STRAIGHT_TORTUOSITY,
        "established at anchor": established,
        "not established": ~established,
    }
    keys = sorted(shared)
    width = max(len(n) for n in names) + 2
    for label, mask in strata.items():
        if mask.sum() < 20:
            print(f"{label:<34} (n={int(mask.sum())}, too few to read)")
            continue
        print(f"{label:<34} n={int(mask.sum())}")
        for metric in ("ade_m", "fde_m"):
            cells = []
            for name in names:
                values = np.array([arms[name][k][metric] for k in keys], dtype=float)
                cells.append(f"{np.median(values[mask]):8.1f}")
            print(f"  {metric:<32}" + "".join(c.rjust(width) for c in cells))
        print()
    print(" " * 34 + "".join(n.rjust(width) for n in names))
    return 0


def _dose_of(argv: list[str], arm: str) -> float:
    for campaign in argv:
        root = Path(campaign)
        if not root.is_absolute():
            root = REPO / campaign
        pred = root / f"{arm}_pred_val"
        if pred.is_dir():
            return _imitation_dose(pred)
    return 0.0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
