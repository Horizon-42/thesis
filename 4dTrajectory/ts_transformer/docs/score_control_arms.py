#!/usr/bin/env python
"""Score any set of control arms on the metrics that diagnosed the bank wiggle.

Every experiment in the queue (docs/2026-08-19_control_bank_wiggle_diagnosis.zh.md) is
judged the same way, so arms from different experiments are directly comparable:

  common-profile share   how much of the bank is the SAME on every flight (truth: 3 %)
  straight-ref bank RMS  bank on genuinely straight references   (truth: 0.55 deg)
  straight-ref reversals bank sign changes there                 (truth: 0)
  per-flight bank skill  correlation with the flown track's bank once both common
                         profiles are removed                    (baseline: -0.073)
  velocity RMS           chart-velocity error                    (baseline: 24.12 m/s)
  ADE / FDE              accuracy, so a "fix" that trades it away is visible

    python 4dTrajectory/ts_transformer/docs/score_control_arms.py <campaign-dir> [...]
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "4dTrajectory" / "ts_transformer"))
from control_inverse_dynamics import actual_controls  # noqa: E402

AERO = np.array([122.6, 2.7, 0.023, 0.0334, 0.8, 0.2])
MAX_THRUST_N = 240_000.0
N_SEGMENTS = 64
STRAIGHT_TORTUOSITY = 1.02


def _tortuosity(rows) -> float:
    metres = np.array([[r["lon"], r["lat"]] for r in rows]) * np.array(
        [111_320.0 * math.cos(math.radians(rows[0]["lat"])), 111_320.0])
    length = float(np.sum(np.linalg.norm(np.diff(metres, axis=0), axis=1)))
    straight = float(np.linalg.norm(metres[-1] - metres[0]))
    return length / straight if straight > 1.0 else np.inf


def score(pred_dir: Path) -> dict | None:
    model, observed, tortuosity = [], [], []
    for path in sorted(pred_dir.glob("*_states.json")):
        payload = json.loads(path.read_text())
        segments = payload["control_segments"]
        future = [r for r in payload["observed_states"] if r["t"] >= 0.0]
        if len(segments) != N_SEGMENTS or len(future) < 8:
            continue
        states = np.array([[r["lat"], r["lon"], r["alt"], r["V"], r["psi"], r["gamma"],
                            r["m"]] for r in future], dtype=float)
        times = np.array([r["t"] for r in future], dtype=float)
        keep = np.concatenate(([True], np.diff(times) > 1e-6))
        states, times = states[keep], times[keep]
        try:
            bank = actual_controls(states, times, aero_params=AERO,
                                   max_thrust_n=MAX_THRUST_N)[:, 1]
        except ValueError:
            continue
        grid = (np.arange(N_SEGMENTS) + 0.5) / N_SEGMENTS * (times[-1] - times[0]) + times[0]
        observed.append(np.degrees(np.interp(grid, times, bank)))
        model.append(np.degrees([s["bank_rad"] for s in segments]))
        tortuosity.append(_tortuosity(future))
    if not model:
        return None

    model_bank = np.array(model)
    observed_bank = np.array(observed)
    straight = np.array(tortuosity) < STRAIGHT_TORTUOSITY
    common = model_bank.mean(axis=0)
    residual_model = model_bank - common
    residual_observed = observed_bank - observed_bank.mean(axis=0)
    skill = [
        np.corrcoef(residual_model[i], residual_observed[i])[0, 1]
        for i in range(len(model_bank))
        if residual_model[i].std() > 1e-9 and residual_observed[i].std() > 1e-9
    ]
    reversals = np.array([
        sum(1 for a, b in zip(row, row[1:]) if a * b < 0 and abs(a - b) > 1.0)
        for row in model_bank[straight]
    ], dtype=float)
    accuracy = json.loads((pred_dir / "summary.json").read_text())["accuracy"]
    return {
        "flights": len(model_bank),
        "straight": int(straight.sum()),
        "common_share_pct": 100 * float(np.sum(common ** 2) * len(model_bank)
                                        / np.sum(model_bank ** 2)),
        "bank_rms_straight_deg": float(np.median(
            np.sqrt((model_bank[straight] ** 2).mean(axis=1)))),
        "reversals_straight": float(np.median(reversals)) if len(reversals) else float("nan"),
        "bank_skill": float(np.median(skill)) if skill else float("nan"),
        "ade_mean": accuracy["ade_m"]["mean"],
        "ade_median": accuracy["ade_m"]["median"],
        "fde_mean": accuracy["fde_m"]["mean"],
        "time_mae": accuracy["final_time_s"]["mae"],
    }


TRUTH = {"common_share_pct": 3.2, "bank_rms_straight_deg": 0.55,
         "reversals_straight": 0.0, "bank_skill": 1.0}
ROWS = [
    ("common-profile share (%)", "common_share_pct", "{:8.1f}"),
    ("straight-ref bank RMS (deg)", "bank_rms_straight_deg", "{:8.2f}"),
    ("straight-ref sign reversals", "reversals_straight", "{:8.1f}"),
    ("per-flight bank skill", "bank_skill", "{:8.3f}"),
    ("ADE mean (m)", "ade_mean", "{:8.1f}"),
    ("ADE median (m)", "ade_median", "{:8.1f}"),
    ("FDE mean (m)", "fde_mean", "{:8.1f}"),
    ("final-time MAE (s)", "time_mae", "{:8.2f}"),
]


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    scored: dict[str, dict] = {}
    for campaign in argv:
        root = Path(campaign)
        if not root.is_absolute():
            root = REPO / root
        for pred_dir in sorted(root.glob("*_pred_*")):
            if not (pred_dir / "summary.json").is_file():
                continue
            result = score(pred_dir)
            if result:
                scored[f"{root.name}/{pred_dir.name.split('_pred_')[0]}"] = result
    if not scored:
        print("no scored arms found")
        return 1

    width = max(len(k) for k in scored) + 2
    print(f"{'metric':<30}{'truth':>9}" + "".join(f"{k:>{width}}" for k in scored))
    for label, key, fmt in ROWS:
        truth = TRUTH.get(key)
        cell = f"{truth:9.2f}" if truth is not None else " " * 9
        print(f"{label:<30}{cell}"
              + "".join(f"{fmt.format(scored[k][key]):>{width}}" for k in scored))
    print(f"\n{'flights / straight refs':<30}{'':>9}"
          + "".join(f"{str(scored[k]['flights']) + '/' + str(scored[k]['straight']):>{width}}"
                    for k in scored))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
