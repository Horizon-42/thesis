#!/usr/bin/env python
"""Paired comparison of two control arms over the flights they both predicted.

Arms are trained on the same split with the same seed, so per-flight differences are
paired and a sign test is the honest summary: an aggregate that moves because a handful
of flights moved a long way is a different claim from one that moves on most flights.

    python 4dTrajectory/ts_transformer/docs/compare_control_arms_paired.py <a_pred_dir> <b_pred_dir>
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from control_inverse_dynamics import actual_controls  # noqa: E402

AERO = np.array([122.6, 2.7, 0.023, 0.0334, 0.8, 0.2])
MAX_THRUST_N = 240_000.0
GRID = 64


def _sign_test(delta: np.ndarray) -> tuple[float, float]:
    """Fraction of flights improved, and the two-sided sign-test p value."""
    better = int((delta < 0).sum())
    trials = int((delta != 0).sum())
    if trials == 0:
        return float("nan"), 1.0
    # Exact two-sided binomial tail against p = 0.5, in log space for large n.
    def log_c(n, k):
        return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    extreme = min(better, trials - better)
    tail = sum(math.exp(log_c(trials, k) - trials * math.log(2))
               for k in range(0, extreme + 1))
    return better / trials, min(1.0, 2 * tail)


def per_flight(pred_dir: Path) -> dict[str, dict]:
    out = {}
    for path in sorted(pred_dir.glob("*_states.json")):
        payload = json.loads(path.read_text())
        predicted = payload["predicted_states"]
        future = [r for r in payload["observed_states"] if r["t"] >= 0.0]
        segments = payload["control_segments"]
        if len(future) < 8 or len(predicted) < 8 or len(segments) < 4:
            continue
        to = np.array([r["t"] for r in future], float)
        keep = np.concatenate(([True], np.diff(to) > 1e-6))
        to = to[keep]
        states = np.array([[r["lat"], r["lon"], r["alt"], r["V"], r["psi"], r["gamma"],
                            r["m"]] for r, k in zip(future, keep) if k], float)
        tp = np.array([r["t"] for r in predicted], float)
        grid_t = to[(to >= tp[0]) & (to <= tp[-1])]
        if len(grid_t) < 4:
            continue
        scale = np.array([111_320.0 * math.cos(math.radians(states[0, 0])), 111_320.0])
        obs_xy = np.column_stack([np.interp(grid_t, to, states[:, 1]),
                                  np.interp(grid_t, to, states[:, 0])]) * scale
        pxy = np.column_stack([
            np.interp(grid_t, tp, [r["lon"] for r in predicted]),
            np.interp(grid_t, tp, [r["lat"] for r in predicted])]) * scale
        try:
            bank = actual_controls(states, to, aero_params=AERO,
                                   max_thrust_n=MAX_THRUST_N,
                                   include_transport=True)[:, 1]
        except ValueError:
            continue
        progress = (np.arange(GRID) + 0.5) / GRID
        observed_bank = np.degrees(np.interp(progress * (to[-1] - to[0]) + to[0], to, bank))
        segment_bank = np.degrees([s["bank_rad"] for s in segments])
        model_bank = segment_bank[np.minimum((progress * len(segments)).astype(int),
                                             len(segments) - 1)]
        out[path.name] = {
            "ade": float(np.linalg.norm(pxy - obs_xy, axis=1).mean()),
            "fde": float(np.linalg.norm(pxy[-1] - obs_xy[-1])),
            "model_bank": model_bank,
            "observed_bank": observed_bank,
        }
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    a, b = (Path(p if Path(p).is_absolute() else REPO / p) for p in argv)
    left, right = per_flight(a), per_flight(b)
    shared = sorted(set(left) & set(right))
    if not shared:
        print("the two arms share no predicted flights")
        return 1
    print(f"{a.name}  ->  {b.name}")
    print(f"{len(shared)} flights predicted by both\n")
    print(f"{'metric':<26}{'A':>11}{'B':>11}{'B better on':>14}{'p':>12}")
    for label, key in (("ADE (m)", "ade"), ("FDE (m)", "fde")):
        va = np.array([left[k][key] for k in shared])
        vb = np.array([right[k][key] for k in shared])
        share, p = _sign_test(vb - va)
        print(f"{label:<26}{np.median(va):11.1f}{np.median(vb):11.1f}"
              f"{100 * share:13.1f}%{p:12.2e}")

    def skills(source):
        model = np.array([source[k]["model_bank"] for k in shared])
        observed = np.array([source[k]["observed_bank"] for k in shared])
        gm, go = model.mean(axis=0), observed.mean(axis=0)
        return np.array([
            np.corrcoef(model[i] - gm, observed[i] - go)[0, 1]
            if (model[i] - gm).std() > 1e-9 and (observed[i] - go).std() > 1e-9 else np.nan
            for i in range(len(shared))
        ])

    sa, sb = skills(left), skills(right)
    valid = ~(np.isnan(sa) | np.isnan(sb))
    share, p = _sign_test(sa[valid] - sb[valid])   # higher skill is better
    print(f"{'bank skill':<26}{np.median(sa[valid]):11.3f}{np.median(sb[valid]):11.3f}"
          f"{100 * share:13.1f}%{p:12.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
