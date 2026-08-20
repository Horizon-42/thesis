#!/usr/bin/env python
"""Score any set of control arms on the metrics that diagnosed the bank wiggle.

Every experiment in the queue (docs/2026-08-19_control_bank_wiggle_diagnosis.zh.md) is
judged the same way, so arms from different experiments are directly comparable:

  common-profile share   how much of the bank is the SAME on every flight
  straight-ref bank RMS  bank on genuinely straight references
  straight-ref reversals bank sign changes there

Each of those three is printed with the SAME metric applied to the flown tracks of that
arm's own airport, because the reference is airport-specific and was previously hardcoded
to KSJC's values (3.2 %, 0.55 deg). KRDU's flown tracks are 1.8 % and 0.41 deg, so scoring
KRDU arms against the KSJC constants overstated how far they still had to go on the share
and understated how far past the data the strongest doses had gone.
  per-flight bank skill  correlation with the flown track's bank once both common
                         profiles are removed. Read it against the two references
                         printed with it, NOT against 1.0: a randomly chosen other real
                         flight already scores the floor, and the same-runway twin shows
                         roughly how much of the future bank is knowable (a yardstick,
                         not a bound -- the 47x arm exceeds it).
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
# Arms may use different segment counts (the n_segments experiment does), so both the
# model and the observed bank are resampled onto ONE normalised-time grid of this width.
# Without it a common-profile share computed at N=16 would not be comparable with one at
# N=64 — and the scorer would silently skip every arm whose N differed from the constant.
COMMON_GRID_POINTS = 64
STRAIGHT_TORTUOSITY = 1.02


def _tortuosity(rows) -> float:
    metres = np.array([[r["lon"], r["lat"]] for r in rows]) * np.array(
        [111_320.0 * math.cos(math.radians(rows[0]["lat"])), 111_320.0])
    length = float(np.sum(np.linalg.norm(np.diff(metres, axis=0), axis=1)))
    straight = float(np.linalg.norm(metres[-1] - metres[0]))
    return length / straight if straight > 1.0 else np.inf


# w = 1.36 is 1x the position term at the converged KRDU baseline; see
# docs/experiments/imitation_arms.json for how that was measured.
IMITATION_WEIGHT_PER_POSITION = 1.36


def _imitation_dose(pred_dir: Path) -> float:
    """The arm's imitation weight as a multiple of the position term, 0.0 if absent."""
    config = pred_dir.parent / pred_dir.name.split("_pred_")[0] / "config.json"
    if not config.is_file():
        return 0.0
    weight = json.loads(config.read_text()).get("control_imitation_loss_weight") or 0.0
    return float(weight) / IMITATION_WEIGHT_PER_POSITION


def _per_flight_skill(pred: np.ndarray, truth: np.ndarray) -> float:
    gp, gt = pred.mean(axis=0), truth.mean(axis=0)
    values = [
        np.corrcoef(pred[i] - gp, truth[i] - gt)[0, 1]
        for i in range(len(truth))
        if (pred[i] - gp).std() > 1e-9 and (truth[i] - gt).std() > 1e-9
    ]
    return float(np.median(values)) if values else float("nan")


def _reference_skills(observed: np.ndarray, runways: np.ndarray,
                      entry: np.ndarray) -> tuple[float, float]:
    """The floor and the ceiling this metric actually has, on THESE flights.

    floor  a randomly chosen other real flight, scored as if it were the prediction.
           Real bank profiles share one dominant mode, so this is well above zero and a
           model below it is not merely weak -- it is not predicting the right object.
    twin   the same-runway flight whose entry state is nearest. It had comparable
           information and then saw a real future, so it is a useful yardstick for how
           much of the future bank is knowable at all.

           It is NOT an upper bound, and has been exceeded: the 47x imitation arm
           reaches +0.735 against a twin of +0.699, because the model reads the whole
           120 s lookback trajectory while the twin match uses only entry position,
           heading and duration. Read it as "roughly where a good predictor should be",
           not as a ceiling.
    """
    if len(observed) < 4:
        return float("nan"), float("nan")
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(20):
        order = rng.permutation(len(observed))
        same = order == np.arange(len(observed))
        order[same] = (order[same] + 1) % len(observed)
        draws.append(_per_flight_skill(observed[order], observed))
    features = (entry - entry.mean(axis=0)) / (entry.std(axis=0) + 1e-9)
    twin = np.zeros_like(observed)
    usable = np.zeros(len(observed), dtype=bool)
    for runway in set(runways.tolist()):
        rows = np.where(runways == runway)[0]
        if len(rows) < 2:
            continue
        for i in rows:
            others = rows[rows != i]
            twin[i] = observed[others[np.argmin(((features[others] - features[i]) ** 2)
                                                .sum(axis=1))]]
            usable[i] = True
    twin_skill = (_per_flight_skill(twin[usable], observed[usable])
                  if usable.sum() >= 4 else float("nan"))
    return float(np.mean(draws)), twin_skill


def score(pred_dir: Path) -> dict | None:
    model, observed, tortuosity, segment_counts = [], [], [], set()
    runways, entry = [], []
    for path in sorted(pred_dir.glob("*_states.json")):
        payload = json.loads(path.read_text())
        segments = payload["control_segments"]
        future = [r for r in payload["observed_states"] if r["t"] >= 0.0]
        if len(segments) < 2 or len(future) < 8:
            continue
        segment_counts.add(len(segments))
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
        progress = (np.arange(COMMON_GRID_POINTS) + 0.5) / COMMON_GRID_POINTS
        grid = progress * (times[-1] - times[0]) + times[0]
        observed.append(np.degrees(np.interp(grid, times, bank)))
        # Piecewise-constant controls: sample the schedule at the same normalised
        # progress, so an N=16 arm and an N=64 arm are compared on one axis.
        segment_bank = np.degrees([s["bank_rad"] for s in segments])
        model.append(segment_bank[np.minimum(
            (progress * len(segments)).astype(int), len(segments) - 1)])
        tortuosity.append(_tortuosity(future))
        runways.append(path.name.split("_")[1])
        entry.append([states[0, 0], states[0, 1], float(times[-1] - times[0]) / 300.0])
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
    del residual_observed
    # Counted on the resampled grid, which is safe: upsampling a piecewise-constant
    # schedule only repeats values, and a repeated value cannot change sign, so the count
    # equals the one on the arm's own N boundaries.
    reversals = np.array([
        sum(1 for a, b in zip(row, row[1:]) if a * b < 0 and abs(a - b) > 1.0)
        for row in model_bank[straight]
    ], dtype=float) if straight.any() else np.array([], dtype=float)
    accuracy = json.loads((pred_dir / "summary.json").read_text())["accuracy"]
    dose = _imitation_dose(pred_dir)
    truth_common = 100 * float(
        np.sum(observed_bank.mean(axis=0) ** 2) * len(observed_bank)
        / np.sum(observed_bank ** 2)
    )
    truth_rms = float(np.median(np.sqrt((observed_bank[straight] ** 2).mean(axis=1))))
    truth_reversals = float(np.median([
        sum(1 for a, b in zip(row, row[1:]) if a * b < 0 and abs(a - b) > 1.0)
        for row in observed_bank[straight]
    ])) if straight.any() else float("nan")
    floor, twin = _reference_skills(observed_bank, np.array(runways), np.array(entry))
    return {
        "flights": len(model_bank),
        "dose": dose,
        "truth_common_share_pct": truth_common,
        "truth_bank_rms_straight_deg": truth_rms,
        "truth_reversals_straight": truth_reversals,
        "skill_floor": floor,
        "skill_twin": twin,
        "straight": int(straight.sum()),
        "n_segments": sorted(segment_counts),
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


# Every reference here is MEASURED on the arm's own flights, never hardcoded: the truth
# values are airport-specific, and bank_skill has no 1.0 to aim at because the future is
# only partly determined by the entry state.
ROWS = [
    ("common-profile share (%)", "common_share_pct", "{:8.1f}"),
    ("  \u2514 flown tracks", "truth_common_share_pct", "{:8.1f}"),
    ("straight-ref bank RMS (deg)", "bank_rms_straight_deg", "{:8.2f}"),
    ("  \u2514 flown tracks", "truth_bank_rms_straight_deg", "{:8.2f}"),
    ("straight-ref sign reversals", "reversals_straight", "{:8.1f}"),
    ("  \u2514 flown tracks", "truth_reversals_straight", "{:8.1f}"),
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

    # Alphabetical glob order once put "imitation_16x" between "baseline" and
    # "imitation_1x", which is an easy column to misread. Order by dose instead.
    scored = dict(sorted(scored.items(), key=lambda kv: kv[1]["dose"]))
    width = max(len(k) for k in scored) + 2
    print(f"{'metric':<30}" + "".join(f"{k:>{width}}" for k in scored))
    for label, key, fmt in ROWS:
        print(f"{label:<30}"
              + "".join(f"{fmt.format(scored[k][key]):>{width}}" for k in scored))
    print(f"\n{'flights / straight refs':<30}"
          + "".join(f"{str(scored[k]['flights']) + '/' + str(scored[k]['straight']):>{width}}"
                    for k in scored))
    print(f"{'segments per arm':<30}"
          + "".join(f"{','.join(map(str, scored[k]['n_segments'])):>{width}}" for k in scored))
    print(f"{'imitation dose (x position)':<30}"
          + "".join(f"{scored[k]['dose']:8.2f}".rjust(width) for k in scored))
    print(f"\n{'-- bank skill references --':<30}")
    for label, key in (("random other flight (floor)", "skill_floor"),
                       ("same-runway twin (ceiling)", "skill_twin")):
        print(f"{label:<30}"
              + "".join(f"{scored[k][key]:8.3f}".rjust(width) for k in scored))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
