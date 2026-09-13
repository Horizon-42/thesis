"""Runway-intent R1: a learned runway head against the causal rules — and what the split does.

Plan `docs/2026-09-13_runway_intent_plan.zh.md` §12 (design) and §11.4 (the pre-registered gates).
Per airport, every arrival whose per-flight split hash is NOT outer-test gives one sample per R0
anchor (the arrival-slice entry and a remaining-path grid) with the causal features of
`data.runway_features`, labelled with its landing runway. One HistGradientBoostingClassifier PER
AIRPORT — not pooled across airports (experiment principle (3), stated: R1 first measures what each
airport's own data teaches). Three trainings:

- ``day_a`` / ``day_b`` — the DAY-BLOCKED split (the user's decision D1 = (a), 2026-09-13): days are
  hashed into folds; the test fold's days are never used (kept for the final test on the sealed
  flights), and the other days are split into train / validation twice (``a`` with the split seed,
  ``b`` with `SECOND_PARTITION_SEED`) — a tree model has no seed variance, the day partition does;
- ``flight`` — today's per-flight hash split over the same non-test days: the leakage control.

Each model is read on its own validation samples against the causal rules recomputed on the same
samples; ``day_a`` and ``flight`` are also read PAIRED on the samples that are validation under
both splits — the flight model has seen those days' other flights, the day model has not, and the
difference is the leakage.

A secondary reading, ``day_a_nowx`` / ``day_b_nowx``: the day-blocked models without the
`DAY_LEVEL_GROUPS` features. Added after the KMSY trial, where ``day_a`` fell below B1 at the entry
anchor (85.5 against 96.5 %) while ``day_b`` beat it: an hourly METAR reading is nearly unique to
its hour, so it can act as a timestamp the model memorises a training day's configuration by —
the same shortcut a per-flight split rewards. The pre-registered gates read the full model.

    python run_ts.py runway_intent_r1 --airport KRDU \\
        --output-dir 4dTrajectory/outputs/POOLED/experiments/runway_intent_r1_20260913/KRDU
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np

from ts_transformer.experiments.support import REPO_ROOT

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from ts_transformer.config import TSConfig  # noqa: E402
from ts_transformer.data.runway_context import (  # noqa: E402
    RULES,
    airport_reference,
    build_airport_context,
    parse_utc,
)
from ts_transformer.data.runway_features import (  # noqa: E402
    ENTRY,
    anchor_features,
    anchors,
    feature_space,
    track_course_at,
)
from ts_transformer.data.splits import split_name_for_dataset_id  # noqa: E402

HARVEST_ROOT = REPO_ROOT / "trajectory_data_process" / "outputs" / "harvest"
METAR_ROOT = REPO_ROOT / "data" / "metar"
SCHEMA = "ts-runway-intent-r1-v1"
SECOND_PARTITION_SEED = 2024
MODELS = ("day_a", "day_b", "flight", "day_a_nowx", "day_b_nowx")
#: The feature groups the ``*_nowx`` models drop: values shared by a whole hour or day.
DAY_LEVEL_GROUPS = ("wind", "time of day")
FOLD_OF = {"day_a": "day_a", "day_b": "day_b", "flight": "flight",
           "day_a_nowx": "day_a", "day_b_nowx": "day_b"}
#: Fixed capacity, no early stopping: sklearn's early stopping holds out RANDOM samples, which
#: would put one day's flights on both sides of its own stopping decision.
HGB_SETTINGS = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
                    early_stopping=False, random_state=0)
PERMUTATION_REPEATS = 3
EPS = 1e-6


def _fraction(text: str) -> float:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") / 2**64


def day_folds(day: str, config: TSConfig) -> dict[str, str]:
    """``{"a": fold, "b": fold}`` for one UTC landing date. The test fold is the SAME in both
    (the split seed; R0's `day_fold`); ``b`` re-splits the non-test days with its own seed so that
    its validation fold is again ~`val_fraction` of all days."""
    base = _fraction(f"{config.resolved_split_seed}:day:{day}")
    if base < config.test_fraction:
        return {"a": "test", "b": "test"}
    a = "val" if base < config.test_fraction + config.val_fraction else "train"
    other = _fraction(f"{SECOND_PARTITION_SEED}:day:{day}")
    b = "val" if other < config.val_fraction / (1.0 - config.test_fraction) else "train"
    return {"a": a, "b": b}


def minority_runways(groups: dict[str, int], majority: Counter) -> list[str]:
    """In every direction group with more than one runway, the runways other than its busiest."""
    out: list[str] = []
    for group in sorted(set(groups.values())):
        members = sorted(r for r in groups if groups[r] == group)
        if len(members) > 1:
            busiest = min(members, key=lambda r: (-majority.get(r, 0), r))
            out += [r for r in members if r != busiest]
    return out


def expected_calibration_error(prob: np.ndarray, truth: np.ndarray, bins: int = 10) -> float:
    confidence = prob.max(axis=1)
    correct = prob.argmax(axis=1) == truth
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(total)


def level_accuracy(picks: np.ndarray, truth: np.ndarray, group_of: np.ndarray, multi: np.ndarray) -> dict[str, Any]:
    """Exact, direction and side-given-direction accuracy of runway-index picks."""
    direction = group_of[picks] == group_of[truth]
    sided = direction & multi[truth]
    return {
        "exact": float((picks == truth).mean()),
        "direction": float(direction.mean()),
        "side_given_direction": float((picks[sided] == truth[sided]).mean()) if sided.any() else None,
        "side_samples": int(sided.sum()),
    }


def score(
    prob: np.ndarray, truth: np.ndarray, rule_picks: dict[str, np.ndarray], b1_prob: np.ndarray,
    anchor_names: np.ndarray, *, group_of: np.ndarray, multi: np.ndarray, minority: np.ndarray,
) -> dict[str, Any]:
    """The model and the rules on the same samples, pooled and per anchor."""
    def block(mask: np.ndarray) -> dict[str, Any]:
        p, y = prob[mask], truth[mask]
        model = level_accuracy(p.argmax(axis=1), y, group_of, multi)
        minor = np.isin(y, minority)
        model.update({
            "nll": float(-np.log(np.clip(p[np.arange(len(y)), y], EPS, 1.0)).mean()),
            "ece": expected_calibration_error(p, y),
            "minority_accuracy": float((p.argmax(axis=1)[minor] == y[minor]).mean()) if minor.any() else None,
            "minority_samples": int(minor.sum()),
        })
        rules = {}
        for rule, picks in rule_picks.items():
            cell = level_accuracy(picks[mask], y, group_of, multi)
            cell["minority_accuracy"] = (
                float((picks[mask][minor] == y[minor]).mean()) if minor.any() else None
            )
            rules[rule] = cell
        b1 = b1_prob[mask]
        return {
            "samples": int(mask.sum()), "model": model, "rules": rules,
            "b1_prob_nll": float(-np.log(np.clip(b1[np.arange(len(y)), y], EPS, 1.0)).mean()),
        }

    out = {"all": block(np.ones(len(truth), dtype=bool))}
    for name in sorted(set(anchor_names), key=lambda a: (a != ENTRY, -float(a[:-2]) if a != ENTRY else 0.0)):
        out[name] = block(anchor_names == name)
    return out


def grouped_permutation(
    model: HistGradientBoostingClassifier, X: np.ndarray, truth: np.ndarray, names: tuple[str, ...],
    groups: dict[str, str], classes: np.ndarray, *, group_of: np.ndarray, multi: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Accuracy lost (exact and side-given-direction) when one feature GROUP's columns are
    permuted together across samples — which kinds of signal the head decides on."""
    def picks(matrix: np.ndarray) -> np.ndarray:
        return classes[model.predict_proba(matrix).argmax(axis=1)]

    base = level_accuracy(picks(X), truth, group_of, multi)
    rng = np.random.default_rng(0)
    out: dict[str, dict[str, float]] = {}
    for group in sorted(set(groups.values())):
        columns = [i for i, name in enumerate(names) if groups[name] == group]
        drops_exact, drops_side = [], []
        for _ in range(PERMUTATION_REPEATS):
            shuffled = X.copy()
            order = rng.permutation(len(X))
            shuffled[:, columns] = X[order][:, columns]
            cell = level_accuracy(picks(shuffled), truth, group_of, multi)
            drops_exact.append(base["exact"] - cell["exact"])
            if base["side_given_direction"] is not None and cell["side_given_direction"] is not None:
                drops_side.append(base["side_given_direction"] - cell["side_given_direction"])
        out[group] = {"exact_drop": float(np.mean(drops_exact)),
                      "side_drop": float(np.mean(drops_side)) if drops_side else None}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--airport", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bins-km", default="30,20,15,10,6,3")
    parser.add_argument("--window-min", type=float, default=30.0)
    parser.add_argument("--sector-window-min", type=float, default=60.0)
    parser.add_argument("--metar-delay-min", type=float, default=10.0)
    parser.add_argument("--calm-kt", type=float, default=3.0)
    args = parser.parse_args(argv)

    airport = args.airport.upper()
    bins_km = [float(value) for value in args.bins_km.split(",")]
    # The split contract (seed, fractions) is the package's default — the locked outer split.
    config = TSConfig()
    manifest_path = HARVEST_ROOT / airport / "arrivals" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    targets = manifest["runway_targets"]
    candidates = sorted(targets)
    courses = {r: float(targets[r]["course_deg"]) for r in candidates}

    def split_of(key: str) -> str:
        return split_name_for_dataset_id(f"{airport}:{key}", config)

    pool = build_airport_context(
        manifest_path, HARVEST_ROOT / airport / "tracks" / "manifest.json",
        sorted((METAR_ROOT / airport).glob("asos_*.csv")), courses, split_of,
        window=timedelta(minutes=args.window_min),
        sector_window=timedelta(minutes=args.sector_window_min),
        metar_delay=timedelta(minutes=args.metar_delay_min),
        calm_kt=args.calm_kt,
    )
    context = pool.rules
    usable = {
        key: flight for key, flight in pool.flights.items()
        if day_folds(flight["landing_time_utc"][:10], config)["a"] != "test"
    }
    # Which operator prefixes get a column: read from the callsigns of every usable flight — a
    # vocabulary, no label — so all three trainings share one feature matrix.
    space = feature_space(candidates, targets, airport_reference(targets), usable.values())
    index_of = {r: i for i, r in enumerate(candidates)}
    group_of = np.array([context.groups[r] for r in candidates])
    group_sizes = Counter(context.groups.values())
    multi = np.array([group_sizes[context.groups[r]] > 1 for r in candidates])
    minority_names = minority_runways(context.groups, context.majority_counts)
    minority = np.array([index_of[r] for r in minority_names], dtype=int)

    rows, truth, anchor_names, flight_keys = [], [], [], []
    folds: dict[str, list[str]] = {"day_a": [], "day_b": [], "flight": []}
    rule_picks: dict[str, list[int]] = {rule: [] for rule in RULES}
    b1_prob: list[np.ndarray] = []
    for key, flight in sorted(usable.items()):
        day = day_folds(flight["landing_time_utc"][:10], config)
        per_flight = split_of(key)
        entry = parse_utc(flight["entry_time_utc"])
        waypoints = flight["waypoints"]
        sector = pool.sectors[key]
        for name, index in anchors(waypoints, bins_km).items():
            t = entry + timedelta(seconds=float(waypoints[index][0]))
            rows.append(anchor_features(space, context, flight, index, sector=sector, anchor_time=t))
            truth.append(index_of[flight["runway"]])
            anchor_names.append(name)
            flight_keys.append(key)
            folds["day_a"].append(day["a"])
            folds["day_b"].append(day["b"])
            folds["flight"].append(per_flight)
            picks = context.picks(t, sector=sector, track_course_deg=track_course_at(waypoints, index))
            for rule in RULES:
                rule_picks[rule].append(index_of[picks[rule].runway])
            counts = Counter(landing.runway for landing in context.recent(t, context.window))
            smoothed = np.array([counts[r] + 1.0 for r in candidates])
            b1_prob.append(smoothed / smoothed.sum())
    X = np.vstack(rows)
    y = np.asarray(truth)
    anchor_arr = np.asarray(anchor_names)
    fold_arr = {name: np.asarray(values) for name, values in folds.items()}
    picks_arr = {rule: np.asarray(values) for rule, values in rule_picks.items()}
    b1_arr = np.vstack(b1_prob)
    keys_arr = np.asarray(flight_keys)

    results: dict[str, Any] = {}
    models: dict[str, HistGradientBoostingClassifier] = {}
    probs: dict[str, np.ndarray] = {}
    kept = np.array([space.groups[n] not in DAY_LEVEL_GROUPS for n in space.names])
    for name in MODELS:
        fold = fold_arr[FOLD_OF[name]]
        train, val = fold == "train", fold == "val"
        columns = kept if name.endswith("_nowx") else np.ones(len(space.names), dtype=bool)
        model = HistGradientBoostingClassifier(**HGB_SETTINGS).fit(X[train][:, columns], y[train])
        prob = np.zeros((len(y), len(candidates)))
        prob[:, model.classes_] = model.predict_proba(X[:, columns])
        models[name], probs[name] = model, prob
        results[name] = {
            "train_samples": int(train.sum()), "val_samples": int(val.sum()),
            "train_flights": int(len(set(keys_arr[train]))), "val_flights": int(len(set(keys_arr[val]))),
            "classes_seen": [candidates[i] for i in model.classes_],
            "features_used": int(columns.sum()),
            "validation": score(prob[val], y[val], {r: p[val] for r, p in picks_arr.items()}, b1_arr[val],
                                anchor_arr[val], group_of=group_of, multi=multi, minority=minority),
        }
    both = (fold_arr["day_a"] == "val") & (fold_arr["flight"] == "val")
    leakage = {
        name: score(probs[name][both], y[both], {r: p[both] for r, p in picks_arr.items()}, b1_arr[both],
                    anchor_arr[both], group_of=group_of, multi=multi, minority=minority)
        for name in ("day_a", "flight")
    }
    val_a = fold_arr["day_a"] == "val"
    importance = grouped_permutation(
        models["day_a"], X[val_a], y[val_a], space.names, dict(space.groups), models["day_a"].classes_,
        group_of=group_of, multi=multi,
    )

    days = sorted({flight["landing_time_utc"][:10] for flight in pool.flights.values()})
    fold_days = {part: Counter(day_folds(d, config)[part] for d in days) for part in ("a", "b")}
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": SCHEMA,
        "airport": airport,
        "candidates": candidates,
        "direction_groups": context.groups,
        "minority_runways": minority_names,
        "features": list(space.names),
        "feature_groups": dict(space.groups),
        "hgb": HGB_SETTINGS,
        "day_level_groups_dropped_by_nowx": list(DAY_LEVEL_GROUPS),
        "split": {
            "sealed": "per-flight outer-test hash flights: never loaded, never context",
            "day_folds": {part: dict(counts) for part, counts in fold_days.items()},
            "second_partition_seed": SECOND_PARTITION_SEED,
            "paired_leakage_samples": int(both.sum()),
            "paired_leakage_flights": int(len(set(keys_arr[both]))),
        },
        "context_pool": {"landings": len(pool.landings), "excluded_outer_test": pool.excluded_outer_test},
        "models": results,
        "leakage": leakage,
        "importance_day_a": importance,
    }
    (out / "runway_intent_r1.json").write_text(json.dumps(document, indent=1), encoding="utf-8")

    lines = [f"{airport}: {len(usable)} usable flights, {len(y)} samples; candidates {candidates}; "
             f"minority {minority_names}; days a {dict(fold_days['a'])} b {dict(fold_days['b'])}"]
    for name in MODELS:
        for anchor in ("all", ENTRY):
            v = results[name]["validation"][anchor]
            m = v["model"]
            sides = [c["side_given_direction"] for c in v["rules"].values() if c["side_given_direction"] is not None]
            best_exact = max(c["exact"] for c in v["rules"].values())
            lines.append(
                f"  {name:<10} {anchor:<5} val {results[name]['val_flights']} flights: "
                f"exact {m['exact']:.1%} (best rule {best_exact:.1%})  direction {m['direction']:.1%} "
                f"(B1 {v['rules']['B1_active_config']['direction']:.1%})  side|dir "
                f"{_pct(m['side_given_direction'])} (best rule {_pct(max(sides) if sides else None)})  "
                f"minority {_pct(m['minority_accuracy'])}  NLL {m['nll']:.3f} "
                f"(B1-prob {v['b1_prob_nll']:.3f})  ECE {m['ece']:.3f}"
            )
    for name in ("day_a", "flight"):
        m = leakage[name]["all"]["model"]
        lines.append(f"  paired ({document['split']['paired_leakage_flights']} flights) {name:<6}: "
                     f"exact {m['exact']:.1%}  side|dir {_pct(m['side_given_direction'])}  NLL {m['nll']:.3f}")
    lines.append("  importance (day_a, exact / side drop): " + ", ".join(
        f"{g} {c['exact_drop']:+.3f}/{_pct(c['side_drop'], signed=True)}" for g, c in importance.items()))
    text = "\n".join(lines)
    (out / "runway_intent_r1.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _pct(value: float | None, *, signed: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:+.3f}" if signed else f"{value:.1%}"


if __name__ == "__main__":
    raise SystemExit(main())
