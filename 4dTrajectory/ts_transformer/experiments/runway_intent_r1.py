"""Runway-intent R1: a learned runway head against the causal rules — and what the split does.

Plan `docs/2026-09-13_runway_intent_plan.zh.md` §12 (design) and §11.4 (the pre-registered gates).
Per airport, every arrival whose per-flight split hash is NOT outer-test gives one sample per
anchor — the arrival-slice entry (the 25 km ring) and the first crossing of each ring in
`RING_RADII_KM` around the airport reference, runway-independent (`ring_anchors`; R0's
remaining-path anchors measure to the true threshold and so leak its along-track distance) — with
the causal features of `data.runway_features`, labelled with its landing runway. One HistGradientBoostingClassifier PER
AIRPORT — not pooled across airports (experiment principle (3), stated: R1 first measures what each
airport's own data teaches). Three trainings:

- ``day_a`` / ``day_b`` — the DAY-BLOCKED split (the user's decision D1 = (a), 2026-09-13): OPERATING
  days (cut at the overnight traffic minimum, `operational_day`) are hashed into folds; the test fold's days are never used (kept for the final test on the sealed
  flights), and the other days are split into train / validation twice (``a`` with the split seed,
  ``b`` with `SECOND_PARTITION_SEED`) — a tree model has no seed variance, the day partition does;
- ``flight`` — today's per-flight hash split over the same non-test days: the leakage control.

Each model is read on its own validation samples against the causal rules recomputed on the same
samples — each partition's rules and minority runways from ITS training days' landings; ``day_a`` and ``flight`` are also read PAIRED on the samples that are validation under
both splits — the flight model has seen those days' other flights, the day model has not, and the
difference is the leakage. Both readings are also cut by OPERATING day (`by_day`): a pooled
number over a handful of validation days can hide one day a rule reads and the head does not.

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
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np

from ts_transformer.experiments.support import REPO_ROOT

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from ts_transformer.config import TSConfig  # noqa: E402
from ts_transformer.data.runway_context import (  # noqa: E402
    OPERATIONAL_DAY_SHIFT,
    RULES,
    AirportContext,
    RunwayContext,
    airport_reference,
    build_airport_context,
    operational_day,
    parse_utc,
)
from ts_transformer.data.runway_features import (  # noqa: E402
    ENTRY,
    FeatureSpace,
    airline,
    anchor_features,
    feature_space,
    minutes_since_each,
    ring_anchors,
    track_course_at,
)
from ts_transformer.data.splits import split_name_for_dataset_id  # noqa: E402

HARVEST_ROOT = REPO_ROOT / "trajectory_data_process" / "outputs" / "harvest"
METAR_ROOT = REPO_ROOT / "data" / "metar"
# v2: operating days, ring anchors, per-partition majority; v3: the per-operating-day blocks
SCHEMA = "ts-runway-intent-r1-v3"
#: Every landing crosses the 6 km ring (the farthest threshold sits 3.65 km from its airport's
#: reference, KSTL 11), so each ring below it exists for every flight.
RING_RADII_KM = (20.0, 15.0, 10.0, 6.0)
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
    for name in sorted(set(anchor_names), key=lambda a: (a != ENTRY, -float(a[1:-2]) if a != ENTRY else 0.0)):
        out[name] = block(anchor_names == name)
    return out


def by_day(
    mask: np.ndarray, days: np.ndarray, truth: np.ndarray, anchors: np.ndarray,
    picks: dict[str, np.ndarray], candidates: list[str],
) -> dict[str, dict[str, Any]]:
    """Per operating day of the masked samples: its flights (one entry sample each), the busiest
    runway among them and its share, and the exact accuracy of every named pick array."""
    out: dict[str, dict[str, Any]] = {}
    for day in sorted(set(days[mask].tolist())):
        on = mask & (days == day)
        labels = Counter(truth[on & (anchors == ENTRY)].tolist())
        flights = sum(labels.values())
        top, count = max(labels.items(), key=lambda item: (item[1], -item[0]))
        out[day] = {
            "flights": flights, "samples": int(on.sum()),
            "top_runway": candidates[top], "top_share": count / flights,
            "exact": {name: float((p[on] == truth[on]).mean()) for name, p in picks.items()},
        }
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


@dataclass(frozen=True)
class RunwaySamples:
    """One airport's sample table: every usable flight x its ring anchors, with R1's causal feature
    vector, the landing runway, each split's fold and every partition's rule picks — the one
    construction R1 and R1.1 (`runway_intent_r11`) read, so their numbers are on the same samples."""

    airport: str
    config: TSConfig
    radii_km: list[float]
    candidates: list[str]
    space: FeatureSpace
    pool: AirportContext
    usable: dict[str, dict[str, Any]]
    contexts: dict[str, RunwayContext]       # partition -> the rules under ITS training majority
    group_of: np.ndarray                     # [C] each candidate's direction group
    multi: np.ndarray                        # [C] the candidate's group has more than one runway
    minority_names: dict[str, list[str]]
    minority: dict[str, np.ndarray]
    X: np.ndarray                            # [n, F] R1's features, columns `space.names`
    y: np.ndarray                            # [n] the landing runway, a candidate index
    anchors: np.ndarray                      # [n] anchor name
    folds: dict[str, np.ndarray]             # partition -> [n] "train" / "val"
    picks: dict[str, dict[str, np.ndarray]]  # partition -> rule -> [n] candidate index
    b1_prob: np.ndarray                      # [n, C] B1 as a probability (add-one window counts)
    keys: np.ndarray                         # [n] flight key
    days: np.ndarray                         # [n] the landing's operating day
    operators: np.ndarray                    # [n] callsign operator prefix ("" = no callsign)
    minutes_since: np.ndarray                # [n, C] minutes since each candidate's last landing


def add_sample_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--airport", required=True)
    parser.add_argument("--ring-radii-km", default=",".join(f"{r:g}" for r in RING_RADII_KM))
    parser.add_argument("--window-min", type=float, default=30.0)
    parser.add_argument("--sector-window-min", type=float, default=60.0)
    parser.add_argument("--metar-delay-min", type=float, default=10.0)
    parser.add_argument("--calm-kt", type=float, default=3.0)


def build_samples(args: argparse.Namespace) -> RunwaySamples:
    airport = args.airport.upper()
    radii_km = [float(value) for value in args.ring_radii_km.split(",")]
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

    def day_of(flight: dict[str, Any]) -> dict[str, str]:
        return day_folds(operational_day(parse_utc(flight["landing_time_utc"])), config)

    usable = {key: flight for key, flight in pool.flights.items() if day_of(flight)["a"] != "test"}
    fold_of_flight = {
        key: {"day_a": day_of(flight)["a"], "day_b": day_of(flight)["b"], "flight": split_of(key)}
        for key, flight in usable.items()
    }
    # Each partition's static majority (B0, the B2 fallback, B4's in-use filter, the minority
    # runways) from ITS training flights only — never another partition's validation labels.
    contexts = {
        part: context.with_majority(Counter(
            flight["runway"] for key, flight in usable.items() if fold_of_flight[key][part] == "train"
        ))
        for part in ("day_a", "day_b", "flight")
    }
    # Which operator prefixes get a column: read from the callsigns of every usable flight — a
    # vocabulary, no label — so all three trainings share one feature matrix.
    space = feature_space(candidates, targets, airport_reference(targets), usable.values())
    index_of = {r: i for i, r in enumerate(candidates)}
    group_of = np.array([context.groups[r] for r in candidates])
    group_sizes = Counter(context.groups.values())
    multi = np.array([group_sizes[context.groups[r]] > 1 for r in candidates])
    minority_names = {
        part: minority_runways(context.groups, ctx.majority_counts) for part, ctx in contexts.items()
    }
    minority = {part: np.array([index_of[r] for r in names], dtype=int)
                for part, names in minority_names.items()}

    rows, truth, anchor_names, flight_keys = [], [], [], []
    folds: dict[str, list[str]] = {"day_a": [], "day_b": [], "flight": []}
    rule_picks: dict[str, dict[str, list[int]]] = {
        part: {rule: [] for rule in RULES} for part in contexts
    }
    b1_prob: list[np.ndarray] = []
    since: list[list[float]] = []
    reference = airport_reference(targets)
    for key, flight in sorted(usable.items()):
        entry = parse_utc(flight["entry_time_utc"])
        waypoints = flight["waypoints"]
        sector = pool.sectors[key]
        for name, index in ring_anchors(waypoints, reference, radii_km).items():
            t = entry + timedelta(seconds=float(waypoints[index][0]))
            rows.append(anchor_features(space, context, flight, index, sector=sector, anchor_time=t))
            truth.append(index_of[flight["runway"]])
            anchor_names.append(name)
            flight_keys.append(key)
            for part in folds:
                folds[part].append(fold_of_flight[key][part])
            course = track_course_at(waypoints, index)
            for part, ctx in contexts.items():
                picks = ctx.picks(t, sector=sector, track_course_deg=course)
                for rule in RULES:
                    rule_picks[part][rule].append(index_of[picks[rule].runway])
            counts = Counter(landing.runway for landing in context.recent(t, context.window))
            smoothed = np.array([counts[r] + 1.0 for r in candidates])
            b1_prob.append(smoothed / smoothed.sum())
            since.append(minutes_since_each(context, t, candidates))
    return RunwaySamples(
        airport=airport, config=config, radii_km=radii_km, candidates=candidates, space=space,
        pool=pool, usable=usable, contexts=contexts, group_of=group_of, multi=multi,
        minority_names=minority_names, minority=minority,
        X=np.vstack(rows), y=np.asarray(truth), anchors=np.asarray(anchor_names),
        folds={name: np.asarray(values) for name, values in folds.items()},
        picks={part: {rule: np.asarray(values) for rule, values in rules.items()}
               for part, rules in rule_picks.items()},
        b1_prob=np.vstack(b1_prob), keys=np.asarray(flight_keys),
        days=np.asarray([operational_day(parse_utc(usable[k]["landing_time_utc"])) for k in flight_keys]),
        operators=np.asarray([airline(usable[k]) for k in flight_keys]),
        minutes_since=np.asarray(since, dtype=np.float64),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_sample_arguments(parser)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    s = build_samples(args)
    airport, config, candidates, space, pool, usable = s.airport, s.config, s.candidates, s.space, s.pool, s.usable
    radii_km, context = s.radii_km, pool.rules
    group_of, multi, minority, minority_names = s.group_of, s.multi, s.minority, s.minority_names
    X, y, anchor_arr, fold_arr, picks_arr = s.X, s.y, s.anchors, s.folds, s.picks
    b1_arr, keys_arr = s.b1_prob, s.keys

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
        part = FOLD_OF[name]
        results[name] = {
            "train_samples": int(train.sum()), "val_samples": int(val.sum()),
            "train_flights": int(len(set(keys_arr[train]))), "val_flights": int(len(set(keys_arr[val]))),
            "classes_seen": [candidates[i] for i in model.classes_],
            "features_used": int(columns.sum()),
            "minority_runways": minority_names[part],
            "validation": score(prob[val], y[val], {r: p[val] for r, p in picks_arr[part].items()},
                                b1_arr[val], anchor_arr[val], group_of=group_of, multi=multi,
                                minority=minority[part]),
        }
    both = (fold_arr["day_a"] == "val") & (fold_arr["flight"] == "val")
    # One reference for both models on the paired flights: the day_a partition's rules and
    # minority set (the flight model is being compared, not re-baselined).
    leakage = {
        name: score(probs[name][both], y[both], {r: p[both] for r, p in picks_arr["day_a"].items()},
                    b1_arr[both], anchor_arr[both], group_of=group_of, multi=multi,
                    minority=minority["day_a"])
        for name in ("day_a", "flight")
    }
    val_a = fold_arr["day_a"] == "val"
    importance = grouped_permutation(
        models["day_a"], X[val_a], y[val_a], space.names, dict(space.groups), models["day_a"].classes_,
        group_of=group_of, multi=multi,
    )

    days_arr = s.days
    validation_by_day = {
        part: by_day(fold_arr[part] == "val", days_arr, y, anchor_arr, {
            "model": probs[part].argmax(axis=1), "nowx": probs[f"{part}_nowx"].argmax(axis=1),
            "B1_active_config": picks_arr[part]["B1_active_config"],
        }, candidates)
        for part in ("day_a", "day_b")
    }
    leakage_by_day = by_day(both, days_arr, y, anchor_arr, {
        "day_a": probs["day_a"].argmax(axis=1), "flight": probs["flight"].argmax(axis=1),
        "B1_active_config": picks_arr["day_a"]["B1_active_config"],
    }, candidates)

    days = sorted({operational_day(parse_utc(f["landing_time_utc"])) for f in pool.flights.values()})
    fold_days = {part: Counter(day_folds(d, config)[part] for d in days) for part in ("a", "b")}
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": SCHEMA,
        "airport": airport,
        "candidates": candidates,
        "direction_groups": context.groups,
        "minority_runways": minority_names,
        "anchors": {"entry": "the arrival-slice entry (25 km ring)",
                    "rings_km": list(radii_km), "rule": "first sample within R km of the airport reference"},
        "operational_day_shift_h": OPERATIONAL_DAY_SHIFT.total_seconds() / 3600,
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
        "validation_by_day": validation_by_day,
        "leakage_by_day": leakage_by_day,
        "importance_day_a": importance,
    }
    (out / "runway_intent_r1.json").write_text(json.dumps(document, indent=1), encoding="utf-8")

    lines = [f"{airport}: {len(usable)} usable flights, {len(y)} samples; candidates {candidates}; "
             f"minority (day_a) {minority_names['day_a']}; days a {dict(fold_days['a'])} b {dict(fold_days['b'])}"]
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
    for part, cells in validation_by_day.items():
        worst = min(cells, key=lambda d: cells[d]["exact"]["model"])
        w = cells[worst]
        median = {n: float(np.median([c["exact"][n] for c in cells.values()])) for n in ("model", "B1_active_config")}
        lines.append(
            f"  {part} per day ({len(cells)} val days): model median {median['model']:.1%}, B1 median "
            f"{median['B1_active_config']:.1%}; worst {worst}: model {w['exact']['model']:.1%}, B1 "
            f"{w['exact']['B1_active_config']:.1%} ({w['flights']} flights, {w['top_runway']} {w['top_share']:.0%})"
        )
    gaps = {d: c for d, c in leakage_by_day.items() if abs(c["exact"]["flight"] - c["exact"]["day_a"]) >= 0.05}
    lines.append("  paired by day, |flight - day_a| >= 5 points: " + (", ".join(
        f"{d} ({c['flights']} flights, {c['top_runway']} {c['top_share']:.0%}) day_a {c['exact']['day_a']:.1%} "
        f"flight {c['exact']['flight']:.1%}" for d, c in gaps.items()) or "none"))
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
