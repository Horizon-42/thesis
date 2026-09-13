"""Runway-intent R0: how well causal context alone names the landing runway (no model, no training).

Plan `docs/2026-09-13_runway_intent_plan.zh.md` §7 R0. For every validation flight of a
checkpoint's split and at several points of its approach — the entry to the arrival slice, then
the first sample with at most D km of flown path left, for each D of a remaining-path grid — the
causal baseline rules of `data.runway_context` (B0 majority, B1 active configuration, B2 B1
inside the course gate, B3 the last landing from the same entry sector, B4 the headwind group)
name a runway, scored against the harvest's label at two levels: the DIRECTION group (the
airport's configuration) and the SIDE inside it (the per-aircraft assignment; read only for
flights whose group has more than one runway). Also: how often the landing direction flips
(per day, and per fold of a day-blocked split — the evidence for the plan's decision D1).

Context pool: the tracks roster's assigned landings (time + runway) and the arrivals roster's
entry sectors, minus every flight whose split hash is OUTER-TEST — its label is never read and
its track never loaded, whether or not it is in the eligible roster. The static majority (B0,
and the in-group fallbacks) counts TRAIN-hash landings only, so no validation label shapes it.

    python run_ts.py runway_intent_r0 --checkpoint <plan head> --airport KRDU \
        --output-dir 4dTrajectory/outputs/POOLED/experiments/runway_intent_r0_20260913/KRDU
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ts_transformer.experiments.support import REPO_ROOT

from ts_transformer.data.runway_context import (  # noqa: E402
    RULES,
    ContextLanding,
    build_airport_context,
    parse_utc,
)
from ts_transformer.data.runway_features import ENTRY, anchors, track_course_at  # noqa: E402
from ts_transformer.data.splits import split_name_for_dataset_id  # noqa: E402
from ts_transformer.training.train import load_checkpoint  # noqa: E402

HARVEST_ROOT = REPO_ROOT / "trajectory_data_process" / "outputs" / "harvest"
METAR_ROOT = REPO_ROOT / "data" / "metar"
SCHEMA = "ts-runway-intent-r0-v1"
#: The landing direction in a bin is the group with the most landings; bins with fewer
#: landings than this are skipped, and a new direction counts as a flip only once it holds
#: for `FLIP_PERSIST_BINS` consecutive counted bins (a single go-around is not a reversal).
FLIP_BIN = timedelta(minutes=15)
FLIP_MIN_LANDINGS = 2
FLIP_PERSIST_BINS = 2
#: Counted bins further apart than this are two operating sessions (overnight, a harvest gap):
#: a different direction after the gap is a CHANGE ACROSS A GAP, counted apart from the flips —
#: the first readout counted them as flips, a third of KSJC's and KSMF's (review, 2026-09-13).
FLIP_MAX_GAP = timedelta(hours=3)


def day_fold(day: str, config: Any) -> str:
    """A day-blocked split with the checkpoint's own seed and fractions (plan §6, D1)."""
    digest = hashlib.sha256(f"{config.resolved_split_seed}:day:{day}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    if fraction < config.test_fraction:
        return "test"
    if fraction < config.test_fraction + config.val_fraction:
        return "val"
    return "train"


def direction_flips(
    landings: list[ContextLanding], groups: dict[str, int]
) -> tuple[dict[str, int], dict[str, int]]:
    """Landing-direction reversals per UTC day, and direction changes across a gap longer than
    `FLIP_MAX_GAP` per UTC day (see `FLIP_BIN` for the rule)."""
    bins: dict[datetime, Counter] = defaultdict(Counter)
    for landing in landings:
        start = landing.time - timedelta(
            minutes=landing.time.minute % 15, seconds=landing.time.second,
            microseconds=landing.time.microsecond,
        )
        bins[start][groups[landing.runway]] += 1
    counted = [
        (start, counts.most_common(1)[0][0])
        for start, counts in sorted(bins.items())
        if sum(counts.values()) >= FLIP_MIN_LANDINGS
    ]
    flips: dict[str, int] = Counter()
    across_gap: dict[str, int] = Counter()
    current: int | None = None
    candidate: int | None = None
    streak = 0
    previous: datetime | None = None
    for start, group in counted:
        if previous is not None and start - previous > FLIP_MAX_GAP:
            if current is not None and group != current:
                across_gap[start.date().isoformat()] += 1
            current, candidate, streak = group, None, 0
        previous = start
        if current is None:
            current = group
            continue
        if group == current:
            candidate, streak = None, 0
            continue
        streak = streak + 1 if group == candidate else 1
        candidate = group
        if streak >= FLIP_PERSIST_BINS:
            flips[start.date().isoformat()] += 1
            current, candidate, streak = group, None, 0
    return dict(flips), dict(across_gap)


def summarise(records: list[dict[str, Any]], groups: dict[str, int]) -> dict[str, Any]:
    """Per anchor × rule: exact, direction and conditional side accuracy, fallback share."""
    group_size = Counter(groups.values())
    table: dict[str, dict[str, Any]] = {}
    for anchor in sorted({r["anchor"] for r in records}, key=_anchor_order):
        rows = [r for r in records if r["anchor"] == anchor]
        entry: dict[str, Any] = {"flights": len(rows), "rules": {}}
        for rule in RULES:
            exact = sum(r["picks"][rule] == r["truth"] for r in rows)
            direction = [r for r in rows if groups[r["picks"][rule]] == groups[r["truth"]]]
            sided = [r for r in direction if group_size[groups[r["truth"]]] > 1]
            entry["rules"][rule] = {
                "exact": exact / len(rows),
                "direction": len(direction) / len(rows),
                "side_given_direction": (
                    sum(r["picks"][rule] == r["truth"] for r in sided) / len(sided) if sided else None
                ),
                "side_flights": len(sided),
                "fallback": sum(r["fallback"][rule] for r in rows) / len(rows),
            }
        table[anchor] = entry
    return table


def per_runway(records: list[dict[str, Any]], anchor: str) -> dict[str, dict[str, Any]]:
    rows = [r for r in records if r["anchor"] == anchor]
    out: dict[str, dict[str, Any]] = {}
    for runway in sorted({r["truth"] for r in rows}):
        mine = [r for r in rows if r["truth"] == runway]
        out[runway] = {"flights": len(mine), **{
            rule: sum(r["picks"][rule] == runway for r in mine) / len(mine) for rule in RULES
        }}
    return out


def _anchor_order(anchor: str) -> tuple[int, float]:
    return (0, 0.0) if anchor == ENTRY else (1, -float(anchor[:-2]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True,
                        help="the checkpoint whose split names the validation flights")
    parser.add_argument("--airport", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bins-km", default="30,20,15,10,6,3")
    parser.add_argument("--window-min", type=float, default=30.0,
                        help="B1/B2/B4: landings counted in this window before the anchor")
    parser.add_argument("--sector-window-min", type=float, default=60.0,
                        help="B3: the same-sector landing must be this recent")
    parser.add_argument("--metar-delay-min", type=float, default=10.0,
                        help="a METAR counts once its observation time is this far behind")
    parser.add_argument("--calm-kt", type=float, default=3.0,
                        help="B4 answers with B1 below this wind speed")
    args = parser.parse_args(argv)

    airport = args.airport.upper()
    bins_km = [float(value) for value in args.bins_km.split(",")]
    manifest_path = HARVEST_ROOT / airport / "arrivals" / "manifest.json"
    tracks_path = HARVEST_ROOT / airport / "tracks" / "manifest.json"
    _model, config, _normalizer, payload = load_checkpoint(args.checkpoint)

    def split_of(key: str) -> str:
        return split_name_for_dataset_id(f"{airport}:{key}", config)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    courses = {runway: float(target["course_deg"]) for runway, target in manifest["runway_targets"].items()}
    pool = build_airport_context(
        manifest_path, tracks_path, sorted((METAR_ROOT / airport).glob("asos_*.csv")), courses, split_of,
        window=timedelta(minutes=args.window_min),
        sector_window=timedelta(minutes=args.sector_window_min),
        metar_delay=timedelta(minutes=args.metar_delay_min),
        calm_kt=args.calm_kt,
    )
    context, by_key, sectors = pool.rules, pool.flights, pool.sectors
    landings, metar_paths = list(pool.landings), list(pool.metar_paths)
    excluded_test = pool.excluded_outer_test

    # ── the validation flights: the checkpoint's own split ───────────────────────────────
    val_keys = [key[len(airport) + 1:] for key in payload["split"]["val"] if key.startswith(f"{airport}:")]
    missing = [key for key in val_keys if key not in by_key]
    if missing:
        raise ValueError(f"{len(missing)} validation flights are not loadable from {manifest_path}")
    records: list[dict[str, Any]] = []
    for key in val_keys:
        flight = by_key[key]
        entry_time = parse_utc(flight["entry_time_utc"])
        waypoints = flight["waypoints"]
        sector = sectors[key]
        for name, index in anchors(waypoints, bins_km).items():
            t = entry_time + timedelta(seconds=float(waypoints[index][0]))
            picks = context.picks(t, sector=sector, track_course_deg=track_course_at(waypoints, index))
            records.append({
                "flight_key": key, "anchor": name, "truth": flight["runway"],
                "picks": {rule: pick.runway for rule, pick in picks.items()},
                "fallback": {rule: pick.fallback for rule, pick in picks.items()},
            })

    summary = summarise(records, context.groups)
    flips, across_gap = direction_flips(landings, context.groups)
    days = sorted({landing.time.date().isoformat() for landing in landings})
    folds: dict[str, dict[str, int]] = defaultdict(lambda: {"days": 0, "flips": 0, "days_with_flip": 0})
    for day in days:
        fold = folds[day_fold(day, config)]
        fold["days"] += 1
        fold["flips"] += flips.get(day, 0)
        fold["days_with_flip"] += flips.get(day, 0) > 0

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": SCHEMA,
        "airport": airport,
        "checkpoint": str(args.checkpoint),
        "candidates": courses,
        "direction_groups": context.groups,
        "rules": list(RULES),
        "settings": {
            "bins_km": bins_km, "window_min": args.window_min,
            "sector_window_min": args.sector_window_min,
            "metar_delay_min": args.metar_delay_min, "calm_kt": args.calm_kt,
            "flip_rule": {"bin_min": FLIP_BIN.total_seconds() / 60,
                          "min_landings": FLIP_MIN_LANDINGS, "persist_bins": FLIP_PERSIST_BINS,
                          "max_gap_h": FLIP_MAX_GAP.total_seconds() / 3600},
        },
        "context_pool": {
            "landings": len(landings), "excluded_outer_test_hash": excluded_test,
            "arrivals_with_sector": len(sectors), "metar_files": [str(p) for p in metar_paths],
            "rule": "tracks-roster assigned landings and arrivals-roster entry sectors, minus every "
                    "flight whose split hash is outer-test; majority from train-hash landings only",
        },
        "validation_flights": len(val_keys),
        "summary": summary,
        "per_runway": {anchor: per_runway(records, anchor) for anchor in (ENTRY, "10km") if
                       any(r["anchor"] == anchor for r in records)},
        "direction_flips": {"per_day": flips, "across_gap_per_day": across_gap,
                            "days": len(days), "day_blocked_folds": dict(folds)},
        "records": records,
    }
    (out / "runway_intent_r0.json").write_text(json.dumps(document, indent=1), encoding="utf-8")
    lines = [
        f"{airport}: {len(val_keys)} validation flights, candidates {sorted(courses)}, "
        f"direction groups {context.groups}",
        f"context: {len(landings)} landings ({excluded_test} outer-test-hash landings excluded), "
        f"{len(sectors)} arrivals with an entry sector, METAR {', '.join(p.name for p in metar_paths)}",
        "",
        "anchor   n     rule                     exact  direction  side|dir (n)   fallback",
    ]
    for anchor, entry in summary.items():
        for rule, cell in entry["rules"].items():
            side = "   n/a   " if cell["side_given_direction"] is None else f"{cell['side_given_direction']:6.1%} ({cell['side_flights']})"
            lines.append(
                f"{anchor:<8} {entry['flights']:<5} {rule:<24} {cell['exact']:6.1%}  {cell['direction']:8.1%}  "
                f"{side:<14} {cell['fallback']:6.1%}"
            )
    lines += ["", f"direction flips: {sum(flips.values())} over {len(days)} days "
                  f"({sum(v > 0 for v in flips.values())} days with at least one); "
                  f"{sum(across_gap.values())} direction changes across a gap > "
                  f"{FLIP_MAX_GAP.total_seconds() / 3600:g} h"]
    for fold in ("train", "val", "test"):
        cell = folds.get(fold, {"days": 0, "flips": 0, "days_with_flip": 0})
        lines.append(f"  day-blocked {fold:<5}: {cell['days']} days, {cell['flips']} flips, "
                     f"{cell['days_with_flip']} days with a flip")
    text = "\n".join(lines)
    (out / "runway_intent_r0.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\nwrote {out / 'runway_intent_r0.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
