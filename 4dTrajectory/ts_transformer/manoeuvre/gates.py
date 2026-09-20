"""The pre-registered gates two-tier v3 still reads (plan §3.3), judged over the readouts'
artefacts — never a number typed in by hand, and never on ONE seed (a single-seed verdict is
refused, plan §3.4).

    grid      stage A1: the (lookback, segment) cell, off the protocol-none lockstep payloads.
    relative  stage A3 / B: a candidate reading against its baseline reading of the SAME
              flights, judged against a seed line the caller names.

Gates T / X / P / E / S belonged to the intent-code second layer and are ARCHIVED 2026-09-20
(`archive/manoeuvre_codes_2026_09/gates_manoeuvre.py`, a verbatim copy; their numbers in
`docs/2026-09-18_manoeuvre_token_results.zh.md` §9–§11).

Every number is a module constant here, named where it comes from; the readers pass the
lockstep artefacts (`experiments/manoeuvre_lockstep.py` payloads) keyed by seed.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ts_transformer.data.approach_difficulty import STRATUM_ALL, STRATUM_STRAIGHT_IN, STRATUM_VECTORED


def _two_seeds(by_seed: Mapping[int, Any], gate: str) -> None:
    if len(by_seed) < 2:
        raise ValueError(f"gate {gate} is judged on two seeds; got {sorted(by_seed)}")


def _require_protocol(by_seed: Mapping[int, dict[str, Any]], expected: str, gate: str) -> None:
    for seed, payload in by_seed.items():
        if payload.get("protocol") != expected:
            raise ValueError(f"gate {gate} reads protocol {expected!r} artefacts; seed {seed}'s is {payload.get('protocol')!r}")


def _cell(payload: dict[str, Any], stratum: str) -> dict[str, Any]:
    cell = payload["strata"][stratum]
    if not cell.get("n"):
        raise ValueError(f"the {stratum!r} stratum is empty in {payload.get('executor_name', '?')}")
    return cell


#: Two-tier v3 stage A1 (§3.3, §5.1): the fully-flyable constraint a cell must meet on both seeds.
GRID_FLYABLE_FLOOR = 0.95
#: The seed line is read off the grid itself (D9): the p75 of the two seeds' absolute difference
#: over the cells (the p50 is recorded beside it).
GRID_SEED_LINE_QUANTILE = 75


def cell_name(lookback_s: float, segment_s: float) -> str:
    return f"L{lookback_s:g}_D{segment_s:g}"


def _seed_line(values: Sequence[tuple[float, float]]) -> dict[str, Any]:
    """The p50 and the p75 of |seed a − seed b| over the cells; the line is the p75
    (`GRID_SEED_LINE_QUANTILE`), read back under that key."""
    deltas = np.abs(np.array([a - b for a, b in values], dtype=np.float64))
    return {"p50": float(np.median(deltas)), "p75": float(np.percentile(deltas, 75)), "cells": int(len(deltas))}


def cell_reading(payload: dict[str, Any]) -> dict[str, float]:
    """The numbers the grid gate reads off one protocol-none lockstep payload."""
    pooled, vectored, straight = payload["strata"][STRATUM_ALL], _cell(payload, STRATUM_VECTORED), _cell(payload, STRATUM_STRAIGHT_IN)
    return {
        "n": payload["flights"], "established_all": pooled["established_share"], "established_vectored": vectored["established_share"],
        "established_straight": straight["established_share"], "fully_flyable": pooled["fully_flyable_share"],
        "vectored_ade_mean_m": vectored["ade_mean_m"], "fde_p50_m": pooled["fde_p50_m"],
        # a receding reading (A3-a) flies only the first executed_s of each segment_s forecast: carried, never silent
        "segment_s": payload["segment_s"], "executed_s": payload["executed_s"],
    }


def gate_grid(cells: Mapping[tuple[float, float], Mapping[int, dict[str, Any]]], *,
              flyable_floor: float = GRID_FLYABLE_FLOOR) -> dict[str, Any]:
    """Two-tier v3 A1: pick the (lookback, segment) cell from ``{(L_s, Δ_s): {seed: protocol-none
    payload}}`` (the L−1 reading; every cell on two seeds).

    §3.3's row: the primary metrics are established over ALL flights and over the VECTORED
    group, the constraint fully flyable ≥ the floor on both seeds; the seed line is the grid's
    own (D9); a cell within the seed line of each seed's best is a tie with it ("leader"); the
    winners are the leaders on both primaries, or — when the two disagree — the leaders on all
    flights; a tie is broken by the shorter segment, then the shorter lookback. Not decisive
    when every eligible cell is a leader: L and Δ are then not the deciding variables in this
    range (§5.1's fallback note), and A2 runs on the shortest cell."""
    if not cells:
        raise ValueError("gate grid needs at least one cell")
    readings: dict[tuple[float, float], dict[str, Any]] = {}
    for cell, by_seed in cells.items():
        _two_seeds(by_seed, "grid")
        _require_protocol(by_seed, "none", "grid")
        per_seed = {seed: cell_reading(p) for seed, p in by_seed.items()}
        readings[cell] = {key: {seed: reading[key] for seed, reading in per_seed.items()} for key in next(iter(per_seed.values()))}
    seeds = sorted(set.intersection(*(set(r["established_all"]) for r in readings.values())))
    if len(seeds) != 2:
        raise ValueError(f"the grid is read on exactly two seeds shared by every cell; the cells share {seeds}")
    metrics = ("established_all", "established_vectored", "vectored_ade_mean_m")
    seed_line = {m: _seed_line([(r[m][seeds[0]], r[m][seeds[1]]) for r in readings.values()]) for m in metrics}
    line_key = f"p{GRID_SEED_LINE_QUANTILE}"
    eligible = [c for c, r in readings.items() if all(r["fully_flyable"][s] >= flyable_floor for s in seeds)]
    ineligible = [c for c in readings if c not in eligible]

    def leaders(metric: str) -> list[tuple[float, float]]:
        best = {s: max(readings[c][metric][s] for c in eligible) for s in seeds}
        return [c for c in eligible if all(readings[c][metric][s] >= best[s] - seed_line[metric][line_key] for s in seeds)]

    def best_on_both(metric: str) -> tuple[float, float] | None:
        bests = {max(eligible, key=lambda c: readings[c][metric][s]) for s in seeds}
        return next(iter(bests)) if len(bests) == 1 else None

    lead_all = leaders("established_all") if eligible else []
    lead_vectored = leaders("established_vectored") if eligible else []
    both = [c for c in lead_all if c in lead_vectored]
    winners = both or lead_all
    selected = min(winners, key=lambda c: (c[1], c[0])) if winners else None      # shorter segment, then shorter lookback
    # decisive = at least one primary separates the eligible cells beyond the seed line
    decisive = bool(eligible) and (len(lead_all) < len(eligible) or len(lead_vectored) < len(eligible))
    if not eligible:
        note = f"no cell is fully flyable ≥ {flyable_floor:g} on both seeds"
    elif not decisive:
        note = ("every eligible cell is within the seed line of the best on both primaries: L and Δ are not the deciding "
                "variables in this range; the shortest cell is taken for A2")
    elif not both:
        note = "the two primaries disagree; established over all flights decides"
    else:
        note = None
    name = cell_name
    return {
        "gate": "grid", "seeds": seeds, "flyable_floor": flyable_floor,
        "seed_line": seed_line, "seed_line_rule": f"the {line_key} of |seed a − seed b| over the cells",
        "readings": {name(*c): r for c, r in readings.items()},
        "eligible": [name(*c) for c in eligible], "ineligible": [name(*c) for c in ineligible],
        "leaders": {"established_all": [name(*c) for c in lead_all], "established_vectored": [name(*c) for c in lead_vectored]},
        "best_on_both_seeds": {m: (None if best_on_both(m) is None else name(*best_on_both(m))) for m in ("established_all", "established_vectored")} if eligible else {},
        "winners": [name(*c) for c in winners],
        "selected": None if selected is None else {"cell": name(*selected), "lookback_s": selected[0], "segment_s": selected[1]},
        "decisive": decisive, "note": note,
        "tie_rule": "within the seed line of each seed's best on all flights AND on the vectored group; if none, on all flights; "
                    "then the shorter segment, then the shorter lookback",
    }


#: The relative gate's metrics (§3.3 rows A3 and B): the two primaries and the vectored ADE.
RELATIVE_METRICS = ("established_all", "established_vectored", "vectored_ade_mean_m")
RELATIVE_PRIMARIES = ("established_all", "established_vectored")


def gate_relative(baseline: Mapping[int, Mapping[str, float]], candidate: Mapping[int, Mapping[str, float]], *,
                  seed_line: Mapping[str, float], flyable_floor: float = GRID_FLYABLE_FLOOR) -> dict[str, Any]:
    """Two-tier v3's relative gate (§3.3 rows A3 and B; D45): a candidate reading against a
    baseline reading of the SAME flights, per seed — ``{seed: cell_reading(...)}`` on both
    sides, the caller having recomputed both over the common flights — judged against a seed
    line the caller names (the grid's own, D39), never a number typed in here.

    improvement = candidate − baseline for the established shares and baseline − candidate for
    the vectored ADE (metres, so positive is better throughout); "not worse" = improvement ≥
    −line, "beyond" = improvement > line. Row A3: on at least one primary both seeds improve and
    one of them beyond the line, the other primary is not worse on both seeds, and the candidate
    is fully flyable ≥ the floor on both seeds. Row B: every metric not worse on both seeds, at
    least one metric beyond the line on both seeds, fully flyable ≥ the floor on both seeds.
    Row B1 (an UPPER-BOUND reading: the second layer fed the truth): at least one metric beyond
    the line on both seeds and fully flyable ≥ the floor — row B without the not-worse clause,
    because an upper bound that buys one thing at the price of another still says the
    information is there; row B, read on what the layer itself says, is where the price counts.
    (Stage B's intent-code version of this row is archived with its layer, 2026-09-20; the
    rewritten stage B tightened the rule to "both established not worse", plan v3 D57, which
    is row B's clause — not yet built.)"""
    seeds = sorted(baseline)
    if len(seeds) != 2 or sorted(candidate) != seeds:
        raise ValueError(f"the relative gate reads two seeds on both sides; got baseline {sorted(baseline)} and candidate {sorted(candidate)}")
    missing = [m for m in RELATIVE_METRICS if m not in seed_line]
    if missing:
        raise ValueError(f"the seed line names every metric of {RELATIVE_METRICS}; missing {missing}")
    if any(seed_line[m] < 0 for m in RELATIVE_METRICS):
        raise ValueError(f"a seed line is a non-negative width; got {dict(seed_line)}")
    improvement = {m: {s: (baseline[s][m] - candidate[s][m]) if m == "vectored_ade_mean_m" else (candidate[s][m] - baseline[s][m])
                       for s in seeds} for m in RELATIVE_METRICS}
    not_worse = {m: {s: improvement[m][s] >= -seed_line[m] for s in seeds} for m in RELATIVE_METRICS}
    beyond = {m: {s: improvement[m][s] > seed_line[m] for s in seeds} for m in RELATIVE_METRICS}
    flyable_ok = all(candidate[s]["fully_flyable"] >= flyable_floor for s in seeds)
    a3_on = [p for p in RELATIVE_PRIMARIES
             if all(improvement[p][s] > 0 for s in seeds) and any(beyond[p][s] for s in seeds)
             and all(not_worse[q][s] for q in RELATIVE_PRIMARIES if q != p for s in seeds)]
    b_beyond = [m for m in RELATIVE_METRICS if all(beyond[m][s] for s in seeds)]
    b_not_worse = all(not_worse[m][s] for m in RELATIVE_METRICS for s in seeds)
    return {
        "gate": "relative", "seeds": seeds, "seed_line": {m: float(seed_line[m]) for m in RELATIVE_METRICS}, "flyable_floor": flyable_floor,
        "baseline": {s: dict(baseline[s]) for s in seeds}, "candidate": {s: dict(candidate[s]) for s in seeds},
        "improvement": improvement, "not_worse": not_worse, "beyond": beyond, "fully_flyable_ok": flyable_ok,
        "verdicts": {
            "a3": {"pass": bool(a3_on) and flyable_ok, "on": a3_on,
                   "rule": "on a primary both seeds improve and one beyond the seed line; the other primary not worse; fully flyable ≥ floor"},
            "b": {"pass": b_not_worse and bool(b_beyond) and flyable_ok, "beyond_on": b_beyond, "not_worse": b_not_worse,
                  "rule": "every metric not worse on both seeds; at least one beyond the seed line on both seeds; fully flyable ≥ floor"},
            "b1": {"pass": bool(b_beyond) and flyable_ok, "beyond_on": b_beyond,
                   "rule": "at least one metric beyond the seed line on both seeds; fully flyable ≥ floor (the truth-token upper bound)"},
        },
    }


__all__ = [
    "GRID_FLYABLE_FLOOR", "GRID_SEED_LINE_QUANTILE", "RELATIVE_METRICS", "RELATIVE_PRIMARIES",
    "cell_name", "cell_reading", "gate_grid", "gate_relative",
]
