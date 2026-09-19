"""The pre-registered gates (plan §3.3), judged over the readouts' artefacts — never a
number typed in by hand, and never on ONE seed (a single-seed verdict is refused, plan §3.4).

    T   the tokenizer: judged by `manoeuvre/readout.py` (gate_t in its artefact) — re-read here.
    X   the executor under the truth's codes (protocol C): fully flyable ≥ 0.95; established ≥
        0.9 × the rule guidance's established along the truth segments (a number MEASURED by the
        guidance run, handed in); vectored ADE mean ≤ 1500 m, straight-in ≤ 250 m (v2 gate L1).
    P   the prior: open-loop (A-truth) vectored displacement p50 at 120 / 180 s ≤ B61's 768 / 1396
        + 125 m, straight-in ≤ 366 / 587 + 60 m (v2 §10.6); and the discrete-vs-continuous
        control on protocol A: the discrete arm's vectored ADE mean and established share not
        worse than the continuous's beyond the seed line on both seeds (a tie is the discrete).
    E   end to end (protocol A): vectored ADE mean ≤ 2745, straight-in ≤ 415, fully flyable ≥
        0.95, established ≥ 0.94 (v2 G3); the PROGRESS line: vectored ADE ≥ 125 m below v2's
        3044 / 3074 and established ≥ 5 points above 0.640 / 0.649, seed by seed.
    S   the segment length: among the candidates that passed T, protocol A's vectored ADE mean
        and established share, best on both seeds; disagreement → established decides; a tie
        within the seed line (125 m / 3 points) → the longer segment.

Every number is a module constant here, named where it comes from; the readers pass the
lockstep artefacts (`experiments/manoeuvre_lockstep.py` payloads) keyed by seed.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ts_transformer.data.approach_difficulty import STRATUM_ALL, STRATUM_STRAIGHT_IN, STRATUM_VECTORED

#: The control path's seed line (CLAUDE.md "How to read results"): pooled ADE 125 m; the
#: established-share line 3 points (plan §3.3 S).
SEED_LINE_ADE_M = 125.0
SEED_LINE_ESTABLISHED = 0.03
#: Gate X (v2 gate L1).
X_FLYABLE = 0.95
X_ESTABLISHED_RATIO = 0.9
X_VECTORED_ADE_M = 1500.0
X_STRAIGHT_ADE_M = 250.0
#: Gate P (v2 §10.6 B61 open-loop p50s + the margins).
P_B61_VECTORED = {"120": 768.0, "180": 1396.0}
P_B61_STRAIGHT = {"120": 366.0, "180": 587.0}
P_VECTORED_MARGIN_M = 125.0
P_STRAIGHT_MARGIN_M = 60.0
#: Gate E (v2 G3) and its progress line (v2 §12.1, pa + B61, seeds 1337 / 2024).
E_VECTORED_ADE_M = 2745.0
E_STRAIGHT_ADE_M = 415.0
E_FLYABLE = 0.95
E_ESTABLISHED = 0.94
E_PROGRESS_VECTORED_ADE_M = {1337: 3044.0, 2024: 3074.0}
E_PROGRESS_ESTABLISHED = {1337: 0.640, 2024: 0.649}
E_PROGRESS_ADE_MARGIN_M = 125.0
E_PROGRESS_ESTABLISHED_MARGIN = 0.05


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


def gate_x(lockstep_c: Mapping[int, dict[str, Any]], *, guidance_established: float) -> dict[str, Any]:
    """Protocol-C payloads keyed by seed; ``guidance_established`` is the rule guidance's
    established share along the truth segments, measured on the same cohort."""
    _two_seeds(lockstep_c, "X")
    _require_protocol(lockstep_c, "C", "X")
    seeds = {}
    for seed, payload in lockstep_c.items():
        vectored, straight = _cell(payload, STRATUM_VECTORED), _cell(payload, STRATUM_STRAIGHT_IN)
        pooled = payload["strata"]["all"]
        seeds[seed] = {
            "fully_flyable": pooled["fully_flyable_share"], "established": pooled["established_share"],
            "vectored_ade_mean_m": vectored["ade_mean_m"], "straight_ade_mean_m": straight["ade_mean_m"],
            "passes": (pooled["fully_flyable_share"] >= X_FLYABLE
                       and pooled["established_share"] >= X_ESTABLISHED_RATIO * guidance_established
                       and vectored["ade_mean_m"] <= X_VECTORED_ADE_M and straight["ade_mean_m"] <= X_STRAIGHT_ADE_M),
        }
    return {"gate": "X", "guidance_established": guidance_established, "established_floor": X_ESTABLISHED_RATIO * guidance_established,
            "seeds": seeds, "passes": all(s["passes"] for s in seeds.values())}


def gate_p_open_loop(lockstep_a_truth: Mapping[int, dict[str, Any]]) -> dict[str, Any]:
    """A-truth payloads keyed by seed: the prior's top-1 flown by the executor from the truth
    history, read at 120 / 180 s against B61's open-loop p50s."""
    _two_seeds(lockstep_a_truth, "P")
    _require_protocol(lockstep_a_truth, "A-truth", "P-open-loop")
    seeds = {}
    for seed, payload in lockstep_a_truth.items():
        vectored, straight = _cell(payload, STRATUM_VECTORED), _cell(payload, STRATUM_STRAIGHT_IN)
        checks = {}
        for lead in ("120", "180"):
            checks[f"vectored_{lead}"] = (vectored["at_p50_m"][lead], vectored["at_p50_m"][lead] is not None
                                          and vectored["at_p50_m"][lead] <= P_B61_VECTORED[lead] + P_VECTORED_MARGIN_M)
            checks[f"straight_{lead}"] = (straight["at_p50_m"][lead], straight["at_p50_m"][lead] is not None
                                          and straight["at_p50_m"][lead] <= P_B61_STRAIGHT[lead] + P_STRAIGHT_MARGIN_M)
        seeds[seed] = {"readings": {k: v[0] for k, v in checks.items()}, "passes": all(v[1] for v in checks.values())}
    return {"gate": "P-open-loop", "b61": {"vectored": P_B61_VECTORED, "straight": P_B61_STRAIGHT},
            "seeds": seeds, "passes": all(s["passes"] for s in seeds.values())}


def gate_p_discrete_vs_continuous(discrete: Mapping[int, dict[str, Any]], continuous: Mapping[int, dict[str, Any]]) -> dict[str, Any]:
    """Protocol-A payloads of the discrete and the continuous prior, keyed by seed: the discrete
    must not be worse than the continuous beyond the seed line on vectored ADE and established,
    on both seeds; a tie is the discrete's (it has the mask and the multi-aircraft interface)."""
    _two_seeds(discrete, "P")
    _two_seeds(continuous, "P")
    _require_protocol(discrete, "A", "P-discrete-vs-continuous")
    _require_protocol(continuous, "A", "P-discrete-vs-continuous")
    seeds = {}
    for seed in sorted(set(discrete) & set(continuous)):
        d, c = _cell(discrete[seed], STRATUM_VECTORED), _cell(continuous[seed], STRATUM_VECTORED)
        d_all, c_all = discrete[seed]["strata"]["all"], continuous[seed]["strata"]["all"]
        seeds[seed] = {
            "discrete_vectored_ade_mean_m": d["ade_mean_m"], "continuous_vectored_ade_mean_m": c["ade_mean_m"],
            "discrete_established": d_all["established_share"], "continuous_established": c_all["established_share"],
            "not_worse": (d["ade_mean_m"] <= c["ade_mean_m"] + SEED_LINE_ADE_M
                          and d_all["established_share"] >= c_all["established_share"] - SEED_LINE_ESTABLISHED),
        }
    if len(seeds) < 2:
        raise ValueError("the discrete and continuous arms share fewer than two seeds")
    return {"gate": "P-discrete-vs-continuous", "seeds": seeds, "passes": all(s["not_worse"] for s in seeds.values())}


def gate_e(lockstep_a: Mapping[int, dict[str, Any]]) -> dict[str, Any]:
    """Protocol-A payloads keyed by seed: the target (v2 G3) and the progress line."""
    _two_seeds(lockstep_a, "E")
    _require_protocol(lockstep_a, "A", "E")
    seeds = {}
    for seed, payload in lockstep_a.items():
        vectored, straight = _cell(payload, STRATUM_VECTORED), _cell(payload, STRATUM_STRAIGHT_IN)
        pooled = payload["strata"]["all"]
        target = (vectored["ade_mean_m"] <= E_VECTORED_ADE_M and straight["ade_mean_m"] <= E_STRAIGHT_ADE_M
                  and pooled["fully_flyable_share"] >= E_FLYABLE and pooled["established_share"] >= E_ESTABLISHED)
        progress = None
        if seed in E_PROGRESS_VECTORED_ADE_M:
            progress = (vectored["ade_mean_m"] <= E_PROGRESS_VECTORED_ADE_M[seed] - E_PROGRESS_ADE_MARGIN_M
                        and pooled["established_share"] >= E_PROGRESS_ESTABLISHED[seed] + E_PROGRESS_ESTABLISHED_MARGIN)
        seeds[seed] = {"vectored_ade_mean_m": vectored["ade_mean_m"], "straight_ade_mean_m": straight["ade_mean_m"],
                       "fully_flyable": pooled["fully_flyable_share"], "established": pooled["established_share"],
                       "target": target, "progress": progress}
    # the progress line exists for the seeds v2 §12.1 ran (1337 / 2024); a seed without one
    # reads "not applicable", never "failed"
    applicable = {seed: s["progress"] for seed, s in seeds.items() if s["progress"] is not None}
    return {"gate": "E", "seeds": seeds, "target": all(s["target"] for s in seeds.values()),
            "progress": all(applicable.values()) if applicable else None, "progress_seeds": sorted(applicable)}


def gate_s(candidates: Mapping[float, Mapping[int, dict[str, Any]]]) -> dict[str, Any]:
    """``{segment_s: {seed: protocol-A payload}}`` for the candidates that passed T: the best
    vectored ADE mean and established on both seeds; disagreement → established; a tie within
    the seed line → the longer segment."""
    if len(candidates) < 2:
        raise ValueError("gate S compares at least two segment lengths")
    readings = {}
    for segment, by_seed in candidates.items():
        _two_seeds(by_seed, "S")
        _require_protocol(by_seed, "A", "S")
        readings[segment] = {
            "vectored_ade_mean_m": {seed: _cell(p, STRATUM_VECTORED)["ade_mean_m"] for seed, p in by_seed.items()},
            "established": {seed: p["strata"]["all"]["established_share"] for seed, p in by_seed.items()},
        }
    seeds = sorted(set.intersection(*(set(r["established"]) for r in readings.values())))
    if len(seeds) < 2:
        raise ValueError("the candidates share fewer than two seeds")

    def best_by(metric: str, lower_is_better: bool) -> float | None:
        winners = []
        for seed in seeds:
            values = {seg: r[metric][seed] for seg, r in readings.items()}
            best = min(values, key=values.get) if lower_is_better else max(values, key=values.get)
            winners.append(best)
        return winners[0] if len(set(winners)) == 1 else None

    by_ade, by_established = best_by("vectored_ade_mean_m", True), best_by("established", False)
    chosen = by_ade if by_ade is not None and by_ade == by_established else by_established
    tie_note = None
    if chosen is not None:
        # a tie within the seed line on BOTH metrics against a longer candidate → the longer one
        for segment in sorted(candidates, reverse=True):
            if segment <= chosen:
                break
            close_ade = all(readings[segment]["vectored_ade_mean_m"][s] <= readings[chosen]["vectored_ade_mean_m"][s] + SEED_LINE_ADE_M for s in seeds)
            close_est = all(readings[segment]["established"][s] >= readings[chosen]["established"][s] - SEED_LINE_ESTABLISHED for s in seeds)
            if close_ade and close_est:
                tie_note, chosen = f"{segment:g} s ties {chosen:g} s within the seed line; the longer segment wins", segment
                break
    return {"gate": "S", "readings": readings, "best_by_ade": by_ade, "best_by_established": by_established,
            "selected_segment_s": chosen, "note": tie_note,
            "tie_rule": "a longer candidate within the seed line on both metrics against the FIRST chosen wins; "
                        "the longest such candidate is taken (ties are judged against the original, not chained)"}


def gate_t(readout_payload: dict[str, Any]) -> dict[str, Any]:
    """The readout artefact's own verdict (`manoeuvre/readout.gate_t`), re-read."""
    return readout_payload["gate_t"]


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
    Gate B1 (stage B's truth-token upper bound, §5.2.2): at least one metric beyond the line on
    both seeds and fully flyable ≥ the floor — row B without the not-worse clause, because a
    truth token that buys one thing at the price of another still carries information the
    prior is worth training for; row B, read on the prior's token, is where the price counts."""
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
    "E_ESTABLISHED", "E_FLYABLE", "E_STRAIGHT_ADE_M", "E_VECTORED_ADE_M", "P_B61_STRAIGHT", "P_B61_VECTORED",
    "RELATIVE_METRICS", "RELATIVE_PRIMARIES",
    "SEED_LINE_ADE_M", "SEED_LINE_ESTABLISHED", "X_ESTABLISHED_RATIO", "X_FLYABLE", "X_STRAIGHT_ADE_M",
    "X_VECTORED_ADE_M", "GRID_FLYABLE_FLOOR", "GRID_SEED_LINE_QUANTILE", "cell_name", "cell_reading", "gate_e", "gate_grid",
    "gate_p_discrete_vs_continuous", "gate_p_open_loop", "gate_relative", "gate_s", "gate_t", "gate_x",
]
