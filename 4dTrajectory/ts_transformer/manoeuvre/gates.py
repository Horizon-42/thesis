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

from typing import Any, Mapping

from ts_transformer.data.approach_difficulty import STRATUM_STRAIGHT_IN, STRATUM_VECTORED

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


__all__ = [
    "E_ESTABLISHED", "E_FLYABLE", "E_STRAIGHT_ADE_M", "E_VECTORED_ADE_M", "P_B61_STRAIGHT", "P_B61_VECTORED",
    "SEED_LINE_ADE_M", "SEED_LINE_ESTABLISHED", "X_ESTABLISHED_RATIO", "X_FLYABLE", "X_STRAIGHT_ADE_M",
    "X_VECTORED_ADE_M", "gate_e", "gate_p_discrete_vs_continuous", "gate_p_open_loop", "gate_s", "gate_t", "gate_x",
]
