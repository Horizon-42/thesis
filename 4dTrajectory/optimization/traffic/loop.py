"""M1: one optimized aircraft in recorded traffic (design §5.4).

The commanded scenario is solved as the constrained batch solves it (the shortest feasible IAF); its
replay is judged against the recorded traffic; each loss it answers for becomes rows (``rows``); the
same IAF is solved again from the last solution, until no such loss is left or ``max_rounds`` re-solves
are spent. A loss that comes back at the same instant (rows hold only at the nodes; between them, or
through replay drift, the record can still be short) asks for twice its margin in the next round. A recorded aircraft already in loss at the window start gets no rows
until its first instant without a loss (MD10). The rows against one aircraft keep one branch for the whole
window (``rows.branch_for``; vertical becomes horizontal when an in-trail loss with it appears).
The window keeps the solve with the fewest counted loss instants (the user's decision, 2026-10-06, MD14; the
earliest on a tie): a re-solve can make it worse, and the last solve is not kept for being last.
A re-solve that fails is tried once more with the other branch (the user's decision, 2026-10-05, MD13)
for each aircraft with a position loss in this round that permits one; the side of a vertical branch
comes from those losses (Claude's reading: an aircraft whose rows already hold keeps its branch). When
the retry fails too, the window ends with the last good solve as its record.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

import scenario_optimization as so
from flight_scenarios import FlightScenario
from scenario_replay import ScenarioOptimization

from . import rules
from .check import Check, Conflict, FlownTrack, Window, check, make_window
from .rows import Branch, branch_for, extra_rows, other_branch, row_spec
from .scene import Traffic

TRAFFIC_RECORD_SCHEMA = "optimization-traffic-v3"
#: The sidecar of a scenario that has no record (its baseline failed): its reason and its timed solves.
TRAFFIC_FAILED_SCHEMA = "optimization-traffic-failed-v1"
SEPARATED_AT_BASELINE, SEPARATED, UNRESOLVED, SOLVE_FAILED, WAKE_AT_FIXED_TIME = (
    "separated_at_baseline", "separated", "unresolved", "solve_failed", "wake_at_fixed_time")


class BaselineFailed(ValueError):
    """The window's baseline solve failed: no record; ``solves`` are the solves it timed (``SolveTime.to_json``)."""

    def __init__(self, message: str, solves: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.solves = solves


def failed_sidecar(flight_key: str, reason: str, solves: list[dict[str, Any]] | None) -> dict[str, Any]:
    """The sidecar of a scenario without a record: why, and the solves it timed (the timing experiment counts
    them, design §8.5); ``solves`` None when the failure lost them (an error outside the solve, a block that
    raised) — counted apart, never as 0 s."""
    return {"schema": TRAFFIC_FAILED_SCHEMA, "flight_key": flight_key, "reason": reason, "solves": solves}


@dataclass(frozen=True)
class LoopSettings:
    """The loop's parameters (design MD6–MD8; start values until step T5 measures them)."""

    step_s: float = 1.0               # MD6: the judge's check step (on UTC multiples)
    row_window_s: float = 60.0        # MD7 W: a loss gets rows on the nodes within this of it
    margin: float = 0.01              # MD7 κ: a row asks for the minimum × (1 + κ)
    max_rounds: int = 5               # MD8 K_max: re-solves before "unresolved"
    reading: str = rules.VISUAL       # MD2: the reading that makes rows; IFR is reported beside it


def free_from(baseline: Check, count: int) -> list[float]:
    """Per recorded aircraft, the first instant from which its losses get rows: 0, or — for one in
    loss at the first check instant — its first instant without a loss (MD10)."""
    times: list[set[float]] = [set() for _ in range(count)]
    for c in baseline.conflicts:
        times[c.other].add(c.t_s)
    first = float(baseline.times_s[0])
    return [next((float(t) for t in baseline.times_s if float(t) not in ts), float("inf")) if first in ts else 0.0
            for ts in times]


def _losses(found: Check, window: Window, active: set) -> list[list[Any]]:
    """``[t, other, kind, required, distance, vertical, responsible, rowed]`` per loss (``rowed``: it
    counts for the loop — the commanded aircraft answers for it and MD10 does not exclude it)."""
    return [[round(c.t_s, 3), window.recorded[c.other].flight_key, c.kind, round(c.required_m, 1),
             round(c.distance_m, 1), round(c.vertical_m, 1), c.responsible, c in active] for c in found.conflicts]


def _branches(active: list[Conflict], branches: dict[int, Branch]) -> dict[int, Branch]:
    """The branch per aircraft after this round: kept once chosen; vertical becomes horizontal when an
    in-trail loss with that aircraft appears."""
    out = dict(branches)
    by_other: dict[int, list[Conflict]] = {}
    for c in active:
        if c.kind != rules.AT_THRESHOLD:
            by_other.setdefault(c.other, []).append(c)
    for other, group in by_other.items():
        chosen = branch_for(group)
        if other not in out or any(c.kind == rules.IN_TRAIL for c in group):
            out[other] = chosen
    return out


def fly_in_traffic(
    scenario: FlightScenario,
    traffic: Traffic,
    *,
    procedure_root: str | Path,
    settings: LoopSettings = LoopSettings(),
    max_duration: float,
    rollout_dt_s: float,
    solve_options: dict[str, Any],
    fixed_duration_s: float | None = None,
    warm: Any = None,
) -> tuple[ScenarioOptimization, dict[str, Any]]:
    """The commanded scenario flown in its traffic: its record (the kept solve, MD14) and the
    traffic sidecar (outcome, rounds with their rows, losses by reading). ``solve_options`` go to
    ``so.solve_iaf``. M2 passes ``fixed_duration_s`` (a flight to a CTA: every solve is fixed-time, so
    a wake-at-the-threshold loss cannot be rowed — it ends the window as ``wake_at_fixed_time``) and
    ``warm`` (the ETA solve: the baseline flies its IAF from its solution). Raises
    :class:`BaselineFailed` when the baseline itself fails."""
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    solve_kw = {**solve_options, "fixed_duration_s": fixed_duration_s}
    solves: list[dict[str, Any]] = []          # every solve_iaf call, timed (design §8.5)

    def record(solve):
        return so.iaf_result(solve, scenario, aircraft, target=target, candidates=len(paths),
                             rollout_dt_s=rollout_dt_s, selection="shortestPath")

    try:
        target, paths, aircraft, min_speed_ms = so.iaf_setup(scenario, procedure_root)
    except ValueError as exc:
        raise BaselineFailed(_first_line(exc), solves=solves) from exc
    if warm is None:
        try:
            best = so.shortest_iaf_solve(scenario, target, paths, aircraft, min_speed_ms, kind="baseline",
                                         max_duration=max_duration, **solve_kw)
        except so.IafSearchFailed as exc:
            raise BaselineFailed(_first_line(exc), solves=[a.to_json() for a in exc.attempts]) from exc
        except ValueError as exc:              # before any solve (the paths' ordering)
            raise BaselineFailed(_first_line(exc), solves=[]) from exc
        solves += [a.to_json() for a in best.attempts]
    else:
        took: list = []
        try:
            with so.timing("slot", so.iaf_name(warm.pc), took):
                best = so.solve_iaf(warm.pc, scenario, target, aircraft, min_speed_ms, max_duration=max_duration,
                                    initial_guess=warm.decision_vector, **solve_kw)
        except ValueError as exc:
            raise BaselineFailed(_first_line(exc), solves=[t.to_json() for t in took]) from exc
        finally:
            solves += [t.to_json() for t in took]
    try:
        result = record(best)
    except ValueError as exc:
        raise BaselineFailed(_first_line(exc), solves=solves) from exc
    window = make_window(scenario, traffic, procedure_root=procedure_root, horizon_s=max_duration)

    def solve(last, rowed, doublings, branches, rows_field):
        """One re-solve of the IAF with the rows of ``rowed`` under ``branches``, warm from ``last``; its
        rows go to ``rounds[-1][rows_field]``, its timed solve to ``solves`` (kind ``resolve`` or ``retry``)."""
        specs = [row_spec(c, window, margin=settings.margin * 2 ** doublings[key],
                          branch=None if c.kind == rules.AT_THRESHOLD else branches[c.other])
                 for key, c in rowed.items()]
        rounds[-1][rows_field] = [asdict(s) for s in specs]
        took: list = []
        try:
            with so.timing("resolve" if rows_field == "rows_next" else "retry", so.iaf_name(last.pc), took):
                solved = so.solve_iaf(
                    last.pc, scenario, target, aircraft, min_speed_ms, max_duration=max_duration,
                    extra_rows=extra_rows(specs, window, last.dense_times, row_window_s=settings.row_window_s),
                    initial_guess=last.decision_vector, **solve_kw)
        finally:
            solves.extend({**t.to_json(), "round": len(rounds) - 1} for t in took)
        return solved, record(solved)
    flown = FlownTrack.from_samples(result.simulator_states)
    judged = check(window, flown, reading=settings.reading, step_s=settings.step_s)
    starts = free_from(judged, len(window.recorded))
    ifr_baseline = check(window, flown, reading=rules.IFR, step_s=settings.step_s)
    rounds: list[dict[str, Any]] = []
    rowed: dict[tuple[int, float, str], Conflict] = {}   # (other, instant, kind) -> the loss, first-seen order
    doublings: dict[tuple[int, float, str], int] = {}    # how many rounds that loss came back after its rows
    branches: dict[int, Branch] = {}
    best_kept = None                                     # (counted loss instants, round, record, flown track)
    while True:
        active = [c for c in judged.conflicts if c.responsible and c.t_s >= starts[c.other]]
        if fixed_duration_s is not None:          # the landing time is fixed: a wake loss is no row
            active_rowable = [c for c in active if c.kind != rules.AT_THRESHOLD]
        else:
            active_rowable = active
        counted = len({c.t_s for c in active})
        # every instant it answers for, MD10 not applied: the census's count of a record (design §10.6), for display
        answered = len({c.t_s for c in judged.conflicts if c.responsible})
        rounds.append({"final_time_s": result.final_time_s, "counted_loss_instants": counted,
                       "answered_loss_instants": answered,
                       "losses": _losses(judged, window, set(active_rowable)), "background_losses": judged.background})
        if best_kept is None or counted < best_kept[0]:   # MD14: the fewest counted losses, the earliest on a tie
            best_kept = (counted, len(rounds) - 1, result, flown)
        if not active:
            outcome = SEPARATED_AT_BASELINE if len(rounds) == 1 else SEPARATED
            break
        if not active_rowable:
            outcome = WAKE_AT_FIXED_TIME
            break
        if len(rounds) > settings.max_rounds:
            outcome = UNRESOLVED
            break
        branches = _branches(active_rowable, branches)
        for key, c in {(c.other, c.t_s, c.kind): c for c in active_rowable}.items():
            doublings[key] = doublings[key] + 1 if key in rowed else 0
            rowed.setdefault(key, c)
        try:
            best, next_result = solve(best, rowed, doublings, branches, "rows_next")
        except ValueError as exc:
            rounds[-1]["next_solve_error"] = str(exc).partition("\n")[0][:200]
            now = [c for c in active_rowable if c.kind != rules.AT_THRESHOLD]
            in_trail = {c.other for c in rowed.values() if c.kind == rules.IN_TRAIL}   # rowed in any round
            flipped = {k: b for k, b in ((k, other_branch(branches[k], [c for c in now if c.other == k]))
                                         for k in sorted({c.other for c in now} - in_trail)) if b is not None}
            if not flipped:
                outcome = SOLVE_FAILED
                break
            rounds[-1]["retried_branches"] = {window.recorded[k].flight_key: asdict(b) for k, b in flipped.items()}
            try:
                best, next_result = solve(best, rowed, doublings, {**branches, **flipped}, "rows_retry")
            except ValueError as retry_exc:
                rounds[-1]["retry_error"] = str(retry_exc).partition("\n")[0][:200]
                outcome = SOLVE_FAILED
                break
            branches = {**branches, **flipped}
        result = next_result
        flown = FlownTrack.from_samples(result.simulator_states)
        judged = check(window, flown, reading=settings.reading, step_s=settings.step_s)
    _counted, kept_round, result, flown = best_kept
    ifr_final = check(window, flown, reading=rules.IFR, step_s=settings.step_s)
    lats = [flown.lat_deg] + [f.lat_deg for f in window.recorded]
    sidecar = {
        "schema": TRAFFIC_RECORD_SCHEMA,
        "flight_key": window.flight_key,
        "settings": asdict(settings),
        "outcome": outcome,
        "rounds": rounds,
        "kept_round": kept_round,                       # MD14: the round whose solve is the record
        "branches": {window.recorded[k].flight_key: asdict(b) for k, b in branches.items()},
        "ifr_losses": {"baseline": _losses(ifr_baseline, window, set()),
                       "final": _losses(ifr_final, window, set())},
        "recorded": [f.flight_key for f in window.recorded],
        # MD10: the first instant that gets rows; null — in loss at every instant of the baseline
        "starts_in_loss": {f.flight_key: (t if t != float("inf") else None)
                           for f, t in zip(window.recorded, starts) if t > 0.0},
        "uncategorised_types": list(window.uncategorised_types),
        "runways_without_faf": sorted(r.ident for r in window.runways.values() if r.d_faf_m is None),
        "frame_east_error_max": window.frame.east_scale_error(np.concatenate(lats)),
        "solves": solves,
        # the whole window: its solves (an M2 aircraft's ETA solves are not in it), the replays and the judge
        "windowWallS": round(time.perf_counter() - started_wall, 3),
        "windowCpuS": round(time.process_time() - started_cpu, 3),
    }
    return result, sidecar


def _first_line(exc: Exception) -> str:
    return str(exc).partition("\n")[0]
