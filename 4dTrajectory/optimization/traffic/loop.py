"""M1: one optimized aircraft in recorded traffic (design §5.4).

The commanded scenario is solved as the constrained batch solves it (the shortest feasible IAF); its
replay is judged against the recorded traffic; each loss it answers for becomes rows (``rows``); the
same IAF is solved again from the last solution, until no such loss is left or ``max_rounds`` re-solves
are spent. A loss that comes back at the same instant (rows hold only at the nodes; between them, or
through replay drift, the record can still be short) asks for twice its margin in the next round. A recorded aircraft already in loss at the window start gets no rows
until its first instant without a loss (MD10). The rows against one aircraft keep one branch for the whole
window (``rows.branch_for``; vertical becomes horizontal when an in-trail loss with it appears).
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

TRAFFIC_RECORD_SCHEMA = "optimization-traffic-v2"
SEPARATED_AT_BASELINE, SEPARATED, UNRESOLVED, SOLVE_FAILED, WAKE_AT_FIXED_TIME = (
    "separated_at_baseline", "separated", "unresolved", "solve_failed", "wake_at_fixed_time")


class BaselineFailed(ValueError):
    """The window's baseline solve failed: no record, no traffic sidecar."""


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
    """The commanded scenario flown in its traffic: its record (the last solve that succeeded) and the
    traffic sidecar (outcome, rounds with their rows, losses by reading). ``solve_options`` go to
    ``so.solve_iaf``. M2 passes ``fixed_duration_s`` (a flight to a CTA: every solve is fixed-time, so
    a wake-at-the-threshold loss cannot be rowed — it ends the window as ``wake_at_fixed_time``) and
    ``warm`` (the ETA solve: the baseline flies its IAF from its solution). Raises
    :class:`BaselineFailed` when the baseline itself fails."""
    solve_kw = {**solve_options, "fixed_duration_s": fixed_duration_s}
    def record(solve):
        return so.iaf_result(solve, scenario, aircraft, target=target, candidates=len(paths),
                             rollout_dt_s=rollout_dt_s, selection="shortestPath")

    try:
        target, paths, aircraft, min_speed_ms = so.iaf_setup(scenario, procedure_root)
        if warm is None:
            best = so.shortest_iaf_solve(scenario, target, paths, aircraft, min_speed_ms,
                                         max_duration=max_duration, **solve_kw)
        else:
            best = so.solve_iaf(warm.pc, scenario, target, aircraft, min_speed_ms, max_duration=max_duration,
                                initial_guess=warm.decision_vector, **solve_kw)
        result = record(best)
    except ValueError as exc:
        raise BaselineFailed(str(exc).partition("\n")[0]) from exc
    window = make_window(scenario, traffic, procedure_root=procedure_root, horizon_s=max_duration)

    def solve(last, rowed, doublings, branches, rows_field):
        """One re-solve of the IAF with the rows of ``rowed`` under ``branches``, warm from ``last``; its
        rows go to ``rounds[-1][rows_field]``, its time to ``rounds[-1]["next_solve_s"]`` (one per attempt)."""
        specs = [row_spec(c, window, margin=settings.margin * 2 ** doublings[key],
                          branch=None if c.kind == rules.AT_THRESHOLD else branches[c.other])
                 for key, c in rowed.items()]
        rounds[-1][rows_field] = [asdict(s) for s in specs]
        started = time.perf_counter()
        try:
            solved = so.solve_iaf(
                last.pc, scenario, target, aircraft, min_speed_ms, max_duration=max_duration,
                extra_rows=extra_rows(specs, window, last.dense_times, row_window_s=settings.row_window_s),
                initial_guess=last.decision_vector, **solve_kw)
            return solved, record(solved)
        finally:
            rounds[-1].setdefault("next_solve_s", []).append(round(time.perf_counter() - started, 3))
    flown = FlownTrack.from_samples(result.simulator_states)
    judged = check(window, flown, reading=settings.reading, step_s=settings.step_s)
    starts = free_from(judged, len(window.recorded))
    ifr_baseline = check(window, flown, reading=rules.IFR, step_s=settings.step_s)
    rounds: list[dict[str, Any]] = []
    rowed: dict[tuple[int, float, str], Conflict] = {}   # (other, instant, kind) -> the loss, first-seen order
    doublings: dict[tuple[int, float, str], int] = {}    # how many rounds that loss came back after its rows
    branches: dict[int, Branch] = {}
    while True:
        active = [c for c in judged.conflicts if c.responsible and c.t_s >= starts[c.other]]
        if fixed_duration_s is not None:          # the landing time is fixed: a wake loss is no row
            active_rowable = [c for c in active if c.kind != rules.AT_THRESHOLD]
        else:
            active_rowable = active
        rounds.append({"final_time_s": result.final_time_s,
                       "losses": _losses(judged, window, set(active_rowable)), "background_losses": judged.background})
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
    ifr_final = check(window, flown, reading=rules.IFR, step_s=settings.step_s)
    lats = [flown.lat_deg] + [f.lat_deg for f in window.recorded]
    sidecar = {
        "schema": TRAFFIC_RECORD_SCHEMA,
        "flight_key": window.flight_key,
        "settings": asdict(settings),
        "outcome": outcome,
        "rounds": rounds,
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
    }
    return result, sidecar
