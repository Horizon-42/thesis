"""M2: a block of arrivals — a landing schedule, then each aircraft flown to its slot, in slot order (design §5.6).

1. ETA: each aircraft's free-time solve without traffic (the shortest feasible IAF): its threshold time.
2. Schedule: first come first served by ETA, each on its recorded runway, placed at the earliest time the
   FAA minima allow (``runway_schedule.earliest_time``) against the slots already placed AND the recorded
   landings of every arrival outside the scheduled set (they are frozen: they fly their records). The
   minima are timed at ONE speed for the block — the slowest target speed of its aircraft
   (``faa_separation`` takes one; the slowest gives the longest time minima). A slot is a controlled time of
   arrival (CTA); "slot order" is this placement order.
3. Slot order: each aircraft flies to its CTA — a fixed-time solve, warm-started from its ETA solve on the same
   IAF — through the M1 loop. Its traffic: the replays of the aircraft before it and the records of every
   arrival outside the scheduled set; an aircraft whose slot solve fails flies its record for the ones after
   it. An aircraft never sees the ones after it. A CTA beyond ``max_duration`` is refused.
4. The block is judged once more, each flown aircraft against all the others: the losses it answers for,
   the losses with it that it does not answer for, and the background near it.
"""

from __future__ import annotations

import bisect
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import scenario_optimization as so
from flight_scenarios import FlightScenario
from trajectory_data_process.harvest.utc import parse_iso_utc_s
from traffic_job_files import PHASE_ETA, PHASE_SCHEDULE, phase_flying

from . import rules
from .check import FlownTrack, check, make_window
from .loop import BaselineFailed, LoopSettings, failed_sidecar, fly_in_traffic
from .scene import RecordedFlight, Traffic

def no_progress(done: int, total: int, flight_key: str) -> None:
    """The default ``on_progress`` of :func:`fly_block`: nobody is watching."""


def no_phase(phase: str) -> None:
    """The default ``on_phase`` of :func:`fly_block`: nobody is watching."""


#: The schema of an M2 sidecar: the M1 sidecar (``loop.TRAFFIC_RECORD_SCHEMA``) plus ``slot`` and
#: ``block_final``.
BLOCK_RECORD_SCHEMA = "optimization-traffic-block-v3"


@dataclass(frozen=True)
class Flown:
    """One aircraft of a block after its slot solve; when it failed, ``result`` is None and ``sidecar`` is its
    failed sidecar (``loop.failed_sidecar``: the reason and the solves it timed)."""

    scenario: FlightScenario
    result: Any
    sidecar: dict[str, Any] | None
    error: str | None


def replay_as_traffic(own: RecordedFlight, t0_utc_s: float, samples) -> RecordedFlight:
    """An optimized aircraft's replay as traffic for the aircraft after it (its record's identity and type)."""
    return RecordedFlight(own.flight_key, own.runway, own.typecode,
                          t0_utc_s + np.array([s.t for s in samples]), np.array([s.lat for s in samples]),
                          np.array([s.lon for s in samples]), np.array([s.alt for s in samples]))


def _t0(scenario: FlightScenario) -> float:
    return parse_iso_utc_s(scenario.source["entry_time_utc"])


def place(arrivals: list, frozen: list, rule_set) -> list:
    """``runway_schedule.schedule`` for one runway per arrival, with ``frozen`` slots placed first (the
    function there has no such argument): each arrival, first come first served, at the earliest time
    every minimum allows against everything placed before it."""
    timeline = sorted(frozen, key=lambda s: (rule_set.approach_s(s), s.key))
    slots = []
    for arrival in rules.fcfs_by_eta(arrivals, 1.0):
        (runway, eta), = arrival.etas.items()
        slot = rules.Slot(arrival.key, runway, rules.earliest_time(timeline, runway, arrival.category, eta, rule_set),
                          eta, arrival.category)
        slots.append(slot)
        bisect.insort(timeline, slot, key=lambda s: (rule_set.approach_s(s), s.key))
    return slots


def fly_block(
    scenarios: list[FlightScenario],
    traffic: Traffic,
    *,
    procedure_root: str | Path,
    settings: LoopSettings,
    max_duration: float,
    rollout_dt_s: float,
    solve_options: dict[str, Any],
    on_progress: Callable[[int, int, str], None] = no_progress,
    on_phase: Callable[[str], None] = no_phase,
) -> tuple[list[Flown], dict[str, Any]]:
    """The block flown (one entry per scenario, in slot order, then the ETA failures) and its summary.

    ``on_progress(done, total, flight_key)`` is called once per aircraft when its record is settled, with the
    aircraft's ``flight_key``: right after its ETA solve for one whose ETA solve fails, else right after its
    slot solve (in slot order). ``total`` is the number of scenarios, so the last call has ``done == total``.

    ``on_phase(phase)`` says what the block is doing (``traffic_job_files.PHASE_*``): the ETA solves, the schedule, then
    "optimizing k of n" before the k-th of the ``n`` slots is flown."""
    done, total = 0, len(scenarios)

    def settled(flight_key: str) -> None:
        nonlocal done
        done += 1
        on_progress(done, total, flight_key)

    on_phase(PHASE_ETA)
    etas, eta_solves, eta_errors, eta_timed = {}, {}, {}, {}       # eta_timed: each aircraft's timed ETA solves
    for s in scenarios:
        key = s.source["flight_key"]
        try:
            target, paths, aircraft, vmin = so.iaf_setup(s, procedure_root)
        except ValueError as exc:
            eta_errors[key], eta_timed[key] = str(exc).partition("\n")[0], []
            settled(key)
            continue
        try:
            best = so.shortest_iaf_solve(s, target, paths, aircraft, vmin, kind="eta", max_duration=max_duration,
                                         **solve_options)
        except so.IafSearchFailed as exc:
            eta_errors[key], eta_timed[key] = str(exc).partition("\n")[0], [a.to_json() for a in exc.attempts]
            settled(key)
            continue
        except ValueError as exc:              # before any solve (the paths' ordering)
            eta_errors[key], eta_timed[key] = str(exc).partition("\n")[0], []
            settled(key)
            continue
        etas[key], eta_solves[key] = _t0(s) + best.final_time, best
        eta_timed[key] = [a.to_json() for a in best.attempts]
    by_key = {s.source["flight_key"]: s for s in scenarios}
    records = [f for f in traffic.flights if f.flight_key not in etas]
    flown: list[Flown] = []
    slots = []
    on_phase(PHASE_SCHEDULE)
    speed = min((by_key[k].target.V for k in etas), default=None)
    if speed is not None:
        rule_set = rules.separation(traffic.runway_targets, speed)
        arrivals = [rules.Arrival(k, {by_key[k].source["runway"]: eta}, {by_key[k].source["runway"]: 0.0},
                                  rules.category(traffic.flight(k).typecode)) for k, eta in etas.items()]
        frozen = [rules.Slot(f.flight_key, f.runway, f.end_utc_s, f.end_utc_s, rules.category(f.typecode))
                  for f in records]
        slots = place(arrivals, frozen, rule_set)

    replays: list[RecordedFlight] = []
    for k, slot in enumerate(slots, start=1):
        on_phase(phase_flying(k, len(slots)))
        s, own = by_key[slot.key], traffic.flight(slot.key)
        duration = slot.time_s - _t0(s)
        try:
            if duration > max_duration:
                raise BaselineFailed(f"CTA {duration:.0f} s after the start, beyond max_duration {max_duration:.0f} s",
                                     solves=[])
            result, sidecar = fly_in_traffic(
                s, Traffic(traffic.airport, (own, *records, *replays), traffic.runway_targets),
                procedure_root=procedure_root, settings=settings, max_duration=max_duration,
                rollout_dt_s=rollout_dt_s, solve_options=solve_options,
                fixed_duration_s=duration, warm=eta_solves[slot.key])
        except BaselineFailed as exc:
            records.append(own)                          # it flies its record for the aircraft after it
            reason = f"BaselineFailed: slot solve (delay {slot.delay_s:.1f} s): {exc}"
            flown.append(Flown(s, None, failed_sidecar(slot.key, reason, eta_timed[slot.key] + exc.solves), reason))
            settled(slot.key)
            continue
        sidecar["schema"] = BLOCK_RECORD_SCHEMA
        sidecar["solves"] = eta_timed[slot.key] + sidecar["solves"]    # its ETA solves first
        sidecar["slot"] = {"eta_utc_s": slot.eta_s, "cta_utc_s": slot.time_s, "delay_s": slot.delay_s}
        replays.append(replay_as_traffic(own, _t0(s), result.simulator_states))
        flown.append(Flown(s, result, sidecar, None))
        settled(slot.key)
    flown += [Flown(by_key[k], None, failed_sidecar(k, f"BaselineFailed: ETA solve: {e}", eta_timed[k]),
                    f"BaselineFailed: ETA solve: {e}") for k, e in eta_errors.items()]

    # the block once more: each flown aircraft against all the others (replays and records)
    final = {}
    for f in (f for f in flown if f.result is not None):
        key = f.scenario.source["flight_key"]
        others = [r for r in replays if r.flight_key != key]
        window = make_window(f.scenario, Traffic(traffic.airport, (traffic.flight(key), *records, *others),
                                                 traffic.runway_targets),
                             procedure_root=procedure_root, horizon_s=max_duration)
        track = FlownTrack.from_samples(f.result.simulator_states)
        final[key] = {}
        for reading in (settings.reading, rules.IFR):
            found = check(window, track, reading=reading, step_s=settings.step_s)
            # distinct loss instants, as every other loss count (the census, the loop's rounds)
            final[key][reading] = {"answered": len({c.t_s for c in found.conflicts if c.responsible}),
                                   "not_answered": len({c.t_s for c in found.conflicts if not c.responsible}),
                                   "background": found.background}
        f.sidecar["block_final"] = final[key]
    summary = {
        "aircraft": len(scenarios),
        "scheduled": len(slots),
        "eta_failed": len(eta_errors),
        "slot_failed": sum(1 for f in flown if f.error and f.error.startswith("BaselineFailed: slot")),
        "schedule_speed_mps": speed,
        "slots": [{"flight_key": sl.key, "runway": sl.runway, "eta_utc_s": sl.eta_s, "cta_utc_s": sl.time_s,
                   "delay_s": sl.delay_s} for sl in slots],
        "outcomes": {f.scenario.source["flight_key"]: (f.error if f.result is None else f.sidecar["outcome"]) for f in flown},
        "final_losses": final,
    }
    return flown, summary
