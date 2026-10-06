"""M2's control flow on a fake solver: the schedule, slot order, who sees whom, failures (no NLP, no data)."""

from types import SimpleNamespace

import numpy as np
import pytest
from geokit import NM_M

from traffic import block, rules
from traffic.check import Check, Conflict
from traffic.loop import BaselineFailed
from traffic.scene import RecordedFlight, Traffic

T0 = 1_800_000_000.0
SPEED = 70.0
TARGETS = {"09": {"lat": 35.0, "lon": -78.0, "course_deg": 90.0}}


def _iso(t):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(t, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _record(key, start, end):
    t = np.linspace(start, end, 5)
    return RecordedFlight(key, "09", None, t, np.full(5, 35.0), np.full(5, -78.1), np.full(5, 900.0))


def _scenario(key, start):
    return SimpleNamespace(source={"flight_key": key, "entry_time_utc": _iso(start), "runway": "09"},
                           target=SimpleNamespace(V=SPEED))


def _run(monkeypatch, *, eta_s, fail_slot=(), fail_eta=(), out_landing=T0 - 400.0, final_conflicts=(),
         on_progress=block.no_progress):
    """Fly A (start T0) and B (start T0 + 5); ``eta_s`` their minimum flight times; OUT a record outside
    the block landing at ``out_landing`` (far before the block by default)."""
    scenarios = [_scenario("A", T0), _scenario("B", T0 + 5.0)]
    traffic = Traffic("KXXX", (_record("A", T0, T0 + 300), _record("B", T0 + 5, T0 + 305),
                               _record("OUT", out_landing - 250, out_landing)), TARGETS)
    seen = {}

    def shortest(scenario, *_a, **_k):
        if scenario.source["flight_key"] in fail_eta:
            raise ValueError("all 1 IAF(s) infeasible")
        return SimpleNamespace(final_time=eta_s[scenario.source["flight_key"]])

    def fly(scenario, window, *, solve_options, fixed_duration_s, warm, **_k):
        key = scenario.source["flight_key"]
        seen[key] = {f.flight_key: f for f in window.flights}
        seen[key + ":duration"] = fixed_duration_s
        seen[key + ":warm"] = warm.final_time
        if key in fail_slot:
            raise BaselineFailed("fixed-time solve failed")
        states = [SimpleNamespace(t=t, lat=35.0, lon=-78.05, alt=500.0) for t in (0.0, fixed_duration_s)]
        return SimpleNamespace(simulator_states=states), {"outcome": "separated_at_baseline"}

    def final_window(scenario, traffic, **_k):
        seen["final:" + scenario.source["flight_key"]] = {f.flight_key for f in traffic.flights}

    monkeypatch.setattr(block, "so", SimpleNamespace(iaf_setup=lambda s, r: ("target", ["IAF"], "aircraft", 60.0),
                                                     shortest_iaf_solve=shortest))
    monkeypatch.setattr(block, "fly_in_traffic", fly)
    monkeypatch.setattr(block, "make_window", final_window)
    monkeypatch.setattr(block, "FlownTrack", SimpleNamespace(from_samples=lambda s: None))
    monkeypatch.setattr(block, "check", lambda *a, **k: Check(np.zeros(1), tuple(final_conflicts), 2))
    flown, summary = block.fly_block(scenarios, traffic, procedure_root="root", settings=block.LoopSettings(),
                                     max_duration=2000.0, rollout_dt_s=0.5, solve_options={}, on_progress=on_progress)
    return flown, summary, seen


def test_the_schedule_spaces_the_follower_and_each_flies_to_its_slot(monkeypatch):
    flown, summary, seen = _run(monkeypatch, eta_s={"A": 200.0, "B": 200.0})     # ETAs 5 s apart
    gap = 3.0 * NM_M / SPEED                                                        # radar minimum, one runway
    assert [s["flight_key"] for s in summary["slots"]] == ["A", "B"]
    assert summary["slots"][1]["cta_utc_s"] == pytest.approx(T0 + 200.0 + gap)
    assert seen["A:duration"] == pytest.approx(200.0) and seen["B:duration"] == pytest.approx(195.0 + gap)
    assert seen["A:warm"] == 200.0                                                  # warm from its ETA solve
    # A sees the records outside the block, never B; B sees A's REPLAY (not its record) and the outside record
    assert set(seen["A"]) == {"A", "OUT"}
    assert set(seen["B"]) == {"A", "B", "OUT"}
    assert seen["B"]["A"].t_utc_s[-1] == pytest.approx(T0 + 200.0)                # the replay ends at A's slot
    assert summary["schedule_speed_mps"] == SPEED
    assert [f.error for f in flown] == [None, None]
    assert flown[1].sidecar["slot"]["delay_s"] == pytest.approx(gap - 5.0)


def test_an_aircraft_whose_slot_solve_fails_flies_its_record_for_the_ones_after_it(monkeypatch):
    flown, summary, seen = _run(monkeypatch, eta_s={"A": 200.0, "B": 200.0}, fail_slot={"A"})
    assert seen["B"]["A"].t_utc_s[-1] == pytest.approx(T0 + 300.0)                # A's record, not a replay
    assert flown[0].error.startswith("BaselineFailed") and flown[1].error is None
    assert set(summary["final_losses"]) == {"B"}


def test_an_eta_failure_leaves_the_aircraft_out_of_the_schedule(monkeypatch):
    flown, summary, seen = _run(monkeypatch, eta_s={"A": 200.0, "B": 200.0}, fail_eta={"A"})
    assert [s["flight_key"] for s in summary["slots"]] == ["B"] and summary["eta_failed"] == 1
    assert set(seen["B"]) == {"A", "B", "OUT"} and seen["B"]["A"].t_utc_s[-1] == pytest.approx(T0 + 300.0)
    assert flown[-1].scenario.source["flight_key"] == "A" and "ETA solve" in flown[-1].error


def test_a_record_landing_near_the_block_is_frozen_into_the_schedule(monkeypatch):
    _flown, summary, _seen = _run(monkeypatch, eta_s={"A": 200.0, "B": 200.0}, out_landing=T0 + 190.0)
    gap = 3.0 * NM_M / SPEED
    assert summary["slots"][0]["cta_utc_s"] == pytest.approx(T0 + 190.0 + gap)    # A waits behind OUT's landing


def test_the_final_check_judges_each_flown_aircraft_against_all_others(monkeypatch):
    loss = dict(kind=rules.RADAR_OR_VERTICAL, required_m=5556.0, distance_m=4000.0, vertical_m=0.0, above=True)
    conflicts = [Conflict(1.0, 0, responsible=True, **loss), Conflict(2.0, 0, responsible=False, **loss)]
    _flown, summary, seen = _run(monkeypatch, eta_s={"A": 200.0, "B": 200.0}, final_conflicts=conflicts)
    assert seen["final:A"] == {"A", "B", "OUT"} and seen["final:B"] == {"A", "B", "OUT"}
    assert summary["final_losses"]["A"][rules.VISUAL] == {"answered": 1, "not_answered": 1, "background": 2}


def test_progress_is_reported_once_per_aircraft_as_each_is_settled(monkeypatch):
    calls = []
    _run(monkeypatch, eta_s={"A": 200.0, "B": 200.0}, on_progress=lambda *call: calls.append(call))
    assert calls == [(1, 2, "A"), (2, 2, "B")]                                      # in slot order


def test_progress_counts_an_eta_failure_and_a_failed_slot_like_any_other_aircraft(monkeypatch):
    calls = []
    _run(monkeypatch, eta_s={"A": 200.0, "B": 200.0}, fail_eta={"A"}, fail_slot={"B"},
         on_progress=lambda *call: calls.append(call))
    # A settles at its failed ETA solve, before the slots; B at its failed slot solve; the last call is done == total
    assert calls == [(1, 2, "A"), (2, 2, "B")]


def test_an_empty_block_has_an_empty_schedule():
    flown, summary = block.fly_block([], Traffic("KXXX", (), TARGETS), procedure_root="root",
                                     settings=block.LoopSettings(), max_duration=2000.0, rollout_dt_s=0.5,
                                     solve_options={})
    assert flown == [] and summary["scheduled"] == 0 and summary["schedule_speed_mps"] is None
