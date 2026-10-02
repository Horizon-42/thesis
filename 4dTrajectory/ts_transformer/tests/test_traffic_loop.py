"""The scene's closed loop (`experiments/traffic_loop`, multi-aircraft design §2.2, §3.4 layer 1) on hand-built
aircraft, and what the step-5 runner (`experiments/traffic_labelled`) reads off a flight's own end and off its runs."""

import dataclasses
import json
import math

import numpy as np
import pytest
from geokit import NM_M

from ts_transformer.inference.runway_schedule import CWT_ON_APPROACH_NM, DEPENDENT, Separation

STEP_S = 2.0
#: One runway "R", and a dependent parallel pair "L1"/"L2" (L2 0.6 NM left of L1, 1.0 NM diagonal), all landing east
#: with their thresholds at the approach clock's origin.
SEPARATION = Separation(same_nm=3.0, speed_mps=70.0, relations={frozenset(("L1", "L2")): DEPENDENT},
                        spacing_nm={frozenset(("L1", "L2")): 0.6}, right_nm={("L1", "L2"): -0.6, ("L2", "L1"): 0.6},
                        diagonal_nm={frozenset(("L1", "L2")): 1.0}, wake_nm=CWT_ON_APPROACH_NM,
                        along_nm={"R": 0.0, "L1": 0.0, "L2": 0.0})


def _controlled(key: str, t0: float, along, *, runway: str = "R", n: float = 0.0, established=True,
                category: str | None = "F", outcome: str = "landed", landing_s: float | None = None,
                angle: float = 0.0):
    """A controlled aircraft from ``t0`` (a step), one step a value of ``along`` (its east, on the approach clock),
    flying its course on its centreline unless said otherwise."""
    from ts_transformer.experiments.traffic_loop import Controlled
    from ts_transformer.prior.scene import Presence

    along = np.asarray(along, dtype=np.float64)
    count = len(along)
    times = t0 + STEP_S * np.arange(count)
    seen = Presence(key, "KXXX", runway, math.inf if landing_s is None else landing_s, True, times, np.abs(along))
    return Controlled(seen, times, along.copy(), np.full(count, n), np.full(count, 500.0), (runway,) * count, along,
                      np.full(count, angle), np.zeros(count), np.full(count, 70.0),
                      np.broadcast_to(np.asarray(established), (count,)).copy(), category, outcome, landing_s)


def _replayed(key: str, times, along, *, landing_s: float, runway: str = "R", captured_s: float = 0.0,
              category: str | None = "F"):
    from ts_transformer.experiments.traffic_census import Track
    from ts_transformer.prior.scene import Presence

    times, along = np.asarray(times, dtype=np.float64), np.asarray(along, dtype=np.float64)
    seen = Presence(key, "KXXX", runway, landing_s, False, times, np.abs(along))
    count = len(times)
    return Track(seen, along.copy(), np.zeros(count), np.full(count, 500.0), along, np.zeros(count),
                 np.zeros(count), np.full(count, 70.0), captured_s, "raw", category)


def _loop(reading: str = "visual"):
    from ts_transformer.experiments.traffic_loop import Loop

    return Loop(SEPARATION, reading, STEP_S)


T = np.arange(0.0, 100.0 + 1e-9, STEP_S)


def test_the_aircraft_behind_is_ended_at_its_first_loss_and_never_judged_again():
    # L and F both established on R, F 4 km behind (under 3 NM) at every step
    leader = _controlled("L", 0.0, -8_000.0 + 70.0 * T)
    follower = _controlled("F", 0.0, -12_000.0 + 70.0 * T)
    run = _loop().run([leader, follower], [])
    assert run.ended == {"F": {"t_s": 0.0, "kind": "in_trail", "relation": "same", "with": "L", "with_controlled": True}}
    assert len(run.episodes) == 1
    episode = run.episodes[0]
    assert (episode["pair"], episode["steps"], episode["ended"]) == (["F", "L"], 1, ["F"])
    assert episode["responsible"] == [{"key": "F", "controlled": True}]
    assert episode["closest_m"] == pytest.approx(4_000.0) and episode["required_m"] == pytest.approx(3.0 * NM_M)
    assert run.steps_judged == 1                              # after the end only L is left
    assert run.scene_seconds == pytest.approx(100.0)


def test_a_replayed_aircraft_is_never_ended_and_a_controlled_one_behind_it_is():
    replayed = _replayed("P", T, -8_000.0 + 70.0 * T, landing_s=8_000.0 / 70.0)
    behind = _controlled("F", 0.0, -12_000.0 + 70.0 * T)
    run = _loop().run([behind], [replayed])
    assert run.ended["F"]["with"] == "P" and run.ended["F"]["with_controlled"] is False
    # the other way round: the replayed one behind answers every step and nobody is ended
    ahead = _controlled("L", 0.0, -8_000.0 + 70.0 * T)
    trailing = _replayed("Q", T, -12_000.0 + 70.0 * T, landing_s=12_000.0 / 70.0)
    run = _loop().run([ahead], [trailing])
    assert run.ended == {} and len(run.episodes) == 1
    assert run.episodes[0]["steps"] == len(T) and run.episodes[0]["responsible"] == [{"key": "Q", "controlled": False}]


def test_two_aircraft_still_vectored_both_answer_and_both_are_ended():
    one = _controlled("A", 0.0, np.full(len(T), -15_000.0), established=False)
    two = _controlled("B", 0.0, np.full(len(T), -15_000.0), n=3_000.0, established=False)
    run = _loop().run([one, two], [])
    assert set(run.ended) == {"A", "B"} and {e["kind"] for e in run.ended.values()} == {"radar_or_vertical"}
    assert sorted(run.episodes[0]["ended"]) == ["A", "B"]


def test_a_parallel_turn_on_is_free_under_the_visual_reading_and_ended_under_ifr():
    # A established on L1; B on L2 (0.6 NM left), turning in at 20° on its own side, 500 m behind: 1.2 km apart
    a = _controlled("A", 0.0, np.full(len(T), -8_000.0), runway="L1")
    b = _controlled("B", 0.0, np.full(len(T), -8_500.0), runway="L2", n=0.6 * NM_M, established=False, angle=20.0)
    assert _loop("visual").run([a, b], []).ended == {}
    ifr = _loop("ifr").run([a, b], [])
    assert ifr.ended == {"B": {"t_s": 0.0, "kind": "radar_or_vertical", "relation": "dependent", "with": "A",
                               "with_controlled": True}}


def test_a_landing_checks_the_wake_minimum_behind_it_between_two_steps():
    """L (F) crosses its threshold at 101 s; I (I) is established 3.5 NM behind it on the approach clock: clear of the
    in-trail minimum (3 NM: TBL 5-5-1 sets none for F → I) at every step, under TBL 5-5-2's 4 NM at the landing."""
    gap = 3.5 * NM_M
    t_leader = np.arange(0.0, 100.0 + 1e-9, STEP_S)
    leader = _controlled("L", 0.0, -70.0 * (101.0 - t_leader), landing_s=101.0)
    t_follower = np.arange(0.0, 160.0 + 1e-9, STEP_S)
    follower = _controlled("I", 0.0, -gap - 70.0 * (101.0 - t_follower), category="I", outcome="timeout")
    run = _loop().run([leader, follower], [])
    assert run.episodes == [] and run.landings_checked == 1
    assert len(run.at_threshold) == 1
    assert run.at_threshold[0]["gap_m"] == pytest.approx(gap) and run.at_threshold[0]["required_m"] == pytest.approx(
        4.0 * NM_M)
    assert run.ended == {"I": {"t_s": 101.0, "kind": "at_threshold", "relation": "same", "with": "L",
                               "with_controlled": True}}
    # an F behind instead: TBL 5-5-2 sets no wake minimum for F behind F
    clear = _controlled("I", 0.0, -gap - 70.0 * (101.0 - t_follower), outcome="timeout")
    assert _loop().run([leader, clear], []).ended == {}


def test_an_aircraft_ended_before_its_landing_is_not_checked_at_it():
    # P (replayed) is 2 km ahead of L at every step, so L is ended at 0 s; L's landing at 101 s is not checked, P's is
    t = np.arange(0.0, 100.0 + 1e-9, STEP_S)
    ahead = _replayed("P", t, 2_000.0 - 70.0 * (101.0 - t), landing_s=72.0)
    leader = _controlled("L", 0.0, -70.0 * (101.0 - t), landing_s=101.0)
    run = _loop().run([leader], [ahead])
    assert run.ended["L"]["t_s"] == 0.0 and run.landings_checked == 1


def test_a_landing_never_takes_a_follower_already_ended():
    """The first landing test's I, ended at 0 s by R (replayed, established, 500 m ahead of it): at L's landing it is
    gone, so the landing finds no follower."""
    gap = 3.5 * NM_M
    t_leader = np.arange(0.0, 100.0 + 1e-9, STEP_S)
    leader = _controlled("L", 0.0, -70.0 * (101.0 - t_leader), landing_s=101.0)
    t_follower = np.arange(0.0, 160.0 + 1e-9, STEP_S)
    follower = _controlled("I", 0.0, -gap - 70.0 * (101.0 - t_follower), category="I", outcome="timeout")
    ahead = _replayed("R", [0.0, 0.5], [-gap - 70.0 * 101.0 + 500.0] * 2, landing_s=0.6)
    run = _loop().run([leader, follower], [ahead])
    assert run.ended["I"]["t_s"] == 0.0 and run.ended["I"]["kind"] == "in_trail"
    assert run.at_threshold == []


def test_a_landing_after_the_leaders_last_step_is_checked_against_a_replayed_aircraft_still_in_the_air():
    """L's last step is 100 s, its crossing 100.5 s; Q (replayed, I) has rows on the steps 0–102 s, past the crossing: the
    landing is checked at step 102 against Q between its rows, and Q is its follower."""
    gap = 3.5 * NM_M
    t = np.arange(0.0, 100.0 + 1e-9, STEP_S)
    leader = _controlled("L", 0.0, -70.0 * (100.5 - t), landing_s=100.5)
    rows = np.arange(0.0, 102.0 + 1e-9, STEP_S)
    follower = _replayed("Q", rows, -gap - 70.0 * (100.5 - rows), landing_s=100.5 + gap / 70.0, category="I")
    run = _loop().run([leader], [follower])
    assert [(a["leader"], a["follower"], a["follower_controlled"]) for a in run.at_threshold] == [("L", "Q", False)]
    assert run.at_threshold[0]["gap_m"] == pytest.approx(gap) and run.ended == {}


def test_a_landing_on_a_step_is_checked_before_that_steps_pairs():
    """L crosses exactly at the 100 s step: I, 3.5 NM behind it (TBL 5-5-2: 4 NM), is ended there before the pair check,
    so R — vectored, 500 m from I and on the scene only at that step — has no pair left to lose separation with."""
    gap = 3.5 * NM_M
    t_leader = np.arange(0.0, 98.0 + 1e-9, STEP_S)
    leader = _controlled("L", 0.0, -70.0 * (100.0 - t_leader), landing_s=100.0)
    t_follower = np.arange(0.0, 160.0 + 1e-9, STEP_S)
    follower = _controlled("I", 0.0, -gap - 70.0 * (100.0 - t_follower), category="I", outcome="timeout")
    joining = _replayed("R", [99.5, 100.5], [-gap + 500.0] * 2, landing_s=200.0, captured_s=math.inf)
    run = _loop().run([leader, follower], [joining])
    assert run.ended["I"] == {"t_s": 100.0, "kind": "at_threshold", "relation": "same", "with": "L",
                              "with_controlled": True}
    assert run.episodes == []


def _judging(reading: str = "visual"):
    from ts_transformer.experiments.traffic_loop import Judging, Run

    run = Run(reading)
    return Judging(SEPARATION, reading, STEP_S, run), run


def test_an_ended_aircraft_still_in_the_scene_is_avoided_and_never_ended_again():
    """Design §9 item 29: P, ended earlier, flies on as a passive aircraft — F 4 km behind it is ended for it (as behind a
    replayed one), and P answering for a loss with L ahead of it is recorded, not ended."""
    ahead = _controlled("L", 0.0, -4_000.0 + 70.0 * T)
    passive = _controlled("P", 0.0, -8_000.0 + 70.0 * T)
    behind = _controlled("F", 0.0, -12_000.0 + 70.0 * T)
    judging, run = _judging()
    ended = judging.step(0.0, [ahead, behind], [], passive=[passive])
    assert ended == run.ended == {"F": {"t_s": 0.0, "kind": "in_trail", "relation": "same", "with": "P",
                                       "with_controlled": False}}
    by_pair = {tuple(e["pair"]): e for e in run.episodes}
    assert by_pair[("L", "P")]["responsible"] == [{"key": "P", "controlled": False}]
    assert by_pair[("L", "P")]["ended"] == [] and by_pair[("F", "P")]["ended"] == ["F"]
    # a step with fewer than two aircraft is not judged
    assert judging.step(2.0, [], [], passive=[passive]) == {} and run.steps_judged == 1


def test_a_passive_aircraft_landing_checks_the_wake_behind_it_and_is_not_ended_as_a_follower():
    """The landing test's geometry: a passive L's landing ends the controlled I behind it; with I passive and L
    controlled, the shortfall is recorded and nobody is ended."""
    gap = 3.5 * NM_M
    t_leader = np.arange(0.0, 100.0 + 1e-9, STEP_S)
    leader = _controlled("L", 0.0, -70.0 * (101.0 - t_leader), landing_s=101.0)
    t_follower = np.arange(0.0, 160.0 + 1e-9, STEP_S)
    follower = _controlled("I", 0.0, -gap - 70.0 * (101.0 - t_follower), category="I", outcome="timeout")
    judging, run = _judging()
    judging.landing(101.0, leader, [follower], [])
    assert run.ended["I"]["kind"] == "at_threshold" and run.at_threshold[0]["follower_controlled"]
    judging, run = _judging()
    judging.landing(101.0, leader, [], [], passive=[follower])
    assert run.ended == {} and [(a["follower"], a["follower_controlled"]) for a in run.at_threshold] == [("I", False)]
    assert run.landings_checked == 1


def test_segments_chain_by_span_and_the_scene_time_is_theirs():
    from ts_transformer.experiments.traffic_loop import segments

    early = _controlled("A", 0.0, np.full(11, -30_000.0))                 # 0–20 s
    overlapping = _controlled("B", 20.0, np.full(11, -60_000.0))          # 20–40 s: joins A's (its first ≤ A's last)
    late = _controlled("C", 100.0, np.full(6, -30_000.0))                 # 100–110 s
    assert [[a.key for a in s] for s in segments([late, overlapping, early])] == [["A", "B"], ["C"]]
    run = _loop().run([early, overlapping, late], [])
    assert run.scene_seconds == pytest.approx(40.0 + 10.0) and run.steps_judged == 1    # A and B together at 20 s


def test_an_aircraft_is_in_the_loop_once_and_a_reading_is_one_of_the_two():
    one = _controlled("A", 0.0, np.full(3, -30_000.0))
    with pytest.raises(ValueError, match="twice"):
        _loop().run([one], [_replayed("A", [0.0, 2.0], [-30_000.0, -30_000.0], landing_s=500.0)])
    with pytest.raises(ValueError, match="reading"):          # the judge refuses it at the first pair
        _loop("vfr").run([one, _controlled("B", 0.0, np.full(3, -60_000.0))], [])


def test_between_two_steps_the_position_is_interpolated_and_the_rest_is_the_earlier_steps():
    from ts_transformer.experiments.traffic_loop import between

    a = _controlled("A", 10.0, [-1_000.0, -900.0, -800.0], established=[False, True, True], angle=5.0)
    moment = between([a], 11.5, STEP_S)
    assert moment.along_m[0] == pytest.approx(-925.0) and moment.e_m[0] == pytest.approx(-925.0)
    assert not moment.established[0]                                       # 11.5 s is after the step at 10 s
    assert between([a], 14.0, STEP_S).along_m[0] == pytest.approx(-800.0)   # the last step: no later one


def _fixture_flight(entry: str):
    from ts_transformer.tests.support import fly_legs, instruction_flight

    legs = [(40, 0.0, 70.0, -3.0)]
    flight = instruction_flight(*fly_legs(legs, 90.0, 1_000.0, -300.0, 0.0))
    return dataclasses.replace(flight, entry_time_utc=entry)


def test_the_recorded_control_is_its_rows_on_the_loop_steps():
    """Its rows on the steps from 11:00:02: row r at 2 s + 2r s, its landing read off its rows on that clock, and it is
    established from the artefact's capture row."""
    from ts_transformer.experiments.traffic_loop import recorded
    from ts_transformer.prior.scene import presence, utc_s
    from ts_transformer.tests.support import instruction_airport

    flight = _fixture_flight("2026-06-01T11:00:02Z")
    geometry = instruction_airport()
    seen = presence(flight, 30, geometry, STEP_S)
    control = recorded(seen, flight, 12, 61.3, geometry, SEPARATION_09, "F", STEP_S)
    start = utc_s("2026-06-01T11:00:00Z")
    # epoch seconds: compared from the start (pytest.approx is relative: 1e-6 of 1.8e9 s is half an hour)
    assert control.times_s[0] - start == pytest.approx(2.0) and len(control.times_s) == 30
    assert np.allclose(np.diff(control.times_s), STEP_S)
    assert control.landing_s - start == pytest.approx(2.0 + 61.3)
    assert control.established.tolist() == [False] * 12 + [True] * 18
    # east of the threshold at the origin, course 090: the approach clock is the east, on the centreline, on the course
    assert np.allclose(control.along_m, flight.e_m[:30]) and np.allclose(control.right_of_course_m, -flight.n_m[:30])
    assert np.allclose(control.track_minus_course_deg, 0.0) and control.runway == ("09",) * 30


SEPARATION_09 = Separation(same_nm=3.0, speed_mps=70.0, along_nm={"09": 0.0}, wake_nm=CWT_ON_APPROACH_NM)


def test_a_flown_aircraft_starts_at_its_first_rows_step_and_lands_on_that_clock():
    from ts_transformer.experiments.traffic_loop import flown
    from ts_transformer.prior.scene import presence, utc_s
    from ts_transformer.tests.support import instruction_airport

    flight = _fixture_flight("2026-06-01T11:00:00Z")
    geometry = instruction_airport()
    seen = presence(flight, 30, geometry, STEP_S)
    e = np.array([-5_000.0, -4_860.0, -4_720.0])
    aircraft = flown(seen, STEP_S, e, np.array([10.0, 5.0, 0.0]), np.full(3, 400.0), np.array([95.0, 92.0, 90.0]),
                     np.full(3, 70.0), np.array([False, False, True]), "09", geometry, SEPARATION_09, "F", "landed", 67.4)
    start = utc_s("2026-06-01T11:00:00Z")
    assert (aircraft.times_s - start).tolist() == pytest.approx([0.0, 2.0, 4.0])
    assert aircraft.landing_s - start == pytest.approx(67.4)
    assert np.allclose(aircraft.along_m, e) and np.allclose(aircraft.right_of_course_m, [-10.0, -5.0, 0.0])
    assert np.allclose(aircraft.track_minus_course_deg, [5.0, 2.0, 0.0])
    assert aircraft.established.tolist() == [False, False, True]
    assert np.allclose(aircraft.along_speed_mps, 70.0 * np.cos(np.radians([5.0, 2.0, 0.0])))


def _flown_states(cycles: int, captured_from: int):
    """One flight flown east along runway 09's centreline at 70 m/s from 8 km out, ``cycles`` 1 s cycles, the executor
    captured after cycle ``captured_from`` and every one after."""
    import torch

    from ts_transformer.autopilot.executor import Flown
    from ts_transformer.autopilot.frame import ALT, GAMMA, LAT, LON, MASS, PSI, SPEED
    from ts_transformer.tests.support import instruction_airport

    geometry = instruction_airport()
    e = -8_000.0 + 70.0 * np.arange(cycles + 1)
    lat, lon = geometry.frame.latlon_from_horizontal(e, np.zeros(len(e)))
    states = np.zeros((1, cycles + 1, 7))
    for column, values in ((LAT, lat), (LON, lon), (ALT, 400.0), (SPEED, 70.0), (PSI, 0.0), (GAMMA, 0.0),
                           (MASS, 60_000.0)):
        states[0, :, column] = values
    captured = torch.zeros((1, cycles), dtype=torch.bool)
    captured[0, captured_from:] = True
    zeros = torch.zeros((1, cycles, 3), dtype=torch.float64)
    return Flown(states=torch.as_tensor(states), commands=zeros, wanted=zeros, limits={}, modes={"captured": captured},
                 done_cycle=torch.tensor([cycles - 1]), sentence_s=torch.zeros((1, cycles), dtype=torch.float64),
                 cycle_s=1.0), e


def test_a_flown_flight_goes_on_the_steps_to_its_own_end_with_the_capture_of_the_cycle_before_each_state():
    from ts_transformer.autopilot.judge import Outcome
    from ts_transformer.experiments.traffic_labelled import flown_aircraft
    from ts_transformer.prior.scene import presence, utc_s
    from ts_transformer.tests.support import instruction_airport

    geometry = instruction_airport()
    flight = _fixture_flight("2026-06-01T11:00:00Z")
    seen = presence(flight, 30, geometry, STEP_S)
    start = utc_s("2026-06-01T11:00:00Z")
    states, e = _flown_states(20, captured_from=6)          # captured from state row 7 on
    landed = flown_aircraft(states, 0, Outcome("landed", 11, {"at_row": 10.4}, {}), -1, seen, "09", geometry,
                            SEPARATION_09, "F", STEP_S)
    # crossed between rows 10 and 11: in to row 10, one state every other row; row 6 is before the capture, row 8 after
    assert landed.outcome == "landed" and (landed.times_s - start).tolist() == pytest.approx(np.arange(0.0, 11.0, 2.0))
    assert landed.established.tolist() == [False, False, False, False, True, True]
    assert np.allclose(landed.along_m, e[0:11:2]) and landed.landing_s - start == pytest.approx(10.4)
    # sunk below the edge at step 1's end (row 4), before the crossing: in to row 4, no landing
    stopped = flown_aircraft(states, 0, Outcome("landed", 11, {"at_row": 10.4}, {}), 1, seen, "09", geometry,
                             SEPARATION_09, "F", STEP_S)
    assert (stopped.outcome, len(stopped.times_s), stopped.landing_s) == ("below_glidepath", 3, None)
    # crossed, but off the runway: no landing
    off = flown_aircraft(states, 0, Outcome("crossed_off_runway", 11, {"at_row": 10.4}, {}), -1, seen, "09", geometry,
                         SEPARATION_09, "F", STEP_S)
    assert (off.outcome, len(off.times_s), off.landing_s) == ("crossed_off_runway", 6, None)


@pytest.mark.parametrize("outcome, end_row, stop_step, expected", [
    ("landed", 301, -1, ("landed", 300)),                  # read at the first row past the threshold: in to the one before
    ("landed", 301, 99, ("below_glidepath", 200)),         # sank at step 99's end (row 200), before the landing
    ("landed", 200, 99, ("landed", 199)),                  # the same row: the judge's outcome
    ("timeout", 450, -1, ("timeout", 450)),                # a timeout is in to its last row
    ("ground_contact", 90, 60, ("ground_contact", 89)),    # the stop after the outcome never counts
])
def test_a_flights_own_end_is_the_earlier_of_its_outcome_and_the_glidepath_stop(outcome, end_row, stop_step, expected):
    from ts_transformer.experiments.traffic_labelled import own_end

    assert own_end(outcome, end_row, stop_step, 2) == expected


def test_the_readout_counts_ended_flights_and_splits_pairs_and_ends_between_the_two_runs():
    from ts_transformer.experiments.traffic_labelled import compared, merged, summary
    from ts_transformer.experiments.traffic_loop import Run

    ended = {"t_s": 0.0, "kind": "in_trail", "relation": "same", "with": "P", "with_controlled": False}
    first = Run("visual", ended={"F": ended}, scene_seconds=1_800.0, steps_judged=10,
                episodes=[{"pair": ["F", "P"], "relation": "same", "kinds": ["in_trail"], "steps": 1, "min_ratio": 0.7,
                           "responsible": [{"key": "F", "controlled": True}], "ended": ["F"]}])
    second = Run("visual", scene_seconds=1_800.0, landings_checked=2,
                 episodes=[{"pair": ["G", "Q"], "relation": "single", "kinds": ["in_trail"], "steps": 3,
                            "min_ratio": 0.9, "responsible": [{"key": "Q", "controlled": False}], "ended": []}],
                 at_threshold=[{"follower_controlled": False}])
    pooled = merged([first, second])
    rows = {key: {"group": group, "outcome": outcome} for key, group, outcome in (
        ("F", "own dynamics", "landed"), ("G", "own dynamics", "landed"), ("H", "stand-in dynamics", "timeout"),
        ("K", "own dynamics", "below_glidepath"))}
    read = summary(pooled, rows)
    assert (read["flights"], read["ended"], read["ended_share"]) == (4, 1, 0.25)
    assert read["outcomes"] == {"lost_separation": 1, "landed": 1, "timeout": 1, "below_glidepath": 1}
    assert read["landed_share"] == 0.25 and read["episodes_between_two_replayed_aircraft"] == 0
    assert read["ended_by_group"]["own dynamics"] == {"flights": 3, "ended": 1, "share": pytest.approx(1 / 3)}
    assert read["ended_beside_a_replayed_aircraft"] == 1 and read["scene_hours"] == pytest.approx(1.0)
    assert (read["episodes_per_scene_hour"], read["episodes_only_a_replayed_aircraft_answers_for"]) == (2.0, 1)
    assert read["at_threshold"] == {"landings_checked": 2, "losses": 1, "followers_ended": 0}
    against = compared(first, second)
    assert against == {"pairs_with_a_loss": {"both": 0, "executor_only": 1, "recorded_only": 1},
                       "ended": {"both": 0, "executor_only": 1, "recorded_only": 0}}


def test_the_pass_line_gates_what_the_executor_adds_over_the_own_dynamics_flights():
    """Of 20 own-dynamics flights the executor's run ends 2 and the recorded control 1 (another): the executor adds
    5 points, over the 3 % line; a stand-in ended in the executor's run alone is not gated."""
    from ts_transformer.experiments.traffic_labelled import EXECUTOR, RECORDED, pass_line
    from ts_transformer.experiments.traffic_loop import Run

    ended = {"t_s": 0.0, "kind": "in_trail", "relation": "same", "with": "X", "with_controlled": True}
    rows = [{"key": f"F{k}", "group": "own dynamics"} for k in range(20)] + [{"key": "S", "group": "stand-in dynamics"}]
    runs = {(EXECUTOR, "visual"): Run("visual", ended={"F0": ended, "F1": ended, "S": ended}),
            (RECORDED, "visual"): Run("visual", ended={"F2": ended})}
    line = pass_line(runs, rows)
    assert line == {"flights": 20, "added_share": pytest.approx(0.05), "passes": False, "ended_share": pytest.approx(0.10),
                    "recorded_ended_share": pytest.approx(0.05), "ended_in_the_executor_run_alone": 2,
                    "ended_in_the_recorded_run_alone": 1}
    # the recorded control ends two others: 10 % ended in each run, so the executor adds nothing and the line passes
    runs[(RECORDED, "visual")] = Run("visual", ended={"F2": ended, "F3": ended})
    again = pass_line(runs, rows)
    assert (again["added_share"], again["passes"], again["ended_in_the_executor_run_alone"]) == (
        pytest.approx(0.0), True, 2)


def _labelled_artefact(tmp_path):
    """Four flights of one vectored approach onto the fixture's runway 09 in a tmp artefact, with a tmp arrivals
    manifest recorded as the one the signals were read from: "a", "b" entering 30 s after it (3 km behind while both
    are vectored at one level), "c" an hour later and "d" two hours later — all labelled."""
    from datetime import timedelta

    from ts_transformer.data.day_split import parse_utc
    from ts_transformer.instructions.artefact import (
        labeller_source_sha256, write_candidates, write_sentences, write_signals, write_spec,
    )
    from ts_transformer.instructions.labeller.read import read_flight
    from ts_transformer.io_utils import file_sha256
    from ts_transformer.tests.support import (
        fixture_days, fly_legs, instruction_airport, instruction_flight, instruction_spec,
    )

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"runway_targets": {"09": {"lat": 35.0, "lon": -78.0, "course_deg": 90.0}}}),
                        encoding="utf-8")
    legs = [(60, 0.0, 100.0, 0.0), (15, -6.0, 100.0, 0.0), (20, 0.0, 90.0, 0.0), (15, -6.0, 85.0, 0.0),
            (120, 0.0, 75.0, -75.0 * np.tan(np.radians(3.0)))]
    base = instruction_flight(*fly_legs(legs, 270.0, 1110.0, -400.0, 0.0))
    landing = parse_utc(base.landing_time_utc)

    def at(name: str, later_s: float):
        lands = landing + timedelta(seconds=later_s)
        enters = lands - timedelta(seconds=float(base.time_s[-1]) + 2.0)
        iso = "%Y-%m-%dT%H:%M:%S.%fZ"
        return dataclasses.replace(base, dataset_id=f"KXXX:{name}", entry_time_utc=enters.strftime(iso),
                                   landing_time_utc=lands.strftime("%Y-%m-%dT%H:%M:%SZ"))

    flights = [at("a", 0.0), at("b", 30.0), at("c", 3_600.0), at("d", 7_200.0)]
    directory = tmp_path / "artefact"
    directory.mkdir()
    write_signals(directory, {"train": flights},
                  {"counts": {"train": {"built_usable": 4}}, "test_days": {"flights_not_opened": 0},
                   "sources": [{"airport": "KXXX", "arrival_manifest_sha256": file_sha256(manifest)}]}, fixture_days())
    write_candidates(directory, {"KXXX": instruction_airport()})
    spec = instruction_spec()
    write_spec(directory, spec, {"n": 1}, {"labeller_source_sha256": labeller_source_sha256(),
                                           "git": {"head": "test", "dirty": False}})
    write_sentences(directory, "train", spec, [read_flight(f, instruction_airport(), spec) for f in flights],
                    [0, 1, 2, 3])
    return directory, manifest, flights


def test_the_runner_flies_the_labelled_words_ends_both_vectored_and_counts_the_one_it_cannot_fly(tmp_path, monkeypatch):
    """The fixture's physics stand in for the rebuilt flights (A320, the executor at the replay's settings, the fixture's
    final); "d" has no identified type, so it is replayed and counted. "a" and "b" meet 3 km apart at one level while
    both are still vectored, in both runs: both answer, both are ended at b's first step; "c", alone, lands."""
    from types import SimpleNamespace

    import torch

    from ts_transformer.autopilot import replay
    from ts_transformer.autopilot.flights import FlightInputs
    from ts_transformer.experiments import traffic_labelled
    from ts_transformer.instructions.words import Words
    from ts_transformer.prior.masks import PROCEDURE_ALTITUDES, ProcedureMasks
    from ts_transformer.prior.scene import utc_s
    from ts_transformer.tests.support import instruction_airport
    from ts_transformer.tests.test_autopilot import _params, _physics, vertical_paths
    from ts_transformer.tests.test_prior_procedure import _final

    directory, manifest, flights = _labelled_artefact(tmp_path)
    geometry = instruction_airport()

    def series(signals):
        typecode = None if signals.dataset_id == "KXXX:d" else "A320"
        return SimpleNamespace(signals=signals, scenario=SimpleNamespace(
            source={"resolved_typecode": typecode, "dynamics_typecode": typecode}, has_dynamics=True,
            initial=SimpleNamespace(m=62_000.0)))

    def inputs(items, *, device, anchor=0):
        rows = [_physics(item.signals, geometry)[0] for item in items]
        return FlightInputs(*(torch.cat([getattr(row, name) for row in rows]) for name in (
            "initial_state", "aero_params", "frame_params", "max_thrust_n")))

    from ts_transformer.instructions.artefact import load_spec

    words = Words(load_spec(directory))
    monkeypatch.setattr(replay, "open_executor", lambda executor, instructions: (
        _params(word_clock="track"), {"sha256": "test"}, words))
    monkeypatch.setattr(replay, "flight_inputs", inputs)
    monkeypatch.setattr(traffic_labelled, "rebuild_series", lambda d, items: [series(item) for item in items])
    monkeypatch.setattr(traffic_labelled, "published_vertical_paths", vertical_paths)
    monkeypatch.setattr(traffic_labelled, "arrival_manifest_path", lambda code: manifest)
    masks = ProcedureMasks((PROCEDURE_ALTITUDES,), {"KXXX": (_final(),)})
    monkeypatch.setattr(traffic_labelled, "ProcedureMasks", SimpleNamespace(build=lambda names, geometries: masks))
    out = tmp_path / "labelled"
    args = ["--instructions", str(directory), "--executor", str(tmp_path / "executor"), "--out", str(out)]
    assert traffic_labelled.main(args) == 0
    payload = json.loads((out / "labelled.json").read_text(encoding="utf-8"))
    assert payload["schema"] == traffic_labelled.SCHEMA and payload["check_reading"] == "visual"
    airport = payload["airports"]["KXXX"]
    assert airport["flights"] == {"with_a_sentence": 4, "flown": 3, "flown_by_group": {"own dynamics": 3},
                                  "not_flown_replayed": {"no identified type": 1}, "background_replayed": 0}
    assert airport["own_outcomes"] == {"landed": 3}
    rows = {row["key"]: row for row in payload["flights"]}
    assert set(rows) == {"KXXX:a", "KXXX:b", "KXXX:c"}                          # "d" is replayed, not a flown row
    first_b = float(np.floor(utc_s(flights[1].entry_time_utc) / 2.0 + 0.5) * 2.0)
    for mode in traffic_labelled.MODES:
        for reading in ("visual", "ifr"):
            a, b = rows["KXXX:a"]["ended"][mode][reading], rows["KXXX:b"]["ended"][mode][reading]
            assert (a["with"], b["with"]) == ("KXXX:b", "KXXX:a")
            assert a["kind"] == b["kind"] == "radar_or_vertical" and a["t_s"] == b["t_s"] == first_b
            assert rows["KXXX:c"]["ended"][mode][reading] is None
            run = airport["runs"][mode][reading]
            assert run["outcomes"] == {"lost_separation": 2, "landed": 1}
            assert run["at_threshold"]["landings_checked"] == 2                  # c's and d's: a and b never land
    # each landing counted from the flight's first step: the control's is its crossing read off its own rows
    from ts_transformer.autopilot.replay import observed_landing_s
    from ts_transformer.instructions.labeller.read import read_flight

    c = flights[2]
    crossing = observed_landing_s(c, read_flight(c, geometry, words.spec, words), geometry)
    landing = rows["KXXX:c"]["landing_after_first_step_s"]
    assert landing["recorded"] == pytest.approx(crossing) and abs(landing["executor"] - crossing) < 30.0
    assert airport["landing_minus_recorded_s"]["n"] == 3
    # both runs end a and b: the executor adds nothing, so the line passes whatever the share ended
    line = {"flights": 3, "added_share": pytest.approx(0.0), "passes": True, "ended_share": pytest.approx(2 / 3),
            "recorded_ended_share": pytest.approx(2 / 3), "ended_in_the_executor_run_alone": 0,
            "ended_in_the_recorded_run_alone": 0}
    assert airport["pass_line"] == line and payload["pass_line"] == {**line, "by_airport": {"KXXX": line}}
    assert airport["executor_against_recorded"]["visual"]["ended"] == {"both": 2, "executor_only": 0,
                                                                         "recorded_only": 0}
    with pytest.raises(SystemExit):                                                   # never overwritten
        traffic_labelled.main(args)
