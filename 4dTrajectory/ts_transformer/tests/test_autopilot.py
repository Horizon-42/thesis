"""The executor (`autopilot/`, design docs/two_tier/two_tier_design.md §5): conventions, words in force, the one-cycle
plant and the exact inverse, the laws, the judge and the replay, on synthetic states of a real airframe. (The tests of
the instruction-v3 executor are archived with it: `archive/two_tier_v3_2026_10/tests/test_autopilot.py`.)"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from aircraft.aero_params import aero_params_for_aircraft
from flight_scenarios.scenario import aircraft_for_code
from ts_transformer.autopilot import inverse
from ts_transformer.autopilot.flights import FlightInputs
from ts_transformer.autopilot.frame import AirportCharts, compass_deg, read_state, wrap180
from ts_transformer.autopilot.plant import Plant
from ts_transformer.autopilot.sentence import DistanceClock, Sentences, TimeClock, TrackClock
from ts_transformer.instructions.labeller.records import Instruction
from ts_transformer.instructions.words import (
    ALTITUDE, ANGLE, COLUMNS, HEADING, RUNWAY, RUNWAY_GO_AROUND, SPEED, UNCHANGED, Words, compass_from_math_rad,
    wrap180 as np_wrap180,
)
from ts_transformer.tests.support import fly_legs, instruction_airport, instruction_flight, instruction_spec as spec

F64 = torch.float64
CPU = torch.device("cpu")


# ---- conventions
def test_the_state_reads_as_the_words_read_a_flight():
    psi = torch.linspace(-7.0, 7.0, 101, dtype=F64)
    assert compass_deg(psi).numpy() == pytest.approx(compass_from_math_rad(psi.numpy()), abs=1e-9)
    angles = torch.linspace(-720.0, 720.0, 97, dtype=F64)
    assert wrap180(angles).numpy() == pytest.approx(np_wrap180(angles.numpy()), abs=1e-9)
    geometry = instruction_airport()
    charts = AirportCharts.of([geometry, geometry], dtype=F64, device=CPU)
    lat, lon = torch.tensor([35.1, 34.8], dtype=F64), torch.tensor([-78.3, -77.6], dtype=F64)
    e, n = charts.horizontal(lat, lon)
    want_e, want_n = geometry.frame.horizontal_from_latlon(lat.numpy(), lon.numpy())
    assert (e.numpy(), n.numpy()) == (pytest.approx(want_e, abs=1e-6), pytest.approx(want_n, abs=1e-6))
    state = torch.tensor([[35.1, -78.3, 900.0, 100.0, math.radians(45.0), math.radians(-3.0), 60000.0]], dtype=F64)
    k = read_state(state, AirportCharts.of([geometry], dtype=F64, device=CPU))
    assert float(k.track_deg) == pytest.approx(45.0) and float(k.height_m) == 900.0
    assert float(k.ground_speed_mps) == pytest.approx(100.0 * math.cos(math.radians(3.0)))


# ---- words in force, and the clocks they are said on
def _grid():
    words = Words(spec())
    grid = np.full((6, len(COLUMNS)), UNCHANGED, dtype=np.int64)
    grid[0] = [0, words.heading_index(180.0), words.altitude_index(1200.0), 0, words.speed_index(100.0)]
    grid[2, HEADING] = words.heading_index(90.0)            # issued at t = 4 s
    grid[3, ALTITUDE] = words.altitude_no_level_off         # issued at t = 6 s
    grid[3, ANGLE] = 3
    grid[4, RUNWAY] = RUNWAY_GO_AROUND                      # t = 8 s
    grid[5, RUNWAY] = 1                                     # t = 10 s: runway 1, which ends the go-around
    return grid


def test_each_word_takes_effect_when_it_is_said_and_stays_in_force():
    """§3.2: the runway in force R is the last candidate said, the go-around state G the runway column's last word being
    "go-around"; a heading word is its relative class (§3.3)."""
    one, words = spec(), Words(spec())
    sentences = Sentences([_grid()], words, step_s=2.0, device=CPU)
    at = [sentences.at(torch.tensor([float(t - t % 2)], dtype=F64)) for t in range(16)]
    relative = np.array([float(w.heading_rel_deg[0]) for w in at])
    assert (relative[:4] == 180.0).all() and (relative[4:] == 90.0).all()
    no_level_off = np.array([bool(w.no_level_off[0]) for w in at])
    assert not no_level_off[:6].any() and no_level_off[6:].all()
    assert float(at[0].altitude_m[0]) == 1200.0 and float(at[6].angle_deg[0]) == one.descent_angle_centres_deg[2]
    go_around = np.array([bool(w.go_around[0]) for w in at])
    assert not go_around[:8].any() and go_around[8:10].all() and not go_around[10:].any()
    assert [int(w.runway[0]) for w in at] == [0] * 10 + [1] * 6       # the go-around keeps R; the runway word sets it
    assert at[9].issued_step[0, RUNWAY] == 4 and at[10].issued_step[0, RUNWAY] == 5
    assert at[3].issued_step[0, HEADING] == 0 and at[4].issued_step[0, HEADING] == 2


def test_a_sentence_must_write_every_column_and_a_candidate_at_step_0():
    words = Words(spec())
    grid = _grid()
    grid[0, SPEED] = UNCHANGED
    with pytest.raises(ValueError, match="step 0 must write every column"):
        Sentences([grid], words, step_s=2.0, device=CPU)
    grid = _grid()
    grid[0, RUNWAY] = RUNWAY_GO_AROUND
    with pytest.raises(ValueError, match="a candidate in the runway column"):
        Sentences([grid], words, step_s=2.0, device=CPU)


def test_a_coarser_row_interval_holds_each_row_for_its_seconds():
    words = Words(spec())
    sentences = Sentences([_grid()], words, step_s=4.0, device=CPU)
    rows = [int(sentences.at(torch.tensor([float(t)], dtype=F64)).issued_step[0, HEADING]) for t in (0, 3.9, 8, 11.9)]
    assert rows == [0, 0, 2, 2]


def test_the_distance_clock_says_a_word_where_the_observed_aircraft_was_told_it():
    from ts_transformer.autopilot.frame import Kinematics

    e = np.arange(5) * 100.0
    clock = DistanceClock.of([e], [np.zeros(5)], 2.0, 1.0, device=CPU)

    def at(position):
        state = Kinematics(*(torch.tensor([value], dtype=F64) for value in (position, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
        return float(clock.now(0, state)[0])

    assert [at(100.0 * t) for t in range(7)] == pytest.approx([0.0, 2.0, 4.0, 6.0, 8.0, 9.0, 10.0])
    assert float(TimeClock(1.0).now(7, Kinematics(*(torch.zeros(1, dtype=F64) for _ in range(8))))[0]) == 7.0


def test_the_track_clock_follows_the_observed_track_forward_one_row_a_cycle_at_most():
    from ts_transformer.autopilot.frame import Kinematics

    clock = TrackClock.of([np.arange(6) * 100.0], [np.zeros(6)], 2.0, 1.0, device=CPU)

    def at(position):
        state = Kinematics(*(torch.tensor([value], dtype=F64) for value in (position, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
        return float(clock.now(0, state)[0])

    assert [at(x) for x in (0.0, 440.0, 460.0, 120.0, 900.0, 900.0, 900.0, 900.0)] == pytest.approx(
        [0.0, 2.0, 4.0, 4.0, 6.0, 8.0, 10.0, 11.0])


def test_a_rebuilt_flight_must_be_the_one_the_signals_were_read_from(monkeypatch):
    from ts_transformer.autopilot import flights

    stored = instruction_flight(*fly_legs([(20, 0.0, 90.0, 0.0)], 90.0, 900.0, -3000.0, 0.0))
    rebuilt = {"signals": stored}
    monkeypatch.setattr(flights, "signals_from_series", lambda series, geometry: rebuilt["signals"])
    flights.require_same_flight(object(), stored, instruction_airport())
    for change, message in ((dict(typecode="B738"), "typecode"), (dict(runway="27"), "runway"),
                            (dict(altitude_m=stored.altitude_m + 0.01), "altitude_m")):
        rebuilt["signals"] = replace(stored, **change)
        with pytest.raises(ValueError, match=message):
            flights.require_same_flight(object(), stored, instruction_airport())


def test_a_flight_is_rebuilt_without_reading_its_manifests_bytes(monkeypatch, tmp_path):
    """D21 (§9.2 #4): a rebuilt flight is the stored one row for row, whatever bytes its manifest has now."""
    import json

    from ts_transformer.autopilot import flights

    stored = instruction_flight(*fly_legs([(20, 0.0, 90.0, 0.0)], 90.0, 900.0, -3000.0, 0.0), dataset_id="KXXX:a")
    (tmp_path / "signals.json").write_text(json.dumps({
        "config": {}, "sources": [{"airport": "KXXX", "arrival_manifest_sha256": "0" * 64}]}), encoding="utf-8")
    monkeypatch.setattr(flights, "load_candidates", lambda directory: {"KXXX": instruction_airport()})
    monkeypatch.setattr(flights, "arrival_manifest_path", lambda airport: tmp_path / "moved.json")
    monkeypatch.setattr(flights, "load_flight_dicts", lambda manifests, **kwargs: [])
    series = SimpleNamespace(dataset_id="KXXX:a")
    monkeypatch.setattr(flights, "build_series", lambda flights_, config, row_start: ([series], None))
    checked = []
    monkeypatch.setattr(flights, "require_same_flight", lambda s, signals, geometry: checked.append(signals.dataset_id))
    assert flights.rebuild_series(tmp_path, [stored]) == [series] and checked == ["KXXX:a"]


# ---- the plant and the inverse
def _a320(states: list[list[float]]) -> tuple[FlightInputs, AirportCharts]:
    aircraft = aircraft_for_code("A320")
    aero = aero_params_for_aircraft(aircraft)
    rows = len(states)
    inputs = FlightInputs(
        initial_state=torch.tensor(states, dtype=F64),
        aero_params=torch.tensor([[aero.S, aero.Cl_max, aero.Cd0, aero.k, aero.stall_threshold, aero.k_stall]] * rows,
                                 dtype=F64),
        frame_params=torch.tensor([[35.0, -78.0, 100.0, 0.0]] * rows, dtype=F64),
        max_thrust_n=torch.full((rows,), float(aircraft.engine.max_thrust_total_n), dtype=F64),
    )
    return inputs, AirportCharts.of([instruction_airport()] * rows, dtype=F64, device=CPU)


def _state(track_deg: float, gamma_deg: float, speed: float = 100.0, height: float = 1500.0) -> list[float]:
    return [35.05, -78.05, height, speed, math.radians(90.0 - track_deg), math.radians(gamma_deg), 62000.0]


def _fly(inputs, charts, track_rate, gamma_rate, accel, previous_bank=None, *, bank_cap_deg=30.0,
         bank_rate_deg_s=1000.0):
    k0 = read_state(inputs.initial_state, charts)
    rows = len(inputs.initial_state)
    previous = torch.zeros(rows, dtype=F64) if previous_bank is None else previous_bank

    def column(value):
        return torch.as_tensor(value, dtype=F64).expand(rows).clone()

    att = inverse.attitude(k0, column(track_rate), column(gamma_rate), previous,
                           bank_cap_rad=math.radians(bank_cap_deg), bank_rate_rad_s=math.radians(bank_rate_deg_s),
                           cycle_s=1.0)
    thr = inverse.thrust(k0, column(accel), att.load_factor, inputs.aero_params, inputs.max_thrust_n)
    command = torch.stack((thr.fraction, att.bank_rad, att.load_factor), dim=1)
    k1 = read_state(Plant(inputs).step(inputs.initial_state, command, 1.0), charts)
    rates = (wrap180(k1.track_deg - k0.track_deg), k1.gamma_rad - k0.gamma_rad, k1.speed_mps - k0.speed_mps)
    return att, thr, rates


def test_the_inverse_flies_the_rates_it_was_asked_for():
    inputs, charts = _a320([_state(45.0, 0.0), _state(45.0, -3.0), _state(300.0, -3.0, 80.0, 900.0),
                            _state(10.0, 2.0, 120.0)])
    asked_track = torch.tensor([2.0, -2.5, 1.5, 0.0], dtype=F64)
    asked_gamma = torch.tensor([0.0, math.radians(0.3), math.radians(-0.2), 0.0], dtype=F64)
    asked_accel = torch.tensor([-0.3, -0.2, 0.0, 0.4], dtype=F64)
    att, thr, (track, gamma, speed) = _fly(inputs, charts, asked_track, asked_gamma, asked_accel)
    assert not any(bool(b.any()) for b in (*att.binds.values(), *thr.binds.values()))
    assert track.numpy() == pytest.approx(asked_track.numpy(), rel=0.02, abs=0.01)
    assert gamma.numpy() == pytest.approx(asked_gamma.numpy(), rel=0.05, abs=math.radians(0.01))
    hold = 9.81 * np.abs(asked_gamma.numpy()) * 1.0 / 2.0
    assert np.all(np.abs(speed.numpy() - asked_accel.numpy()) <= hold + 0.005)
    assert float(att.bank_rad[0]) < 0.0 < float(att.bank_rad[1])


def test_each_limit_binds_on_the_quantity_it_bounds():
    inputs, charts = _a320([_state(90.0, 0.0)] * 4)
    att, _, (track, gamma, _) = _fly(inputs, charts, 10.0, 0.0, 0.0, bank_cap_deg=25.0)
    assert att.binds["bank_cap"].all() and float(att.bank_rad[0]) == pytest.approx(-math.radians(25.0))
    capped_rate = math.degrees(9.81 * math.tan(math.radians(25.0)) / 100.0)
    assert float(track[0]) == pytest.approx(capped_rate, rel=0.02) and abs(float(gamma[0])) < math.radians(0.02)
    att, _, _ = _fly(inputs, charts, 3.0, 0.0, 0.0, torch.zeros(4, dtype=F64), bank_rate_deg_s=2.0)
    assert att.binds["bank_rate"].all() and float(att.bank_rad[0]) == pytest.approx(-math.radians(2.0))
    att, _, _ = _fly(inputs, charts, 0.0, math.radians(20.0), 0.0)
    assert att.binds["load_factor"].all() and float(att.load_factor[0]) == inverse.LOAD_FACTOR_MAX
    _, thr, _ = _fly(inputs, charts, 0.0, 0.0, 6.0)
    assert thr.binds["thrust_max"].all() and float(thr.fraction[0]) == pytest.approx(1.0)
    _, thr, _ = _fly(inputs, charts, 0.0, 0.0, -8.0)
    assert thr.binds["thrust_min"].all() and not thr.binds["stall"].any()


# ---- the parameters
def _params(**changes):
    from ts_transformer.autopilot.params import ExecutorParams
    from ts_transformer.experiments.executor_spec import ROLL_RATE_DEG_S

    base = ExecutorParams(cycle_s=1.0, heading_time_constant_s=4.0, bank_rate_deg_s=ROLL_RATE_DEG_S,
                          path_time_constant_s=2.0, path_rate_factor=2.0, timeout_factor=1.5, word_clock="time",
                          decision_cone_share=1.0, decision_glidepath_tolerance_m=60.0)
    return replace(base, **changes)


def test_the_parameters_are_checked_against_the_designs_constraints():
    one, params = spec(), _params()
    params.check(one, 2.0)
    params.check(one, 8.0)
    for change, message in ((dict(heading_time_constant_s=1.5), "under 2 Δt"),
                            (dict(path_time_constant_s=1.0), "under 2 Δt"),
                            (dict(timeout_factor=0.0), "positive"),
                            (dict(decision_cone_share=0.0), "positive"),
                            (dict(decision_glidepath_tolerance_m=math.nan), "finite"),
                            (dict(bank_rate_deg_s=math.nan), "finite"),
                            (dict(word_clock="sundial"), "word_clock")):
        with pytest.raises(ValueError, match=message):
            replace(params, **change).check(one, 2.0)
    with pytest.raises(ValueError, match="row interval 2.5 s"):
        params.check(one, 2.5)


def test_the_standards_roll_rate_stops_the_executors_own_turns_within_the_bank_limit():
    from aerodynamic_model.torch_dynamics import GRAVITY_MPS2
    from ts_transformer.autopilot import derive
    from ts_transformer.autopilot.lateral import rate_for_error, stopping_rate_deg_s
    from ts_transformer.experiments.executor_spec import ROLL_RATE_DEG_S

    one = spec(turn_rate_max_deg_s=10.0)
    floor = derive.stopping_roll_rate_deg_s(one, 4.0)
    assert floor == pytest.approx(math.degrees(math.tan(math.radians(32.0)) / 8.0)) and floor < ROLL_RATE_DEG_S
    speed = torch.linspace(50.0, 160.0, 23, dtype=F64)[:, None]
    error = torch.linspace(0.5, 180.0, 360, dtype=F64)[None, :]
    bank_capped = GRAVITY_MPS2 * math.tan(math.radians(32.0)) / speed

    def excess(p_deg_s: float) -> float:
        params = _params(bank_rate_deg_s=p_deg_s)
        own = torch.minimum(rate_for_error(error, params, one), torch.rad2deg(bank_capped))
        return float((own - stopping_rate_deg_s(error, speed, params)).max())

    assert excess(floor) <= 1e-9 and excess(ROLL_RATE_DEG_S) <= 0.0 and excess(0.9 * floor) > 0.0


# ---- the lateral law
def test_the_lateral_relative_mirrors_the_labellers():
    from ts_transformer.autopilot import lateral
    from ts_transformer.instructions.airport import relative_to_runway

    rng = np.random.default_rng(7)
    candidate = instruction_airport().candidates[0]
    e, n, track = rng.uniform(-20000, 5000, 50), rng.uniform(-9000, 9000, 50), rng.uniform(0, 360, 50)
    k = read_state(torch.zeros(50, 7, dtype=F64), AirportCharts.of([instruction_airport()] * 50, dtype=F64, device=CPU))
    k = type(k)(**{**k.__dict__, "e_m": torch.tensor(e), "n_m": torch.tensor(n), "track_deg": torch.tensor(track)})
    ours = lateral.relative(k, torch.tensor(candidate.threshold_e_m, dtype=F64),
                            torch.tensor(candidate.threshold_n_m, dtype=F64), torch.tensor(candidate.course_deg, dtype=F64))
    ref = relative_to_runway(e, n, track, np.zeros(50), candidate)
    assert ours[0].numpy() == pytest.approx(ref.before_threshold_m) and ours[1].numpy() == pytest.approx(ref.right_of_course_m)
    assert ours[2].numpy() == pytest.approx(ref.track_minus_course_deg)


def test_the_executors_own_turn_takes_the_shorter_way_within_the_vocabularys_rates_and_eases_out():
    from ts_transformer.autopilot.lateral import rate_for_error

    params, one = _params(), spec()
    track, target = torch.tensor([10.0, 350.0, 100.0, 100.0], dtype=F64), torch.tensor([350.0, 10.0, 104.0, 100.0],
                                                                                        dtype=F64)
    rate = rate_for_error(wrap180(target - track), params, one)
    assert rate.tolist() == pytest.approx([-one.turn_rate_max_deg_s, one.turn_rate_max_deg_s,
                                           4.0 / params.heading_time_constant_s, 0.0])


def test_a_heading_word_is_flown_to_arrive_when_its_lead_runs_out_no_faster_than_the_bank_can_stop():
    from aerodynamic_model.torch_dynamics import GRAVITY_MPS2
    from ts_transformer.autopilot.lateral import stopping_rate_deg_s, word_rate

    params, one = _params(), spec()
    error = torch.tensor([6.0, 6.0, 6.0, 20.0, -6.0], dtype=F64)
    to_go = torch.tensor([4.0, 3.0, -5.0, 4.0, 1.0], dtype=F64)
    speed = torch.full((5,), 140.0, dtype=F64)
    stop = float(stopping_rate_deg_s(torch.tensor([6.0], dtype=F64), torch.tensor([140.0], dtype=F64), params))
    assert stop == pytest.approx(math.degrees(math.sqrt(2 * GRAVITY_MPS2 * math.radians(params.bank_rate_deg_s) / 140.0
                                                        * math.radians(6.0))))
    assert word_rate(error, to_go, speed, params, one).tolist() == pytest.approx([1.5, 2.0, stop, 3.5, -stop])


def _lateral_state(track_deg: float, n_m: float = 0.0):
    geometry = _two_runways()
    lat, lon = geometry.frame.latlon_from_horizontal(-8000.0, n_m)
    return read_state(torch.tensor([[lat, lon, 900.0, 80.0, math.radians(90.0 - track_deg), 0.0, 60000.0]], dtype=F64),
                      AirportCharts.of([geometry], dtype=F64, device=CPU))


def _two_runways():
    """Runway 09 (the frame's origin, course 090) and 12 (course 120.3°, off the 5° grid like KSTL's)."""
    from ts_transformer.instructions.airport import AirportGeometry

    ends = [{"ident": "09", "threshold_e_m": 0.0, "threshold_n_m": 0.0, "course_deg": 90.0},
            {"ident": "12", "threshold_e_m": 500.0, "threshold_n_m": -3000.0, "course_deg": 120.3}]
    return AirportGeometry.from_dict({
        "code": "KXXX", "reference": {"lat": 35.0, "lon": -78.0, "elevation_m": 100.0},
        "candidates": [{**end, "elevation_m": 100.0, "length_m": 3000.0} for end in ends], "runway_ends": ends})


def test_a_heading_word_is_converted_with_the_course_of_r_when_heard_and_a_runway_change_does_not_turn():
    """§3.3, §5.4 (D8): a word is θ = course(R) + 5k° at the hearing; the executor keeps that absolute target, so a
    change of R alone does not turn the aircraft; the next word is converted with the new R's course."""
    from ts_transformer.autopilot.lateral import Lateral, Runways

    one, words = spec(), Words(spec())
    geometry = _two_runways()
    runways = Runways.of([geometry], one, dtype=F64, device=CPU)
    lateral = Lateral(1, _params(), one, CPU)
    no = torch.tensor([False])

    def error(relative_deg, issued, runway, track_deg, time_s):
        state = _lateral_state(track_deg)
        _, _ = lateral.rate(state, torch.tensor([relative_deg], dtype=F64), torch.tensor([issued]),
                            torch.tensor([runway]), no, torch.tensor([0]), runways, time_s)
        return float(lateral.target_unwrapped[0] - lateral.track_unwrapped[0]), float(lateral.word_deg[0])

    assert error(words.heading_relative_deg(0), 0, 0, 90.0, 0.0) == pytest.approx((0.0, 90.0))       # on R = 09's course
    # R becomes 12 (course 120.3°), the heading word unchanged: the target stays 090
    assert error(words.heading_relative_deg(0), 0, 1, 90.0, 1.0) == pytest.approx((0.0, 90.0))
    # a new word of class 0 under R = 12: its course, 120.3° — off the absolute 5° grid
    assert error(words.heading_relative_deg(0), 3, 1, 90.0, 6.0) == pytest.approx((30.3, 120.3))
    # class 70 (−10°) under 12: 110.3°, measured from the word before (a turn said word by word keeps its way)
    assert error(words.heading_relative_deg(70), 4, 1, 100.0, 8.0) == pytest.approx((10.3, 110.3))


# ---- whole flights
#: The test airport's runways' published TCH, glidepath and decision altitude (the fleet's DAs above the threshold
#: run 61–129 m).
TEST_TCH_M, TEST_GLIDEPATH_DEG, TEST_DA_M = 15.0, 3.0, 60.0


def vertical_paths(geometry):
    from ts_transformer.autopilot.runway_data import VerticalPath

    return tuple(VerticalPath(TEST_TCH_M, TEST_GLIDEPATH_DEG, TEST_DA_M) for _ in geometry.candidates)


def _physics(signals, geometry, approach_ias=None):
    """One A320 flight flown from ``signals``' row 0: ``(inputs, runways, charts, approach IAS)``."""
    from ts_transformer.autopilot.lateral import Runways
    from ts_transformer.autopilot.speed import approach_speed_ias_mps

    aircraft = aircraft_for_code("A320")
    aero, mass = aero_params_for_aircraft(aircraft), 62000.0
    lat, lon = geometry.frame.latlon_from_horizontal(signals.e_m[0], signals.n_m[0])
    gamma = math.atan2(signals.vertical_rate_mps[0], signals.ground_speed_mps[0])
    state = [lat, lon, signals.altitude_m[0], signals.ground_speed_mps[0] / math.cos(gamma),
             math.radians(90.0 - signals.track_deg[0]), gamma, mass]
    candidate = geometry.candidates[0]
    tlat, tlon = geometry.frame.latlon_from_horizontal(candidate.threshold_e_m, candidate.threshold_n_m)
    inputs = FlightInputs(
        initial_state=torch.tensor([state], dtype=F64),
        aero_params=torch.tensor([[aero.S, aero.Cl_max, aero.Cd0, aero.k, aero.stall_threshold, aero.k_stall]], dtype=F64),
        frame_params=torch.tensor([[tlat, tlon, candidate.elevation_m, 0.0]], dtype=F64),
        max_thrust_n=torch.tensor([aircraft.engine.max_thrust_total_n], dtype=F64))
    return (inputs, Runways.of([geometry], spec(), dtype=F64, device=CPU),
            AirportCharts.of([geometry], dtype=F64, device=CPU),
            torch.tensor([approach_speed_ias_mps("A320", mass) if approach_ias is None else approach_ias], dtype=F64))


def _said(grid, words, geometry, runway_index=0):
    """A grid's words as instructions on its rows (`replay.sentence_on_interval`'s): a heading word's track under the
    runway."""
    course = geometry.candidates[runway_index].course_deg
    out = []
    for row, column in zip(*np.nonzero(grid != UNCHANGED)):
        value = int(grid[row, column])
        info = {"target_deg": words.heading_track_deg(value, course)} if column == HEADING else {}
        out.append(Instruction(int(column), value, int(row), "said", info))
    return out


def _fly_sentence(signals, grid=None, params=None, approach_ias=None, clock="distance", reading=None):
    """Read ``signals`` with the labeller, fly its sentence (or ``grid``) from row 0 on ``clock`` (the distance clock by
    default: the words said where the observed aircraft was, so the executor's own pace does not move them) and judge
    it."""
    from ts_transformer.autopilot.executor import GO_AROUND_EXTRA_S, fly
    from ts_transformer.autopilot.judge import judge
    from ts_transformer.instructions.labeller.read import read_flight

    one = spec()
    words, geometry = Words(one), instruction_airport()
    params = params or _params()
    reading = read_flight(signals, geometry, one, words) if reading is None else reading
    grid = reading.words if grid is None else grid
    inputs, runways, charts, approach = _physics(signals, geometry, approach_ias)
    rows = len(grid)
    clocks = {"time": lambda: TimeClock(params.cycle_s),
              "track": lambda: TrackClock.of([signals.e_m[:rows]], [signals.n_m[:rows]], one.step_s, params.cycle_s,
                                             device=CPU),
              "distance": lambda: DistanceClock.of([signals.e_m[:rows]], [signals.n_m[:rows]], one.step_s,
                                                   params.cycle_s, device=CPU)}
    flown = fly(inputs, Sentences([grid], words, step_s=one.step_s, device=CPU), clocks[clock](), runways, charts,
                approach, params, words, time_limit_s=torch.tensor([rows * one.step_s * params.timeout_factor], dtype=F64),
                reserve_s=GO_AROUND_EXTRA_S * int((grid[:, RUNWAY] == RUNWAY_GO_AROUND).sum()))
    verdict = judge(flown, 0, geometry, vertical_paths(geometry), _said(grid, words, geometry), one.step_s, signals, one,
                    words, params)
    return flown, verdict, reading


def _turn(degrees: float, speed_mps: float, per_row_deg: float = 4.5) -> list[tuple[int, float, float, float]]:
    side = math.copysign(1.0, degrees)
    steady = int(round((abs(degrees) - 4.0 * per_row_deg) / per_row_deg))
    return [(2, side * per_row_deg / 3.0, speed_mps, 0.0), (2, side * per_row_deg * 2.0 / 3.0, speed_mps, 0.0),
            (steady, side * per_row_deg, speed_mps, 0.0),
            (2, side * per_row_deg * 2.0 / 3.0, speed_mps, 0.0), (2, side * per_row_deg / 3.0, speed_mps, 0.0)]


DOWNWIND_BASE_FINAL = [(60, 0.0, 100.0, 0.0), *_turn(-90.0, 100.0), (20, 0.0, 90.0, 0.0), *_turn(-90.0, 85.0),
                       (120, 0.0, 75.0, -75.0 * np.tan(np.radians(3.0)))]


def _downwind():
    signals = instruction_flight(*fly_legs(DOWNWIND_BASE_FINAL, 270.0, 1110.0, -400.0, 0.0))
    _, _, reading = _fly_sentence(signals)
    return signals, reading


def test_a_downwind_base_final_sentence_is_flown_onto_the_final_by_its_words_and_lands():
    """D2, D3: the words alone — the turn onto the final said word by word, the final held by class 0, the descent by
    "no level-off" at the class's angle — bring the aircraft to the runway (said where the observed aircraft was: the
    distance clock); the DA check passes and every word is inside its envelope. Nothing captures (no such mode exists)."""
    from ts_transformer.autopilot.executor import MODES

    signals, _ = _downwind()
    flown, verdict, reading = _fly_sentence(signals)
    assert verdict.outcome == "landed" and verdict.flew_the_sentence
    assert abs(verdict.crossing["cross_m"]) < 60.0 and 0.0 < verdict.crossing["height_m"] < 100.0
    assert verdict.crossing["decision"]["passed"]
    assert set(MODES) == {"go_around", "flying_course", "level_captured"}
    assert not any(verdict.limits[name]["cycles"] for name in ("thrust_max", "thrust_min", "stall", "load_factor"))
    assert all(h["inside"] == h["rows"] for h in verdict.words["heading"] if h["rows"])


def test_class_0_holds_the_course_and_keeps_the_offset_the_turn_left():
    """§3.3: "a word of class 0 gives no lateral correction. The aircraft keeps its lateral offset until the model says
    a word of ±5°" — on the time clock the executor's turn onto the final ends 300-odd metres right of the line (it
    flies the words at its own pace), and it stays there to the threshold: no law brings it back (D2, D3)."""
    from ts_transformer.autopilot.judge import flown_track
    from ts_transformer.instructions.airport import relative_to_runway

    signals, reading = _downwind()
    flown, verdict, _ = _fly_sentence(signals, clock="time")
    geometry = instruction_airport()
    track = flown_track(flown.states[0, : verdict.end_row + 1].numpy(), geometry)
    relative = relative_to_runway(track["e"], track["n"], track["track"], track["height"], geometry.candidates[0])
    last_word = max(i.row for i in reading.instructions if i.column == HEADING)
    right = relative.right_of_course_m[last_word * 2 + 20:]           # 20 s after the last heading word (class 0)
    assert right.min() > 200.0 and right.max() - right.min() < 1.0
    assert verdict.outcome == "crossed_off_runway" and verdict.crossing["cross_m"] == pytest.approx(right[-1], abs=1.0)


def test_no_level_off_flies_the_class_angle_without_levelling_off():
    """§5.5: "no level-off" + a descent class flies the class's nominal angle; no level is captured and no aim at a
    crossing point bends it (D3, D9)."""
    signals, reading = _downwind()
    flown, verdict, _ = _fly_sentence(signals)
    words = Words(spec())
    said = [i.row for i in reading.instructions if i.column == ALTITUDE and i.value == words.altitude_no_level_off]
    angle = [i for i in reading.instructions if i.column == ANGLE and i.row >= said[0]]
    gamma = flown.states[0, 1:, 5].numpy()
    end = verdict.end_row - 1
    steady = slice(said[0] * 2 + 15, end)
    assert gamma[steady] == pytest.approx(-math.radians(words.angle_deg(angle[0].value)), abs=1e-3)
    assert not flown.modes["level_captured"][0, steady].any()


def test_a_sentence_said_a_step_at_a_time_is_flown_as_the_whole_sentence():
    from ts_transformer.autopilot.executor import Executor, fly
    from ts_transformer.autopilot.sentence import Spoken

    one, geometry = spec(), instruction_airport()
    words, params = Words(one), _params()
    signals, reading = _downwind()
    grid = reading.words
    inputs, runways, charts, approach = _physics(signals, geometry)
    limit = torch.tensor([len(grid) * one.step_s * params.timeout_factor], dtype=F64)
    whole = fly(inputs, Sentences([grid], words, step_s=one.step_s, device=CPU), TimeClock(params.cycle_s), runways,
                charts, approach, params, words, time_limit_s=limit, reserve_s=0.0)
    executor = Executor(inputs, runways, charts, approach, params, words, step_s=one.step_s, time_limit_s=limit)
    spoken = Spoken(1, words, step_s=one.step_s, device=CPU)
    last = grid[-1:].copy()
    last[:] = UNCHANGED
    for step in range(executor.cycles // executor.step_rows + 1):
        spoken.say(grid[step: step + 1] if step < len(grid) else last)
        for _ in range(executor.step_rows):
            if executor.count < executor.cycles and not bool(executor.done.all()):
                executor.cycle(spoken.at(torch.tensor([step * one.step_s], dtype=F64)),
                               torch.tensor([executor.count * params.cycle_s], dtype=F64))
    stepped = executor.flown()
    assert torch.equal(stepped.states, whole.states) and torch.equal(stepped.done_cycle, whole.done_cycle)
    assert torch.equal(stepped.commands, whole.commands) and torch.equal(stepped.runway, whole.runway)
    with pytest.raises(ValueError, match="the step just said"):
        spoken.at(torch.tensor([0.0], dtype=F64))
    with pytest.raises(ValueError, match="step 0 must write every column"):
        Spoken(1, words, step_s=one.step_s, device=CPU).say(last)
    first = grid[:1].copy()
    first[0, RUNWAY] = RUNWAY_GO_AROUND                        # rule 1: the first step says a candidate
    with pytest.raises(ValueError, match="a candidate in the runway column"):
        Spoken(1, words, step_s=one.step_s, device=CPU).say(first)


def test_a_turn_said_word_by_word_is_flown_without_levelling():
    legs = [(30, 0.0, 100.0, 0.0), *_turn(180.0, 100.0), (20, 0.0, 100.0, 0.0), *_turn(-90.0, 100.0),
            (100, 0.0, 75.0, -75.0 * np.tan(np.radians(3.0)))]
    signals = instruction_flight(*fly_legs(legs, 0.0, 950.0, -400.0, 0.0))
    flown, verdict, reading = _fly_sentence(signals)
    turn = [i for i in reading.instructions if i.column == HEADING and 0 < i.row < 80]
    assert len(turn) > 25
    bank = flown.commands[0, :, 1].numpy()
    heard = flown.sentence_s[0].numpy()
    first, last = (int(np.searchsorted(heard, row * 2.0)) for row in (turn[0].row, turn[-1].row))
    assert (bank[first + 8: last] < -math.radians(5.0)).all()
    assert all(h["inside"] == h["rows"] for h in verdict.words["heading"])


def test_a_fast_turn_said_word_by_word_is_followed_on_every_clock():
    from ts_transformer.autopilot.sentence import CLOCKS

    legs = [(40, 0.0, 80.0, 0.0), *_turn(-90.0, 80.0, 6.0), (20, 0.0, 80.0, 0.0), *_turn(-90.0, 80.0, 6.0),
            (120, 0.0, 70.0, -70.0 * np.tan(np.radians(3.0)))]
    signals = instruction_flight(*fly_legs(legs, 270.0, 1110.0, -400.0, 0.0))
    for clock in CLOCKS:
        _, verdict, _ = _fly_sentence(signals, clock=clock)
        assert all(h["inside"] == h["rows"] for h in verdict.words["heading"]), clock


def test_descents_level_off_at_their_targets_inside_the_tubes():
    legs = [(100, 0.0, 75.0, 0.0), (80, 0.0, 75.0, -75.0 * np.tan(np.radians(2.1))), (120, 0.0, 75.0, 0.0),
            (100, 0.0, 75.0, -75.0 * np.tan(np.radians(3.0)))]
    signals = instruction_flight(*fly_legs(legs, 90.0, 1500.0, -400.0, 0.0))
    _, verdict, reading = _fly_sentence(signals)
    targets = [i for i in reading.instructions if i.column == ALTITUDE]
    assert len(targets) == 3 and len(verdict.words["vertical"]) == 3
    assert all(word["contained"] for word in verdict.words["vertical"])


def test_the_outcome_alone_is_the_verdict_s_first_layer():
    from ts_transformer.autopilot.judge import outcome_of

    signals, _ = _downwind()
    flown, verdict, _ = _fly_sentence(signals)
    geometry = instruction_airport()
    ended = outcome_of(flown, 0, geometry, vertical_paths(geometry), spec(), _params())
    assert (ended.outcome, ended.end_row, ended.crossing, ended.limits) == (
        verdict.outcome, verdict.end_row, verdict.crossing, verdict.limits)


def test_the_words_the_executor_refuses():
    signals, reading = _downwind()
    words = Words(spec())
    climb = reading.words.copy()
    climb[5, ALTITUDE], climb[5, ANGLE] = words.altitude_no_level_off, words.angle_climb
    with pytest.raises(ValueError, match="without a descent class"):
        _fly_sentence(signals, climb)
    unspecified = reading.words.copy()
    unspecified[5, SPEED] = words.speed_unspecified
    with pytest.raises(ValueError, match="publishes no approach speed"):
        _fly_sentence(signals, unspecified, approach_ias=math.nan)


def test_the_flight_meets_the_ground_when_told_below_the_threshold():
    words = Words(spec())
    signals, reading = _downwind()
    below = reading.words.copy()
    below[1, ALTITUDE], below[1, ANGLE] = words.altitude_index(0.0), 4
    below[2:, ALTITUDE] = UNCHANGED
    below[2:, ANGLE] = UNCHANGED
    _, grounded, _ = _fly_sentence(signals, below)
    assert grounded.outcome == "ground_contact"
    limited = grounded.limits["path_rate_limited"]
    assert limited["cycles"] > 0 and limited["wanted_minus_given"] > 1e-4


# ---- the go-around (§3.2, §5.4, §5.5, D10)
def test_a_go_around_climbs_at_200_ft_per_nm_on_rs_course_holds_its_speed_and_lands_after_a_runway_word():
    """D10: "go-around" climbs at the missed-approach gradient (the climb replaces "no level-off"), flies the course of R,
    holds the airspeed, gives the flight 900 s more, and a crossing under it is no event; the runway said again ends it,
    and the words that follow bring the aircraft round to land."""
    from ts_transformer.autopilot.executor import GO_AROUND_EXTRA_S
    from ts_transformer.autopilot.vertical import GO_AROUND_CLIMB_RAD

    signals, reading = _downwind()
    grid = reading.words.copy()
    go = len(grid) - 15                                         # on the final, "no level-off" in force
    grid[go, RUNWAY] = RUNWAY_GO_AROUND
    flown, verdict, _ = _fly_sentence(signals, grid)
    gamma, speed = flown.states[0, 1:, 5].numpy(), flown.states[0, 1:, 3].numpy()
    track = compass_from_math_rad(flown.states[0, 1:, 4].numpy())
    after = int(np.argmax(flown.modes["go_around"][0].numpy()))          # the cycle that heard it
    assert after > 0 and flown.modes["go_around"][0, after:].all()
    assert gamma[after + 25: after + 90] == pytest.approx(GO_AROUND_CLIMB_RAD, abs=1e-3)
    assert abs(speed[after + 60] - speed[after]) < 1.0
    assert np.abs(np_wrap180(track[after + 5: after + 90] - 90.0)).max() < 2.0
    # over the threshold under the go-around: no event; the flight flies on to its extended limit
    assert verdict.outcome == "timeout"
    assert int(flown.done_cycle[0]) + 1 == pytest.approx(len(grid) * 2.0 * 1.5 + GO_AROUND_EXTRA_S)


def test_a_heading_word_said_at_or_after_a_go_around_is_flown():
    words = Words(spec())
    signals, reading = _downwind()

    def track_deg(grid):
        flown, _, _ = _fly_sentence(signals, grid, clock="time")
        return compass_from_math_rad(flown.states[0, 1:, 4].numpy())

    def off(track, target):
        return np.abs((track - target + 180.0) % 360.0 - 180.0)

    go = len(reading.words) - 15
    after = reading.words.copy()
    after[go, RUNWAY] = RUNWAY_GO_AROUND
    after[go + 10, HEADING] = words.heading_class(0.0, 90.0)     # 10 steps later: left onto north
    track = track_deg(after)
    assert off(track[go * 2 + 10: (go + 10) * 2], 90.0).max() < 2.0
    assert off(track[(go + 10) * 2 + 60: (go + 10) * 2 + 80], 0.0).max() < 2.0
    assert 0.0 < track[(go + 10) * 2 + 20] < 90.0                  # turned left, the shorter way


def test_the_runway_word_after_a_go_around_ends_it_and_a_level_word_holds():
    """§3.2, §5.5: a climb word under the go-around climbs at 200 ft per NM to its level and holds it; a runway word ends
    G, after which a climb word flies the climb class's own centre."""
    from ts_transformer.autopilot.vertical import GO_AROUND_CLIMB_RAD

    words = Words(spec())
    signals, reading = _downwind()
    go = len(reading.words) - 15
    grid = np.vstack([reading.words, np.full((200, len(COLUMNS)), UNCHANGED)])
    grid[go, RUNWAY] = RUNWAY_GO_AROUND
    grid[go + 2, ALTITUDE], grid[go + 2, ANGLE] = words.altitude_index(900.0), words.angle_climb
    grid[go + 160, RUNWAY] = 0                                            # the climb to 900 m takes ~110 steps
    grid[go + 165, ALTITUDE], grid[go + 165, ANGLE] = words.altitude_index(1260.0), words.angle_climb
    flown, _, _ = _fly_sentence(signals, grid, clock="time")
    gamma, height = flown.states[0, 1:, 5].numpy(), flown.states[0, 1:, 2].numpy()
    assert flown.modes["go_around"][0, go * 2: (go + 160) * 2].all()
    assert not flown.modes["go_around"][0, (go + 160) * 2:].any()
    assert gamma[(go + 2) * 2 + 20: (go + 2) * 2 + 40] == pytest.approx(GO_AROUND_CLIMB_RAD, abs=1e-3)
    held = slice((go + 160) * 2 - 20, (go + 165) * 2)                     # the level reached, held to the next word
    assert height[held] == pytest.approx(900.0, abs=3.0) and flown.modes["level_captured"][0, held].all()
    climbing = slice((go + 165) * 2 + 20, (go + 165) * 2 + 40)
    assert gamma[climbing] == pytest.approx(math.radians(spec().climb_angle_centre_deg), abs=1e-3)


def test_the_vertical_law_knows_how_long_its_own_go_around_climb_takes():
    from ts_transformer.autopilot.vertical import GO_AROUND_CLIMB_GRADIENT, Vertical

    words, params = Words(spec()), _params()
    vertical = Vertical(1, params, words, CPU)
    signals, reading = _downwind()
    grid = reading.words.copy()
    go = len(grid) - 15
    grid[go, RUNWAY] = RUNWAY_GO_AROUND
    flown, _, _ = _fly_sentence(signals, grid, clock="time")
    start = go * 2
    height, speed, gamma = (flown.states[0, start:, c].numpy() for c in (2, 3, 5))
    for climb_m in (5.0, 50.0, 150.0):
        flew_s = float(np.argmax(height >= height[0] + climb_m)) * params.cycle_s
        reference = vertical.go_around_climb_s(float(height[0]), float(gamma[0]), float(speed[0]), height[0] + climb_m)
        assert abs(reference - flew_s) <= 2.0 * params.cycle_s
    assert vertical.go_around_climb_s(500.0, -0.05, 70.0, 500.0) == 0.0
    with pytest.raises(ValueError, match="flying forward"):
        vertical.go_around_climb_s(400.0, -0.05, 0.0, 500.0)
    alone = 7.0 / (70.0 * GO_AROUND_CLIMB_GRADIENT)
    assert vertical.go_around_climb_s(493.0, -math.radians(3.0), 70.0, 500.0) > 3.0 * alone


def test_a_go_around_word_past_the_reserve_is_refused():
    from ts_transformer.autopilot.executor import Executor

    words, params = Words(spec()), _params()
    signals, reading = _downwind()
    grid = reading.words.copy()
    grid[3, RUNWAY] = RUNWAY_GO_AROUND
    inputs, runways, charts, approach = _physics(signals, instruction_airport())
    executor = Executor(inputs, runways, charts, approach, params, words, step_s=2.0,
                        time_limit_s=torch.tensor([500.0], dtype=F64), reserve_s=0.0)
    sentences = Sentences([grid], words, step_s=2.0, device=CPU)
    with pytest.raises(ValueError, match="past the executor's reserve"):
        for cycle in range(10):
            executor.cycle(sentences.at(torch.tensor([float(cycle - cycle % 2)], dtype=F64)),
                           torch.tensor([float(cycle)], dtype=F64))


# ---- the judge (§5.8)
def _parallels():
    """09R (the frame's origin), 09L 1,500 m north, a crossing runway 03 whose threshold lies 2 km past 09R's, 200 m
    south of its centreline, and 27L, 09R's other end."""
    from ts_transformer.instructions.airport import AirportGeometry

    ends = [{"ident": "09R", "threshold_e_m": 0.0, "threshold_n_m": 0.0, "course_deg": 90.0},
            {"ident": "09L", "threshold_e_m": 0.0, "threshold_n_m": 1500.0, "course_deg": 90.0},
            {"ident": "03", "threshold_e_m": 2000.0, "threshold_n_m": -200.0, "course_deg": 30.0},
            {"ident": "27L", "threshold_e_m": 3000.0, "threshold_n_m": 0.0, "course_deg": 270.0}]
    return AirportGeometry.from_dict({
        "code": "KXXX", "reference": {"lat": 35.0, "lon": -78.0, "elevation_m": 100.0},
        "candidates": [{**end, "elevation_m": 100.0, "length_m": 3000.0} for end in ends], "runway_ends": ends})


def _judged(n_m, height_m, *, runway=0, go_around=False, track_deg=90.0, e_from=-3000.0, e_to=200.0,
            glide=False, geometry=None, params=None):
    """A straight flight at ``n_m`` north of 09R's centreline from ``e_from`` to ``e_to``, ``height_m`` above the
    threshold — or on the published glidepath to ``height_m`` at the threshold (``glide``) — under R = ``runway`` and
    G = ``go_around`` in every cycle."""
    from ts_transformer.autopilot.judge import _outcome

    geometry = geometry or _parallels()
    e = np.arange(e_from, e_to + 1.0, 70.0)
    height = 100.0 + (height_m + np.maximum(-e, 0.0) * math.tan(math.radians(3.0)) if glide else np.full(len(e), height_m))
    track = {"e": e, "n": np.full(len(e), n_m), "height": height, "track": np.full(len(e), track_deg)}
    rows = len(e)
    return _outcome(np.ones((rows, 7)), track, np.full(rows - 1, runway), np.full(rows - 1, go_around),
                    np.zeros(rows, dtype=bool), geometry, vertical_paths(geometry), spec(), params or _params())


def test_each_outcome_on_a_hand_built_track():
    from ts_transformer.autopilot.judge import CROSSINGS, OUTCOMES, runway_lateral_limit_m

    geometry, one = _parallels(), spec()
    assert runway_lateral_limit_m(geometry, 0, one) == pytest.approx(350.0 * 0.3048)
    assert set(CROSSINGS) < set(OUTCOMES)
    kind, _, crossing = _judged(-20.0, 15.0, glide=True)
    assert kind == "landed" and crossing["runway_index"] == 0 and crossing["decision"]["passed"]
    assert _judged(-5.0, 180.0)[0] == "crossed_too_high"                 # an approach crossing 180 m up
    assert _judged(-300.0, 20.0, glide=True)[0] == "crossed_off_runway"     # inside the screen's 750 m, off the runway
    assert _judged(-1300.0, 20.0)[0] == "timeout"                         # beyond 09R's screen: no approach crossing
    assert _judged(-20.0, 20.0)[0] == "unstable_at_minimums"              # level at 20 m: never descended through the DA
    # lined up with 09L over its threshold, R = 09R: another runway, at any height
    kind, _, crossing = _judged(1480.0, 250.0)
    assert kind == "crossed_other_runway" and crossing["runway_index"] == 1
    assert _judged(1200.0, 20.0)[0] == "timeout"                          # 300 m beside 09L is not over it
    # under R = 09L the same crossing is an approach crossing (no runway lock, D12: the judge reads R at the crossing)
    assert _judged(1480.0, 15.0, runway=1, glide=True)[0] == "landed"
    # not lined up (60° across the plane): no event
    assert _judged(-20.0, 15.0, glide=True, track_deg=30.0)[0] == "timeout"
    # down 09R past its end: 03's plane at 60° and 27L's the wrong way are no events
    assert _judged(-20.0, 15.0, glide=True, e_to=4000.0)[0] == "landed"
    assert _judged(0.0, 20.0, e_from=500.0, e_to=4000.0)[0] == "timeout"


def test_a_crossing_under_a_go_around_is_not_an_event():
    """§5.8: while G is true a crossing of R is not an event — the low pass flies on."""
    assert _judged(-20.0, 15.0, glide=True, go_around=True)[0] == "timeout"


def test_the_decision_altitude_check_passes_and_fails():
    """§5.8 (D3, O2): at the DA point the lateral offset within the share of the FAS cone and the height within the
    tolerance of the published glidepath (straight-line reference); a failed check is `unstable_at_minimums`."""
    from ts_transformer.instructions.airport import curvature_radius_m

    geometry = _parallels()
    kind, _, crossing = _judged(-20.0, 15.0, glide=True)
    decision = crossing["decision"]
    assert kind == "landed" and decision["lateral_ok"] and decision["vertical_ok"]
    # on the flat glidepath: the straight line lies d²/(2R) above it at the DA point (centimetres there; 31 m at 20 km)
    d = (TEST_DA_M - TEST_TCH_M) / math.tan(math.radians(3.0))
    radius = curvature_radius_m(geometry.frame.lat0, 90.0)
    assert decision["above_glidepath_m"] == pytest.approx(-d ** 2 / (2.0 * radius), abs=1.0)
    from ts_transformer.instructions.airport import glidepath_height_m
    flat = TEST_TCH_M + 20_000.0 * math.tan(math.radians(3.0))
    assert float(glidepath_height_m(20_000.0, TEST_TCH_M, 3.0, radius)) - flat == pytest.approx(31.4, abs=0.2)
    assert decision["cone_half_width_m"] > 106.7
    tight = _params(decision_glidepath_tolerance_m=0.01)
    kind, _, crossing = _judged(-20.0, 15.0, glide=True, params=tight)
    assert kind == "unstable_at_minimums" and not crossing["decision"]["vertical_ok"]
    narrow = _params(decision_cone_share=0.1)
    kind, _, crossing = _judged(-20.0, 15.0, glide=True, params=narrow)
    assert kind == "unstable_at_minimums" and not crossing["decision"]["lateral_ok"]


def test_two_events_at_one_row_follow_the_tables_order():
    """§5.8: at one row the outcome is the first in the table's order — a stall before a landing, and an approach
    crossing of R before another runway's crossing on the same row."""
    from ts_transformer.autopilot.judge import EVENT_ORDER, _outcome, flown_track

    assert EVENT_ORDER == ("dynamics_failure", "ground_contact", "crossed_too_high", "crossed_off_runway",
                           "unstable_at_minimums", "landed", "crossed_other_runway")
    one, geometry = spec(), instruction_airport()
    signals, _ = _downwind()
    flown, verdict, _ = _fly_sentence(signals)
    states = flown.states[0, : verdict.end_row + 1].numpy()
    track = flown_track(states, geometry)
    last = verdict.end_row
    runway = flown.runway[0, :last].numpy()
    go_around = flown.modes["go_around"][0, :last].numpy()
    stalled = np.zeros(len(states), dtype=bool)
    args = (geometry, vertical_paths(geometry), one, _params())
    assert _outcome(states, track, runway, go_around, stalled, *args)[:2] == ("landed", last)
    stalled[last] = True
    assert _outcome(states, track, runway, go_around, stalled, *args)[:2] == ("dynamics_failure", last)
    # 09 and a runway "08" on the same threshold, 10° apart: both planes crossed lined up at one row
    from ts_transformer.instructions.airport import AirportGeometry

    ends = [{"ident": "09", "threshold_e_m": 0.0, "threshold_n_m": 0.0, "course_deg": 90.0},
            {"ident": "08", "threshold_e_m": 0.0, "threshold_n_m": 0.0, "course_deg": 80.0}]
    twin = AirportGeometry.from_dict({"code": "KXXX", "reference": {"lat": 35.0, "lon": -78.0, "elevation_m": 100.0},
                                      "candidates": [{**end, "elevation_m": 100.0, "length_m": 3000.0} for end in ends],
                                      "runway_ends": ends})
    kind, _, crossing = _judged(-20.0, 15.0, glide=True, geometry=twin)
    assert kind == "landed" and crossing["runway_index"] == 0
    kind, _, crossing = _judged(-20.0, 15.0, glide=True, geometry=twin, runway=1)        # R = 08: 08 first
    assert kind == "landed" and crossing["runway_index"] == 1


def test_the_judge_fails_words_the_flown_track_falls_behind():
    _, verdict, _ = _fly_sentence(_downwind()[0], params=_params(bank_rate_deg_s=0.5))
    outside = [h for h in verdict.words["heading"] if h["rows"] and h["inside"] < h["rows"]]
    assert outside and not verdict.words["all_contained"] and not verdict.flew_the_sentence


def test_a_dynamics_failure_judges_its_words_only_on_the_states_before_it():
    from ts_transformer.autopilot.judge import judge

    one, words, geometry = spec(), Words(spec()), instruction_airport()
    signals, _ = _downwind()
    flown, _, reading = _fly_sentence(signals)
    failed_at = 120
    states = flown.states.clone()
    states[0, failed_at:] = float("nan")
    broken = replace(flown, states=states, done_cycle=torch.tensor([failed_at - 1]))
    verdict = judge(broken, 0, geometry, vertical_paths(geometry), _said(reading.words, words, geometry), one.step_s,
                    signals, one, words, _params())
    assert verdict.outcome == "dynamics_failure" and verdict.end_row == failed_at
    judged = [h for h in verdict.words["heading"] if h["rows"] > 0]
    assert judged and all(h["inside"] == h["rows"] for h in judged)


def test_a_word_the_clock_never_reached_is_not_reached():
    signals, reading = _downwind()
    _, verdict, _ = _fly_sentence(signals, params=_params(timeout_factor=0.3), clock="time")
    later = [i for i in _said(reading.words, Words(spec()), instruction_airport()) if i.row * 2.0 >= verdict.end_row]
    assert later and verdict.words["not_reached"] == len(later)


def test_words_said_on_one_flown_row_leave_the_later_one_of_each_column():
    from ts_transformer.autopilot.judge import said_at

    words = [Instruction(ALTITUDE, 30, 10, "target"), Instruction(ALTITUDE, 20, 12, "target"),
             Instruction(HEADING, 5, 10, "per-step", {"target_deg": 25.0}),
             Instruction(HEADING, 9, 11, "per-step", {"target_deg": 45.0}), Instruction(SPEED, 14, 20, "target")]
    kept, superseded = said_at(words, [7, 7, 7, 7, 15])
    assert superseded == 1
    assert [(w.column, w.value, w.row, w.info["sentence_row"]) for w in kept] == [
        (ALTITUDE, 20, 7, 12), (HEADING, 5, 7, 10), (HEADING, 9, 7, 11), (SPEED, 14, 15, 20)]


def test_the_word_count_leaves_out_the_words_the_judge_did_not_judge():
    from ts_transformer.autopilot.judge import Verdict
    from ts_transformer.autopilot.replay import word_results

    heading = [{"row": 0, "rows": 5, "inside": 5}, {"row": 9, "rows": 3, "inside": 2}, {"row": 12, "rows": 0, "inside": 0}]
    words = {"not_reached": 0, "superseded_before_flown": 0, "heading": heading, "vertical": [{"contained": True}],
             "speed": [{"contained": False}], "all_contained": False}
    judged, not_judged = word_results(Verdict("landed", 50, None, {}, words, flown_rows=51))
    assert not_judged == 1
    assert judged == [("heading", True), ("heading", False), ("altitude", True), ("speed", False)]


# ---- speed
def test_a_speed_word_is_flown_at_the_vocabularys_pace_both_ways():
    from ts_transformer.autopilot.frame import Kinematics
    from ts_transformer.autopilot.speed import Speed, speed_change_mps2

    one = spec()
    speed = Speed(torch.tensor([65.0], dtype=F64), one)
    state = Kinematics(*(torch.tensor([v], dtype=F64) for v in (0.0, 0.0, 100.0, 90.0, 90.0, 0.0, 90.0, 60000.0)))
    aero = torch.tensor([[122.6, 2.5, 0.02, 0.04, 0.9, 0.2]], dtype=F64)

    def rate(word_mps, go_around=False):
        return float(speed.rate(state, torch.tensor([word_mps], dtype=F64), torch.tensor([False]),
                                torch.tensor([go_around]), torch.ones(1, dtype=F64), aero,
                                torch.tensor([50_000.0], dtype=F64))[0][0])

    assert rate(70.0) == pytest.approx(-speed_change_mps2(one)) and rate(110.0) == pytest.approx(speed_change_mps2(one))
    assert rate(70.0, go_around=True) == pytest.approx(0.0)                 # a go-around holds the airspeed


def test_the_pilots_own_speed_is_reached_by_the_threshold():
    from ts_transformer.autopilot.frame import Kinematics
    from ts_transformer.autopilot.speed import Speed, speed_change_mps2

    one = spec()
    speed = Speed(torch.tensor([65.0], dtype=F64), one)
    state = Kinematics(*(torch.tensor([v], dtype=F64) for v in (0.0, 0.0, 100.0, 90.0, 90.0, 0.0, 90.0, 60000.0)))
    aero = torch.tensor([[122.6, 2.5, 0.02, 0.04, 0.9, 0.2]], dtype=F64)

    def rate(straight_m):
        return float(speed.rate(state, torch.tensor([float("nan")], dtype=F64), torch.tensor([True]),
                                torch.tensor([False]), torch.ones(1, dtype=F64), aero,
                                torch.tensor([straight_m], dtype=F64))[0][0])

    far, near = rate(50_000.0), rate(3_000.0)
    assert far == pytest.approx(-speed_change_mps2(one)) and -one.speed_accel_max_mps2 <= near < far


# ---- multi-aircraft batches and the single flight
def test_a_multi_aircraft_batch_flies_each_flight_from_its_own_start_as_it_flies_alone():
    from ts_transformer.autopilot.executor import Executor, fly
    from ts_transformer.autopilot.lateral import Runways
    from ts_transformer.instructions.labeller.read import read_flight

    one, geometry = spec(), instruction_airport()
    words, params = Words(one), _params()
    flights = [instruction_flight(*fly_legs(DOWNWIND_BASE_FINAL, 270.0, 1110.0, -400.0, 0.0)),
               instruction_flight(*fly_legs(DOWNWIND_BASE_FINAL, 270.0, 1180.0, -900.0, 0.0))]
    grids = [read_flight(f, geometry, one, words).words for f in flights]
    physics = [_physics(f, geometry) for f in flights]
    limits = [len(g) * one.step_s * params.timeout_factor for g in grids]
    alone = [fly(inputs, Sentences([grid], words, step_s=one.step_s, device=CPU), TimeClock(params.cycle_s), runways,
                 charts, approach, params, words, time_limit_s=torch.tensor([limit], dtype=F64), reserve_s=0.0)
             for (inputs, runways, charts, approach), grid, limit in zip(physics, grids, limits)]
    step_rows = int(round(one.step_s / params.cycle_s))
    start = torch.tensor([0, 7 * step_rows])
    inputs = FlightInputs(*(torch.cat([getattr(p[0], name) for p in physics]) for name in (
        "initial_state", "aero_params", "frame_params", "max_thrust_n")))
    executor = Executor(inputs, Runways.of([geometry] * 2, one, dtype=F64, device=CPU),
                        AirportCharts.of([geometry] * 2, dtype=F64, device=CPU), torch.cat([p[3] for p in physics]),
                        params, words, step_s=one.step_s, time_limit_s=torch.tensor(limits, dtype=F64),
                        start_cycle=start)
    sentences = Sentences(grids, words, step_s=one.step_s, device=CPU)
    first = inputs.initial_state[1].clone()
    heard = torch.zeros(2, dtype=F64)
    while executor.count < executor.cycles and not bool(executor.done.all()):
        own = executor.own_cycle()
        now_s = own.clamp(min=0).to(F64) * params.cycle_s
        heard = torch.where((own % step_rows == 0) & (own >= 0), now_s, heard)
        executor.cycle(sentences.at(heard), now_s)
        if executor.count <= start[1]:
            assert torch.equal(executor.state[1], first) and not bool(executor.done[1])
    together = executor.flown()
    for j, solo in enumerate(alone):
        done = int(solo.done_cycle[0])
        assert int(together.done_cycle[j]) == done
        assert torch.equal(together.states[j, :done + 2], solo.states[0, :done + 2])
        assert torch.equal(together.runway[j, :done + 1], solo.runway[0, :done + 1])
        for name in solo.modes:
            assert torch.equal(together.modes[name][j, :done + 1], solo.modes[name][0, :done + 1]), name


def test_words_spoken_to_a_multi_aircraft_batch_count_from_each_flights_first_step():
    from ts_transformer.autopilot.sentence import Spoken

    words = Words(spec())
    every = np.zeros((2, len(COLUMNS)), dtype=np.int64)
    spoken = Spoken(2, words, step_s=2.0, device=CPU, start_step=np.array([0, 2]))
    half = every.copy()
    half[1, :] = UNCHANGED
    spoken.say(half)
    spoken.say(half)
    with pytest.raises(ValueError, match="step 0 must write every column"):
        spoken.say(half)
    later = every.copy()
    later[0, :] = UNCHANGED
    spoken.say(later)
    spoken.at(torch.tensor([2 * 2.0, 0.0], dtype=F64))
    said = spoken.sentences()
    assert said.shape == (2, 3, len(COLUMNS)) and (said[1, 1:] == UNCHANGED).all() and (said[1, 0] == 0).all()


def test_the_single_flight_executor_flies_what_the_batch_flies():
    """§12.4: `autopilot.single` mirrors the batch's cycle — a go-around, a runway word and "no level-off" included —
    to round-off."""
    from ts_transformer.autopilot import single
    from ts_transformer.autopilot.executor import GO_AROUND_EXTRA_S

    words, params, geometry = Words(spec()), _params(), instruction_airport()
    signals, reading = _downwind()
    go = len(reading.words) - 15
    grid = np.vstack([reading.words, np.full((60, len(COLUMNS)), UNCHANGED)])
    grid[go, RUNWAY] = RUNWAY_GO_AROUND
    grid[go + 40, RUNWAY] = 0
    batch, _, _ = _fly_sentence(signals, grid, clock="time")
    inputs, _, _, approach = _physics(signals, geometry)
    executor = single.SingleExecutor(inputs, geometry, float(approach[0]), params, words, step_s=2.0,
                                     time_limit_s=len(grid) * 2.0 * params.timeout_factor, reserve_s=GO_AROUND_EXTRA_S)
    sentence = single.Sentence(grid, words, step_s=2.0)
    for cycle in range(executor.cycles):
        executor.cycle(sentence.at(float(cycle - cycle % 2)), float(cycle))
        if executor.done:
            break
    flown = executor.flown()
    done = int(batch.done_cycle[0])
    assert int(flown.done_cycle[0]) == done
    assert np.abs(flown.states[0, :done + 2, 2].numpy() - batch.states[0, :done + 2, 2].numpy()).max() < 1e-6
    assert torch.equal(flown.runway[0, :done + 1], batch.runway[0, :done + 1])
    for name in batch.modes:
        assert torch.equal(flown.modes[name][0, :done + 1], batch.modes[name][0, :done + 1]), name


# ---- the spec, the hash and the replay
def test_the_executor_spec_is_written_once_and_opened_only_for_code_that_flew_its_reference(tmp_path, monkeypatch):
    import json
    import platform

    from ts_transformer.autopilot import spec as executor_spec
    from ts_transformer.tests.support import passed_executor

    params = _params()
    source = {"executor_source_sha256": executor_spec.executor_source_sha256(), "python": platform.python_version(),
              "git": {"head": "x", "dirty": False}}
    executor_spec.write_spec(tmp_path, params, "vocabulary", {"data": {}}, source)
    loaded, record = executor_spec.load_spec(tmp_path)
    assert loaded == params and record["sha256"] == executor_spec.params_sha256(params)
    assert record["schema"] == "ts-executor-spec-v7"
    with pytest.raises(FileExistsError):
        executor_spec.write_spec(tmp_path, params, "vocabulary", {}, source)
    assert executor_spec.params_sha256(replace(params, decision_cone_share=0.5)) != record["sha256"]
    with pytest.raises(ValueError, match=r"has not been checked against .* reference tracks"):
        executor_spec.require_conforming_executor(tmp_path)
    passed_executor(tmp_path)
    executor_spec.require_conforming_executor(tmp_path)
    stored = json.loads((tmp_path / "spec.json").read_text())
    for broken, message in (({**stored, "params": {**stored["params"], "bank_rate_deg_s": 9.0}}, "do not hash"),
                            ({**stored, "params": {k: v for k, v in stored["params"].items()
                                                   if k != "decision_cone_share"}}, "missing"),
                            ({**stored, "schema": "ts-executor-spec-v6"}, "is not a ts-executor-spec-v7 file")):
        (tmp_path / "spec.json").write_text(json.dumps(broken))
        with pytest.raises(ValueError, match=message):
            executor_spec.load_spec(tmp_path)


def test_the_executor_hash_reads_the_logic_and_covers_what_it_imports_from_the_repository():
    from ts_transformer.autopilot import spec as executor_spec
    from ts_transformer.io_utils import logic

    assert logic('"""doc"""\nx = 1  # note\n') == logic("x = 1\n")
    labels = [label for label, _ in executor_spec.executor_source_files()]
    package = {path.name for path in executor_spec.PACKAGE.glob("*.py")} - {"spec.py"}
    assert {f"autopilot/{name}" for name in package} <= set(labels) and "autopilot/spec.py" not in labels
    for needed in ("aerodynamic_model.torch_dynamics", "aircraft.reference_speeds",
                   "ts_transformer.outputs.dynamics.rollout", "ts_transformer.outputs.envelope", "geokit",
                   "flight_scenarios.fas_geometry"):
        assert needed in labels
    assert set(executor_spec.REACHED_MODULES) <= set(labels)
    assert not any(label.startswith(("ts_transformer.instructions", "ts_transformer.io_utils", "evaluation.cli"))
                   for label in labels)


def test_an_executor_spec_is_opened_only_against_its_own_vocabulary_and_a_conforming_labeller(tmp_path, monkeypatch):
    from ts_transformer.autopilot import replay
    from ts_transformer.autopilot import spec as executor_spec
    from ts_transformer.tests.support import passed_executor

    one = spec()
    source = {"executor_source_sha256": executor_spec.executor_source_sha256(), "python": "3",
              "git": {"head": "x", "dirty": False}}
    executor_spec.write_spec(tmp_path / "spec", _params(), one.sha256, {}, source)
    passed_executor(tmp_path / "spec")
    monkeypatch.setattr(replay.artefact, "load_spec", lambda directory: one)
    labeller = {"conforms": True}

    def require(directory):
        if not labeller["conforms"]:
            raise ValueError("the labeller code on disk has not been checked")
    monkeypatch.setattr(replay, "require_conforming_labeller", require)
    params, _, _ = replay.open_executor(tmp_path / "spec", tmp_path / "artefact")
    assert params == _params()
    labeller["conforms"] = False
    with pytest.raises(ValueError, match="labeller code on disk"):
        replay.open_executor(tmp_path / "spec", tmp_path / "artefact")
    labeller["conforms"] = True
    monkeypatch.setattr(replay.artefact, "load_spec", lambda directory: spec(heading_tolerance_deg=4.0))
    with pytest.raises(ValueError, match="measured against vocabulary"):
        replay.open_executor(tmp_path / "spec", tmp_path / "artefact")


def test_each_candidate_reads_its_runways_published_vertical_path_and_decision_altitude(monkeypatch):
    from ts_transformer.autopilot import runway_data
    from ts_transformer.autopilot.runway_data import VerticalPath

    geometry = _two_runways()
    idents = [c.ident for c in geometry.candidates]
    published = {ident: (15.0 + k, 3.0 + 0.5 * k, 60.0 + 10 * k) for k, ident in enumerate(idents)}

    def runways(guided=True):
        return [SimpleNamespace(ident=ident, threshold_crossing_height_m=tch, published_glidepath_deg=glidepath,
                                decision_height_above_threshold_m=da,
                                published_minima=SimpleNamespace(vertically_guided=guided, note="no plate"))
                for ident, (tch, glidepath, da) in reversed(published.items())]

    monkeypatch.setattr(runway_data, "load_airport", lambda code, **_: SimpleNamespace(runways=runways()))
    paths = tuple(VerticalPath(*published[ident]) for ident in idents)
    assert runway_data.vertical_paths(geometry, runways()) == paths
    assert runway_data.published_vertical_paths(geometry) == paths
    with pytest.raises(ValueError, match="publishes no decision altitude"):
        runway_data.vertical_paths(geometry, runways(guided=False))
    with pytest.raises(ValueError, match="not among the harvest's runways"):
        runway_data.vertical_paths(geometry, [r for r in runways() if r.ident != idents[0]])
    published[idents[0]] = (None, 3.0, 60.0)
    with pytest.raises(ValueError, match="publishes no threshold crossing height or glidepath"):
        runway_data.vertical_paths(geometry, runways())


def test_a_flight_is_flown_on_its_own_or_a_stand_ins_dynamics_only_with_a_published_approach_speed(monkeypatch):
    from ts_transformer.autopilot import replay

    def series(resolved, dynamics):
        source = {"resolved_typecode": resolved, "dynamics_typecode": dynamics}
        return SimpleNamespace(scenario=SimpleNamespace(source=source, initial=SimpleNamespace(m=62000.0),
                                                        has_dynamics=dynamics is not None))

    assert replay.group_of(series("A320", "A320")) == replay.OWN
    assert replay.group_of(series(None, None)) == "no identified type"
    assert replay.group_of(series("GLF4", "CRJ9")) == replay.STAND_IN
    assert replay.group_of(series("PC12", None)) == "no aircraft dynamics"
    monkeypatch.setattr(replay, "approach_speed_ias_mps", lambda typecode, mass: math.nan)
    assert replay.group_of(series("A320", "A320")) == "type publishes no approach speed"


def _batch(signals, reading, interval_s=2.0):
    from ts_transformer.autopilot import replay

    geometry, words = instruction_airport(), Words(spec())
    sentence = replay.sentence_on_interval(reading, signals, interval_s, geometry, words)
    return replay.Batch(signals=[replay.from_row(signals, sentence.first_row)], series=[], readings=[reading],
                        sentences=[sentence], row_interval_s=interval_s, geometries=[geometry],
                        vertical_paths=[vertical_paths(geometry)], approach_ias_mps=[], groups=[replay.OWN], drawn={})


def test_a_sentence_on_a_coarser_interval_starts_on_its_utc_grid():
    """§4.8 (D11, D25): the batch's sentence is the labelled one at 2 s; at 4 s it starts on the first row on a UTC
    multiple of 4 s, the observed flight from there."""
    signals, reading = _downwind()
    two = _batch(signals, reading).sentences[0]
    assert two.first_row == 0 and np.array_equal(two.grid, reading.words)
    assert [(i.column, i.value, i.row) for i in two.instructions] == [
        (i.column, i.value, i.row) for i in sorted(reading.instructions, key=lambda i: (i.row, i.column))]
    shifted = replace(signals, entry_time_utc="2026-06-01T11:00:02Z")
    batch = _batch(shifted, reading, 4.0)
    four = batch.sentences[0]
    assert four.first_row == 1 and len(four.grid) == len(range(1, len(reading.words), 2))
    assert batch.signals[0].e_m[0] == signals.e_m[1] and batch.signals[0].entry_time_utc == "2026-06-01T11:00:04Z"


def test_a_batch_readout_counts_what_was_flown_and_how_far_it_lies_from_the_observed_track():
    from ts_transformer.autopilot import replay

    signals, reading = _downwind()
    flown, verdict, _ = _fly_sentence(signals)
    batch = _batch(signals, reading)
    summary = replay.summary([verdict])
    judged = replay.word_results(verdict)[0]
    assert summary["outcomes"] == {"landed": 1} and summary["words_judged"] == len(judged)
    assert summary["words_inside_share"] == 1.0 and summary["words_inside_share_by_column"]["heading"] == 1.0
    assert summary["decision_checks"] == {"approach_crossings_low": 1, "no_da_point": 0, "passed": 1,
                                          "lateral_failed": 0, "vertical_failed": 0}
    aligned = replay.alignment(batch, flown, [verdict])
    assert aligned["mean_horizontal_distance_m"]["p50"] < 1000.0 and aligned["mean_vertical_distance_m"]["p50"] < 60.0
    landing = aligned["landing_time_minus_observed_s"]
    observed = (len(reading.words) - 1) * 2.0 + 400.0 / 75.0
    assert landing["n"] == 1 and landing["p50"] == pytest.approx(verdict.crossing["at_row"] - observed)


def test_a_replayed_flight_becomes_a_control_record_and_the_readout_reads_no_criterion():
    from ts_transformer.autopilot.plant import EXECUTOR_DYNAMICS
    from ts_transformer.experiments.executor_replay import executor_forecast, readout_table

    signals, _ = _downwind()
    flown, verdict, _ = _fly_sentence(signals)
    inputs, _ = _a320([[0.0] * 7])
    series = _observed_series(instruction_airport())
    forecast = executor_forecast(flown, 0, verdict, inputs, series)
    assert forecast.anchor == 0 and forecast.n_steps == verdict.end_row == len(forecast.controls)
    assert forecast.truncated_at_threshold and not forecast.horizon_capped
    assert forecast.control_parameterization == EXECUTOR_DYNAMICS.control_thrust_parameterization
    base = {"airport": "KXXX", "group": "own dynamics", "stratum": "vectored", "words_not_reached": 0,
            "heading_words_not_judged": 0, "heading_words_told_with_a_skipped_word": {"judged": 0, "inside": 0},
            "crossing": None}
    rows = [{**base, "kind": "with go-around", "outcome": "landed",
             "words": [("heading", True)] * 9 + [("altitude", False)], "observed_verdict": "pass",
             "replay_verdict": "pass", "crossing": {"decision": {"passed": True}}},
            {**base, "kind": "without go-around", "outcome": "unstable_at_minimums", "words": None,
             "observed_verdict": "fail", "replay_verdict": "fail", "crossing": {"decision": None}}]
    table = readout_table(rows)
    cell = table["own dynamics"]["all"]["all"]["all"]
    assert (cell["flights"], cell["landed"], cell["words_inside"]) == (2, 0.5, 0.9)
    assert cell["words_inside_by_column"] == {"heading": 1.0, "altitude": 0.0, "speed": None}
    assert cell["decision_checks"] == {"n": 2, "no_da_point": 1, "passed": 1}
    assert table["own dynamics"]["KXXX"]["vectored"]["with go-around"]["flights"] == 1
    assert "clears" not in cell


def _observed_series(geometry):
    from ts_transformer.data.coordinate_frames import ENUFrame
    from ts_transformer.data.dataset import FlightSeries

    candidate = geometry.candidates[0]
    lat, lon = geometry.frame.latlon_from_horizontal(candidate.threshold_e_m, candidate.threshold_n_m)
    scenario = SimpleNamespace(source={"arr_airport": "KXXX", "runway": "09", "resolved_typecode": "A320"},
                               initial=SimpleNamespace(m=62000.0),
                               target=SimpleNamespace(latitude=lat, longitude=lon, psi=0.0),
                               aircraft=SimpleNamespace(code="A320"))
    return FlightSeries(flight_id="TEST1", scenario=scenario, frame=ENUFrame(lat0=lat, lon0=lon, alt0=candidate.elevation_m),
                        times=np.zeros(1), values=np.zeros((1, 6)))


def test_draw_reads_a_seeded_permutation_until_each_airport_is_full(monkeypatch):
    from ts_transformer.autopilot import replay
    from ts_transformer.autopilot.runway_data import VerticalPath

    flights = [SimpleNamespace(airport="KAAA" if i % 3 else "KBBB", dataset_id=f"F{i}") for i in range(30)]
    stored = [np.full((3, len(COLUMNS)), i, dtype=np.int16) for i in range(30)]
    sentences = {"signal_index": np.arange(30), "offsets": np.arange(31) * 3,
                 "words": np.concatenate(stored), "runway_index": np.zeros(30, dtype=np.int64)}
    typecode = {i: ("A320", "A320") if i % 4 else ("CRJ7", "A320") for i in range(30)}
    geometries = {code: SimpleNamespace(code=code, candidates=(SimpleNamespace(ident="09"),)) for code in ("KAAA", "KBBB")}
    monkeypatch.setattr(replay, "load_candidates", lambda d: geometries)
    paths = {"KAAA": (VerticalPath(15.0, 3.0, 60.0),), "KBBB": (VerticalPath(16.0, 3.5, 70.0),)}
    monkeypatch.setattr(replay, "published_vertical_paths", lambda geometry: paths[geometry.code])
    monkeypatch.setattr(replay, "load_signals", lambda d, split: flights)
    monkeypatch.setattr(replay, "load_sentences", lambda d, split, spec: sentences)
    monkeypatch.setattr(replay, "rebuild_series", lambda d, items: [SimpleNamespace(scenario=SimpleNamespace(
        source=dict(zip(("resolved_typecode", "dynamics_typecode"), typecode[int(f.dataset_id[1:])])),
        initial=SimpleNamespace(m=60000.0), has_dynamics=True)) for f in items])
    reread = {"differ": None}
    monkeypatch.setattr(replay, "read_flight", lambda f, g, s, w: SimpleNamespace(
        words=stored[int(f.dataset_id[1:])] + (1 if f.dataset_id == reread["differ"] else 0), runway_index=0))
    monkeypatch.setattr(replay, "batch_of", lambda drawn, keep, readings, interval, words: drawn)
    one, words = spec(), Words(spec())
    first = replay.draw(Path("x"), "train", one, words, per_airport=3, seed=5, row_interval_s=2.0)
    again = replay.draw(Path("x"), "train", one, words, per_airport=3, seed=5, row_interval_s=2.0)
    assert [f.dataset_id for f in first.signals] == [f.dataset_id for f in again.signals]
    assert Counter(f.airport for f in first.signals) == {"KAAA": 3, "KBBB": 3}
    with pytest.raises(ValueError, match="too few eligible flights"):
        replay.draw(Path("x"), "train", one, words, per_airport=9, seed=5, row_interval_s=2.0)
    reread["differ"] = first.signals[0].dataset_id
    with pytest.raises(ValueError, match="differs from the stored one"):
        replay.draw(Path("x"), "train", one, words, per_airport=3, seed=5, row_interval_s=2.0)


def test_the_spec_runner_takes_the_decision_altitude_tolerances_from_the_user_only():
    """O2, D7: the two tolerances are required on the command line, with no default."""
    from ts_transformer.experiments import executor_spec

    for missing in ([], ["--decision-cone-share", "1.0"], ["--decision-glidepath-tolerance-m", "30"]):
        with pytest.raises(SystemExit):
            executor_spec.main(["--instructions", "x", "--dir", "y", "--word-clock", "track", *missing])


def test_a_sentence_on_a_coarser_interval_is_flown_and_judged_on_its_own_rows():
    """§4.8: at Δ = 4 s from the first row on a UTC multiple of 4 s the executor hears a row every 4 s, and the judge
    files each word at the flown row (the data's 2 s) where it was heard: every heading word is judged."""
    from ts_transformer.autopilot import replay
    from ts_transformer.autopilot.executor import fly
    from ts_transformer.autopilot.judge import judge

    one, words, params, geometry = spec(), Words(spec()), _params(), instruction_airport()
    signals, reading = _downwind()
    shifted = replace(signals, entry_time_utc="2026-06-01T11:00:02Z")
    batch = _batch(shifted, reading, 4.0)
    sentence, observed = batch.sentences[0], batch.signals[0]
    assert sentence.first_row == 1
    inputs, runways, charts, approach = _physics(observed, geometry)
    e_m, n_m = replay.observed_rows(observed, sentence, 4.0, one.step_s)
    assert len(e_m) == min(len(sentence.grid) * 2, observed.n_rows)    # the clock reads the data's 2 s rows
    flown = fly(inputs, Sentences([sentence.grid], words, step_s=4.0, device=CPU),
                DistanceClock.of([e_m], [n_m], one.step_s, params.cycle_s, device=CPU), runways, charts, approach,
                params, words, time_limit_s=torch.tensor([len(sentence.grid) * 4.0 * 1.5], dtype=F64), reserve_s=0.0)
    verdict = judge(flown, 0, geometry, vertical_paths(geometry), sentence.instructions, 4.0, observed, one, words,
                    params)
    headings = [h for h in verdict.words["heading"] if h["rows"]]
    assert len(headings) == sum(i.column == HEADING for i in sentence.instructions)
    assert verdict.outcome in ("landed", "unstable_at_minimums", "crossed_off_runway", "crossed_too_high")


def test_a_multi_aircraft_batch_with_go_arounds_flies_each_as_it_flies_alone():
    """Executor design §12.4 with D10: two flights, each with a go-around and a runway word after it, staggered by 7
    steps — each flies what it flies alone, its time limit grown by its own go-around."""
    from ts_transformer.autopilot.executor import GO_AROUND_EXTRA_S, Executor, fly
    from ts_transformer.autopilot.lateral import Runways

    one, geometry = spec(), instruction_airport()
    words, params = Words(one), _params()
    signals, reading = _downwind()
    grids = []
    for go in (len(reading.words) - 15, len(reading.words) - 25):
        grid = np.vstack([reading.words, np.full((60, len(COLUMNS)), UNCHANGED)])
        grid[go, RUNWAY], grid[go + 30, RUNWAY] = RUNWAY_GO_AROUND, 0
        grids.append(grid)
    physics = _physics(signals, geometry)
    limits = [len(g) * one.step_s * params.timeout_factor for g in grids]
    alone = [fly(physics[0], Sentences([g], words, step_s=2.0, device=CPU), TimeClock(params.cycle_s), physics[1],
                 physics[2], physics[3], params, words, time_limit_s=torch.tensor([limit], dtype=F64),
                 reserve_s=GO_AROUND_EXTRA_S) for g, limit in zip(grids, limits)]
    inputs = FlightInputs(*(torch.cat([getattr(physics[0], name)] * 2) for name in (
        "initial_state", "aero_params", "frame_params", "max_thrust_n")))
    executor = Executor(inputs, Runways.of([geometry] * 2, one, dtype=F64, device=CPU),
                        AirportCharts.of([geometry] * 2, dtype=F64, device=CPU), torch.cat([physics[3]] * 2), params,
                        words, step_s=2.0, time_limit_s=torch.tensor(limits, dtype=F64),
                        start_cycle=torch.tensor([0, 14]), reserve_s=GO_AROUND_EXTRA_S)
    sentences = Sentences(grids, words, step_s=2.0, device=CPU)
    heard = torch.zeros(2, dtype=F64)
    while executor.count < executor.cycles and not bool(executor.done.all()):
        own = executor.own_cycle()
        now_s = own.clamp(min=0).to(F64) * params.cycle_s
        heard = torch.where((own % 2 == 0) & (own >= 0), now_s, heard)
        executor.cycle(sentences.at(heard), now_s)
        executor.halt(executor.done)
    together = executor.flown()
    assert executor.time_limit_s.tolist() == pytest.approx([limit + GO_AROUND_EXTRA_S for limit in limits])
    for j, solo in enumerate(alone):
        done = int(solo.done_cycle[0])
        assert int(together.done_cycle[j]) == done
        assert torch.equal(together.states[j, :done + 2], solo.states[0, :done + 2])
        for name in solo.modes:
            assert torch.equal(together.modes[name][j, :done + 1], solo.modes[name][0, :done + 1]), name
