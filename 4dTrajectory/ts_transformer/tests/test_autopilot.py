"""The executor (`autopilot/`, executor design): conventions, words in force, the one-cycle plant and
the exact inverse, on synthetic states of a real airframe."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from aircraft.aero_params import aero_params_for_aircraft
from flight_scenarios.scenario import aircraft_for_code
from ts_transformer.autopilot import inverse
from ts_transformer.autopilot.flights import FlightInputs
from ts_transformer.autopilot.frame import AirportCharts, compass_deg, read_state, wrap180
from ts_transformer.autopilot.plant import Plant
from ts_transformer.autopilot.sentence import Delays, DistanceClock, Sentences, TimeClock, TrackClock
from ts_transformer.instructions.words import (
    ALTITUDE, ANGLE, APPROACH, APPROACH_CLEARED, APPROACH_NOT_CLEARED, HEADING, SPEED, UNCHANGED, Words,
    compass_from_math_rad, wrap180 as np_wrap180,
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
    # a state flying north-east, descending 3° at 100 m/s
    state = torch.tensor([[35.1, -78.3, 900.0, 100.0, math.radians(45.0), math.radians(-3.0), 60000.0]], dtype=F64)
    k = read_state(state, AirportCharts.of([geometry], dtype=F64, device=CPU))
    assert float(k.track_deg) == pytest.approx(45.0) and float(k.height_m) == 900.0
    assert float(k.ground_speed_mps) == pytest.approx(100.0 * math.cos(math.radians(3.0)))


# ---- words in force, and the clocks they are said on
def _grid():
    words = Words(spec())
    grid = np.full((5, 6), UNCHANGED, dtype=np.int64)
    grid[0] = [0, APPROACH_NOT_CLEARED, words.heading_index(270.0), words.altitude_index(1200.0), 0,
               words.speed_index(100.0)]
    grid[2, HEADING] = words.heading_index(180.0)           # issued at t = 4 s
    grid[2, APPROACH] = APPROACH_CLEARED
    grid[3, ALTITUDE] = words.altitude_land                 # issued at t = 6 s
    grid[3, ANGLE] = 3
    grid[4, SPEED] = words.speed_unspecified                # issued at t = 8 s
    return grid


def test_each_word_takes_effect_its_columns_delay_after_its_step_and_stays_in_force():
    one, words = spec(), Words(spec())
    sentences = Sentences([_grid()], words, device=CPU)
    delays = Delays(heading_s=3.0, vertical_s=1.0, speed_s=0.0)
    at = [sentences.at(torch.tensor([float(t)], dtype=F64), delays) for t in range(20)]
    heading = np.array([float(w.heading_deg[0]) for w in at])
    assert (heading[:7] == 270.0).all() and (heading[7:] == 180.0).all()        # 4 s + 3 s
    assert at[6].approach[0] == APPROACH_NOT_CLEARED and at[7].approach[0] == APPROACH_CLEARED
    land = np.array([bool(w.land[0]) for w in at])
    assert not land[:7].any() and land[7:].all()                                 # 6 s + 1 s
    assert float(at[0].altitude_m[0]) == 1200.0 and float(at[7].angle_deg[0]) == one.descent_angle_centres_deg[2]
    unspecified = np.array([bool(w.unspecified[0]) for w in at])
    assert not unspecified[:8].any() and unspecified[8:].all()
    assert all(int(w.runway[0]) == 0 for w in at)
    assert at[6].issued_step[0, HEADING] == 0 and at[7].issued_step[0, HEADING] == 2
    assert at[19].issued_step[0, SPEED] == 4                                     # held after the last step


def test_a_sentence_must_write_every_column_at_step_0():
    words = Words(spec())
    grid = np.full((3, 6), UNCHANGED, dtype=np.int64)
    grid[0, :5] = [0, 0, 0, 0, 0]
    with pytest.raises(ValueError, match="step 0 must write every column"):
        Sentences([grid], words, device=CPU)


def test_the_distance_clock_says_a_word_where_the_observed_aircraft_was_told_it():
    """Observed: 100 m every 2 s row. An executor twice as fast reaches row 3's position in 3 s, not 6 s;
    past the observed path's end sentence time runs on at the executor's own pace."""
    from ts_transformer.autopilot.frame import Kinematics

    e = np.arange(5) * 100.0
    clock = DistanceClock.of([e], [np.zeros(5)], 2.0, 1.0, device=CPU)

    def at(position):
        state = Kinematics(*(torch.tensor([value], dtype=F64) for value in (position, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
        return float(clock.now(0, state)[0])

    times = [at(100.0 * t) for t in range(7)]                                   # 100 m a second
    assert times == pytest.approx([0.0, 2.0, 4.0, 6.0, 8.0, 9.0, 10.0])        # past the end: the executor's pace
    assert float(TimeClock(1.0).now(7, Kinematics(*(torch.zeros(1, dtype=F64) for _ in range(8))))[0]) == 7.0


def test_a_rebuilt_flight_must_be_the_one_the_signals_were_read_from(monkeypatch):
    from dataclasses import replace
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
    geometry = instruction_airport()
    return inputs, AirportCharts.of([geometry] * rows, dtype=F64, device=CPU)


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
    """§13 E2: one cycle forward through the dynamics gives the rates the inverse solved for, to
    within what the zero-order hold costs (V and γ move inside the cycle while the controls do not)."""
    inputs, charts = _a320([_state(45.0, 0.0), _state(45.0, -3.0), _state(300.0, -3.0, 80.0, 900.0),
                            _state(10.0, 2.0, 120.0)])
    asked_track = torch.tensor([2.0, -2.5, 1.5, 0.0], dtype=F64)             # compass deg/s, + = right
    asked_gamma = torch.tensor([0.0, math.radians(0.3), math.radians(-0.2), 0.0], dtype=F64)
    asked_accel = torch.tensor([-0.3, -0.2, 0.0, 0.4], dtype=F64)
    att, thr, (track, gamma, speed) = _fly(inputs, charts, asked_track, asked_gamma, asked_accel)
    assert not any(bool(b.any()) for b in (*att.binds.values(), *thr.binds.values()))
    assert track.numpy() == pytest.approx(asked_track.numpy(), rel=0.02, abs=0.01)
    assert gamma.numpy() == pytest.approx(asked_gamma.numpy(), rel=0.05, abs=math.radians(0.01))
    # the speed rate moves with the gravity term while γ turns inside the cycle: g · |γ̇| · Δt / 2
    hold = 9.81 * np.abs(asked_gamma.numpy()) * 1.0 / 2.0
    assert np.all(np.abs(speed.numpy() - asked_accel.numpy()) <= hold + 0.005)
    # a right (clockwise) turn is a NEGATIVE bank in the dynamics' convention, a left turn positive
    assert float(att.bank_rad[0]) < 0.0 < float(att.bank_rad[1])


def test_each_limit_binds_on_the_quantity_it_bounds():
    inputs, charts = _a320([_state(90.0, 0.0)] * 4)
    # a bank cap costs turn rate and keeps the path: 10°/s at 100 m/s wants ~60° of bank
    att, _, (track, gamma, _) = _fly(inputs, charts, 10.0, 0.0, 0.0, bank_cap_deg=25.0)
    assert att.binds["bank_cap"].all() and float(att.bank_rad[0]) == pytest.approx(-math.radians(25.0))
    capped_rate = math.degrees(9.81 * math.tan(math.radians(25.0)) / 100.0)
    assert float(track[0]) == pytest.approx(capped_rate, rel=0.02) and abs(float(gamma[0])) < math.radians(0.02)
    # the bank moves at most its rate per cycle
    att, _, _ = _fly(inputs, charts, 3.0, 0.0, 0.0, torch.zeros(4, dtype=F64), bank_rate_deg_s=2.0)
    assert att.binds["bank_rate"].all() and float(att.bank_rad[0]) == pytest.approx(-math.radians(2.0))
    # a pull the envelope does not allow is clamped to its load factor
    att, _, _ = _fly(inputs, charts, 0.0, math.radians(20.0), 0.0)
    assert att.binds["load_factor"].all() and float(att.load_factor[0]) == inverse.LOAD_FACTOR_MAX
    # more acceleration than the engines give, and more deceleration than idle and drag give
    _, thr, _ = _fly(inputs, charts, 0.0, 0.0, 6.0)
    assert thr.binds["thrust_max"].all() and float(thr.fraction[0]) == pytest.approx(1.0)
    _, thr, _ = _fly(inputs, charts, 0.0, 0.0, -8.0)
    assert thr.binds["thrust_min"].all() and not thr.binds["stall"].any()


def test_the_bank_cap_stays_inside_the_graders_envelope():
    inputs, charts = _a320([_state(0.0, 0.0)])
    k = read_state(inputs.initial_state, charts)
    with pytest.raises(ValueError, match="bank cap"):
        inverse.attitude(k, torch.zeros(1, dtype=F64), torch.zeros(1, dtype=F64), torch.zeros(1, dtype=F64),
                         bank_cap_rad=math.radians(50.0), bank_rate_rad_s=1.0, cycle_s=1.0)


# ---- the lateral law and its mirrors of the labeller
def test_the_lateral_mirrors_equal_the_labellers_own_functions():
    from ts_transformer.autopilot import lateral
    from ts_transformer.instructions import envelope
    from ts_transformer.instructions.airport import relative_to_runway

    one = spec()
    rng = np.random.default_rng(7)
    heading, course = rng.uniform(0, 360, 400), rng.uniform(0, 360, 400)
    right, before = rng.uniform(-8000, 8000, 400), rng.uniform(-2000, 30000, 400)
    right[:40] = rng.uniform(-60, 60, 40)                       # some inside the corridor's width
    for tolerance in (0.0, one.heading_tolerance_deg):
        mine = lateral.converges(*(torch.tensor(a, dtype=F64) for a in (heading, course, right, before)), tolerance, one)
        theirs = [envelope.heading_converges(h, c, r, b, tolerance, one.corridor_half_width_m, one.corridor_widening_deg)
                  for h, c, r, b in zip(heading, course, right, before)]
        assert mine.tolist() == theirs
    assert lateral.corridor_half_width(torch.tensor(before, dtype=F64), one).numpy() == pytest.approx(
        envelope.corridor_half_width_m(before, one.corridor_half_width_m, one.corridor_widening_deg))
    candidate = instruction_airport().candidates[0]
    e, n, track = rng.uniform(-20000, 5000, 50), rng.uniform(-9000, 9000, 50), rng.uniform(0, 360, 50)
    k = read_state(torch.zeros(50, 7, dtype=F64), AirportCharts.of([instruction_airport()] * 50, dtype=F64, device=CPU))
    k = type(k)(**{**k.__dict__, "e_m": torch.tensor(e), "n_m": torch.tensor(n), "track_deg": torch.tensor(track)})
    ours = lateral.relative(k, torch.tensor(candidate.threshold_e_m, dtype=F64),
                            torch.tensor(candidate.threshold_n_m, dtype=F64), torch.tensor(candidate.course_deg, dtype=F64))
    ref = relative_to_runway(e, n, track, np.zeros(50), candidate)
    assert ours[0].numpy() == pytest.approx(ref.before_threshold_m) and ours[1].numpy() == pytest.approx(ref.right_of_course_m)
    assert ours[2].numpy() == pytest.approx(ref.track_minus_course_deg)


def test_the_heading_law_turns_the_shorter_way_at_the_steady_rate_and_eases_out():
    from ts_transformer.autopilot.lateral import heading_rate

    params = _params()
    rate = heading_rate(torch.tensor([10.0, 350.0, 100.0, 100.0], dtype=F64),
                        torch.tensor([350.0, 10.0, 104.0, 100.0], dtype=F64), params)
    # 10 → 350 is 20° LEFT (not 340° right); 350 → 10 is 20° right; 4° to go eases to 4/τ_ψ
    assert rate.tolist() == pytest.approx([-params.turn_rate_deg_s, params.turn_rate_deg_s,
                                           4.0 / params.heading_time_constant_s, 0.0])


def test_the_parameters_are_checked_against_the_designs_constraints():
    from dataclasses import replace

    one, params = spec(), _params()
    params.check(one)
    for change, message in ((dict(heading_time_constant_s=1.5), "under 2 Δt"),
                            (dict(turn_rate_deg_s=3.0, heading_time_constant_s=4.0), "split turn"),
                            (dict(bank_cap_deg=40.0), "φ_cap"),
                            (dict(delays=Delays(12.0, 0.0, 0.0)), "may start late")):
        with pytest.raises(ValueError, match=message):
            replace(params, **change).check(one)


# ---- whole flights
def _params(**changes):
    from dataclasses import replace
    from ts_transformer.autopilot.params import ExecutorParams

    base = ExecutorParams(cycle_s=1.0, turn_rate_deg_s=2.15, bank_cap_deg=25.0, heading_time_constant_s=4.0,
                          bank_rate_deg_s=2.0, path_time_constant_s=2.0, path_rate_factor=2.0, decel_mps2=0.24,
                          accel_mps2=0.16, unspecified_decel_mps2=0.26, land_aim_height_m=20.8, land_window_low_m=8.4,
                          land_window_high_m=36.5,
                          delays=Delays(0.0, 0.0, 0.0),
                          timeout_factor=1.5, word_clock="time")
    return replace(base, **changes)


def _fly_sentence(signals, grid=None, params=None, approach_ias=None, early_words=False):
    """Read ``signals`` with the labeller, fly its sentence (or ``grid``) from row 0, judge it; the A320's
    published approach speed unless ``approach_ias`` is given."""
    from ts_transformer.autopilot.executor import fly
    from ts_transformer.autopilot.judge import judge
    from ts_transformer.autopilot.lateral import Runways
    from ts_transformer.autopilot.speed import approach_speed_ias_mps
    from ts_transformer.instructions.labeller.read import read_flight

    one, words, geometry = spec(), Words(spec()), instruction_airport()
    params = params or _params()
    reading = read_flight(signals, geometry, one, words)
    grid = reading.words if grid is None else grid
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
    limit = len(grid) * one.step_s * params.timeout_factor
    flown = fly(inputs, Sentences([grid], words, device=CPU), TimeClock(params.cycle_s),
                Runways.of([geometry], dtype=F64, device=CPU),
                AirportCharts.of([geometry], dtype=F64, device=CPU),
                torch.tensor([approach_speed_ias_mps("A320", mass) if approach_ias is None else approach_ias],
                             dtype=F64), params, words,
                time_limit_s=torch.tensor([limit], dtype=F64), early_words=early_words)
    return flown, judge(flown, 0, geometry, 0, reading, signals, one, words), reading


DOWNWIND_BASE_FINAL = [(60, 0.0, 100.0, 0.0), (15, -6.0, 100.0, 0.0), (20, 0.0, 90.0, 0.0), (15, -6.0, 85.0, 0.0),
                       (120, 0.0, 75.0, -75.0 * np.tan(np.radians(3.0)))]


def test_a_downwind_base_final_sentence_is_flown_to_the_runway():
    """§13 E6: a whole flight — downwind, base, the capture, the final — lands on the pointed runway with
    every word inside its envelope, and the capture turn ends on the line without crossing it."""
    signals = instruction_flight(*fly_legs(DOWNWIND_BASE_FINAL, 270.0, 1110.0, -400.0, 0.0))
    flown, verdict, _ = _fly_sentence(signals)
    assert verdict.outcome == "landed" and verdict.flew_the_sentence
    assert abs(verdict.crossing["cross_m"]) < 1.0 and 0.0 < verdict.crossing["height_m"] < 100.0
    assert verdict.words["corridor"]["inside"] == verdict.words["corridor"]["rows"] > 0
    # the line is held from the north: the flown track lags the bank, so it may cross it by metres, no more
    captured = flown.modes["captured"][0].numpy()
    k = read_state(flown.states[0, 1:][captured], AirportCharts.of([instruction_airport()] * int(captured.sum()),
                                                                   dtype=F64, device=CPU))
    assert float(k.n_m.min()) > -5.0
    assert not any(verdict.limits[name]["cycles"] for name in ("thrust_max", "thrust_min", "stall", "load_factor"))


def test_a_flight_on_the_final_from_row_0_is_captured_at_once_and_lands():
    straight = [(40, 0.0, 90.0, 0.0), (30, 0.0, 80.0, 0.0), (110, 0.0, 72.0, -72.0 * np.tan(np.radians(3.0)))]
    signals = instruction_flight(*fly_legs(straight, 90.0, 950.0, -300.0, 0.0))     # ~50 m over the threshold
    flown, verdict, _ = _fly_sentence(signals)
    assert bool(flown.modes["captured"][0, 0]) and verdict.outcome == "landed" and verdict.flew_the_sentence


def test_a_split_turn_is_flown_without_levelling_between_its_parts():
    legs = [(30, 0.0, 100.0, 0.0), (30, 6.0, 100.0, 0.0), (20, 0.0, 100.0, 0.0), (15, -6.0, 100.0, 0.0),
            (100, 0.0, 75.0, -75.0 * np.tan(np.radians(3.0)))]
    signals = instruction_flight(*fly_legs(legs, 0.0, 950.0, -400.0, 0.0))
    flown, verdict, reading = _fly_sentence(signals)
    parts = [i for i in reading.instructions if i.kind == "turn-split"]
    assert len(parts) == 2
    # from the first part's row to where the flown track reaches the second part's target, the bank
    # stays on the right-turn side (negative in the dynamics' convention) — the turn does not stop
    bank = flown.commands[0, :, 1].numpy()
    first, second = parts[0].row * 2, parts[1].row * 2
    assert (bank[first + 12: second + 12] < -math.radians(5.0)).all()
    assert verdict.outcome == "landed"


def test_the_flights_outcome_is_the_first_event_it_meets():
    from ts_transformer.instructions.words import APPROACH_CLEARED as CLEARED

    signals = instruction_flight(*fly_legs(DOWNWIND_BASE_FINAL, 270.0, 1110.0, -400.0, 0.0))
    _, verdict, reading = _fly_sentence(signals)
    # never cleared: the executor holds the base heading past the line; told to descend to land it never goes
    # below the straight line to the threshold, and flies on until its time runs out
    grid = reading.words.copy()
    grid[grid[:, APPROACH] == CLEARED, APPROACH] = UNCHANGED
    flown, uncleared, _ = _fly_sentence(signals, grid)
    assert uncleared.outcome == "timeout" and not flown.modes["captured"][0].any()
    assert not uncleared.flew_the_sentence
    # told to descend below the threshold's elevation (a target of 0 m MSL; the threshold is at 100 m), it meets
    # the ground before the threshold
    words = Words(spec())
    below = reading.words.copy()
    below[1, ALTITUDE], below[1, ANGLE] = words.altitude_index(0.0), 4          # at the steepest class, at once
    below[2:, ALTITUDE] = UNCHANGED
    below[2:, ANGLE] = UNCHANGED
    _, grounded, _ = _fly_sentence(signals, below)
    assert grounded.outcome == "ground_contact"
    # the steepest class at once asks more than γ̇_max: layer 1 records the rate the law wanted before its limit
    limited = grounded.limits["path_rate_limited"]
    assert limited["cycles"] > 0 and limited["wanted_minus_given"] > 1e-4


def test_descents_level_off_at_their_targets_inside_the_tubes():
    """§5.3: a descent to a target levels off at it without passing it; the next descent word releases it."""
    legs = [(100, 0.0, 75.0, 0.0), (80, 0.0, 75.0, -75.0 * np.tan(np.radians(2.1))), (120, 0.0, 75.0, 0.0),
            (100, 0.0, 75.0, -75.0 * np.tan(np.radians(3.0)))]
    signals = instruction_flight(*fly_legs(legs, 90.0, 1500.0, -400.0, 0.0))
    _, verdict, reading = _fly_sentence(signals)
    targets = [i for i in reading.instructions if i.column == ALTITUDE]
    assert len(targets) == 3 and len(verdict.words["vertical"]) == 3
    assert all(word["contained"] for word in verdict.words["vertical"])


# ---- E7: the parameters and the executor spec
def test_the_data_parameters_are_read_from_the_labellers_reading_of_each_flight():
    from ts_transformer.autopilot import measure
    from ts_transformer.instructions.labeller.read import admit, read_flight

    one, words, geometry = spec(), Words(spec()), instruction_airport()
    signals = instruction_flight(*fly_legs(DOWNWIND_BASE_FINAL, 270.0, 1110.0, -400.0, 0.0))
    reading = read_flight(signals, geometry, one, words)
    out = measure.flight_measurements(admit(signals, geometry, one), reading, geometry, one, words)
    # the base turn, 90° at 6°/row (3°/s; the turn onto the final is the capture, not a word): one steady rate;
    # at 100 m/s it is not in the fast band the bank cap is read from
    assert out["turn_steady_rate_deg_s"] == [pytest.approx(3.0, abs=0.05)]
    assert out["turn_fast_bank_deg"] == []
    # the legs change speed in a step: no transition lasts the MIN_SPAN_S a rate is read from
    assert out["transition_accel_mps2"] == []
    # the final is a straight 3° line, so the extrapolated crossing is the line's height at the threshold
    height = signals.altitude_m[-1] - geometry.candidates[0].elevation_m
    assert out["crossing_height_m"] == [pytest.approx(height - 400.0 * math.tan(math.radians(3.0)), abs=0.5)]
    # the measured sentence must be the stored one
    stored = [(reading.words, reading.runway_index)]
    measure.measure_flights([signals], stored, one, words, {"KXXX": geometry})
    changed = reading.words.copy()
    changed[1, SPEED] = words.speed_unspecified
    with pytest.raises(ValueError, match="differs from the stored one"):
        measure.measure_flights([signals], [(changed, 0)], one, words, {"KXXX": geometry})


def test_the_data_parameters_are_the_pooled_medians_rounded_as_the_spec_says():
    from ts_transformer.autopilot import measure

    one = spec()
    pooled = {"turn_steady_rate_deg_s": [2.1, 2.16, 2.3], "turn_fast_bank_deg": [24.2, 24.6, 40.0],
              "transition_accel_mps2": [-0.3, -0.24, -0.2, 0.1, 0.16, 0.5], "unspecified_slope_mps2": [-0.26, -0.3, -0.2],
              "crossing_height_m": [16.0, 20.84, 26.0]}
    measured = measure.measured_values(pooled, one)
    assert measured["values"] == {"turn_rate_deg_s": 2.15, "bank_cap_deg": 25.0, "decel_mps2": 0.24, "accel_mps2": 0.16,
                                  "unspecified_decel_mps2": 0.26, "land_aim_height_m": 20.8,
                                  "land_window_low_m": 16.5, "land_window_high_m": 25.5}
    assert measured["counts"]["decelerations"] == measured["counts"]["accelerations"] == 3
    # never past the vocabulary's own bank ceiling
    steep = measure.measured_values({**pooled, "turn_fast_bank_deg": [40.0]}, one)
    assert steep["values"]["bank_cap_deg"] == one.turn_bank_max_deg


def test_method_a_takes_the_gentlest_roll_out_and_roll_rate_the_constraints_allow():
    from dataclasses import replace

    from ts_transformer.autopilot import derive

    one = spec()
    assert derive.heading_time_constant_s(2.15, one, 1.0) == 4.5              # 10° / 2.15°/s = 4.65 → 4.5
    with pytest.raises(ValueError, match="under 2 Δt"):
        derive.heading_time_constant_s(6.0, one, 1.0)
    params = _params(heading_time_constant_s=4.5)
    largest = one.heading_max_turn_deg
    slow, fast = (derive.turn_overshoot_deg(replace(params, bank_rate_deg_s=p), 140.0, largest) for p in (1.0, 5.0))
    assert slow > one.heading_tolerance_deg > fast >= 0.0                        # a slower roll passes further
    rate, overshoots = derive.roll_rate_deg_s(params, one)
    assert max(overshoots.values()) <= one.heading_tolerance_deg
    less = replace(params, bank_rate_deg_s=rate - derive.ROLL_RATE_STEP_DEG_S)
    assert rate == derive.ROLL_RATE_STEP_DEG_S or max(
        derive.turn_overshoot_deg(less, speed, largest) for speed in derive.ROLL_CHECK_SPEEDS_MPS) > one.heading_tolerance_deg


def test_a_received_word_is_matched_to_the_nearest_reread_word_of_its_column_and_value():
    from ts_transformer.autopilot.observe import word_leads

    received = [(HEADING, 7, 100.0), (ALTITUDE, 3, 200.0), (SPEED, 5, 300.0)]
    reread = [(HEADING, 7, 96.0), (HEADING, 7, 150.0), (HEADING, 8, 100.0), (ALTITUDE, 3, 206.0), (SPEED, 5, 400.0)]
    assert word_leads(received, reread, 30.0) == [(HEADING, 4.0), (ALTITUDE, -6.0)]


def _observed_series(geometry):
    from types import SimpleNamespace

    from ts_transformer.data.coordinate_frames import ENUFrame
    from ts_transformer.data.dataset import FlightSeries

    candidate = geometry.candidates[0]
    lat, lon = geometry.frame.latlon_from_horizontal(candidate.threshold_e_m, candidate.threshold_n_m)
    scenario = SimpleNamespace(source={"arr_airport": "KXXX", "runway": "09", "resolved_typecode": "A320"},
                               initial=SimpleNamespace(m=62000.0),
                               target=SimpleNamespace(latitude=lat, longitude=lon, psi=0.0), aircraft=SimpleNamespace(code="A320"))
    return FlightSeries(flight_id="TEST1", scenario=scenario, frame=ENUFrame(lat0=lat, lon0=lon, alt0=candidate.elevation_m),
                        times=np.zeros(1), values=np.zeros((1, 6)))


def test_the_observation_operator_reads_a_flown_track_as_the_data_plane_reads_a_radar_track():
    """Method B's O: the flown track, sampled once a cycle, through the data plane's own fit and grid, reads
    back as the track that was flown, and the labeller re-reads its words from it: the turn within a row of
    where it was said, the descent LATER than the executor began it (the reading needs height lost before it
    sees a descent — method B's finding, flown with no delay)."""
    from ts_transformer.autopilot.observe import flight_leads, observe
    from ts_transformer.autopilot.judge import flown_track

    one, words, geometry = spec(), Words(spec()), instruction_airport()
    signals = instruction_flight(*fly_legs(DOWNWIND_BASE_FINAL, 270.0, 1110.0, -400.0, 0.0))
    flown, verdict, reading = _fly_sentence(signals)
    states = flown.states[0, : verdict.end_row + 1].numpy()
    seen = observe(states, 1.0, _observed_series(geometry), geometry, one.step_s)
    truth = flown_track(states, geometry)
    rows = (seen.time_s / 1.0).astype(int)
    assert len(seen.time_s) == len(states[::2]) and seen.time_s[1] - seen.time_s[0] == one.step_s
    assert np.abs(seen.e_m - truth["e"][rows]).max() < 5.0 and np.abs(seen.altitude_m - truth["height"][rows]).max() < 2.0
    leads, received = flight_leads(states, flown.sentence_s[0].numpy(), 1.0, _observed_series(geometry), geometry,
                                   reading, one, words, 30.0)
    said = [i for i in reading.instructions if i.row > 0 and i.column in (HEADING, ALTITUDE, ANGLE, SPEED)]
    assert sum(received.values()) == len(said) == 5 and received[HEADING] == 1
    by_column = dict(leads)
    assert abs(by_column[HEADING]) <= one.step_s
    assert by_column[ALTITUDE] < 0.0 and by_column[ANGLE] < 0.0


def test_the_executor_spec_is_written_once_and_refused_unless_it_is_the_current_executors(tmp_path):
    import json
    from dataclasses import replace

    from ts_transformer.autopilot import spec as executor_spec

    params = _params(delays=Delays(2.0, 0.0, 0.0))
    source = {"executor_source_sha256": executor_spec.executor_source_sha256(), "git": {"head": "x", "dirty": False}}
    executor_spec.write_spec(tmp_path, params, "vocabulary", {"data": {}}, source)
    loaded, record = executor_spec.load_spec(tmp_path)
    assert loaded == params and record["sha256"] == executor_spec.params_sha256(params)
    executor_spec.require_current_executor(record)
    with pytest.raises(FileExistsError):
        executor_spec.write_spec(tmp_path, params, "vocabulary", {}, source)
    assert executor_spec.params_sha256(replace(params, bank_rate_deg_s=3.0)) != record["sha256"]
    with pytest.raises(ValueError, match="other executor code"):
        executor_spec.require_current_executor({**record, "source": {**source, "executor_source_sha256": "0" * 64}})
    stored = json.loads((tmp_path / "spec.json").read_text())
    for broken, message in (({**stored, "params": {**stored["params"], "bank_rate_deg_s": 9.0}}, "do not hash"),
                            ({**stored, "params": {**stored["params"], "extra_s": 1.0}}, "extra"),
                            ({**stored, "schema": "ts-executor-spec-v0"}, f"is not a {executor_spec.EXECUTOR_SPEC_SCHEMA} file")):
        (tmp_path / "spec.json").write_text(json.dumps(broken))
        with pytest.raises(ValueError, match=message):
            executor_spec.load_spec(tmp_path)


def test_a_flight_is_flown_on_its_own_or_a_stand_ins_dynamics_only_with_a_published_approach_speed(monkeypatch):
    from types import SimpleNamespace

    from ts_transformer.autopilot import replay

    def series(resolved, dynamics):
        source = {"resolved_typecode": resolved, "dynamics_typecode": dynamics}
        return SimpleNamespace(scenario=SimpleNamespace(source=source, initial=SimpleNamespace(m=62000.0),
                                                        has_dynamics=dynamics is not None))

    assert replay.group_of(series("A320", "A320")) == replay.OWN
    assert replay.group_of(series(None, None)) == "no identified type"
    # the performance index flies a GLF4 as a CRJ9: a stand-in, reported and never gated
    assert replay.group_of(series("GLF4", "CRJ9")) == replay.STAND_IN
    # an excluded type is kept by the signals (`all-flights`) but has nothing to fly on
    assert replay.group_of(series("PC12", None)) == "no aircraft dynamics"
    monkeypatch.setattr(replay, "approach_speed_ias_mps", lambda typecode, mass: math.nan)
    assert replay.group_of(series("A320", "A320")) == "type publishes no approach speed"


def test_a_batch_readout_counts_what_was_flown_and_how_far_it_lies_from_the_observed_track():
    from ts_transformer.autopilot import replay

    signals = instruction_flight(*fly_legs(DOWNWIND_BASE_FINAL, 270.0, 1110.0, -400.0, 0.0))
    flown, verdict, reading = _fly_sentence(signals)
    batch = replay.Batch(signals=[signals], series=[], readings=[reading], geometries=[instruction_airport()],
                         approach_ias_mps=[], groups=[replay.OWN], drawn={})
    summary = replay.summary([verdict])
    judged = replay.word_results(verdict)[0]
    assert summary == {"flights": 1, "outcomes": {"landed": 1}, "landed_share": 1.0, "flew_the_sentence_share": 1.0,
                       "words_judged": len(judged), "words_inside_share": 1.0, "heading_words_not_judged": 0,
                       "flights_with_unjudged_words": 0, "word_failures": {}}
    aligned = replay.alignment(batch, flown, [verdict])
    # the executor flies the words, not the legs: it stays within a kilometre of the synthetic flight and lands
    # later (after "unspecified" it slows to the A320's published approach speed, under the legs' 75 m/s)
    assert aligned["mean_horizontal_distance_m"]["p50"] < 1000.0 and aligned["mean_vertical_distance_m"]["p50"] < 30.0
    # the observed landing: the last row, 400 m short, carried on at 75 m/s; the flown one: its interpolated crossing
    landing = aligned["landing_time_minus_observed_s"]
    observed = (len(reading.words) - 1) * 2.0 + 400.0 / 75.0
    assert landing["n"] == 1 and landing["p50"] == pytest.approx(verdict.crossing["at_row"] - observed)
    assert 0.0 < landing["p50"] < 40.0


def test_the_sensitivity_moves_one_parameter_at_a_time_and_marks_a_word_acted_on_early_as_a_probe():
    from ts_transformer.experiments.executor_sensitivity import variants

    params = _params(heading_time_constant_s=4.5, bank_rate_deg_s=2.5, delays=Delays(2.0, 0.0, 0.0))
    table = {name: (moved, probe) for name, moved, probe in variants(params)}
    assert table["spec"] == (params, False)
    assert {n for n in table if n.startswith("heading_time")} == {f"heading_time_constant_s={t:g}" for t in (2, 3, 4)}
    assert "path_rate_factor=2" not in table and "bank_rate_deg_s=2" in table
    assert table["delays.heading_s=-2"][1] and not table["delays.heading_s=6"][1]
    assert table["delays.vertical_s=-4"][1] and table["delays.vertical_s=-4"][0].delays == Delays(2.0, -4.0, 0.0)
    for name, (moved, probe) in table.items():
        if probe:
            with pytest.raises(ValueError, match="before it is said"):
                moved.check(spec())
        elif name.startswith(("heading", "bank", "path", "delays")) or name == "spec":
            moved.check(spec())


def test_a_replayed_flight_becomes_a_control_record_evaluation_can_grade_and_its_words_are_counted_one_by_one():
    from ts_transformer.autopilot.plant import EXECUTOR_DYNAMICS
    from ts_transformer.autopilot.replay import word_results
    from ts_transformer.experiments.executor_replay import executor_forecast, gate_table

    signals = instruction_flight(*fly_legs(DOWNWIND_BASE_FINAL, 270.0, 1110.0, -400.0, 0.0))
    flown, verdict, reading = _fly_sentence(signals)
    judged, not_judged = word_results(verdict)
    assert not_judged == 0
    assert judged and all(ok for _, ok in judged)
    assert sum(column == "heading" for column, _ in judged) == sum(h["words"] for h in verdict.words["heading"])
    assert ("approach", True) in judged
    # the forecast from row 0: one state per cycle to the outcome's row, the schedule and its newtons
    inputs, _ = _a320([[0.0] * 7])
    series = _observed_series(instruction_airport())
    forecast = executor_forecast(flown, 0, verdict, inputs, series)
    assert forecast.anchor == 0 and forecast.n_steps == verdict.end_row == len(forecast.controls)
    assert forecast.final_time_s == pytest.approx(verdict.end_row * flown.cycle_s)
    assert forecast.truncated_at_threshold and not forecast.horizon_capped
    assert forecast.control_parameterization == EXECUTOR_DYNAMICS.control_thrust_parameterization
    fraction = flown.commands[0, : verdict.end_row, 0].numpy()
    assert forecast.controls[:, 0] == pytest.approx(fraction * float(inputs.max_thrust_n[0]))
    assert forecast.controls[:, 1:] == pytest.approx(flown.commands[0, : verdict.end_row, 1:].numpy())
    # the gates count per word and pair evaluation with the observed verdict
    base = {"airport": "KXXX", "group": "own dynamics", "stratum": "vectored", "words_not_reached": 0,
            "heading_words_not_judged": 0}
    rows = [{**base, "outcome": "landed", "words": [("heading", True)] * 19 + [("speed", False)],
             "observed_verdict": "pass", "replay_verdict": "pass"},
            {**base, "outcome": "ground_contact", "words": None, "observed_verdict": "fail", "replay_verdict": "fail"}]
    cell = gate_table(rows)["own dynamics"]["KXXX"]["vectored"]
    assert (cell["landed"], cell["words_inside"], cell["replay_passes_where_observed_passes"]) == (0.5, 0.95, 1.0)
    assert cell["clears"] == {"landed": False, "words": True, "evaluation": True}
    assert cell["flights_with_unjudged_words"] == 1


# ---- the E3–E6 review's cases
def _downwind():
    signals = instruction_flight(*fly_legs(DOWNWIND_BASE_FINAL, 270.0, 1110.0, -400.0, 0.0))
    _, _, reading = _fly_sentence(signals)
    return signals, reading


def test_a_late_clearance_is_captured_past_the_line_and_the_line_law_brings_it_back():
    """Review 1: cleared inside its lead, the capture turn crosses the line; it hands over to the line law
    instead of flying away at a constant angle."""
    signals, reading = _downwind()
    clear = int(np.nonzero(reading.words[:, APPROACH] == APPROACH_CLEARED)[0][0])
    for late in (38, 44):
        grid = reading.words.copy()
        grid[clear, APPROACH] = UNCHANGED
        grid[clear + late, APPROACH] = APPROACH_CLEARED
        flown, verdict, _ = _fly_sentence(signals, grid)
        assert verdict.outcome == "landed" and abs(verdict.crossing["cross_m"]) < 5.0 and verdict.flew_the_sentence


def _orbit():
    legs = [(40, 0.0, 100.0, 0.0), (60, 6.0, 100.0, 0.0), (40, 0.0, 90.0, 0.0), (15, 6.0, 85.0, 0.0),
            (110, 0.0, 75.0, -75.0 * np.tan(np.radians(3.0)))]
    return instruction_flight(*fly_legs(legs, 0.0, 1300.0, -400.0, 0.0))


def test_an_orbit_is_flown_the_way_its_first_part_set_and_judged_as_one_whole_turn():
    """Review 4 and 5: a 360° right orbit said in three parts. The executor turns slower than the observed
    aircraft, so it lags the last part by more than 180°; it still turns right all the way (the word is
    measured from the word in force), and the judge reads a 360° turn and its hold's funnel the same way."""
    flown, verdict, reading = _fly_sentence(_orbit())
    assert [i.kind for i in reading.instructions if i.column == HEADING] == ["initial"] + ["turn-split"] * 3
    track = np.degrees(np.unwrap(np.radians(compass_from_math_rad(flown.states[0, :, 4].numpy()))))
    assert np.diff(track).min() > -0.5 and track[-1] - track[0] == pytest.approx(450.0, abs=5.0)
    (_, orbit) = verdict.words["heading"]
    assert orbit["words"] == 3 and all(orbit["turn"].values()) and orbit["hold"]["inside"] == orbit["hold"]["rows"]
    assert verdict.outcome == "landed" and verdict.flew_the_sentence


def test_the_hold_funnel_mirror_equals_the_labellers_under_180_degrees():
    from ts_transformer.autopilot.judge import hold_funnel
    from ts_transformer.instructions.labeller.lateral import heading_spans
    from ts_transformer.instructions.labeller.read import admit, span_funnel

    one, words, geometry = spec(), Words(spec()), instruction_airport()
    signals, reading = _downwind()
    flight = admit(signals, geometry, one)
    spans = [s for s in heading_spans(reading.instructions, reading.checks["turns"], None, reading.capture_row)
             if s.turn is not None and s.held]
    assert spans
    for span in spans:
        target = words.heading_deg(span.word.value)
        track = float(flight.smoothed.track_deg[span.word.row])
        _, theirs = span_funnel(flight, span, target, one)
        _, ours = hold_funnel(flight, span, target, float(np_wrap180(target - track)), one)
        assert ours.outline == pytest.approx(theirs.outline)


def test_a_turn_the_capture_takes_over_at_once_is_superseded_not_failed():
    """Review 3: the clearance comes with the base turn and the executor is already inside its capture lead."""
    legs = [(60, 0.0, 100.0, 0.0), (15, -6.0, 100.0, 0.0), (6, 0.0, 90.0, 0.0), (15, -6.0, 85.0, 0.0),
            (120, 0.0, 75.0, -75.0 * np.tan(np.radians(3.0)))]
    _, verdict, _ = _fly_sentence(instruction_flight(*fly_legs(legs, 270.0, 1110.0, -400.0, 0.0)))
    (_, base) = verdict.words["heading"]
    assert base["superseded"] and base["turn"] is None
    assert verdict.words["capture_turn"]["progress_ok"] and verdict.outcome == "landed" and verdict.flew_the_sentence
    # a probe acting on the words 2 s early captures before the base turn's own row: the word never flew
    signals = instruction_flight(*fly_legs(legs, 270.0, 1110.0, -400.0, 0.0))
    _, early, _ = _fly_sentence(signals, params=_params(delays=Delays(-2.0, 0.0, 0.0)), early_words=True)
    assert [h["superseded"] for h in early.words["heading"]] == [False, True]


def test_the_judge_fails_a_turn_flown_slower_than_the_vocabulary_allows():
    """Review 12: layer 2 catches a violation, not only passes a good flight — a 5° bank cap turns the orbit at
    under 0.5°/s at 100 m/s, below the vocabulary's slowest turn."""
    _, verdict, _ = _fly_sentence(_orbit(), params=_params(bank_cap_deg=5.0))
    (_, orbit) = verdict.words["heading"]
    assert orbit["turn"] == {"reached": True, "progress_ok": True, "rate_ok": False}
    assert not verdict.words["all_contained"] and not verdict.flew_the_sentence


def test_crossings_are_told_apart_by_capture_and_by_the_landing_condition():
    signals, reading = _downwind()
    level = reading.words.copy()
    level[1:, ALTITUDE] = UNCHANGED
    level[1:, ANGLE] = UNCHANGED
    _, high, _ = _fly_sentence(signals, level)                  # captured, never descends: 1000 m over the threshold
    assert high.outcome == "crossed_off_runway" and high.crossing["height_m"] > 500.0
    straight = [(40, 0.0, 90.0, 0.0), (30, 0.0, 80.0, 0.0), (110, 0.0, 72.0, -72.0 * np.tan(np.radians(3.0)))]
    aligned = instruction_flight(*fly_legs(straight, 90.0, 950.0, -300.0, 0.0))
    _, _, on_final = _fly_sentence(aligned)
    grid = on_final.words.copy()
    grid[grid[:, APPROACH] == APPROACH_CLEARED, APPROACH] = APPROACH_NOT_CLEARED
    flown, uncleared, _ = _fly_sentence(aligned, grid)
    assert uncleared.outcome == "crossed_without_capture" and not flown.modes["captured"][0].any()


def test_a_go_around_cancels_the_capture_climbs_and_holds_its_speed():
    from ts_transformer.instructions.words import APPROACH_GO_AROUND

    signals, reading = _downwind()
    grid = reading.words.copy()
    grid[150, APPROACH] = APPROACH_GO_AROUND
    flown, verdict, _ = _fly_sentence(signals, grid)
    after = 150 * 2 + 5
    assert flown.modes["go_around"][0, after] and not flown.modes["captured"][0, after:].any()
    height, speed = flown.states[0, after:, 2].numpy(), flown.states[0, after:, 3].numpy()
    assert height[60] > height[0] + 50.0 and abs(speed[60] - speed[0]) < 1.0
    assert verdict.outcome == "timeout"


def test_a_cleared_heading_that_misses_the_line_is_bent_by_its_tolerance():
    """§4.3: cleared on a heading that just misses the line, the executor flies it bent toward the line."""
    # 2 km north of an eastbound final, 41 km out, flying the course: parallel, it never reaches the line;
    # 4.5° to the right it crosses it 25 km later, ahead of the threshold
    legs = [(100, 0.0, 100.0, 0.0), (5, 4.0, 100.0, 0.0), (29, 0.0, 100.0, 0.0), (5, -4.0, 90.0, 0.0),
            (100, 0.0, 75.0, -75.0 * np.tan(np.radians(3.0)))]
    signals = instruction_flight(*fly_legs(legs, 90.0, 1600.0, -400.0, 0.0))
    _, _, reading = _fly_sentence(signals)
    grid = reading.words.copy()
    grid[0, APPROACH] = APPROACH_CLEARED                       # cleared at once, on the 090° it is flying
    flown, _, _ = _fly_sentence(signals, grid)
    bent = flown.modes["bent"][0].numpy()
    assert bent[:150].all()
    track = compass_from_math_rad(flown.states[0, 1:151, 4].numpy())
    assert track[60:150] == pytest.approx(90.0 + spec().heading_tolerance_deg, abs=0.2)


def test_the_words_the_executor_refuses():
    from ts_transformer.autopilot.lateral import Lateral, Runways
    from ts_transformer.instructions.airport import AirportGeometry

    signals, reading = _downwind()
    words = Words(spec())
    climb = reading.words.copy()
    climb[5, ALTITUDE], climb[5, ANGLE] = words.altitude_land, words.angle_climb
    with pytest.raises(ValueError, match="without a descent class"):
        _fly_sentence(signals, climb)
    unspecified = reading.words.copy()
    unspecified[5, SPEED] = words.speed_unspecified
    with pytest.raises(ValueError, match="publishes no approach speed"):
        _fly_sentence(signals, unspecified, approach_ias=math.nan)
    # the runway pointer moved once cleared (§4.6)
    data = instruction_airport().to_dict()
    data["candidates"].append({**data["candidates"][0], "ident": "09R", "threshold_n_m": -1500.0})
    geometry = AirportGeometry.from_dict(data)
    runways = Runways.of([geometry], dtype=F64, device=CPU)
    lateral = Lateral(1, _params(), spec(), CPU)
    state = read_state(torch.tensor([[35.0, -78.05, 900.0, 80.0, 0.0, 0.0, 60000.0]], dtype=F64),
                       AirportCharts.of([geometry], dtype=F64, device=CPU))
    heading, issued, bank = torch.tensor([90.0], dtype=F64), torch.tensor([0]), torch.zeros(1, dtype=F64)
    cleared = torch.tensor([APPROACH_CLEARED])
    lateral.rate(state, heading, issued, cleared, torch.tensor([0]), runways, bank, 0.05)
    with pytest.raises(ValueError, match="runway pointer changed after the clearance"):
        lateral.rate(state, heading, issued, cleared, torch.tensor([1]), runways, bank, 0.05)


def test_a_stall_is_a_dynamics_failure_and_wins_a_row_it_shares_with_a_crossing():
    """Review 7 and the event order: the dynamics' stall cut-off binding is a failure (§8.1), and at one row a
    failure comes before the ground and the ground before a crossing."""
    from ts_transformer.autopilot.judge import _outcome, flown_track

    one, geometry = spec(), instruction_airport()
    signals, _ = _downwind()
    flown, verdict, _ = _fly_sentence(signals)
    states = flown.states[0, : verdict.end_row + 1].numpy()
    track = flown_track(states, geometry)
    captured = np.concatenate(([False], flown.modes["captured"][0, : verdict.end_row].numpy()))
    stalled = np.zeros(len(states), dtype=bool)
    assert _outcome(states, track, captured, stalled, geometry, 0, one)[:2] == ("landed", verdict.end_row)
    stalled[200] = True
    assert _outcome(states, track, captured, stalled, geometry, 0, one)[:2] == ("dynamics_failure", 200)
    stalled[:] = False
    stalled[verdict.end_row] = True
    assert _outcome(states, track, captured, stalled, geometry, 0, one)[:2] == ("dynamics_failure", verdict.end_row)


# ---- the E7–E8 review's cases
def test_a_sensitivity_probe_flies_its_words_early_only_when_named():
    """Review H1: a negative delay is refused unless the flight is flown as a probe, and then the words act
    that much earlier."""
    signals, _ = _downwind()
    early = _params(delays=Delays(0.0, -4.0, 0.0))
    with pytest.raises(ValueError, match="before it is said"):
        _fly_sentence(signals, params=early)
    on_time, _, reading = _fly_sentence(signals)
    ahead, _, _ = _fly_sentence(signals, params=early, early_words=True)
    descent = [i.row for i in reading.instructions if i.column == ANGLE and i.row > 0][0] * 2
    gamma_on_time, gamma_ahead = on_time.states[0, :, 5].numpy(), ahead.states[0, :, 5].numpy()
    assert gamma_ahead[descent - 1] < gamma_on_time[descent - 1] - 1e-3        # already pitching down


def test_a_stand_ins_unspecified_speed_is_its_types_as_published():
    """Review H2: a flight on a stand-in's dynamics carries the stand-in's mass; its own type's published speed
    is taken unscaled, never scaled by another airframe's mass."""
    from types import SimpleNamespace

    from ts_transformer.autopilot import replay
    from ts_transformer.autopilot.speed import approach_speed_ias_mps

    def series(resolved, dynamics, mass):
        return SimpleNamespace(scenario=SimpleNamespace(
            source={"resolved_typecode": resolved, "dynamics_typecode": dynamics}, initial=SimpleNamespace(m=mass),
            has_dynamics=True))

    own = series("A320", "A320", 60000.0)
    assert replay.flight_approach_ias_mps(own, replay.OWN) == approach_speed_ias_mps("A320", 60000.0)
    stand_in = series("CRJ7", "A320", 66300.0)
    assert replay.group_of(stand_in) == replay.STAND_IN
    published = approach_speed_ias_mps("CRJ7", None)
    assert replay.flight_approach_ias_mps(stand_in, replay.STAND_IN) == published < 80.0
    assert approach_speed_ias_mps("CRJ7", 66300.0) > published


def test_the_word_count_leaves_out_the_words_the_judge_did_not_judge():
    """Review M1: a superseded turn, or a word with neither a turn nor a hold judged, is counted apart."""
    from ts_transformer.autopilot.judge import Verdict
    from ts_transformer.autopilot.replay import word_results

    heading = [{"row": 0, "kind": "initial", "words": 1, "turn": None, "superseded": False, "hold": {"rows": 5, "inside": 5}},
               {"row": 9, "kind": "turn", "words": 1, "turn": None, "superseded": True, "hold": None},
               {"row": 20, "kind": "turn-split", "words": 2, "turn": {"reached": True, "progress_ok": True, "rate_ok": False},
                "superseded": False, "hold": "not judged: slowest turn unfinished"},
               {"row": 40, "kind": "turn", "words": 1, "turn": None, "superseded": False, "hold": None}]
    words = {"not_reached": 0, "superseded_before_flown": 0, "heading": heading, "capture_turn": None,
             "intercepting_off_word_cycles": 0, "aim_left_tube_cycles": 0,
             "corridor": {"cleared": False, "entered": False, "rows": 0, "inside": 0},
             "vertical": [{"contained": True}], "speed": [{"contained": False}], "all_contained": False}
    judged, not_judged = word_results(Verdict("landed", 50, None, {}, words, flown_rows=51))
    assert not_judged == 2
    assert judged == [("heading", True), ("heading", False), ("heading", False), ("altitude", True), ("speed", False)]


def test_the_executor_hash_covers_the_package_and_what_it_imports_from_the_repository():
    """Review M2: the dynamics, the envelope and the approach-speed table decide a flown track; the instruction
    language has its own hash."""
    from ts_transformer.autopilot import spec as executor_spec
    from ts_transformer.repo_layout import REPO_ROOT

    root = REPO_ROOT.resolve()
    files = {path.relative_to(root).as_posix() for path in executor_spec.executor_source_files()}
    package = {path.name for path in executor_spec.PACKAGE.glob("*.py")} - {"spec.py"}
    assert {f"4dTrajectory/ts_transformer/autopilot/{name}" for name in package} <= files
    assert "4dTrajectory/ts_transformer/autopilot/spec.py" not in files
    for needed in ("aerodynamic_model/torch_dynamics.py", "aircraft/reference_speeds.py", "flight_scenarios/start_state.py",
                   "4dTrajectory/ts_transformer/outputs/dynamics/rollout.py", "4dTrajectory/ts_transformer/outputs/envelope.py",
                   "4dTrajectory/ts_transformer/data/channels.py"):
        assert needed in files
    assert not any("/instructions/" in f or f.endswith(("io_utils.py", "repo_layout.py")) for f in files)


def test_an_executor_spec_is_opened_only_against_its_own_vocabulary_and_labeller(tmp_path, monkeypatch):
    """Review M3: the vocabulary sha, the artefact's labeller and the spec's recorded labeller must all agree."""
    from ts_transformer.autopilot import replay
    from ts_transformer.autopilot import spec as executor_spec

    one = spec()
    source = {"executor_source_sha256": executor_spec.executor_source_sha256(), "labeller_source_sha256": "a" * 64,
              "git": {"head": "x", "dirty": False}}
    executor_spec.write_spec(tmp_path / "spec", _params(), one.sha256, {}, source)
    monkeypatch.setattr(replay.artefact, "load_spec", lambda directory: one)
    monkeypatch.setattr(replay.artefact, "require_current_labeller", lambda directory: None)
    monkeypatch.setattr(replay.artefact, "labeller_source_sha256", lambda: "a" * 64)
    params, _, _ = replay.open_executor(tmp_path / "spec", tmp_path / "artefact")
    assert params == _params()
    monkeypatch.setattr(replay.artefact, "labeller_source_sha256", lambda: "b" * 64)
    with pytest.raises(ValueError, match="other labeller code"):
        replay.open_executor(tmp_path / "spec", tmp_path / "artefact")
    monkeypatch.setattr(replay.artefact, "load_spec", lambda directory: spec(heading_tolerance_deg=4.0))
    with pytest.raises(ValueError, match="measured against vocabulary"):
        replay.open_executor(tmp_path / "spec", tmp_path / "artefact")


def test_method_bs_delays_are_the_groups_median_leads_floored_at_zero():
    from ts_transformer.autopilot.observe import delays_from_leads

    leads = {HEADING: [2.0, 2.0, 4.0], ALTITUDE: [-6.0, -4.0], ANGLE: [-6.0], SPEED: [-9.0, 3.0, -8.0]}
    assert delays_from_leads(leads) == Delays(heading_s=2.0, vertical_s=0.0, speed_s=0.0)
    assert delays_from_leads({**leads, ALTITUDE: [5.0, 7.0], ANGLE: [6.0]}).vertical_s == 6.0
    with pytest.raises(ValueError, match="matched no word of speed_s"):
        delays_from_leads({**leads, SPEED: []})


def test_draw_reads_a_seeded_permutation_until_each_airport_is_full(monkeypatch):
    """`replay.draw`: the seed fixes the flights, each airport stops at its count, 0 takes every flight, a short
    airport is refused, and a re-read that differs from the stored sentence stops the draw."""
    from types import SimpleNamespace

    from ts_transformer.autopilot import replay

    flights = [SimpleNamespace(airport="KAAA" if i % 3 else "KBBB", dataset_id=f"F{i}") for i in range(30)]
    stored = [np.full((3, 6), i, dtype=np.int16) for i in range(30)]
    sentences = {"signal_index": np.arange(30), "offsets": np.arange(31) * 3,
                 "words": np.concatenate(stored), "runway_index": np.zeros(30, dtype=np.int64)}
    typecode = {i: ("A320", "A320") if i % 4 else ("CRJ7", "A320") for i in range(30)}
    monkeypatch.setattr(replay, "load_candidates", lambda d: {"KAAA": "geo-a", "KBBB": "geo-b"})
    monkeypatch.setattr(replay, "load_signals", lambda d, split: flights)
    monkeypatch.setattr(replay, "load_sentences", lambda d, split, spec: sentences)
    monkeypatch.setattr(replay, "rebuild_series", lambda d, items: [SimpleNamespace(scenario=SimpleNamespace(
        source=dict(zip(("resolved_typecode", "dynamics_typecode"), typecode[int(f.dataset_id[1:])])),
        initial=SimpleNamespace(m=60000.0), has_dynamics=True)) for f in items])
    reread = {"differ": None}
    monkeypatch.setattr(replay, "read_flight", lambda f, g, s, w: SimpleNamespace(
        words=stored[int(f.dataset_id[1:])] + (1 if f.dataset_id == reread["differ"] else 0), runway_index=0))
    one, words = spec(), Words(spec())
    first = replay.draw(Path("x"), "train", one, words, per_airport=3, seed=5)
    again = replay.draw(Path("x"), "train", one, words, per_airport=3, seed=5)
    assert [f.dataset_id for f in first.signals] == [f.dataset_id for f in again.signals]
    assert Counter(f.airport for f in first.signals) == {"KAAA": 3, "KBBB": 3}
    assert set(first.groups) == {replay.OWN}
    every = replay.draw(Path("x"), "train", one, words, per_airport=0, seed=5, groups=(replay.OWN, replay.STAND_IN))
    assert len(every.signals) == 30 and every.drawn["by_group"] == {replay.OWN: 22, replay.STAND_IN: 8}
    with pytest.raises(ValueError, match="too few eligible flights"):
        replay.draw(Path("x"), "train", one, words, per_airport=9, seed=5)
    reread["differ"] = first.signals[0].dataset_id
    with pytest.raises(ValueError, match="differs from the stored one"):
        replay.draw(Path("x"), "train", one, words, per_airport=3, seed=5)


def test_a_split_turn_said_only_in_part_is_judged_toward_the_part_said():
    """Review: when the flight ends after the first part of a 360° orbit was said, the turn judged is that part's
    120°, not the whole orbit's end — read here on the observed orbit, which reaches 120° and turns on."""
    from ts_transformer.autopilot.judge import _heading_words, said_at
    from ts_transformer.instructions.labeller.read import admit, read_flight

    one, words, geometry = spec(), Words(spec()), instruction_airport()
    signals = _orbit()
    reading = read_flight(signals, geometry, one, words)
    first = [i for i in reading.instructions if i.kind == "turn-split"][0]
    said = [i for i in reading.instructions if i.column == HEADING and i.row <= first.row]
    said, _ = said_at(said, [i.row for i in said])                              # told where the observed aircraft was
    (_, part) = _heading_words(admit(signals, geometry, one), said, reading.checks["turns"], None, one, words)
    assert part["words"] == 1 and part["turn"]["reached"] and part["turn"]["progress_ok"]


def test_words_said_on_one_flown_row_leave_the_later_one_of_each_column():
    """A clock running ahead of the sentence can say two words of a column on one flown row: the later one flies."""
    from ts_transformer.autopilot.judge import said_at
    from ts_transformer.instructions.labeller.records import Instruction

    words = [Instruction(ALTITUDE, 30, 10, "target"), Instruction(ALTITUDE, 20, 12, "target"),
             Instruction(HEADING, 5, 10, "turn-split", {"part": 1}), Instruction(HEADING, 9, 11, "turn-split", {"part": 2}),
             Instruction(SPEED, 14, 20, "target")]
    kept, superseded = said_at(words, [7, 7, 7, 7, 15])
    assert superseded == 1
    assert [(w.column, w.value, w.row, w.info["sentence_row"]) for w in kept] == [
        (ALTITUDE, 20, 7, 12), (HEADING, 5, 7, 10), (HEADING, 9, 7, 11), (SPEED, 14, 15, 20)]


def test_the_track_clock_follows_the_observed_track_forward_one_row_a_cycle_at_most():
    """The nearest point ahead, never back, at most one row a cycle (no word is skipped), and on at the executor's
    pace past the track's end."""
    from ts_transformer.autopilot.frame import Kinematics

    e = np.arange(6) * 100.0
    clock = TrackClock.of([e], [np.zeros(6)], 2.0, 1.0, device=CPU)

    def at(position):
        state = Kinematics(*(torch.tensor([value], dtype=F64) for value in (position, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
        return float(clock.now(0, state)[0])

    assert [at(x) for x in (0.0, 440.0, 460.0, 120.0, 900.0, 900.0, 900.0, 900.0)] == pytest.approx(
        [0.0, 2.0, 4.0, 4.0, 6.0, 8.0, 10.0, 11.0])


def test_the_landing_aim_leaves_the_words_tube_only_when_the_tube_misses_the_landing_window():
    """The aim stays inside the word's tube where the tube meets the landing window; where it does not, the window."""
    from ts_transformer.autopilot.frame import Kinematics
    from ts_transformer.autopilot.vertical import Vertical

    words = Words(spec())
    params = _params()                                   # window 8.4–36.5 m, aim 20.8 m
    vertical = Vertical(1, params, words, CPU)

    def crossing_for(anchor_above_threshold_m, angle_class):
        vertical.anchor_height_m = torch.tensor([float("nan")], dtype=F64)
        vertical.issued = torch.full((1, 2), -1, dtype=torch.long)
        vertical.flown_m = torch.zeros(1, dtype=F64)
        state = Kinematics(*(torch.tensor([v], dtype=F64) for v in (0.0, 0.0, 100.0 + anchor_above_threshold_m, 75.0,
                                                                     90.0, 0.0, 75.0, 60000.0)))
        rate, _, modes = vertical.rate(state, torch.tensor([float("nan")], dtype=F64), torch.tensor([True]),
                                       torch.tensor([angle_class]), torch.tensor([words.angle_deg(angle_class)], dtype=F64),
                                       torch.tensor([[0, 0]]), torch.tensor([3000.0], dtype=F64),
                                       torch.tensor([100.0], dtype=F64), torch.tensor([3000.0], dtype=F64),
                                       torch.tensor([True]), torch.tensor([False]))
        return bool(modes["aim_left_tube"][0])

    # 3000 m out, the 3° class from 160 m over the threshold: its tube reaches the threshold near 0 m — the window
    assert not crossing_for(160.0, 3)
    # the shallowest descent class from 600 m: the tube crosses 500+ m over the threshold, no landing: the aim leaves it
    assert crossing_for(600.0, 1)


def test_the_pilots_own_speed_is_reached_by_the_threshold():
    """"Unspecified": at least the deceleration that reaches the approach speed over the straight line left."""
    from ts_transformer.autopilot.frame import Kinematics
    from ts_transformer.autopilot.speed import Speed

    one = spec()
    speed = Speed(torch.tensor([65.0], dtype=F64), _params(), one)
    state = Kinematics(*(torch.tensor([v], dtype=F64) for v in (0.0, 0.0, 100.0, 90.0, 90.0, 0.0, 90.0, 60000.0)))
    aero = torch.tensor([[122.6, 2.5, 0.02, 0.04, 0.9, 0.2]], dtype=F64)

    def rate(straight_m):
        return float(speed.rate(state, torch.tensor([float("nan")], dtype=F64), torch.tensor([True]),
                                torch.tensor([False]), torch.ones(1, dtype=F64), aero,
                                torch.tensor([straight_m], dtype=F64))[0][0])

    far, near = rate(50_000.0), rate(3_000.0)
    assert far == pytest.approx(-_params().unspecified_decel_mps2)            # far out: the pilot's own rate
    assert -one.speed_accel_max_mps2 <= near < far                          # close in: harder, within the vocabulary


def test_an_executor_intercepting_off_its_word_fails_that_word():
    """Review 2026-09-24: cleared on a heading that cannot reach the line even bent, the executor intercepts on its
    own; when that is more than the tolerance off the word, the word failed and the flight did not fly as said."""
    legs = [(100, 0.0, 100.0, 0.0), (5, 4.0, 100.0, 0.0), (29, 0.0, 100.0, 0.0), (5, -4.0, 90.0, 0.0),
            (100, 0.0, 75.0, -75.0 * np.tan(np.radians(3.0)))]
    signals = instruction_flight(*fly_legs(legs, 90.0, 1600.0, -400.0, 0.0))
    _, _, reading = _fly_sentence(signals)
    words = Words(spec())
    grid = reading.words.copy()
    grid[0, APPROACH] = APPROACH_CLEARED
    grid[0, HEADING] = words.heading_index(60.0)                            # away from the line: cannot reach it
    grid[1:, HEADING] = UNCHANGED
    flown, verdict, _ = _fly_sentence(signals, grid)
    assert flown.modes["intercepting_off_word"][0].any()
    assert verdict.words["intercepting_off_word_cycles"] > 0 and not verdict.flew_the_sentence


def test_a_word_the_clock_never_reached_is_not_reached():
    """Review 2026-09-24: a word past the flight's end was never said — not counted as flown at the last row."""
    signals, reading = _downwind()
    params = _params(timeout_factor=0.3)                                    # the flight stops a third of the way
    _, verdict, _ = _fly_sentence(signals, params=params)
    later = [i for i in reading.instructions if i.row * 2.0 >= verdict.end_row]
    assert later and verdict.words["not_reached"] == len(later)
