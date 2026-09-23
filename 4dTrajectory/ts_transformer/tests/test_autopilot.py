"""The executor (`autopilot/`, executor design): conventions, words in force, the one-cycle plant and
the exact inverse, on synthetic states of a real airframe."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from aircraft.aero_params import aero_params_for_aircraft
from flight_scenarios.scenario import aircraft_for_code
from ts_transformer.autopilot import inverse
from ts_transformer.autopilot.flights import FlightInputs
from ts_transformer.autopilot.frame import AirportCharts, compass_deg, read_state, wrap180
from ts_transformer.autopilot.plant import Plant
from ts_transformer.autopilot.sentence import Delays, words_in_force
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


# ---- words in force
def test_each_word_takes_effect_its_columns_delay_after_its_step_and_stays_in_force():
    one, words = spec(), Words(spec())
    grid = np.full((5, 6), UNCHANGED, dtype=np.int64)
    grid[0] = [0, APPROACH_NOT_CLEARED, words.heading_index(270.0), words.altitude_index(1200.0), 0,
               words.speed_index(100.0)]
    grid[2, HEADING] = words.heading_index(180.0)           # issued at t = 4 s
    grid[2, APPROACH] = APPROACH_CLEARED
    grid[3, ALTITUDE] = words.altitude_land                 # issued at t = 6 s
    grid[3, ANGLE] = 3
    grid[4, SPEED] = words.speed_unspecified                # issued at t = 8 s
    delays = Delays(heading_s=3.0, vertical_s=1.0, speed_s=0.0)
    force = words_in_force([grid], words, delays, cycle_s=1.0, cycles=20, device=CPU)
    heading = force.heading_deg[0].numpy()
    assert (heading[:7] == 270.0).all() and (heading[7:] == 180.0).all()        # 4 s + 3 s
    assert force.approach[0, 6] == APPROACH_NOT_CLEARED and force.approach[0, 7] == APPROACH_CLEARED
    land = force.land[0].numpy()
    assert not land[:7].any() and land[7:].all()                                 # 6 s + 1 s
    assert float(force.altitude_m[0, 0]) == 1200.0 and float(force.angle_deg[0, 7]) == one.descent_angle_centres_deg[2]
    assert not force.unspecified[0, :8].any() and force.unspecified[0, 8:].all()
    assert (force.runway[0] == 0).all()
    assert force.issued_step[0, 6, HEADING] == 0 and force.issued_step[0, 7, HEADING] == 2
    assert force.issued_step[0, 19, SPEED] == 4                                  # held after the last step


def test_a_sentence_must_write_every_column_at_step_0():
    words = Words(spec())
    grid = np.full((3, 6), UNCHANGED, dtype=np.int64)
    grid[0, :5] = [0, 0, 0, 0, 0]
    with pytest.raises(ValueError, match="step 0 must write every column"):
        words_in_force([grid], words, Delays(0.0, 0.0, 0.0), cycle_s=1.0, cycles=4, device=CPU)


# ---- the rebuilt flight
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
                          accel_mps2=0.16, unspecified_decel_mps2=0.26, land_aim_height_m=20.8,
                          delays=Delays(0.0, 0.0, 0.0),
                          timeout_factor=1.5)
    return replace(base, **changes)


def _fly_sentence(signals, grid=None, params=None):
    """Read ``signals`` with the labeller, fly its sentence (or ``grid``) from row 0, judge it."""
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
    force = words_in_force([grid], words, params.delays, cycle_s=params.cycle_s, cycles=int(math.ceil(limit)), device=CPU)
    flown = fly(inputs, force, Runways.of([geometry], dtype=F64, device=CPU),
                AirportCharts.of([geometry], dtype=F64, device=CPU),
                torch.tensor([approach_speed_ias_mps("A320", mass)], dtype=F64), params, words,
                time_limit_s=torch.tensor([limit], dtype=F64))
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
    # never across the centreline to the south of an eastbound final approached from the north
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
    # never cleared: the executor holds the base heading past the line; told to descend to land it meets
    # the ground, told nothing more it flies on until its time runs out
    grid = reading.words.copy()
    grid[grid[:, APPROACH] == CLEARED, APPROACH] = UNCHANGED
    flown, uncleared, _ = _fly_sentence(signals, grid)
    assert uncleared.outcome == "ground_contact" and not flown.modes["captured"][0].any()
    grid[1:, ALTITUDE] = UNCHANGED
    grid[1:, ANGLE] = UNCHANGED
    _, level, _ = _fly_sentence(signals, grid)
    assert level.outcome == "timeout" and not level.flew_the_sentence
    words = Words(spec())
    steep = reading.words.copy()
    steep[1, ALTITUDE], steep[1, ANGLE] = words.altitude_land, 4       # descend to land at the steepest class, at once
    _, grounded, _ = _fly_sentence(signals, steep)
    assert grounded.outcome == "ground_contact"


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
    # two 90° turns at 6°/row (3°/s); the middle half of each is the steady rate, and at 100 m/s neither
    # is in the fast band the bank cap is read from
    assert np.median(out["turn_mid_rate_deg_s"]) == pytest.approx(3.0, abs=0.05)
    assert out["turn_mid_fast_bank_deg"] == []
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
    pooled = {"turn_mid_rate_deg_s": [2.1, 2.16, 2.3], "turn_mid_fast_bank_deg": [24.2, 24.6, 40.0],
              "transition_accel_mps2": [-0.3, -0.24, -0.2, 0.1, 0.16, 0.5], "unspecified_slope_mps2": [-0.26, -0.3, -0.2],
              "crossing_height_m": [16.0, 20.84, 26.0]}
    measured = measure.measured_values(pooled, one)
    assert measured["values"] == {"turn_rate_deg_s": 2.15, "bank_cap_deg": 25.0, "decel_mps2": 0.24, "accel_mps2": 0.16,
                                  "unspecified_decel_mps2": 0.26, "land_aim_height_m": 20.8}
    assert measured["counts"]["decelerations"] == measured["counts"]["accelerations"] == 3
    # never past the vocabulary's own bank ceiling
    steep = measure.measured_values({**pooled, "turn_mid_fast_bank_deg": [40.0]}, one)
    assert steep["values"]["bank_cap_deg"] == one.turn_bank_max_deg


def test_method_a_takes_the_gentlest_roll_out_and_roll_rate_the_constraints_allow():
    from dataclasses import replace

    from ts_transformer.autopilot import derive

    one = spec()
    assert derive.heading_time_constant_s(2.15, one, 1.0) == 4.5              # 10° / 2.15°/s = 4.65 → 4.5
    with pytest.raises(ValueError, match="under 2 Δt"):
        derive.heading_time_constant_s(6.0, one, 1.0)
    params = _params(heading_time_constant_s=4.5)
    slow, fast = (derive.turn_overshoot_deg(replace(params, bank_rate_deg_s=p), 140.0) for p in (1.0, 5.0))
    assert slow > one.heading_tolerance_deg > fast >= 0.0                        # a slower roll passes further
    rate, overshoots = derive.roll_rate_deg_s(params, one)
    assert max(overshoots.values()) <= one.heading_tolerance_deg
    assert derive.turn_overshoot_deg(replace(params, bank_rate_deg_s=rate - derive.ROLL_RATE_STEP_DEG_S), 140.0) \
        > one.heading_tolerance_deg or rate == derive.ROLL_RATE_STEP_DEG_S


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
    scenario = SimpleNamespace(source={"arr_airport": "KXXX", "runway": "09"}, initial=SimpleNamespace(m=62000.0),
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
    leads, received = flight_leads(states, 1.0, _observed_series(geometry), geometry, reading, one, words, 30.0)
    said = [i for i in reading.instructions if i.row > 0 and i.column in (HEADING, ALTITUDE, ANGLE, SPEED)]
    assert received == len(said) == 5
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
                            ({**stored, "schema": "ts-executor-spec-v0"}, "is not a ts-executor-spec-v1 file")):
        (tmp_path / "spec.json").write_text(json.dumps(broken))
        with pytest.raises(ValueError, match=message):
            executor_spec.load_spec(tmp_path)


def test_only_flights_on_their_own_types_dynamics_with_a_published_approach_speed_are_flown(monkeypatch):
    from types import SimpleNamespace

    from ts_transformer.autopilot import replay

    def series(resolved, dynamics):
        source = {"resolved_typecode": resolved, "dynamics_typecode": dynamics}
        return SimpleNamespace(scenario=SimpleNamespace(source=source, initial=SimpleNamespace(m=62000.0)))

    assert replay.exclusion(series("A320", "A320")) is None
    assert replay.exclusion(series(None, "A320")) == "no identified type"
    assert replay.exclusion(series("A20N", "A320")) == "flown on a stand-in's dynamics"
    monkeypatch.setattr(replay, "approach_speed_ias_mps", lambda typecode, mass: math.nan)
    assert replay.exclusion(series("A320", "A320")) == "type publishes no approach speed"


def test_a_batch_readout_counts_what_was_flown_and_how_far_it_lies_from_the_observed_track():
    from ts_transformer.autopilot import replay

    signals = instruction_flight(*fly_legs(DOWNWIND_BASE_FINAL, 270.0, 1110.0, -400.0, 0.0))
    flown, verdict, reading = _fly_sentence(signals)
    batch = replay.Batch(signals=[signals], series=[], readings=[reading], geometries=[instruction_airport()],
                         approach_ias_mps=[], drawn={})
    summary = replay.summary([verdict])
    assert summary == {"flights": 1, "outcomes": {"landed": 1}, "landed_share": 1.0, "flew_the_sentence_share": 1.0,
                       "word_failures": {}}
    aligned = replay.alignment(batch, flown, [verdict])
    # the executor flies the words, not the legs: it stays within a kilometre of the synthetic flight and lands
    # later (after "unspecified" it slows to the A320's published approach speed, under the legs' 75 m/s)
    assert aligned["mean_horizontal_distance_m"]["p50"] < 1000.0 and aligned["mean_vertical_distance_m"]["p50"] < 30.0
    landing = aligned["landing_time_minus_observed_s"]
    assert landing["n"] == 1 and 0.0 < landing["p50"] < 40.0


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
