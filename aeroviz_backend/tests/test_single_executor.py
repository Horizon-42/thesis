"""The single-flight executor (`ts_transformer.autopilot.single`) against the batched torch executor it mirrors
(`ts_transformer.autopilot`): the executor tests' own synthetic flights flown by both — the same cycles, sentence times,
modes and limits, the same verdict for every word, the states apart by round-off only — and the pieces one at a time:
the word lookup, the clocks, torch's semantics on floats, a dynamics failure. Synthetic flights only — nothing here opens
an artefact, a spec or the frontend's data (the spec's reference tracks check it on real flights:
`autopilot.conformance`)."""

import dataclasses
import math
import random
import unittest

import numpy as np
import torch

from ts_transformer.autopilot import single
from aeroviz_backend.autopilot_segment.check_single import same
from aeroviz_backend.autopilot_segment.fly import fly_until
from ts_transformer.autopilot import replay
from ts_transformer.autopilot.executor import LIMITS, MODES, fly
from ts_transformer.autopilot.frame import AirportCharts, compass_deg, read_state, wrap180
from ts_transformer.autopilot.judge import judge
from ts_transformer.autopilot.lateral import Runways
from ts_transformer.autopilot.sentence import CLOCKS, DistanceClock, Sentences, TimeClock, TrackClock
from ts_transformer.instructions.labeller.read import read_flight
from ts_transformer.instructions.words import (
    ALTITUDE, ANGLE, APPROACH, APPROACH_CLEARED, APPROACH_GO_AROUND, HEADING, RUNWAY, SPEED, UNCHANGED, Words,
)
from ts_transformer.tests import test_autopilot as executor_tests
from ts_transformer.tests.support import fly_legs, instruction_airport, instruction_flight, instruction_spec

CPU = torch.device("cpu")
F64 = torch.float64
NEVER = lambda: False  # noqa: E731
#: How far apart the two executors' states may be — round-off, where the elementary functions differ in the last bit.
STATE_ROUNDOFF = {"lat_lon_deg": 1e-10, "alt_m": 1e-6, "speed_mps": 1e-7, "angle_rad": 1e-9}


def turn(degrees: float, speed: float, per_row: float = 4.5):
    return executor_tests._turn(degrees, speed, per_row)


def flights() -> dict[str, object]:
    """The executor tests' flights: a downwind, base and final; a straight-in captured at once; a 180° turn said word by
    word; descents levelled off at their targets; an orbit; fast turns at the bank limit."""
    glide = lambda speed: -speed * np.tan(np.radians(3.0))  # noqa: E731
    return {
        "downwind_base_final": instruction_flight(*fly_legs(executor_tests.DOWNWIND_BASE_FINAL, 270.0, 1110.0, -400.0, 0.0)),
        "straight_in": instruction_flight(*fly_legs(
            [(40, 0.0, 90.0, 0.0), (30, 0.0, 80.0, 0.0), (110, 0.0, 72.0, glide(72.0))], 90.0, 950.0, -300.0, 0.0)),
        "turn_word_by_word": instruction_flight(*fly_legs(
            [(30, 0.0, 100.0, 0.0), *turn(180.0, 100.0), (20, 0.0, 100.0, 0.0), *turn(-90.0, 100.0),
             (100, 0.0, 75.0, glide(75.0))], 0.0, 950.0, -400.0, 0.0)),
        "descents": instruction_flight(*fly_legs(
            [(100, 0.0, 75.0, 0.0), (80, 0.0, 75.0, -75.0 * np.tan(np.radians(2.1))), (120, 0.0, 75.0, 0.0),
             (100, 0.0, 75.0, glide(75.0))], 90.0, 1500.0, -400.0, 0.0)),
        "orbit": executor_tests._orbit(),
        "fast_turns": instruction_flight(*fly_legs(
            [(40, 0.0, 110.0, 0.0), *turn(-90.0, 110.0, 5.6), (20, 0.0, 110.0, 0.0), *turn(-90.0, 110.0, 5.6),
             (120, 0.0, 70.0, glide(70.0))], 270.0, 1110.0, -400.0, 0.0)),
    }


class BothExecutorsTest(unittest.TestCase):
    """Each flight's sentence flown from row 0 by `executor.fly` and by the single-flight executor on the backend's
    drive (`fly.fly_until`), and judged: the labelled sentences on every word clock, then sentences and starts changed
    until every mode and every limit binds somewhere (`test_every_mode_and_limit_is_flown_by_both`)."""

    def setUp(self):
        self.spec, self.geometry = instruction_spec(), instruction_airport()
        self.words, self.params = Words(self.spec), executor_tests._params()

    def reading(self, signals):
        return read_flight(signals, self.geometry, self.spec, self.words)

    def both(self, signals, grid=None, clock="track", *, geometry=None, start=None, runway=0):
        """``geometry``: another airport for the same flight (its runway ``runway`` judged); ``start``: changes to the
        initial geodetic row, ``{index: value}``."""
        spec, params, words = self.spec, self.params, self.words
        geometry = geometry or self.geometry
        reading = self.reading(signals)
        grid = reading.words if grid is None else grid
        inputs, _runways, _charts, approach = executor_tests._physics(signals, self.geometry)
        if start:
            state = inputs.initial_state.clone()
            for index, value in start.items():
                state[0, index] = value
            inputs = dataclasses.replace(inputs, initial_state=state)
        runways = Runways.of([geometry], [executor_tests.vertical_paths(geometry)], dtype=F64, device=CPU)
        charts = AirportCharts.of([geometry], dtype=F64, device=CPU)
        limit = len(grid) * spec.step_s * params.timeout_factor
        rows = len(grid)
        e_m, n_m = signals.e_m[:rows], signals.n_m[:rows]
        clocks = {"time": lambda: TimeClock(params.cycle_s),
                  "track": lambda: TrackClock.of([e_m], [n_m], spec.step_s, params.cycle_s, device=CPU),
                  "distance": lambda: DistanceClock.of([e_m], [n_m], spec.step_s, params.cycle_s, device=CPU)}
        theirs = fly(inputs, Sentences([grid], words, device=CPU), clocks[clock](), runways, charts, approach, params,
                     words, time_limit_s=torch.tensor([limit], dtype=F64))
        executor = single.SingleExecutor(inputs, geometry, executor_tests.vertical_paths(geometry), float(approach[0]),
                                         params, words, time_limit_s=limit)
        ours_clock = (single.TimeClock(params.cycle_s) if clock == "time"
                      else single.word_clock(executor_tests._params(word_clock=clock), e_m, n_m, spec.step_s))
        self.assertFalse(fly_until(executor, single.Sentence(grid, words), ours_clock, spec.step_s, None, NEVER))
        ours = executor.flown()
        verdicts = [judge(run, 0, geometry, runway, reading, signals, spec, words) for run in (ours, theirs)]
        return ours, theirs, verdicts

    def assert_same_flight(self, ours, theirs, verdicts, name):
        self.assertTrue(torch.equal(ours.done_cycle, theirs.done_cycle), name)
        self.assertEqual(ours.commands.shape, theirs.commands.shape, name)
        self.assertTrue(torch.equal(ours.sentence_s, theirs.sentence_s), name)
        for group in ("modes", "limits"):
            for key, flags in getattr(theirs, group).items():
                self.assertTrue(torch.equal(getattr(ours, group)[key], flags), f"{name}: {key}")
        finite = torch.isfinite(theirs.states).all(dim=2)[0]
        self.assertTrue(torch.equal(torch.isfinite(ours.states).all(dim=2)[0], finite), name)
        apart = (ours.states - theirs.states).abs()[0, finite]
        self.assertLessEqual(float(apart[:, :2].max()), STATE_ROUNDOFF["lat_lon_deg"], name)
        self.assertLessEqual(float(apart[:, 2].max()), STATE_ROUNDOFF["alt_m"], name)
        self.assertLessEqual(float(apart[:, 3].max()), STATE_ROUNDOFF["speed_mps"], name)
        self.assertLessEqual(float(apart[:, 4:6].max()), STATE_ROUNDOFF["angle_rad"], name)
        self.assertTrue(torch.allclose(ours.commands, theirs.commands, rtol=0.0, atol=1e-8, equal_nan=True), name)
        self.assertTrue(torch.allclose(ours.wanted, theirs.wanted, rtol=0.0, atol=1e-8, equal_nan=True), name)
        ours_verdict, their_verdict = verdicts
        self.assertEqual((ours_verdict.outcome, ours_verdict.end_row, ours_verdict.flew_the_sentence),
                         (their_verdict.outcome, their_verdict.end_row, their_verdict.flew_the_sentence), name)
        self.assertEqual(replay.word_results(ours_verdict), replay.word_results(their_verdict), name)
        self.assertTrue(same(ours_verdict.limits, their_verdict.limits), name)
        self.assertTrue(same(ours_verdict.words, their_verdict.words), name)
        self.assertTrue(same(ours_verdict.crossing, their_verdict.crossing), name)

    def test_every_flight_is_the_torch_executors_on_every_clock(self):
        for name, signals in flights().items():
            for clock in CLOCKS:
                ours, theirs, verdicts = self.both(signals, clock=clock)
                self.assert_same_flight(ours, theirs, verdicts, f"{name} on the {clock} clock")
                self.assertGreater(ours.commands.shape[1], 100, name)

    def variants(self):
        """``(name, both's arguments)``: sentences and starts that make every mode and limit bind."""
        words, all_flights = self.words, flights()
        straight, downwind = all_flights["straight_in"], all_flights["downwind_base_final"]
        grid = self.reading(straight).words
        parallel = instruction_flight(*fly_legs(
            [(100, 0.0, 100.0, 0.0), (5, 4.0, 100.0, 0.0), (29, 0.0, 100.0, 0.0), (5, -4.0, 90.0, 0.0),
             (100, 0.0, 75.0, -75.0 * np.tan(np.radians(3.0)))], 90.0, 1600.0, -400.0, 0.0))
        bent = self.reading(parallel).words.copy()
        bent[0, APPROACH] = APPROACH_CLEARED                      # cleared at once on a heading that just misses the line
        away = bent.copy()
        away[0, HEADING] = words.heading_index(60.0)              # a heading that cannot reach it even bent
        away[1:, HEADING] = UNCHANGED
        around = self.reading(downwind).words.copy()
        around[150, APPROACH] = APPROACH_GO_AROUND
        vectored = around.copy()                                  # step 8: vectored and climbing after it, cleared again
        vectored[165, HEADING] = words.heading_index(0.0)
        vectored[175, ALTITUDE], vectored[175, ANGLE] = words.altitude_index(1200.0), words.angle_climb
        vectored[200, HEADING] = words.heading_index(180.0)
        vectored[240, APPROACH] = APPROACH_CLEARED
        steep = grid.copy()                                       # "descend to land" from row 0 at the steepest class
        steep[:, [ALTITUDE, ANGLE]] = UNCHANGED
        steep[0, ALTITUDE], steep[0, ANGLE] = words.altitude_land, words.n_descent
        slow = grid.copy()                                        # the slowest speed word from row 0
        slow[:, SPEED] = UNCHANGED
        slow[0, SPEED] = 0
        pointed = grid.copy()
        pointed[:, RUNWAY] = UNCHANGED
        return [("bent", {"signals": parallel, "grid": bent}), ("intercepting off its word", {"signals": parallel, "grid": away}),
                ("go-around", {"signals": downwind, "grid": around}),
                ("go-around, vectored, climbing, cleared again", {"signals": downwind, "grid": vectored}),
                ("steepest class to land", {"signals": straight, "grid": steep}),
                ("slowest speed", {"signals": straight, "grid": slow}), ("fast turns", {"signals": all_flights["fast_turns"]}),
                ("40 m/s start", {"signals": straight, "start": {3: 40.0}}),
                ("5 m/s start, a dynamics failure", {"signals": straight, "start": {3: 5.0}}),
                ("climbing start", {"signals": straight, "start": {5: 0.5}}),
                ("descending start", {"signals": straight, "start": {5: -0.5}}),
                ("pointed at a parallel runway", {"signals": straight, "grid": _pointed(pointed, 1),
                                                  "geometry": executor_tests._parallels(), "runway": 1}),
                ("pointed at the runway's other end", {"signals": straight, "grid": _pointed(pointed, 3),
                                                       "geometry": executor_tests._parallels(), "runway": 3})]

    def test_every_mode_and_limit_is_flown_by_both(self):
        bound: set[str] = set()
        outcomes: set[str] = set()
        for name, arguments in self.variants():
            ours, theirs, verdicts = self.both(**arguments, clock="time")
            self.assert_same_flight(ours, theirs, verdicts, name)
            bound |= {key for key, flags in {**theirs.modes, **theirs.limits}.items() if bool(flags.any())}
            outcomes.add(verdicts[1].outcome)
        self.assertEqual(bound, set(MODES) | set(LIMITS))
        self.assertIn("dynamics_failure", outcomes)

    def test_a_late_clearance_is_the_torch_executors(self):
        signals = flights()["downwind_base_final"]
        grid = self.reading(signals).words
        clear = int(np.nonzero(grid[:, APPROACH] == APPROACH_CLEARED)[0][0])
        late = grid.copy()
        late[clear, APPROACH], late[clear + 10, APPROACH] = UNCHANGED, APPROACH_CLEARED
        ours, theirs, verdicts = self.both(signals, late)
        self.assert_same_flight(ours, theirs, verdicts, "late clearance")

    def test_a_flight_flown_alone_is_many_times_faster(self):
        """Not a benchmark: the reason the module exists — a flight's cycles in a few milliseconds, not a second."""
        import time
        signals = flights()["downwind_base_final"]
        started = time.perf_counter()
        self.both(signals, clock="time")
        both_s = time.perf_counter() - started
        spec, params = self.spec, self.params
        grid = self.reading(signals).words
        inputs, _runways, _charts, approach = executor_tests._physics(signals, self.geometry)
        executor = single.SingleExecutor(inputs, self.geometry, executor_tests.vertical_paths(self.geometry),
                                         float(approach[0]), params, self.words,
                                         time_limit_s=len(grid) * spec.step_s * params.timeout_factor)
        started = time.perf_counter()
        fly_until(executor, single.Sentence(grid, self.words), single.TimeClock(params.cycle_s), spec.step_s, None, NEVER)
        alone_s = time.perf_counter() - started
        self.assertLess(alone_s * 10.0, both_s)


def _pointed(grid, runway):
    changed = grid.copy()
    changed[0, RUNWAY] = runway
    return changed


class PiecesTest(unittest.TestCase):
    def setUp(self):
        self.spec = instruction_spec()
        self.words = Words(self.spec)

    def test_the_words_in_force_are_the_torch_lookups(self):
        signals = flights()["downwind_base_final"]
        grid = read_flight(signals, instruction_airport(), self.spec, self.words).words
        ours, theirs = single.Sentence(grid, self.words), Sentences([grid], self.words, device=CPU)
        for heard_s in np.arange(-3.0, len(grid) * 2.0 + 10.0, 0.5):
            mine, torch_words = ours.at(float(heard_s)), theirs.at(torch.tensor([heard_s], dtype=F64))
            for field in ("runway", "approach", "heading_deg", "land", "angle_class", "angle_deg", "unspecified"):
                self.assertEqual(getattr(mine, field), getattr(torch_words, field)[0].item(), (heard_s, field))
            for field in ("altitude_m", "speed_mps"):
                value = getattr(torch_words, field)[0].item()
                self.assertTrue(getattr(mine, field) == value or (math.isnan(value) and math.isnan(getattr(mine, field))))
            self.assertEqual(mine.issued_step, tuple(torch_words.issued_step[0].tolist()))

    def test_the_clocks_are_the_torch_clocks(self):
        rng = np.random.default_rng(7)
        e_m, n_m = np.cumsum(rng.normal(150.0, 30.0, 60)), np.cumsum(rng.normal(0.0, 60.0, 60))
        for name, ours, theirs in (
                ("track", single.TrackClock(e_m, n_m, 2.0, 1.0), TrackClock.of([e_m], [n_m], 2.0, 1.0, device=CPU)),
                ("distance", single.DistanceClock(e_m, n_m, 2.0, 1.0), DistanceClock.of([e_m], [n_m], 2.0, 1.0, device=CPU))):
            e, n = float(e_m[0]), float(n_m[0])
            for cycle in range(200):
                e, n = e + 70.0 + 10.0 * math.sin(cycle / 7.0), n + 20.0 * math.cos(cycle / 5.0)
                state = single.Kin(e, n, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                torch_state = type("K", (), {"e_m": torch.tensor([e], dtype=F64), "n_m": torch.tensor([n], dtype=F64)})
                self.assertAlmostEqual(ours.now(cycle, state), float(theirs.now(cycle, torch_state)[0]), places=9,
                                       msg=f"{name} cycle {cycle}")

    def test_a_distance_clock_over_a_zero_length_step_is_torchs(self):
        """The observed path's last step of zero length: torch divides by it (inf, clamped to the row's end)."""
        e_m, n_m = np.array([0.0, 100.0, 100.0]), np.zeros(3)
        ours, theirs = single.DistanceClock(e_m, n_m, 2.0, 1.0), DistanceClock.of([e_m], [n_m], 2.0, 1.0, device=CPU)
        for cycle, e in enumerate((0.0, 60.0, 120.0, 150.0, 190.0)):
            state = single.Kin(e, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            torch_state = type("K", (), {"e_m": torch.tensor([e], dtype=F64), "n_m": torch.tensor([0.0], dtype=F64)})
            self.assertEqual(ours.now(cycle, state), float(theirs.now(cycle, torch_state)[0]), cycle)

    def test_the_state_is_read_as_the_torch_executor_reads_it(self):
        geometry = instruction_airport()
        chart = single.Chart(geometry.frame.lat0, geometry.frame.lon0, geometry.frame.m_per_deg_lon)
        torch_chart = AirportCharts.of([geometry], dtype=F64, device=CPU)
        rng = random.Random(3)
        for _ in range(500):
            row = (geometry.frame.lat0 + rng.uniform(-0.3, 0.3), geometry.frame.lon0 + rng.uniform(-0.3, 0.3),
                   rng.uniform(0.0, 3000.0), rng.uniform(50.0, 150.0), rng.uniform(-7.0, 7.0), rng.uniform(-0.2, 0.2),
                   60000.0)
            ours, theirs = chart.read(row), read_state(torch.tensor([row], dtype=F64), torch_chart)
            for field in ("e_m", "n_m", "height_m", "speed_mps", "track_deg", "gamma_rad", "ground_speed_mps", "mass_kg"):
                self.assertAlmostEqual(getattr(ours, field), float(getattr(theirs, field)[0]), places=9, msg=field)
        for angle in (-721.3, -180.0, -0.0, 0.0, 179.999, 180.0, 540.5):
            self.assertAlmostEqual(single._wrap180(angle), float(wrap180(torch.tensor([angle], dtype=F64))[0]), places=12)
        self.assertAlmostEqual(chart.read((0.0, 0.0, 0.0, 1.0, -8.0, 0.0, 1.0)).track_deg,
                               float(compass_deg(torch.tensor([-8.0], dtype=F64))[0]), places=12)

    def test_torchs_semantics_on_floats(self):
        nan = math.nan
        for a, b in ((1.0, 2.0), (2.0, 1.0), (nan, 1.0), (1.0, nan), (-0.0, 0.0)):
            for ours, theirs in ((single._min(a, b), torch.minimum), (single._max(a, b), torch.maximum)):
                value = float(theirs(torch.tensor(a, dtype=F64), torch.tensor(b, dtype=F64)))
                self.assertTrue(ours == value or (math.isnan(ours) and math.isnan(value)), (a, b))
        self.assertTrue(math.isnan(single._clamp(nan, 0.0, 1.0)))
        self.assertEqual((single._clamp(-1.0, 0.0), single._clamp(2.0, high=1.0)), (0.0, 1.0))
        for x in (-2.0, -0.0, 0.0, 3.0, nan):
            self.assertEqual(single._sign(x), float(torch.sign(torch.tensor(x, dtype=F64))), x)
        for a, b in ((1.0, 0.0), (-1.0, 0.0), (0.0, 0.0), (1.0, -0.0), (nan, 0.0), (3.0, 2.0)):
            ours, value = single._divide(a, b), float(torch.tensor(a, dtype=F64) / torch.tensor(b, dtype=F64))
            self.assertTrue(ours == value or (math.isnan(ours) and math.isnan(value)), (a, b))

    def test_dynamics_that_leave_the_reals_end_in_a_non_finite_state(self):
        signals = flights()["straight_in"]
        inputs, *_ = executor_tests._physics(signals, instruction_airport())
        plant = single.Plant(inputs.aero_params[0].tolist(), inputs.frame_params[0].tolist(), float(inputs.max_thrust_n[0]))
        stopped = list(inputs.initial_state[0].tolist())
        stopped[3] = 0.0                                    # no airspeed: the chart's velocity basis divides by zero
        failed = plant.step(stopped, (0.5, 0.0, 1.0), 1.0)
        self.assertTrue(all(math.isnan(value) for value in failed[:6]))
        self.assertEqual(failed[6], stopped[6])             # the mass, whose rate is zero, as torch carries it
        moving = plant.step(inputs.initial_state[0].tolist(), (0.5, 0.0, 1.0), 1.0)
        self.assertTrue(all(math.isfinite(value) for value in moving))


if __name__ == "__main__":
    unittest.main()
