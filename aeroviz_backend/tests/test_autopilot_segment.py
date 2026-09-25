"""The live executor segment (`aeroviz_backend.autopilot_segment`): which segment a selected word is, what the executor
is told, where the flight is stopped, which of the judge's results is the selected word's, the answer written, the
service's lookups and caches, the endpoint's statuses, and the frontend's copies of this module's names. Synthetic
readings and verdicts only — nothing here opens an artefact, a spec or the frontend's data."""

import json
import os
import re
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from aeroviz_backend.autopilot_segment import fly as fly_module, payload as payload_module
from aeroviz_backend.autopilot_segment.backend import FLIGHT_CACHE_SIZE, AutopilotSegmentBackend
from aeroviz_backend.autopilot_segment.errors import NotFlyable, NotListed, RequestRefused, Superseded
from aeroviz_backend.autopilot_segment.fly import FlightContext, FlownSegment, fly_batch_until, fly_until
from aeroviz_backend.autopilot_segment.payload import (
    SCHEMA, SEGMENT_END, band_cut_by_stop, heading_facts, segment_payload, track_payload,
)
from aeroviz_backend.autopilot_segment.segment import segment_of, segment_reading, segment_signals, told_words
from aeroviz_backend.autopilot_segment.verdict import STATUSES, HeadingFacts, selected_heading, word_verdict
from aeroviz_backend.http_server import AeroVizBackendApp
from aeroviz_backend.paths import REPO_ROOT
from ts_transformer.autopilot.sentence import row_at
from ts_transformer.autopilot.executor import LIMITS, MODES, Flown
from ts_transformer.autopilot.frame import ALT, GAMMA, LAT, LON, MASS, PSI, SPEED as SPEED_STATE
from ts_transformer.autopilot.judge import Verdict
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.labeller.read import Reading
from ts_transformer.instructions.labeller.records import Instruction
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.words import ALTITUDE, ANGLE, APPROACH, HEADING, RUNWAY, SPEED, UNCHANGED

U = UNCHANGED
#: No newer request ever comes in.
NEVER = lambda: False  # noqa: E731
#: The heading lead in steps (4 s at 2 s a step), as `fly_segment` passes it.
LEAD = 2


def reading() -> Reading:
    """Ten steps: step 0 says all six; heading words at 3 and 6; the clearance at 5; an altitude word at 4 and an angle
    word at 7; a speed word at 8."""
    said = [
        Instruction(RUNWAY, 1, 0, "initial", {"ident": "23R"}),
        Instruction(APPROACH, 0, 0, "initial"),
        Instruction(HEADING, 10, 0, "initial", {"target_deg": 50.0}),
        Instruction(ALTITUDE, 7, 0, "initial"),
        Instruction(ANGLE, 0, 0, "initial"),
        Instruction(SPEED, 4, 0, "initial"),
        Instruction(HEADING, 12, 3, "per-step", {"target_deg": 60.0}),
        Instruction(ALTITUDE, 5, 4, "target"),
        Instruction(APPROACH, 1, 5, "clear"),
        Instruction(HEADING, 14, 6, "per-step", {"target_deg": 70.0}),
        Instruction(ANGLE, 2, 7, "target"),
        Instruction(SPEED, 3, 8, "target"),
    ]
    grid = np.full((10, 6), U, dtype=np.int16)
    for word in said:
        grid[word.row, word.column] = word.value
    return Reading(dataset_id="KXXX:test", airport="KXXX", runway_index=1, words=grid, instructions=said,
                   capture_row=6, join_row=5, unspecified_row=9, cut_at_crossing=True, checks={"heading": []})


class SegmentTest(unittest.TestCase):
    def test_a_segment_runs_from_its_word_to_the_next_word_of_its_column_and_one_silent_step_more(self):
        segment = segment_of(reading(), ALTITUDE, 0, LEAD)
        self.assertEqual((segment.row, segment.end_row, segment.stop_row, segment.to_landing), (0, 4, 4, False))
        # step 0: the six words in force; then steps 1–3 as said (the heading word at 3); then step 4, silent
        np.testing.assert_array_equal(segment.grid, [[1, 0, 10, 7, 0, 4], [U] * 6, [U] * 6, [U, U, 12, U, U, U], [U] * 6])

    def test_a_heading_word_is_flown_a_lead_past_the_next_heading_word_which_is_told_as_the_sentence_says(self):
        segment = segment_of(reading(), HEADING, 3, LEAD)
        self.assertEqual((segment.end_row, segment.stop_row, segment.to_landing), (6, 8, False))
        np.testing.assert_array_equal(segment.grid, [[1, 0, 12, 7, 0, 4],
                                                     [U, U, U, 5, U, U],
                                                     [U, 1, U, U, U, U],
                                                     [U, U, 14, U, U, U],
                                                     [U, U, U, U, 2, U],
                                                     [U] * 6])
        self.assertEqual([(w.column, w.row, w.value, w.kind) for w in segment.instructions],
                         [(RUNWAY, 0, 1, "initial"), (APPROACH, 0, 0, "initial"), (HEADING, 0, 12, "per-step"),
                          (ALTITUDE, 0, 7, "initial"), (ANGLE, 0, 0, "initial"), (SPEED, 0, 4, "initial"),
                          (ALTITUDE, 1, 5, "target"), (APPROACH, 2, 1, "clear"), (HEADING, 3, 14, "per-step"),
                          (ANGLE, 4, 2, "target")])
        # the selected word keeps its own diagnostics (the heading word's target)
        self.assertEqual(segment.instructions[HEADING].info["target_deg"], 60.0)
        # a heading word whose lead runs past the sentence is flown to the landing
        self.assertTrue(segment_of(reading(), HEADING, 6, LEAD).to_landing)

    def test_the_words_told_are_listed_by_step_then_column_at_the_flights_steps(self):
        told = told_words(segment_of(reading(), HEADING, 3, LEAD))
        self.assertEqual([(w["row"], w["column"]) for w in told],
                         [(3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (4, ALTITUDE), (5, APPROACH), (6, HEADING),
                          (7, ANGLE)])

    def test_a_columns_last_word_is_flown_to_the_landing(self):
        segment = segment_of(reading(), ANGLE, 7, LEAD)
        self.assertEqual((segment.end_row, segment.stop_row, segment.to_landing, len(segment.grid)), (10, 10, True, 3))
        np.testing.assert_array_equal(segment.grid[0], [1, 1, 14, 5, 2, 4])
        np.testing.assert_array_equal(segment.grid[1:], [[U, U, U, U, U, 3], [U, U, U, U, U, U]])
        self.assertTrue(segment_reading(reading(), segment).cut_at_crossing)

    def test_a_segment_starts_only_where_its_word_is_said_and_has_a_step_to_fly(self):
        # a request the view cannot make: answered as one (HTTP 400)
        with self.assertRaisesRegex(RequestRefused, "no heading word is said at step 4"):
            segment_of(reading(), HEADING, 4, LEAD)
        with self.assertRaisesRegex(RequestRefused, "not a step"):
            segment_of(reading(), HEADING, 10, LEAD)
        with self.assertRaisesRegex(RequestRefused, "not one of the six"):
            segment_of(reading(), 6, 0, LEAD)
        last = reading()
        last.words[9, SPEED] = 9
        last.instructions.append(Instruction(SPEED, 9, 9, "unspecified"))
        with self.assertRaisesRegex(RequestRefused, "the sentence's last, has no step after it to fly"):
            segment_of(last, SPEED, 9, LEAD)

    def test_the_segments_reading_and_observed_rows_are_renumbered_from_its_first_step(self):
        segment = segment_of(reading(), HEADING, 3, LEAD)
        renumbered = segment_reading(reading(), segment)
        self.assertEqual((renumbered.join_row, renumbered.capture_row, renumbered.unspecified_row), (2, 3, 6))
        self.assertFalse(renumbered.cut_at_crossing)
        self.assertEqual(renumbered.checks, {})
        rows = np.arange(10, dtype=np.float64)
        signals = FlightSignals(dataset_id="KXXX:test", airport="KXXX", runway="23R", typecode="A320",
                                entry_time_utc="2026-09-01T00:00:00Z", landing_time_utc="2026-09-01T00:05:00Z",
                                time_s=2.0 * rows, e_m=rows, n_m=rows, altitude_m=rows, track_deg=rows,
                                ground_speed_mps=rows, vertical_rate_mps=rows)
        np.testing.assert_array_equal(segment_signals(signals, segment).e_m, [3.0, 4.0, 5.0, 6.0, 7.0, 8.0])


def flown(sentence_s: list[float], done_cycle: int) -> Flown:
    cycles = len(sentence_s)
    return Flown(states=torch.zeros(1, cycles + 1, 7, dtype=torch.float64),
                 commands=torch.zeros(1, cycles, 3, dtype=torch.float64),
                 wanted=torch.zeros(1, cycles, 3, dtype=torch.float64),
                 limits={name: torch.zeros(1, cycles, dtype=torch.bool) for name in LIMITS},
                 modes={name: torch.zeros(1, cycles, dtype=torch.bool) for name in MODES},
                 done_cycle=torch.tensor([done_cycle]), sentence_s=torch.tensor([sentence_s], dtype=torch.float64),
                 cycle_s=1.0)


class FakeExecutor:
    """The stepper's surface `fly_until` drives: cycles flown, a done flag after ``done_after`` cycles."""

    def __init__(self, cycles: int, done_after: int | None = None) -> None:
        self.cycles, self.step_rows, self.count, self.heard = cycles, 2, 0, []
        self.done_after = done_after
        self.done = torch.zeros(1, dtype=torch.bool)

    def now(self):
        return None

    def cycle(self, force, sentence_s):
        self.heard.append(float(force))
        self.count += 1
        self.done = torch.tensor([self.done_after is not None and self.count >= self.done_after])


class FakeClock:
    def __init__(self, times: list[float]) -> None:
        self.times = times

    def now(self, cycle, state):
        return torch.tensor([self.times[cycle]], dtype=torch.float64)


class FakeSentences:
    def at(self, heard_s):
        return heard_s[0]


class StopTest(unittest.TestCase):
    def test_it_stops_before_the_first_cycle_that_starts_a_step_the_clock_puts_at_the_stop(self):
        executor = FakeExecutor(cycles=10)
        # the time clock: step 3 starts at cycle 6, which is never flown
        self.assertTrue(fly_until(executor, FakeSentences(), FakeClock([float(c) for c in range(10)]), 2.0, 3, NEVER))
        self.assertEqual(executor.count, 6)
        # the words of each step are heard on the cycle that starts it, as `executor.fly` hears them
        self.assertEqual(executor.heard, [0.0, 0.0, 2.0, 2.0, 4.0, 4.0])

    def test_a_clock_that_runs_ahead_stops_on_a_cycle_that_starts_a_step(self):
        executor = FakeExecutor(cycles=6)
        # cycle 3 reads step 1.95 but starts no step; cycle 4 starts one at step 2.1
        self.assertTrue(fly_until(executor, FakeSentences(), FakeClock([0.0, 0.5, 1.2, 3.9, 4.2, 6.0]), 2.0, 2, NEVER))
        self.assertEqual(executor.count, 4)

    def test_a_flight_done_first_or_out_of_cycles_does_not_reach_its_stop(self):
        done = FakeExecutor(cycles=10, done_after=3)
        self.assertFalse(fly_until(done, FakeSentences(), FakeClock([float(c) for c in range(10)]), 2.0, 3, NEVER))
        self.assertEqual(done.count, 3)
        short = FakeExecutor(cycles=4)
        self.assertFalse(fly_until(short, FakeSentences(), FakeClock([float(c) for c in range(4)]), 2.0, 3, NEVER))
        self.assertEqual(short.count, 4)

    def test_a_newer_request_stops_it_before_the_next_cycle(self):
        executor = FakeExecutor(cycles=10)
        asked = iter([False, False, True])
        with self.assertRaisesRegex(Superseded, "stopped after 2 cycles"):
            fly_until(executor, FakeSentences(), FakeClock([float(c) for c in range(10)]), 2.0, None, lambda: next(asked))
        self.assertEqual(executor.count, 2)

    def test_without_a_stop_it_flies_to_the_outcome(self):
        executor = FakeExecutor(cycles=10, done_after=7)
        self.assertFalse(fly_until(executor, FakeSentences(), FakeClock([float(c) for c in range(10)]), 2.0, None, NEVER))
        self.assertEqual(executor.count, 7)


class StepperTest(unittest.TestCase):
    """`fly_until` IS `executor.fly` up to its stop — the real executor on a synthetic downwind, base and final (the
    executor tests' own flight): without a stop state for state, with one the whole flight cut at the first cycle that
    starts a step the clock puts at the stop. If `fly`'s loop changes, this fails before the copy goes stale."""

    def setUp(self):
        from ts_transformer.instructions.labeller.read import read_flight
        from ts_transformer.instructions.words import Words
        from ts_transformer.tests import test_autopilot as executor_tests
        from ts_transformer.tests.support import fly_legs, instruction_airport, instruction_flight, instruction_spec

        self.spec, geometry = instruction_spec(), instruction_airport()
        self.words, self.params = Words(self.spec), executor_tests._params()
        self.signals = instruction_flight(*fly_legs(executor_tests.DOWNWIND_BASE_FINAL, 270.0, 1110.0, -400.0, 0.0))
        self.grid = read_flight(self.signals, geometry, self.spec, self.words).words
        self.physics = executor_tests._physics(self.signals, geometry)
        self.limit = torch.tensor([len(self.grid) * self.spec.step_s * self.params.timeout_factor], dtype=torch.float64)

    def clock(self, name: str):
        from ts_transformer.autopilot.sentence import TimeClock, TrackClock
        if name == "time":
            return TimeClock(self.params.cycle_s)
        rows = len(self.grid)
        return TrackClock.of([self.signals.e_m[:rows]], [self.signals.n_m[:rows]], self.spec.step_s, self.params.cycle_s,
                             device=torch.device("cpu"))

    def fly(self, name: str, stop: int | None):
        from ts_transformer.autopilot.executor import Executor, fly
        from ts_transformer.autopilot.sentence import Sentences
        inputs, runways, charts, approach = self.physics
        sentences = lambda: Sentences([self.grid], self.words, device=torch.device("cpu"))  # noqa: E731
        whole = fly(inputs, sentences(), self.clock(name), runways, charts, approach, self.params, self.words,
                    time_limit_s=self.limit)
        executor = Executor(inputs, runways, charts, approach, self.params, self.words, time_limit_s=self.limit)
        reached = fly_until(executor, sentences(), self.clock(name), self.spec.step_s, stop, NEVER)
        return whole, executor.flown(), reached, executor.step_rows

    def test_without_a_stop_it_is_executor_fly_state_for_state(self):
        for name in ("time", "track"):
            whole, stepped, reached, _ = self.fly(name, None)
            self.assertFalse(reached)
            self.assertTrue(torch.equal(stepped.states, whole.states), name)
            self.assertTrue(torch.equal(stepped.commands, whole.commands), name)
            self.assertTrue(torch.equal(stepped.done_cycle, whole.done_cycle), name)
            self.assertTrue(torch.equal(stepped.sentence_s, whole.sentence_s), name)

    def test_with_a_stop_it_is_executor_fly_cut_where_the_clock_first_starts_a_step_there(self):
        for name in ("time", "track"):
            stop = 20
            whole, stepped, reached, step_rows = self.fly(name, stop)
            self.assertTrue(reached)
            starts = row_at(whole.sentence_s[0].numpy(), self.spec.step_s)[::step_rows]
            cycle = int(np.searchsorted(starts, stop)) * step_rows
            self.assertEqual(stepped.commands.shape[1], cycle, name)
            self.assertTrue(torch.equal(stepped.states, whole.states[:, : cycle + 1]), name)
            self.assertTrue(torch.equal(stepped.commands, whole.commands[:, :cycle]), name)


class SetupTest(unittest.TestCase):
    """`fly_batch_until` sets the executor up as `replay.fly_sentences` sets up `executor.fly` — the same inputs, runways,
    charts, approach speeds, parameters, words, time limit and word clock — with its own stepper in place of `fly`."""

    def test_the_executor_is_set_up_as_the_replay_sets_it_up(self):
        calls: dict[str, list] = {"inputs": [], "sentences": [], "runways": [], "charts": []}

        def recorder(name, value):
            def record(*args, **kwargs):
                calls[name].append((args, kwargs))
                return value
            return record

        grid = np.arange(42).reshape(7, 6)
        batch = SimpleNamespace(inputs=recorder("inputs", "the inputs"), readings=[SimpleNamespace(words=grid)],
                                geometries=["the geometry"], crossing_heights=[(15.0,)], approach_ias_mps=[70.0])
        params, words = SimpleNamespace(timeout_factor=1.5), SimpleNamespace(spec=SimpleNamespace(step_s=2.0))
        recorded = {}

        class Stop(Exception):
            pass

        def record(name, stops=True):
            def recorder(*args, **kwargs):
                recorded[name] = (args, kwargs)
                if stops:
                    raise Stop
            return recorder

        clock = mock.Mock(return_value="the clock")
        with mock.patch.object(fly_module, "Sentences", recorder("sentences", "the sentences")), \
                mock.patch("ts_transformer.autopilot.replay.Sentences", recorder("sentences", "the sentences")), \
                mock.patch("ts_transformer.autopilot.replay.word_clock", clock), \
                mock.patch.object(fly_module.Runways, "of", recorder("runways", "the runways")), \
                mock.patch.object(fly_module.AirportCharts, "of", recorder("charts", "the charts")), \
                mock.patch.object(fly_module, "Executor", record("executor", stops=False)), \
                mock.patch.object(fly_module, "fly_until", record("fly_until")), \
                mock.patch("ts_transformer.autopilot.replay.fly", record("fly")):
            with self.assertRaises(Stop):
                fly_batch_until(batch, params, words, None, NEVER)
            with self.assertRaises(Stop):
                fly_module.replay.fly_sentences(batch, params, words, device=fly_module.DEVICE)
        (inputs, runways, charts, ias, *rest), kwargs = recorded["executor"]
        (fly_inputs, _sentences, _clock, fly_runways, fly_charts, fly_ias, *fly_rest), fly_kwargs = recorded["fly"]
        self.assertEqual((inputs, runways, charts), (fly_inputs, fly_runways, fly_charts))
        self.assertTrue(torch.equal(ias, fly_ias))
        self.assertEqual(rest, fly_rest)
        self.assertTrue(torch.equal(kwargs["time_limit_s"], fly_kwargs["time_limit_s"]))
        # every piece built from the same arguments on both paths (the grids compared as arrays)
        for name in ("inputs", "runways", "charts"):
            self.assertEqual(len(calls[name]), 2, name)
            self.assertEqual(calls[name][0], calls[name][1], name)
        (ours, our_kwargs), (theirs, their_kwargs) = calls["sentences"]
        self.assertEqual(len(ours[0]), 1)
        np.testing.assert_array_equal(ours[0][0], theirs[0][0])
        self.assertEqual((ours[1:], our_kwargs), (theirs[1:], their_kwargs))
        # the word clock: asked for once by each, the same way, and handed to the stepper as to `fly`
        self.assertEqual(clock.call_args_list[0], clock.call_args_list[1])
        self.assertEqual(recorded["fly_until"][0][1:3], ("the sentences", "the clock"))
        self.assertEqual((_sentences, _clock), ("the sentences", "the clock"))


SPEC = SimpleNamespace(heading_lead_s=4.0, heading_tolerance_deg=4.5, step_s=2.0, rows_exact=lambda seconds: int(round(seconds / 2.0)))
#: A heading word's flown facts: never left to intercept on its own, twelve rows judged.
HELD = HeadingFacts(off_word_cycles=0, judged_rows=12)
WORDS = SimpleNamespace(speed_mps=lambda value: None if value == 9 else 60.0 + value)


def verdict(**words) -> Verdict:
    judged = {"heading": [], "capture_turn": None, "intercepting_off_word_cycles": 0,
              "corridor": {"cleared": False, "entered": False, "rows": 0, "inside": 0},
              "vertical": [], "speed": [], **words}
    return Verdict("timeout", 12, None, {}, judged, flown_rows=13)


class WordVerdictTest(unittest.TestCase):
    def test_a_heading_word_is_inside_when_every_judged_row_is(self):
        segment = segment_of(reading(), HEADING, 3, LEAD)
        # the next heading word, told at the segment's step 3, has no row of its own in it
        inside = word_verdict(verdict(heading=[{"row": 0, "rows": 4, "inside": 4}, {"row": 3, "rows": 0, "inside": 0}]),
                              segment, SPEC, WORDS, HELD)
        outside = word_verdict(verdict(heading=[{"row": 0, "rows": 4, "inside": 3}]), segment, SPEC, WORDS, HELD)
        unjudged = word_verdict(verdict(heading=[{"row": 0, "rows": 0, "inside": 0}]), segment, SPEC, WORDS, HELD)
        self.assertEqual([inside["status"], outside["status"], unjudged["status"]], ["inside", "outside", "not judged"])
        self.assertEqual((outside["checks"][0]["inside"], outside["checks"][0]["rows"]), (3, 4))
        self.assertRegex(unjudged["reason"], "at or past the clearance the executor was told or its capture")
        # its rows begin past the flown track its judge read
        short = word_verdict(verdict(heading=[{"row": 0, "rows": 0, "inside": 0}]), segment, SPEC, WORDS,
                             HeadingFacts(off_word_cycles=0, judged_rows=2))
        self.assertRegex(short["reason"], "past the end of the flown track its judge read")

    def test_a_heading_word_left_to_intercept_the_final_on_its_own_fails_whatever_its_rows(self):
        segment = segment_of(reading(), HEADING, 3, LEAD)
        left = HeadingFacts(off_word_cycles=3, judged_rows=12)
        with_rows = word_verdict(verdict(heading=[{"row": 0, "rows": 4, "inside": 4}]), segment, SPEC, WORDS, left)
        # the common case: the last heading word before the clearance, whose lead carries its rows past it
        without_rows = word_verdict(verdict(heading=[{"row": 0, "rows": 0, "inside": 0}]), segment, SPEC, WORDS, left)
        self.assertEqual((with_rows["status"], len(with_rows["checks"])), ("outside", 2))
        self.assertEqual((without_rows["status"], [check["ok"] for check in without_rows["checks"]]), ("outside", [False]))
        self.assertRegex(without_rows["checks"][0]["name"], "left for 3 cycles to intercept the final on its own")

    def test_the_cycles_left_to_intercept_are_the_selected_words_only_before_the_next_heading_word_is_heard(self):
        segment = segment_of(reading(), HEADING, 3, LEAD)
        run = flown([float(c) for c in range(10)], 9)
        # the next heading word, at the segment's step 3, is heard at cycle 6 (2 s steps of 1 s cycles)
        judged = SimpleNamespace(smoothed=SimpleNamespace(track_deg=np.zeros(5)))

        def facts(cycles: list[int], end_row: int) -> HeadingFacts:
            off = torch.zeros(1, 10, dtype=torch.bool)
            off[0, cycles] = True
            result = FlownSegment(segment=segment, reading=segment_reading(reading(), segment), signals=None,
                                  flown=replace(run, modes={**run.modes, "intercepting_off_word": off}),
                                  verdict=Verdict("timeout", end_row, None, {}, {}, flown_rows=end_row + 1),
                                  reached_end=True, fly_s=0.0, judge_s=0.0)
            return heading_facts(result, judged, SPEC)

        # cycle 5 is the word's last; cycle 6 is the next heading word's first
        self.assertEqual(facts([5, 6], end_row=9), HeadingFacts(off_word_cycles=1, judged_rows=5))
        # and none at or past the outcome: an outcome at state row 4 leaves cycles 0–3 flown, so of 3, 4 and 5 only 3 counts
        self.assertEqual(facts([3, 4, 5], end_row=4).off_word_cycles, 1)

    def test_the_clearance_is_its_capture_turn_and_corridor(self):
        segment = segment_of(reading(), APPROACH, 5, LEAD)
        judged = word_verdict(verdict(capture_turn={"progress_ok": True, "rate_ok": True},
                                      corridor={"cleared": True, "entered": True, "rows": 5, "inside": 4}),
                              segment, SPEC, WORDS, None)
        self.assertEqual(judged["status"], "outside")
        self.assertEqual([check["ok"] for check in judged["checks"]], [True, True, True, False])
        self.assertEqual(word_verdict(verdict(), segment_of(reading(), APPROACH, 0, LEAD), SPEC, WORDS, None)["status"],
                         "no check")

    def test_an_altitude_word_is_its_tube_and_an_angle_word_every_tube_it_anchors(self):
        tubes = [{"row": 0, "rows": 3, "inside": 3, "contained": True, "target_m": 210.0},
                 {"row": 2, "rows": 5, "inside": 4, "contained": False, "target_m": None}]
        altitude = word_verdict(verdict(vertical=tubes[:1]), segment_of(reading(), ALTITUDE, 4, LEAD), SPEC, WORDS, None)
        angle = word_verdict(verdict(vertical=tubes), segment_of(reading(), ANGLE, 7, LEAD), SPEC, WORDS, None)
        self.assertEqual(altitude["status"], "inside")
        self.assertEqual(angle["status"], "outside")
        self.assertEqual([check["ok"] for check in angle["checks"]], [True, False])
        self.assertEqual([check["name"] for check in angle["checks"]],
                         ["in the tube of the altitude word 210 m", "in the tube of the altitude word descend to land"])

    def test_a_speed_word_is_its_transition_and_band_and_the_pilots_own_speed_has_none(self):
        span = {"row": 0, "transition_ok": True, "cut_before_arrival": False, "band_rows": 4, "band_inside": 4,
                "contained": True}
        self.assertEqual(word_verdict(verdict(speed=[span]), segment_of(reading(), SPEED, 8, LEAD), SPEC, WORDS,
                                      None)["status"], "inside")
        own = reading()
        own.words[8, SPEED] = 9
        own.instructions[-1] = Instruction(SPEED, 9, 8, "unspecified")
        self.assertEqual(word_verdict(verdict(), segment_of(own, SPEED, 8, LEAD), SPEC, WORDS, None)["status"], "no check")

    def test_a_flown_segment_the_gate_refuses_judges_nothing(self):
        refused = Verdict("timeout", 12, None, {}, None, flown_rows=13, refused="too short")
        self.assertEqual(word_verdict(refused, segment_of(reading(), HEADING, 3, LEAD), SPEC, WORDS, None)["status"],
                         "not judged")

    def test_the_selected_heading_word_is_the_one_told_at_step_0(self):
        judged = {"heading": [{"row": 3, "rows": 0, "inside": 0}, {"row": 0, "rows": 5, "inside": 2}]}
        self.assertEqual(selected_heading(judged), {"row": 0, "rows": 5, "inside": 2})


def geometry() -> AirportGeometry:
    return AirportGeometry.from_dict({
        "code": "KXXX", "reference": {"lat": 35.0, "lon": -78.0, "elevation_m": 100.0},
        "candidates": [{"ident": "09", "threshold_e_m": 0.0, "threshold_n_m": 0.0, "course_deg": 90.0,
                        "elevation_m": 100.0, "length_m": 3000.0}],
        "runway_ends": [{"ident": "09", "threshold_e_m": 0.0, "threshold_n_m": 0.0, "course_deg": 90.0}],
    })


class TrackTest(unittest.TestCase):
    def payload(self, outcome: str):
        """Five states flying east at 100 m/s from the segment of the heading word said at step 3, banked 10° left."""
        cycles = 4
        states = torch.zeros(1, cycles + 1, 7, dtype=torch.float64)
        states[0, :, LAT] = 35.0
        states[0, :, LON] = -78.0 + torch.arange(cycles + 1, dtype=torch.float64) * 100.0 / (111320.0 * 0.8191520)
        states[0, :, ALT], states[0, :, SPEED_STATE], states[0, :, PSI] = 900.0, 100.0, 0.0
        states[0, :, GAMMA], states[0, :, MASS] = 0.0, 60000.0
        base = flown([float(c) for c in range(cycles)], cycles - 1)
        commands = torch.zeros(1, cycles, 3, dtype=torch.float64)
        commands[0, :, 1] = np.radians(10.0)
        run = replace(base, states=states, commands=commands)
        segment = segment_of(reading(), HEADING, 3, LEAD)
        result = FlownSegment(segment=segment, reading=segment_reading(reading(), segment), signals=None, flown=run,
                              verdict=Verdict(outcome, cycles, None, {}, None, flown_rows=cycles + 1), reached_end=True,
                              fly_s=0.1, judge_s=0.01)
        # the observed smoothed track reads 450° (one turn up) and has flown 1200 m by step 3
        context = FlightContext(signals=None, series=None, reading=reading(), geometry=geometry(), crossing_heights=(15.0,),
                                group="own dynamics", approach_ias_mps=70.0,
                                observed_track_deg=np.full(10, 450.0), observed_distance_m=400.0 * np.arange(10))
        return track_payload(result, context, 2.0)

    def test_the_flown_segment_reads_on_the_flights_own_clock_distance_and_heading_branch(self):
        track, shift = self.payload("timeout")
        self.assertEqual(track["tS"], [6.0, 7.0, 8.0, 9.0, 10.0])
        self.assertEqual(shift, 360.0)
        np.testing.assert_allclose(track["trackDeg"], 450.0, atol=1e-6)
        np.testing.assert_allclose(track["distanceM"], [1200.0, 1300.0, 1400.0, 1500.0, 1600.0], atol=0.6)
        # the commands: one per cycle, the dynamics' left bank read as a negative right bank
        self.assertEqual(track["bankRightDeg"], [-10.0] * 4)
        self.assertEqual(len(track["loadFactor"]), 4)

    def test_a_dynamics_failure_leaves_the_failed_state_out(self):
        track, _ = self.payload("dynamics_failure")
        self.assertEqual(len(track["tS"]), 4)
        self.assertEqual(len(track["thrustFraction"]), 3)


def signals10() -> FlightSignals:
    rows = np.arange(10, dtype=np.float64)
    return FlightSignals(dataset_id="KXXX:test", airport="KXXX", runway="09", typecode="A320",
                         entry_time_utc="2026-09-01T00:00:00Z", landing_time_utc="2026-09-01T00:05:00Z",
                         time_s=2.0 * rows, e_m=100.0 * rows, n_m=0.0 * rows, altitude_m=np.full(10, 900.0),
                         track_deg=np.full(10, 90.0), ground_speed_mps=np.full(10, 100.0), vertical_rate_mps=0.0 * rows)


class PayloadTest(unittest.TestCase):
    """The answer for a segment flown to its stop: the reason (`SEGMENT_END` exactly when it got there before any
    event), where it was against the observed aircraft there, the observed time over the same steps, the flown time."""

    def answer(self, reached: bool, outcome: str = "timeout"):
        segment = segment_of(reading(), ALTITUDE, 0, LEAD)           # steps 0–4, stopped at 4
        cycles = 8
        states = torch.zeros(1, cycles + 1, 7, dtype=torch.float64)
        states[0, :, LAT] = 35.0
        states[0, :, LON] = -78.0 + torch.arange(cycles + 1, dtype=torch.float64) * 90.0 / (111320.0 * 0.8191520)
        states[0, :, ALT], states[0, :, SPEED_STATE], states[0, :, MASS] = 890.0, 95.0, 60000.0
        run = replace(flown([float(c) for c in range(cycles)], cycles - 1), states=states)
        judged = {"heading": [], "capture_turn": None, "intercepting_off_word_cycles": 0,
                  "corridor": {"cleared": False, "entered": False, "rows": 0, "inside": 0},
                  "vertical": [{"row": 0, "rows": 4, "inside": 4, "contained": True, "target_m": 1110.0}], "speed": []}
        limits = {"cycles": {"cycles": cycles}, "bank_rate": {"cycles": 2}, "bank_cap": {"cycles": 0}}
        result = FlownSegment(segment=segment, reading=segment_reading(reading(), segment), signals=None, flown=run,
                              verdict=Verdict(outcome, cycles, None, limits, judged, flown_rows=cycles + 1),
                              reached_end=reached, fly_s=0.1, judge_s=0.01)
        context = FlightContext(signals=signals10(), series=None, reading=reading(), geometry=geometry(),
                                crossing_heights=(15.0,), group="own dynamics", approach_ias_mps=70.0,
                                observed_track_deg=np.full(10, 90.0), observed_distance_m=200.0 * np.arange(10))
        words = SimpleNamespace(spec=SPEC, speed_mps=WORDS.speed_mps)
        return segment_payload(result, context, words)

    def test_a_segment_that_got_to_its_stop_ends_there_and_says_where_it_was_against_the_observed_aircraft(self):
        body = self.answer(reached=True)
        self.assertEqual(body["end"]["reason"], SEGMENT_END)
        self.assertEqual((body["end"]["flownS"], body["segment"]["observedS"], body["end"]["reachedSegmentEnd"]), (8.0, 8.0, True))
        # the flown end, 8 × 90 m east at 890 m, 95 m/s; the observed aircraft at step 4: 400 m east at 900 m, 100 m/s
        self.assertAlmostEqual(body["end"]["offsetFromObserved"]["horizontalM"], 320.0, delta=1.0)
        self.assertEqual((body["end"]["offsetFromObserved"]["aboveM"], body["end"]["offsetFromObserved"]["groundSpeedMps"]),
                         (-10.0, -5.0))
        self.assertEqual(body["word"]["status"], "inside")
        self.assertIsNone(body["word"]["heading"])
        self.assertEqual(body["limits"], {"cycles": 8, "bound": {"bank_rate": 2, "bank_cap": 0}})

    def test_a_heading_word_carries_its_band_on_the_flown_rows_a_dynamics_failures_too(self):
        segment = segment_of(reading(), HEADING, 3, LEAD)            # said at step 3, its target 60°, stopped at 8
        cycles = 12
        states = torch.zeros(1, cycles + 1, 7, dtype=torch.float64)
        states[0, :, LAT] = 35.0
        states[0, :, LON] = -78.0 + torch.arange(cycles + 1, dtype=torch.float64) * 90.0 / (111320.0 * 0.8191520)
        states[0, :, ALT], states[0, :, SPEED_STATE], states[0, :, MASS] = 890.0, 95.0, 60000.0
        run = replace(flown([float(c) for c in range(cycles)], cycles - 1), states=states)
        judged = {"heading": [{"row": 0, "rows": 4, "inside": 4}], "capture_turn": None, "intercepting_off_word_cycles": 0,
                  "corridor": {"cleared": False, "entered": False, "rows": 0, "inside": 0}, "vertical": [], "speed": []}
        result = FlownSegment(segment=segment, reading=segment_reading(reading(), segment), signals=None, flown=run,
                              verdict=Verdict("dynamics_failure", cycles, None, {"cycles": {"cycles": cycles}}, judged,
                                              flown_rows=cycles + 1),
                              reached_end=False, fly_s=0.1, judge_s=0.01)
        context = FlightContext(signals=signals10(), series=None, reading=reading(), geometry=geometry(),
                                crossing_heights=(15.0,), group="own dynamics", approach_ias_mps=70.0,
                                observed_track_deg=np.full(10, 90.0), observed_distance_m=200.0 * np.arange(10))
        # the flown track as its judge read it: seven steps, all on the word's 60°
        admitted = SimpleNamespace(smoothed=SimpleNamespace(track_deg=np.full(7, 60.0)))
        with mock.patch.object(payload_module, "read_flown", lambda *args, **kwargs: admitted):
            body = segment_payload(result, context, SimpleNamespace(spec=SPEC, speed_mps=WORDS.speed_mps))
        band = body["word"]["heading"]
        self.assertEqual((band["firstRow"], band["stopRow"], band["inside"]), (5, 9, [1, 1, 1, 1]))
        self.assertEqual(band["targetOnTrackDeg"], 60.0)
        self.assertEqual(len(body["judgedTrackDeg"]), 7)
        self.assertEqual((body["word"]["status"], body["end"]["reason"]), ("inside", "dynamics_failure"))
        # the failed state is left out of the track, and the judged steps are among its points
        self.assertEqual(len(body["track"]["tS"]), cycles)

    def test_a_heading_band_the_stop_cut_short_is_refused_and_one_it_did_not_is_not(self):
        def cut(clock_s: list[float], lead: int, band_rows: int, judged_rows: int, outcome: str = "timeout"):
            """Fly the heading word said at step 3 (the next one at the segment's step 3) to its stop as `fly_until`
            flies it, on ``clock_s``; its judge's band (``band_rows`` from the lead) and the track it read
            (``judged_rows``: one row a flown step through the last state)."""
            segment = segment_of(reading(), HEADING, 3, lead)
            executor = FakeExecutor(cycles=len(clock_s))
            reached = fly_until(executor, FakeSentences(), FakeClock(clock_s), SPEC.step_s, segment.stop_row - 3, NEVER)
            judged = {"heading": [{"row": 0, "rows": band_rows, "inside": band_rows}], "capture_turn": None,
                      "intercepting_off_word_cycles": 0,
                      "corridor": {"cleared": False, "entered": False, "rows": 0, "inside": 0}, "vertical": [], "speed": []}
            result = FlownSegment(segment=segment, reading=segment_reading(reading(), segment), signals=None,
                                  flown=flown(executor.heard, executor.count - 1),
                                  verdict=Verdict(outcome, executor.count, None, {}, judged, flown_rows=executor.count + 1),
                                  reached_end=reached, fly_s=0.0, judge_s=0.0)
            spec = SimpleNamespace(**{**vars(SPEC), "heading_lead_s": lead * SPEC.step_s})
            return reached, executor.count, band_cut_by_stop(result, judged_rows, spec)

        # the time clock: the next word heard at cycle 6 (flown step 3), the stop at cycle 10; the band ends at step 5
        # of a 6-step track
        self.assertEqual(cut([float(c) for c in range(20)], 2, 3, 6), (True, 10, False))
        # the track clock at its cap, a step a cycle: heard at cycle 4 (step 2), stopped at cycle 6; the band ends at
        # step 4, the track's last — exactly where it should
        self.assertEqual(cut([2.0 * c for c in range(20)], 2, 2, 4), (True, 6, False))
        # a clock with no cap (the distance clock) jumps past the next word to the stop within one step: the word is
        # never heard, and the band runs to the track's end, short of where it should end
        jump = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 11.0, 12.0, 13.0, 14.0]
        self.assertEqual(cut(jump, 2, 2, 4), (True, 6, True))
        # ... unless an event ended the flight there (the executor flies on past an uncaptured crossing): the judge
        # judged it to the event
        self.assertEqual(cut(jump, 2, 2, 4, "crossed_without_capture"), (True, 6, False))
        # a three-step lead on the track clock: heard at cycle 4 (step 2), its lead ends at step 5, the track at 4
        self.assertEqual(cut([2.0 * c for c in range(20)], 3, 1, 4), (True, 6, True))

    def test_an_event_before_the_stop_is_the_end_and_has_no_offset(self):
        body = self.answer(reached=False, outcome="ground_contact")
        self.assertEqual((body["end"]["reason"], body["end"]["offsetFromObserved"]), ("ground_contact", None))
        # out of cycles before the stop: the time limit, not the segment's end
        self.assertEqual(self.answer(reached=False)["end"]["reason"], "timeout")


class BackendTest(unittest.TestCase):
    def write_set(self, root: Path, kind: str = "vocabulary-readback", reading_rule: str | None = None, **sample) -> None:
        """A set's index and sample as `instruction_training_export` writes them (the fields the checks read)."""
        from ts_transformer.instructions.training_files import INDEX_SCHEMA, SAMPLE_SCHEMA
        from ts_transformer.instructions.spec import READING_RULE
        training = root / "KXXX" / "training"
        (training / "a_set").mkdir(parents=True)
        (training / "index.json").write_text(json.dumps({"schema": INDEX_SCHEMA, "airport": "KXXX", "sets": [
            {"id": "a_set", "kind": kind, "readingRule": reading_rule or READING_RULE, "file": "a_set/sample.json"}]}))
        (training / "a_set" / "sample.json").write_text(json.dumps({
            "schema": SAMPLE_SCHEMA, "setId": "a_set", "airport": "KXXX", "vocabulary": {"readingRule": READING_RULE},
            "producedBy": {"artefact": "4dTrajectory/outputs/POOLED/instruction_language/an_artefact"},
            "cohort": {"split": "val"}, "flights": [{"flightKey": "F_23R_abc_T", "datasetId": "KXXX:F_23R_abc_T"}],
            **sample}))

    def test_a_set_names_its_artefact_and_split(self):
        with TemporaryDirectory() as tmp:
            self.write_set(Path(tmp))
            backend = AutopilotSegmentBackend(airports_root=Path(tmp), executor_root=Path(tmp) / "none")
            artefact, split, sample = backend.training_set("KXXX", "a_set")
            self.assertEqual((artefact.parts[-2:], split), (("instruction_language", "an_artefact"), "val"))
            with self.assertRaisesRegex(NotListed, "lists no set another_set"):
                backend.training_set("KXXX", "another_set")
            with self.assertRaisesRegex(NotListed, "KYYY has no Training export"):
                backend.training_set("KYYY", "a_set")
            # the airport is a path segment: only an airport code is one
            with self.assertRaisesRegex(RequestRefused, "not an airport code"):
                backend.training_set("../KXXX", "a_set")

    def test_only_a_sample_of_this_vocabulary_drawn_from_the_exports_split_is_flown(self):
        for change, refusal in [({"schema": "aeroviz-training-sample-v6"}, "is a aeroviz-training-sample-v6 file"),
                                ({"reading_rule": "instruction-v2"}, "set of instruction-v2, not"),
                                ({"cohort": {"split": "test"}}, "split test, read under instruction-v3; expected"),
                                ({"vocabulary": {"readingRule": "instruction-v2"}}, "read under instruction-v2; expected")]:
            with TemporaryDirectory() as tmp:
                self.write_set(Path(tmp), **change)
                backend = AutopilotSegmentBackend(airports_root=Path(tmp), executor_root=Path(tmp) / "none")
                with self.assertRaisesRegex(ValueError, refusal):
                    backend.training_set("KXXX", "a_set")

    def test_only_a_readback_sets_flights_are_flown(self):
        with TemporaryDirectory() as tmp:
            self.write_set(Path(tmp), kind="prior-generated")
            backend = AutopilotSegmentBackend(airports_root=Path(tmp), executor_root=Path(tmp) / "none")
            with self.assertRaisesRegex(ValueError, "prior-generated"):
                backend.training_set("KXXX", "a_set")

    def test_without_one_executor_spec_for_the_artefact_nothing_is_flown(self):
        with TemporaryDirectory() as tmp:
            (Path(tmp) / "executor" / "old").mkdir(parents=True)
            (Path(tmp) / "executor" / "old" / "spec.json").write_text(json.dumps({"schema": "ts-executor-spec-v1"}))
            backend = AutopilotSegmentBackend(airports_root=Path(tmp), executor_root=Path(tmp) / "executor")
            with self.assertRaisesRegex(ValueError, r"0 executor specs .* Refused: \[.old: .*is not a ts-executor-spec"):
                backend.executor_for(Path(tmp) / "artefact")

    def test_the_spec_is_chosen_again_when_a_spec_is_added_moved_or_rewritten(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "executor"
            (root / "one").mkdir(parents=True)
            (root / "one" / "spec.json").write_text("{}")
            backend = AutopilotSegmentBackend(airports_root=Path(tmp), executor_root=root)
            opened = []

            def open_executor(directory, artefact):
                opened.append(directory.name)
                return "params", {"sha256": directory.name}, "words"

            with mock.patch("ts_transformer.autopilot.replay.open_executor", open_executor):
                self.assertEqual(backend.executor_for(Path(tmp) / "artefact")[0].name, "one")
                backend.executor_for(Path(tmp) / "artefact")
                self.assertEqual(opened, ["one"])                     # kept while nothing changed
                (root / "one").rename(root / "two")
                self.assertEqual(backend.executor_for(Path(tmp) / "artefact")[0].name, "two")
                self.assertEqual(opened, ["one", "two"])
                # rewritten in place: a new time of writing
                stat = (root / "two" / "spec.json").stat()
                os.utime(root / "two" / "spec.json", ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
                backend.executor_for(Path(tmp) / "artefact")
                self.assertEqual(opened, ["one", "two", "two"])
                (root / "three").mkdir()
                (root / "three" / "spec.json").write_text("{}")
                with self.assertRaisesRegex(ValueError, r"2 executor specs .* \(\['three', 'two'\]\); one is needed"):
                    backend.executor_for(Path(tmp) / "artefact")

    def test_a_rebuilt_flight_is_kept_for_the_next_requests_and_the_oldest_let_go(self):
        backend = AutopilotSegmentBackend(airports_root=Path("/nonexistent"), executor_root=Path("/nonexistent"))
        with mock.patch("aeroviz_backend.autopilot_segment.backend.open_flight", lambda artefact, split, key, words: key):
            self.assertEqual(backend.flight(Path("a"), "val", "K:0", None), ("K:0", False))
            self.assertEqual(backend.flight(Path("a"), "val", "K:0", None), ("K:0", True))
            for index in range(1, FLIGHT_CACHE_SIZE + 1):
                backend.flight(Path("a"), "val", f"K:{index}", None)
            self.assertEqual(backend.flight(Path("a"), "val", "K:0", None), ("K:0", False))

    def test_a_request_names_a_column_and_an_integer_step(self):
        backend = AutopilotSegmentBackend(airports_root=Path("/nonexistent"), executor_root=Path("/nonexistent"))
        request = {"clientId": "page", "seq": 1, "airport": "KXXX", "setId": "a_set", "flightKey": "F", "column": "heading", "row": 3}
        with self.assertRaisesRegex(RequestRefused, "none of"):
            backend.fly({**request, "column": "track"})
        with self.assertRaisesRegex(RequestRefused, "integer step"):
            backend.fly({**request, "row": 3.0})
        with self.assertRaisesRegex(RequestRefused, "no 'row'"):
            backend.fly({key: value for key, value in request.items() if key != "row"})
        with self.assertRaisesRegex(RequestRefused, "no 'clientId'"):
            backend.fly({key: value for key, value in request.items() if key != "clientId"})
        for seq in (-1, 1.0, True):
            with self.assertRaisesRegex(RequestRefused, "seq must be the page's request number"):
                backend.fly({**request, "seq": seq})

    def test_a_page_s_later_request_supersedes_its_earlier_ones_whatever_order_they_arrive_in(self):
        backend = AutopilotSegmentBackend(airports_root=Path("/nonexistent"), executor_root=Path("/nonexistent"))
        request = {"clientId": "page", "seq": 2, "airport": "KXXX", "setId": "a_set", "flightKey": "F", "column": "heading",
                   "row": 3}
        with self.assertRaises(NotListed):                     # it goes on (and finds no Training export here)
            backend.fly(request)
        # the page's request 1 arrives after its request 2: refused at once
        with self.assertRaisesRegex(Superseded, "request 2 came in before its request 1"):
            backend.fly({**request, "seq": 1})
        # another page's numbers are its own
        with self.assertRaises(NotListed):
            backend.fly({**request, "clientId": "another page", "seq": 1})

    def test_a_later_request_supersedes_one_still_waiting(self):
        backend = AutopilotSegmentBackend(airports_root=Path("/nonexistent"), executor_root=Path("/nonexistent"))
        request = {"clientId": "page", "seq": 1, "airport": "KXXX", "setId": "a_set", "flightKey": "F", "column": "heading",
                   "row": 3}

        class Flying:
            """Another segment flying: while this request waits for it, the page sends ``newer_from``'s request 2."""

            def __init__(self, newer_from: str) -> None:
                self.newer_from = newer_from

            def __enter__(self):
                backend._claim(self.newer_from, 2)

            def __exit__(self, *exc):
                return False

        backend._lock = Flying("page")
        with self.assertRaisesRegex(Superseded, "while this one waited"):
            backend.fly(request)
        # another page's request does not
        backend._latest.clear()
        backend._lock = Flying("another page")
        with self.assertRaises(NotListed):
            backend.fly(request)

    def test_the_flight_is_asked_each_cycle_whether_a_later_request_came_in(self):
        backend = AutopilotSegmentBackend(airports_root=Path("/nonexistent"), executor_root=Path("/nonexistent"))
        request = {"clientId": "page", "seq": 1, "airport": "KXXX", "setId": "a_set", "flightKey": "F", "column": "heading",
                   "row": 3}
        asked: list[bool] = []

        def flying(context, params, words, column, row, superseded):
            asked.append(superseded())
            backend._claim("another page", 7)                  # another page's request: not this page's
            asked.append(superseded())
            backend._claim("page", 2)                          # this page's next click
            asked.append(superseded())
            raise Superseded("stopped")

        sample = {"flights": [{"flightKey": "F", "datasetId": "KXXX:F"}]}
        with mock.patch.object(backend, "training_set", lambda airport, set_id: (Path("a"), "val", sample)), \
                mock.patch.object(backend, "executor_for", lambda artefact: (Path("e"), None, {}, None)), \
                mock.patch.object(backend, "flight", lambda artefact, split, key, words: (None, False)), \
                mock.patch("aeroviz_backend.autopilot_segment.backend.fly_segment", flying):
            with self.assertRaises(Superseded):
                backend.fly(request)
        self.assertEqual(asked, [False, False, True])


class FakeAutopilot:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls, self.error = [], error

    def fly(self, payload):
        self.calls.append(payload)
        if self.error is not None:
            raise self.error
        return {"ok": True, "end": {"reason": SEGMENT_END}}


class EndpointTest(unittest.TestCase):
    def app(self, autopilot) -> AeroVizBackendApp:
        return AeroVizBackendApp(simulation_backend=object(), optimization_backend=object(),
                                 dynamics_comparison_backend=object(), observed_trajectory_backend=object(),
                                 autopilot_segment_backend=autopilot)

    def test_the_endpoint_delegates_the_request(self):
        autopilot = FakeAutopilot()
        status, payload, event = self.app(autopilot).handle_post("/autopilot/segment", {"row": 3})
        self.assertEqual((status, payload["end"]["reason"], event), (200, SEGMENT_END, None))
        self.assertEqual(autopilot.calls, [{"row": 3}])

    def test_a_bad_request_is_a_400_a_set_or_flight_not_listed_a_404_and_anything_else_a_500_with_its_reason(self):
        post = lambda error: self.app(FakeAutopilot(error)).handle_post("/autopilot/segment", {})[:2]  # noqa: E731
        self.assertEqual(post(RequestRefused("no 'row'")), (400, {"ok": False, "error": "no 'row'"}))
        self.assertEqual(post(NotListed("no set")), (404, {"ok": False, "error": "no set"}))
        self.assertEqual(post(NotFlyable("no aircraft dynamics")), (422, {"ok": False, "error": "no aircraft dynamics"}))
        self.assertEqual(post(Superseded("a newer request")), (409, {"ok": False, "error": "a newer request"}))
        # a listed flight the backend could not fly — a spec refused, a re-read that differs, a file gone — is its fault
        self.assertEqual(post(ValueError("2 executor specs")), (500, {"ok": False, "error": "ValueError: 2 executor specs"}))
        self.assertEqual(post(FileNotFoundError("npz")), (500, {"ok": False, "error": "FileNotFoundError: npz"}))


def ts_constant(path: Path, name: str):
    """A `export const NAME = …` of a frontend file, as JSON."""
    match = re.search(rf"export const {name} = (\[[^\]]*\]|\"[^\"]*\")", path.read_text(encoding="utf-8"))
    if match is None:
        raise AssertionError(f"{path.name} has no {name}")
    return json.loads(re.sub(r",\s*\]", "]", match.group(1)))


class MirrorTest(unittest.TestCase):
    """The frontend's copies of this module's names (`trainingAutopilot.ts`), of the judge's outcomes and the prior
    readout's rules (`trainingOverlays.ts`) and of the strata (`trainingSample.ts`): one name on both sides, or the reader
    refuses every file that carries it."""

    def test_the_frontend_reader_mirrors_the_backends_names(self):
        from ts_transformer.autopilot.judge import OUTCOMES
        data = REPO_ROOT / "aeroviz-4d" / "src" / "data"
        self.assertEqual(ts_constant(data / "trainingAutopilot.ts", "TRAINING_AUTOPILOT_SCHEMA"), SCHEMA)
        self.assertEqual(ts_constant(data / "trainingAutopilot.ts", "TRAINING_AUTOPILOT_STATUSES"), list(STATUSES))
        self.assertEqual(ts_constant(data / "trainingAutopilot.ts", "TRAINING_AUTOPILOT_SEGMENT_END"), SEGMENT_END)
        self.assertEqual(ts_constant(data / "trainingOverlays.ts", "TRAINING_EXECUTOR_OUTCOMES"), list(OUTCOMES))

    def test_the_frontend_reader_mirrors_the_training_files_names_no_other_test_pins(self):
        from ts_transformer.instructions.readout import STRATA
        from ts_transformer.prior.readout import RULES
        data = REPO_ROOT / "aeroviz-4d" / "src" / "data"
        self.assertEqual(ts_constant(data / "trainingSample.ts", "TRAINING_STRATA"), list(STRATA))
        self.assertEqual(ts_constant(data / "trainingOverlays.ts", "TRAINING_PRIOR_RULES"), list(RULES))


if __name__ == "__main__":
    unittest.main()
