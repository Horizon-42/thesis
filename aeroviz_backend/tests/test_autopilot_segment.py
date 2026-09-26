"""The live executor segment (`aeroviz_backend.autopilot_segment`): which segment a selected word is, what the executor
is told, where the flight is stopped, which of the judge's results is the selected word's, the answer written, the
service's lookups and caches, the endpoint's statuses, and the frontend's copies of this module's names. Synthetic
readings and verdicts only — nothing here opens an artefact, a spec or the frontend's data."""

import json
import math
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
    SCHEMA, SEGMENT_END, band_cut_by_stop, heading_facts, heading_payload, segment_payload, track_payload,
)
from aeroviz_backend.autopilot_segment.segment import (
    ModelSentence, judged_reading, model_reading, model_segment, model_sentence, model_steps_max, segment_of,
    segment_reading, segment_signals, sentence_instructions, told_words,
)
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
            result = FlownSegment(segment=segment, reading=segment_reading(reading(), segment), signals=None, model=None,
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
        self.assertEqual(selected_heading(judged, 0), {"row": 0, "rows": 5, "inside": 2})


def geometry() -> AirportGeometry:
    return AirportGeometry.from_dict({
        "code": "KXXX", "reference": {"lat": 35.0, "lon": -78.0, "elevation_m": 100.0},
        "candidates": [{"ident": "09", "threshold_e_m": 0.0, "threshold_n_m": 0.0, "course_deg": 90.0,
                        "elevation_m": 100.0, "length_m": 3000.0}],
        "runway_ends": [{"ident": "09", "threshold_e_m": 0.0, "threshold_n_m": 0.0, "course_deg": 90.0}],
    })


def geometry2() -> AirportGeometry:
    """`geometry` with a second runway, 27, for a model that changes its runway."""
    return AirportGeometry.from_dict({
        "code": "KXXX", "reference": {"lat": 35.0, "lon": -78.0, "elevation_m": 100.0},
        "candidates": [{"ident": ident, "threshold_e_m": east, "threshold_n_m": 0.0, "course_deg": course,
                        "elevation_m": 100.0, "length_m": 3000.0}
                       for ident, east, course in (("09", 0.0, 90.0), ("27", 3000.0, 270.0))],
        "runway_ends": [{"ident": ident, "threshold_e_m": east, "threshold_n_m": 0.0, "course_deg": course}
                        for ident, east, course in (("09", 0.0, 90.0), ("27", 3000.0, 270.0))],
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
        result = FlownSegment(segment=segment, reading=segment_reading(reading(), segment), signals=None, model=None, flown=run,
                              verdict=Verdict(outcome, cycles, None, {}, None, flown_rows=cycles + 1), reached_end=True,
                              fly_s=0.1, judge_s=0.01)
        # the observed smoothed track reads 450° (one turn up) and has flown 1200 m by step 3
        context = FlightContext(signals=None, series=None, reading=reading(), geometry=geometry(), crossing_heights=(15.0,),
                                group="own dynamics", approach_ias_mps=70.0,
                                observed_track_deg=np.full(10, 450.0), observed_distance_m=400.0 * np.arange(10),
                                hae_minus_msl_m=-32.0)
        return track_payload(result, context, 2.0)

    def test_the_flown_segment_reads_on_the_flights_own_clock_distance_and_heading_branch(self):
        track, shift = self.payload("timeout")
        self.assertEqual(track["tS"], [6.0, 7.0, 8.0, 9.0, 10.0])
        self.assertEqual(shift, 360.0)
        # drawn at the observed track's height: MSL plus the flight's runway's HAE − MSL offset
        self.assertEqual((track["altitudeM"], track["altitudeHaeM"]), ([900.0] * 5, [868.0] * 5))
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
        result = FlownSegment(segment=segment, reading=segment_reading(reading(), segment), signals=None, model=None, flown=run,
                              verdict=Verdict(outcome, cycles, None, limits, judged, flown_rows=cycles + 1),
                              reached_end=reached, fly_s=0.1, judge_s=0.01)
        context = FlightContext(signals=signals10(), series=None, reading=reading(), geometry=geometry(),
                                crossing_heights=(15.0,), group="own dynamics", approach_ias_mps=70.0,
                                observed_track_deg=np.full(10, 90.0), observed_distance_m=200.0 * np.arange(10),
                                hae_minus_msl_m=-32.0)
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
        result = FlownSegment(segment=segment, reading=segment_reading(reading(), segment), signals=None, model=None, flown=run,
                              verdict=Verdict("dynamics_failure", cycles, None, {"cycles": {"cycles": cycles}}, judged,
                                              flown_rows=cycles + 1),
                              reached_end=False, fly_s=0.1, judge_s=0.01)
        context = FlightContext(signals=signals10(), series=None, reading=reading(), geometry=geometry(),
                                crossing_heights=(15.0,), group="own dynamics", approach_ias_mps=70.0,
                                observed_track_deg=np.full(10, 90.0), observed_distance_m=200.0 * np.arange(10),
                                hae_minus_msl_m=-32.0)
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
            result = FlownSegment(segment=segment, reading=segment_reading(reading(), segment), signals=None, model=None,
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
        request = {"clientId": "page", "seq": 1, "airport": "KXXX", "setId": "a_set", "flightKey": "F", "column": "heading", "row": 3,
                   "sentence": None}
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
                   "row": 3, "sentence": None}
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
                   "row": 3, "sentence": None}

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
                   "row": 3, "sentence": None}
        asked: list[bool] = []

        def flying(context, params, words, column, row, superseded, model):
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

    def test_a_models_sentence_is_read_under_its_time_limit_flown_and_answered_on_the_time_clock(self):
        backend = AutopilotSegmentBackend(airports_root=Path("/nonexistent"), executor_root=Path("/nonexistent"))
        params = SimpleNamespace(word_clock="track", cycle_s=1.0, timeout_factor=1.5)
        words = SimpleNamespace(**vars(MODEL_WORDS), spec=SimpleNamespace(sha256="v" * 64))
        context = SimpleNamespace(geometry=geometry2(), reading=SimpleNamespace(words=np.zeros((20, 6))), group="own dynamics")
        record = {"sha256": "s" * 64, "source": {"executor_source_sha256": "c" * 64}}
        flown_with = []

        def flying(context_, params_, words_, column, row, superseded, model):
            flown_with.append(model)
            return SimpleNamespace(fly_s=0.0, judge_s=0.0, flown=SimpleNamespace(commands=torch.zeros(1, 3, 3)))

        sample = {"flights": [{"flightKey": "F", "datasetId": "KXXX:F"}]}
        answers = []
        with mock.patch.object(backend, "training_set", lambda airport, set_id: (Path("a"), "val", sample)), \
                mock.patch.object(backend, "executor_for", lambda artefact: (Path("e"), params, record, words)), \
                mock.patch.object(backend, "flight", lambda artefact, split, key, words_: (context, False)), \
                mock.patch("aeroviz_backend.autopilot_segment.backend.fly_segment", flying), \
                mock.patch("aeroviz_backend.autopilot_segment.backend.segment_payload", lambda *args: {}):
            for seq, sentence in enumerate((model_record(), None), start=1):
                answers.append(backend.fly({"clientId": "page", "seq": seq, "airport": "KXXX", "setId": "a_set",
                                            "flightKey": "F", "column": "heading", "row": 10, "sentence": sentence}))
            # over the steps its time limit lets it say (12 observed steps left from step 8 × 1.5: 19): refused by name
            with self.assertRaisesRegex(RequestRefused, "lets it say 19"):
                backend.fly({"clientId": "page", "seq": 3, "airport": "KXXX", "setId": "a_set", "flightKey": "F",
                             "column": "heading", "row": 10, "sentence": model_record(rows=8 + 20)})
        model, truth = flown_with
        self.assertEqual((model.overlay_id, model.sample, model.first_row, model.rows), ("generation_x", 2, 8, 18))
        self.assertIsNone(truth)
        self.assertEqual([answer["executor"]["wordClock"] for answer in answers], ["time", "track"])


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


# ---- a MODEL's word: its own sentence, flown again from its first step
def model_record(**changes) -> dict:
    """A model's sentence of a 20-step flight, from step 8 (every column said there): a heading word at 10, the clearance
    at 12, another heading word at 14, a runway change at 15, the speed left to the pilot at 16; it ends at step 18."""
    events = [{"row": 8, "column": column, "value": value} for column, value in enumerate([0, 0, 10, 7, 0, 4])]
    events += [{"row": 10, "column": HEADING, "value": 12}, {"row": 12, "column": APPROACH, "value": 1},
               {"row": 14, "column": HEADING, "value": 14}, {"row": 15, "column": RUNWAY, "value": 1},
               {"row": 16, "column": SPEED, "value": 9}]
    return {"overlayId": "generation_x", "sample": 2, "firstRow": 8, "rows": 18, "events": events, **changes}


MODEL_WORDS = SimpleNamespace(
    class_counts=lambda: {"approach": 3, "heading": 72, "altitude": 182, "angle": 6, "speed": 10},
    heading_deg=lambda value: 5.0 * value, speed_unspecified=9)


class ModelSegmentTest(unittest.TestCase):
    def sentence(self, **changes) -> ModelSentence:
        return model_sentence(model_record(**changes), MODEL_WORDS, runways=2, observed_rows=20, timeout_factor=1.5)

    def test_a_models_sentence_is_read_from_its_first_step_at_the_flights_own_steps(self):
        sentence = self.sentence()
        self.assertEqual((sentence.overlay_id, sentence.sample, sentence.first_row, sentence.rows), ("generation_x", 2, 8, 18))
        self.assertEqual(sentence.grid.shape, (10, 6))
        self.assertEqual(list(sentence.grid[0]), [0, 0, 10, 7, 0, 4])
        self.assertEqual(sentence.grid[2, HEADING], 12)          # step 10
        self.assertEqual(sentence.grid[7, RUNWAY], 1)            # step 15

    def test_a_sentence_that_is_not_one_is_refused_by_name(self):
        record = model_record()
        cases = [
            (dict(firstRow=9, events=[{**event, "row": 9} for event in record["events"][:6]]),
             "starts at step 9, not the prior's first predicted step 8"),
            (dict(rows=8), "not after its first step"),
            (dict(events=[*record["events"][1:], record["events"][0]]), "not in (step, column) order"),
            (dict(events=record["events"][1:]), "does not say every column"),
            (dict(events=[*record["events"][:6], {"row": 10, "column": HEADING, "value": 72}]), "is not a word of"),
            (dict(events=[*record["events"][:6], {"row": 18, "column": HEADING, "value": 3}]), "is not a word of"),
            (dict(events=[*record["events"][:6], {"row": 9, "column": RUNWAY, "value": 2}]), "is not a word of"),
            (dict(events=[*record["events"][:6], {"row": 9, "column": HEADING}]), "is not {row, column, value}"),
            (dict(sample=1.5), "must be a whole number"),
            (dict(overlayId=7), "overlayId must be a string"),
            # 12 observed steps left from step 8 × 1.5: 1 + 18 steps said at most
            (dict(rows=8 + 20), "its flight's time limit lets it say 19"),
        ]
        for change, message in cases:
            with self.subTest(change=list(change)), self.assertRaisesRegex(RequestRefused, re.escape(message)):
                self.sentence(**change)
        self.assertEqual(self.sentence(rows=8 + 19).rows, 27)
        with self.assertRaisesRegex(RequestRefused, "leave none to fly after the sentence's first"):
            model_sentence(model_record(rows=9, events=record["events"][:6]), MODEL_WORDS, runways=2, observed_rows=9,
                           timeout_factor=1.5)
        with self.assertRaisesRegex(RequestRefused, "has no"):
            model_sentence({"overlayId": "x"}, MODEL_WORDS, runways=2, observed_rows=20, timeout_factor=1.5)

    def test_a_models_steps_are_bounded_by_its_free_generations_own_cap(self):
        from ts_transformer.prior.generate import rows_for
        from ts_transformer.prior.scene import N_LOOK

        params = SimpleNamespace(timeout_factor=1.5)
        for observed_rows, first in ((20, 8), (121, 8), (57, 8)):
            context = FlightContext(signals=None, series=None, reading=SimpleNamespace(words=np.zeros((observed_rows, 6))),
                                    geometry=None, crossing_heights=(), group="own dynamics", approach_ias_mps=70.0,
                                    observed_track_deg=None, observed_distance_m=None, hae_minus_msl_m=-32.0)
            limit = fly_module.model_time_limit_s(context, first, params, 2.0)
            self.assertEqual(model_steps_max(observed_rows, first, 1.5), rows_for(limit, 2.0) - N_LOOK)

    def test_a_models_segment_is_its_sentence_from_its_first_step_to_the_words_stop(self):
        segment = model_segment(self.sentence(), HEADING, 10, LEAD, MODEL_WORDS)
        # said at 10, the next heading word at 14, stopped a lead later; the executor starts at the sentence's first step
        self.assertEqual((segment.row, segment.end_row, segment.stop_row, segment.start_row, segment.to_landing),
                         (10, 14, 16, 8, False))
        self.assertEqual(segment.word_step, 2)
        self.assertEqual(len(segment.grid), 16 - 8 + 1)          # to the stop, and one silent step there
        self.assertTrue((segment.grid[-1] == U).all())
        self.assertEqual(segment.selected().value, 12)
        # the words told: the model's, from its first step, at the flight's steps
        told = told_words(segment)
        self.assertEqual(told[0], {"row": 8, "column": RUNWAY, "value": 0})
        self.assertEqual([(w["row"], w["column"]) for w in told[6:]],
                         [(10, HEADING), (12, APPROACH), (14, HEADING), (15, RUNWAY)])

    def test_a_models_columns_last_word_is_flown_to_its_outcome_and_a_word_it_did_not_say_is_refused(self):
        last = model_segment(self.sentence(), SPEED, 16, LEAD, MODEL_WORDS)
        self.assertEqual((last.stop_row, last.to_landing, len(last.grid)), (18, True, 10))
        # said at the sentence's last step: the step the flight was still flying in when it ended — flown to its outcome
        record = model_record()
        final = model_segment(self.sentence(events=[*record["events"], {"row": 17, "column": ALTITUDE, "value": 5}]),
                              ALTITUDE, 17, LEAD, MODEL_WORDS)
        self.assertEqual((final.word_step, final.stop_row, final.to_landing, len(final.grid)), (9, 18, True, 10))
        with self.assertRaisesRegex(RequestRefused, "says no heading word at step 11"):
            model_segment(self.sentence(), HEADING, 11, LEAD, MODEL_WORDS)
        with self.assertRaisesRegex(RequestRefused, "not a step of the model's sentence"):
            model_segment(self.sentence(), HEADING, 7, LEAD, MODEL_WORDS)

    def test_the_judge_reads_a_models_words_as_it_reads_the_labellers(self):
        words = sentence_instructions(self.sentence().grid, MODEL_WORDS)
        clear = [w for w in words if w.column == APPROACH and w.value == 1]
        self.assertEqual([(w.row, w.kind) for w in clear], [(4, "clear")])
        self.assertEqual({w.info["target_deg"] for w in words if w.column == HEADING}, {50.0, 60.0, 70.0})
        self.assertTrue(all(w.kind == "said" for w in words if w.column != APPROACH or w.value != 1))

    def test_a_models_reading_points_at_the_runway_its_sentence_points_at_last(self):
        """The runway its free generation's outcome was read on, so a segment ends where the sample did — even one that
        stops before the change (27 from step 15)."""
        sentence, signals = self.sentence(), signals10()
        before = model_segment(sentence, HEADING, 10, LEAD, MODEL_WORDS)       # stops at 16: after the change
        early = model_segment(sentence, HEADING, 8, LEAD, MODEL_WORDS)         # stops at 12: before it
        after = model_segment(sentence, APPROACH, 12, LEAD, MODEL_WORDS)       # to the outcome
        self.assertEqual(sentence.last_runway, 1)
        self.assertEqual([model_reading(signals, segment, sentence, MODEL_WORDS).runway_index for segment in (before, early, after)],
                         [1, 1, 1])
        reading_ = model_reading(signals, after, sentence, MODEL_WORDS)
        self.assertEqual((reading_.join_row, reading_.unspecified_row), (4, 8))

    def test_a_models_time_limit_is_its_free_generations(self):
        from ts_transformer.experiments.prior_free_generation import limits_s
        from ts_transformer.prior.scene import N_LOOK

        params = SimpleNamespace(timeout_factor=1.5)
        truth = SimpleNamespace(words=np.zeros((60, 6)))
        batch = SimpleNamespace(readings=[truth])
        context = FlightContext(signals=None, series=None, reading=truth, geometry=None, crossing_heights=(), group="own dynamics",
                                approach_ias_mps=70.0, observed_track_deg=None, observed_distance_m=None,
                                hae_minus_msl_m=-32.0)
        self.assertEqual(fly_module.model_time_limit_s(context, N_LOOK, params, 2.0), limits_s(batch, params, 2.0)[0])

    def test_a_model_word_is_its_own_step_s_result_in_the_judges_verdict(self):
        segment = model_segment(self.sentence(), HEADING, 10, LEAD, MODEL_WORDS)
        judged = verdict(heading=[{"row": 0, "rows": 2, "inside": 2}, {"row": 2, "rows": 4, "inside": 3},
                                  {"row": 6, "rows": 0, "inside": 0}])
        body = word_verdict(judged, segment, SPEC, WORDS, HELD)
        self.assertEqual(body["status"], "outside")
        self.assertEqual(body["checks"][0]["inside"], 3)
        self.assertEqual(selected_heading(judged.words, 2), {"row": 2, "rows": 4, "inside": 3})

    def test_an_angle_word_is_judged_on_the_tube_it_anchors_from_its_own_step(self):
        """The judge counts a tube from its altitude word's step: for a model's angle word said later, the altitude word in
        force is said again at the angle word's step in the reading the judge reads (never in what the executor flies),
        and only the tubes from that step count."""
        record = model_record()
        events = sorted([*record["events"], {"row": 11, "column": ANGLE, "value": 2}],
                        key=lambda event: (event["row"], event["column"]))
        sentence = self.sentence(events=events)
        angle = model_segment(sentence, ANGLE, 11, LEAD, MODEL_WORDS)
        flown_reading = model_reading(signals10(), angle, sentence, MODEL_WORDS)
        judged = judged_reading(flown_reading, angle)
        added = [word for word in judged.instructions if word not in flown_reading.instructions]
        self.assertEqual([(w.column, w.value, w.row, w.kind) for w in added], [(ALTITUDE, 7, 3, "in force")])
        self.assertIs(judged.words, flown_reading.words)
        # the truth's, a model's word at its first step, and a model's altitude word: read as flown
        truth = segment_of(reading(), ANGLE, 7, LEAD)
        self.assertIs(judged_reading(segment_reading(reading(), truth), truth).instructions,
                      segment_reading(reading(), truth).instructions)
        first = model_segment(sentence, ANGLE, 8, LEAD, MODEL_WORDS)
        self.assertEqual(len(judged_reading(model_reading(signals10(), first, sentence, MODEL_WORDS), first).instructions),
                         len(first.instructions))
        # the tubes before the word's step are other angle words': the verdict reads the ones from it on
        tubes = [{"row": 0, "rows": 3, "inside": 1, "contained": False, "target_m": 1110.0},
                 {"row": 3, "rows": 4, "inside": 4, "contained": True, "target_m": 1110.0}]
        body = word_verdict(verdict(vertical=tubes), angle, SPEC, WORDS, None)
        self.assertEqual((body["status"], [check["rows"] for check in body["checks"]]), ("inside", [4]))

    def test_the_labellers_own_tube_check_judges_a_models_angle_word_from_its_step(self):
        """The labeller's `tube_checks` on a flight level at 1200 m for five steps, then down at 3°: the model said 600 m
        and level at its first step (8), 3° at 13. On what was flown, one tube from step 8 — its five level rows outside
        the 600 m level band count against the 3° word; on what its judge reads, the 3° word's own tube from step 13,
        every row inside."""
        from ts_transformer.instructions.labeller.vertical import tube_checks
        from ts_transformer.instructions.words import ANGLE_LEVEL, Words
        from ts_transformer.tests.support import instruction_spec

        spec = instruction_spec()
        words = Words(spec)
        events = [{"row": 8, "column": column, "value": value}
                  for column, value in enumerate([0, 0, 18, words.altitude_index(600.0), ANGLE_LEVEL, 10])]
        events.append({"row": 13, "column": ANGLE, "value": words.angle_index(3.0)})
        sentence = model_sentence(model_record(rows=28, events=events), words, runways=1, observed_rows=40, timeout_factor=1.5)
        segment = model_segment(sentence, ANGLE, 13, LEAD, words)            # to the outcome: 20 steps from step 8
        flown_reading = model_reading(signals10(), segment, sentence, words)
        distance = 200.0 * np.arange(20)
        altitude = np.where(np.arange(20) < 5, 1200.0, 1200.0 - (distance - distance[5]) * math.tan(math.radians(3.0)))

        def read(reading_: Reading) -> list[tuple]:
            return [(tube["row"], tube["rows"], tube["inside"], tube["contained"])
                    for tube in tube_checks(reading_.instructions, distance, altitude, spec, words)]

        self.assertEqual(read(flown_reading), [(0, 20, 15, False)])
        judged = judged_reading(flown_reading, segment)
        self.assertEqual(read(judged), [(0, 5, 0, False), (5, 15, 15, True)])
        body = word_verdict(verdict(vertical=tube_checks(judged.instructions, distance, altitude, spec, words)), segment, SPEC,
                            WORDS, None)
        self.assertEqual((body["status"], [(check["inside"], check["rows"]) for check in body["checks"]]), ("inside", [(15, 15)]))

    def test_a_models_second_clearance_is_not_judged_on_the_first_ones_capture(self):
        record = model_record()
        events = sorted([*record["events"], {"row": 13, "column": APPROACH, "value": 2},
                         {"row": 15, "column": APPROACH, "value": 1}], key=lambda event: (event["row"], event["column"]))
        sentence = self.sentence(events=events)
        again = model_segment(sentence, APPROACH, 15, LEAD, MODEL_WORDS)
        body = word_verdict(verdict(capture_turn={"progress_ok": True, "rate_ok": True},
                                    corridor={"cleared": True, "entered": True, "rows": 5, "inside": 5}),
                            again, SPEC, WORDS, None)
        self.assertEqual(body["status"], "not judged")
        self.assertRegex(body["reason"], "a clearance after an earlier one")
        first = word_verdict(verdict(capture_turn={"progress_ok": True, "rate_ok": True},
                                     corridor={"cleared": True, "entered": True, "rows": 5, "inside": 5}),
                             model_segment(sentence, APPROACH, 12, LEAD, MODEL_WORDS), SPEC, WORDS, None)
        self.assertEqual(first["status"], "inside")

    def test_a_models_heading_word_is_read_from_its_own_step(self):
        """Heard at cycle 4 (its step 2 of 1 s cycles, 2 s steps), the next heading word at cycle 12: the cycles it was
        left to intercept on its own count between; its band starts a lead after its step, in the flight's steps."""
        segment = model_segment(self.sentence(), HEADING, 10, LEAD, MODEL_WORDS)
        run = flown([float(c) for c in range(16)], 15)
        off = torch.zeros(1, 16, dtype=torch.bool)
        off[0, 3:14] = True
        judged = verdict(heading=[{"row": 0, "rows": 2, "inside": 2}, {"row": 2, "rows": 4, "inside": 4},
                                  {"row": 6, "rows": 0, "inside": 0}])
        result = FlownSegment(segment=segment, reading=model_reading(signals10(), segment, self.sentence(), MODEL_WORDS),
                              signals=None,
                              model=self.sentence(), flown=replace(run, modes={**run.modes, "intercepting_off_word": off}),
                              verdict=replace(judged, end_row=16), reached_end=True, fly_s=0.0, judge_s=0.0)
        admitted = SimpleNamespace(smoothed=SimpleNamespace(track_deg=np.full(9, 60.0)))
        self.assertEqual(heading_facts(result, admitted, SPEC), HeadingFacts(off_word_cycles=8, judged_rows=9))
        band, track = heading_payload(result, admitted, SPEC, 0.0)
        self.assertEqual((band["firstRow"], band["stopRow"], band["inside"]), (12, 16, [1, 1, 1, 1]))
        self.assertEqual(len(track), 9 - 2)                     # the judged track from the word's step on

    def test_a_models_segment_is_returned_from_its_word_on(self):
        """Eight cycles flown from the sentence's first step (8): the heading word said at 10 starts at cycle 4."""
        segment = model_segment(self.sentence(), HEADING, 10, LEAD, MODEL_WORDS)
        cycles = 8
        states = torch.zeros(1, cycles + 1, 7, dtype=torch.float64)
        states[0, :, LAT] = 35.0
        states[0, :, LON] = -78.0 + torch.arange(cycles + 1, dtype=torch.float64) * 100.0 / (111320.0 * 0.8191520)
        states[0, :, ALT], states[0, :, SPEED_STATE], states[0, :, MASS] = 900.0, 100.0, 60000.0
        run = replace(flown([float(c) for c in range(cycles)], cycles - 1), states=states,
                      commands=torch.zeros(1, cycles, 3, dtype=torch.float64))
        result = FlownSegment(segment=segment, reading=None, signals=None, model=self.sentence(), flown=run,
                              verdict=Verdict("timeout", cycles, None, {}, None, flown_rows=cycles + 1), reached_end=True,
                              fly_s=0.1, judge_s=0.01)
        context = FlightContext(signals=None, series=None, reading=reading(), geometry=geometry(), crossing_heights=(15.0,),
                                group="own dynamics", approach_ias_mps=70.0,
                                observed_track_deg=np.full(20, 90.0), observed_distance_m=400.0 * np.arange(20),
                                hae_minus_msl_m=-32.0)
        track, _ = track_payload(result, context, 2.0)
        self.assertEqual(track["tS"], [20.0, 21.0, 22.0, 23.0, 24.0])
        # from the word on, at the observed track's height: MSL plus the flight's runway's HAE − MSL offset
        self.assertEqual((track["altitudeM"], track["altitudeHaeM"]), ([900.0] * 5, [868.0] * 5))
        # the distance flown from the observed flight's at the sentence's first step (3200 m), 100 m a cycle
        np.testing.assert_allclose(track["distanceM"], [3600.0, 3700.0, 3800.0, 3900.0, 4000.0], atol=0.6)
        self.assertEqual(len(track["loadFactor"]), 4)


class ModelFlightTest(unittest.TestCase):
    """`fly_segment` for a model's word: the flight's own dynamics only; the executor set up as the free generation sets
    it up — the time clock and the generation's time limit — and judged against the model's runway, on the reading with
    an angle word's tube anchored at its step; a word said as or after the flight ended refused."""

    def context(self, group: str = "own dynamics") -> FlightContext:
        rows = np.arange(20, dtype=np.float64)
        signals = FlightSignals(dataset_id="KXXX:test", airport="KXXX", runway="09", typecode="A320",
                                entry_time_utc="2026-09-01T00:00:00Z", landing_time_utc="2026-09-01T00:05:00Z",
                                time_s=2.0 * rows, e_m=100.0 * rows, n_m=0.0 * rows, altitude_m=np.full(20, 900.0),
                                track_deg=np.full(20, 90.0), ground_speed_mps=np.full(20, 100.0),
                                vertical_rate_mps=0.0 * rows)
        return FlightContext(signals=signals, series=None, reading=SimpleNamespace(words=np.zeros((20, 6))),
                             geometry=geometry2(), crossing_heights=(15.0, 15.0), group=group, approach_ias_mps=70.0,
                             observed_track_deg=np.full(20, 90.0), observed_distance_m=200.0 * rows, hae_minus_msl_m=-32.0)

    def fly(self, column: int, row: int, end_row: int, group: str = "own dynamics", events: list | None = None,
            read_rows: int | None = None):
        """``read_rows``: the rows the judge read through the labeller's gate — None, a gate that refused the track."""
        record = model_record() if events is None else model_record(events=events)
        sentence = model_sentence(record, MODEL_WORDS, runways=2, observed_rows=20, timeout_factor=1.5)
        asked = {}

        def fly_batch_until(batch, params, words, stop_steps, superseded, *, model_limit_s=None):
            asked.update(stop_steps=stop_steps, model_limit_s=model_limit_s)
            return flown([float(c) for c in range(20)], 19), stop_steps is not None, 0.0

        def judge(run, index, geometry, runway_index, judged, signals, spec, words):
            asked.update(runway_index=runway_index, judged=judged, runway=signals.runway)
            return (Verdict("timeout", end_row, None, {}, None, flown_rows=end_row + 1, refused="not read here")
                    if read_rows is None else replace(verdict(), end_row=end_row, flown_rows=end_row + 1))

        words = SimpleNamespace(**vars(MODEL_WORDS), spec=SPEC)
        params = SimpleNamespace(timeout_factor=1.5, cycle_s=1.0)
        admitted = SimpleNamespace(signals=SimpleNamespace(n_rows=read_rows))
        with mock.patch.object(fly_module, "segment_batch", lambda *args: None), \
                mock.patch.object(fly_module, "fly_batch_until", fly_batch_until), \
                mock.patch.object(fly_module, "judge", judge), \
                mock.patch.object(fly_module, "read_flown", lambda *args: admitted):
            result = fly_module.fly_segment(self.context(group), params, words, column, row, NEVER, sentence)
        return result, asked

    def test_it_flies_on_the_flights_own_dynamics_only(self):
        with self.assertRaisesRegex(RequestRefused, "own dynamics only"):
            self.fly(HEADING, 10, 16, group="stand-in dynamics")

    def test_it_is_flown_under_the_generations_limit_and_judged_on_the_runway_the_model_points_at(self):
        result, asked = self.fly(HEADING, 10, 16)               # said at 10, stopped at 16: after the change to 27 at 15
        # 12 observed steps left from step 8, 2 s each, × 1.5; from the sentence's first step to the stop
        self.assertEqual((asked["model_limit_s"], asked["stop_steps"]), (36.0, 8))
        self.assertEqual((asked["runway_index"], asked["runway"]), (1, "27"))
        self.assertIs(result.reading, asked["judged"])          # a heading word: judged on what was flown
        _, early = self.fly(HEADING, 8, 12)                      # stopped at 12, before the change: the sentence's last too
        self.assertEqual((early["runway_index"], early["runway"]), (1, "27"))

    def test_an_angle_word_is_judged_on_its_tube_from_its_step(self):
        record = model_record()
        events = sorted([*record["events"], {"row": 11, "column": ANGLE, "value": 2}],
                        key=lambda event: (event["row"], event["column"]))
        result, asked = self.fly(ANGLE, 11, 19, events=events)
        extra = [word for word in asked["judged"].instructions if word not in result.reading.instructions]
        self.assertEqual([(w.column, w.row, w.kind) for w in extra], [(ALTITUDE, 3, "in force")])

    def test_a_word_said_as_or_after_the_flight_ended_is_refused(self):
        # the heading word at 10 is the flown sentence's step 2: cycle 4 of 1 s cycles
        with self.assertRaisesRegex(RequestRefused, "had ended .* by the time it said this word at step 10"):
            self.fly(HEADING, 10, end_row=4)
        result, _ = self.fly(HEADING, 10, end_row=5)
        self.assertEqual(result.segment.word_step, 2)

    def test_a_word_past_where_the_labellers_gate_cut_the_flown_track_is_refused(self):
        """The gate cuts a flown track at a landing passage it finds on the 2 s rows, which can come before the outcome
        read every cycle: the heading word at the flown sentence's step 2 is judged on three rows read, not on two."""
        with self.assertRaisesRegex(RequestRefused, "cuts the model's flown track at a landing passage after step 9, before "
                                                    "it said this word at step 10"):
            self.fly(HEADING, 10, end_row=16, read_rows=2)
        result, _ = self.fly(HEADING, 10, end_row=16, read_rows=3)
        self.assertEqual(result.segment.row, 10)

    def test_the_drive_is_the_time_clock_under_the_given_limit(self):
        """`fly_batch_until` for a model: its stepper handed the time clock (the spec's word clock not even asked for)
        and the executor the limit given."""
        grid = np.arange(42).reshape(7, 6)
        batch = SimpleNamespace(inputs=lambda *args, **kwargs: "the inputs", readings=[SimpleNamespace(words=grid)],
                                geometries=["the geometry"], crossing_heights=[(15.0,)], approach_ias_mps=[70.0])
        params, words = SimpleNamespace(timeout_factor=1.5, cycle_s=1.0), SimpleNamespace(spec=SimpleNamespace(step_s=2.0))
        recorded = {}

        class Stop(Exception):
            pass

        def executor(*args, **kwargs):
            recorded["executor"] = kwargs
            return SimpleNamespace()

        def stepper(executor_, sentences, clock, *args):
            recorded["clock"] = clock
            raise Stop

        clock = mock.Mock(side_effect=AssertionError("the spec's word clock asked for"))
        with mock.patch.object(fly_module, "Sentences", lambda *args, **kwargs: "the sentences"), \
                mock.patch("ts_transformer.autopilot.replay.word_clock", clock), \
                mock.patch.object(fly_module.Runways, "of", lambda *args, **kwargs: "the runways"), \
                mock.patch.object(fly_module.AirportCharts, "of", lambda *args, **kwargs: "the charts"), \
                mock.patch.object(fly_module, "Executor", executor), \
                mock.patch.object(fly_module, "fly_until", stepper), self.assertRaises(Stop):
            fly_batch_until(batch, params, words, None, NEVER, model_limit_s=36.0)
        self.assertEqual(float(recorded["executor"]["time_limit_s"][0]), 36.0)
        self.assertIsInstance(recorded["clock"], fly_module.TimeClock)


class FreeGenerationTest(unittest.TestCase):
    """A model's sentence flown on the backend's drive — `fly_until` with the time clock, the words said as a grid —
    IS the flight the free generation flew while the model said them (`prior_free_generation.speak_and_fly`, its own
    loop, a scripted speaker in the prior's place): the real executor on the executor tests' synthetic downwind, base
    and final, state for state. The export writes those words; the frontend re-flies them."""

    def test_the_backends_drive_flies_the_free_generations_flight(self):
        from ts_transformer.autopilot.executor import Executor
        from ts_transformer.autopilot.sentence import Sentences, TimeClock
        from ts_transformer.experiments import prior_free_generation as generation
        from ts_transformer.instructions.labeller.read import read_flight
        from ts_transformer.instructions.words import Words
        from ts_transformer.tests import test_autopilot as executor_tests
        from ts_transformer.tests.support import fly_legs, instruction_airport, instruction_flight, instruction_spec

        spec, geometry_ = instruction_spec(), instruction_airport()
        words, params = Words(spec), executor_tests._params()
        signals = instruction_flight(*fly_legs(executor_tests.DOWNWIND_BASE_FINAL, 270.0, 1110.0, -400.0, 0.0))
        script = read_flight(signals, geometry_, spec, words).words
        inputs, runways, charts, approach = executor_tests._physics(signals, geometry_)
        limit = len(script) * spec.step_s * params.timeout_factor

        class ScriptedSpeaker:
            """Says the labelled sentence a step at a time, as the prior would say its own; nothing after it."""

            def __init__(self, *args, **kwargs):
                self.step, self.forbidden = 0, {}

            def append(self, *args, **kwargs):
                pass

            def speak(self, active, runway_locked):
                row = script[self.step] if self.step < len(script) else np.full(6, U)
                self.step += 1
                return np.where(active[:, None], np.where(row == U, 0, row.astype(np.int64) + 1)[None, :], 0)

        with mock.patch.object(generation, "Speaker", ScriptedSpeaker):
            spoken, said, _, _ = generation.speak_and_fly(None, [], [], inputs, runways, charts, approach, [limit], words,
                                                          params, None, generator=None, temperature=1.0)
        step_rows = int(round(spec.step_s / params.cycle_s))
        grid = said[0][: generation.steps_said(spoken, 0, said.shape[1], step_rows)]
        executor = Executor(inputs, runways, charts, approach, params, words,
                            time_limit_s=torch.tensor([limit], dtype=torch.float64))
        self.assertFalse(fly_until(executor, Sentences([grid], words, device=torch.device("cpu")), TimeClock(params.cycle_s),
                                   spec.step_s, None, NEVER))
        again = executor.flown()
        self.assertTrue(torch.equal(again.done_cycle, spoken.done_cycle))
        # through the cycle the flight ended in — all the export keeps: the generation's loop flies out the rest of that
        # step, its cycles after the end, which the stepper does not
        cycles = int(spoken.done_cycle[0]) + 1
        self.assertTrue(torch.equal(again.states[:, : cycles + 1], spoken.states[:, : cycles + 1]))
        self.assertTrue(torch.equal(again.commands[:, :cycles], spoken.commands[:, :cycles]))
        self.assertTrue(torch.equal(again.sentence_s[:, :cycles], spoken.sentence_s[:, :cycles]))
        # and what the judge reads of the flight besides its states: the modes and the limits that bound, cycle by cycle
        for name in (*again.modes, *again.limits):
            ours = (again.modes if name in again.modes else again.limits)[name]
            theirs = (spoken.modes if name in spoken.modes else spoken.limits)[name]
            self.assertTrue(torch.equal(ours[:, :cycles], theirs[:, :cycles]), name)


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
