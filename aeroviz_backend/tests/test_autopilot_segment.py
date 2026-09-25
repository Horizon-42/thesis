"""The live executor segment (`aeroviz_backend.autopilot_segment`): which segment a selected word is, what the executor
is told, where the flight is cut, and which of the judge's results is the selected word's. Synthetic readings and
verdicts only — nothing here opens an artefact, a spec or the frontend's data."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch

from aeroviz_backend.autopilot_segment import (
    SEGMENT_END,
    AutopilotSegmentBackend,
    FlightContext,
    FlownSegment,
    cut,
    end_cycle,
    segment_of,
    segment_reading,
    segment_signals,
    selected_heading,
    told_words,
    track_payload,
    word_verdict,
)
from aeroviz_backend.http_server import AeroVizBackendApp
from ts_transformer.autopilot.executor import LIMITS, MODES, Flown
from ts_transformer.autopilot.frame import ALT, GAMMA, LAT, LON, MASS, PSI, SPEED as SPEED_STATE
from ts_transformer.autopilot.judge import Verdict
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.labeller.read import Reading
from ts_transformer.instructions.labeller.records import Instruction
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.words import ALTITUDE, ANGLE, APPROACH, HEADING, RUNWAY, SPEED, UNCHANGED

U = UNCHANGED
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
        Instruction(HEADING, 12, 3, "track", {"target_deg": 60.0}),
        Instruction(ALTITUDE, 5, 4, "target"),
        Instruction(APPROACH, 1, 5, "clear"),
        Instruction(HEADING, 14, 6, "track", {"target_deg": 70.0}),
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
                         [(RUNWAY, 0, 1, "initial"), (APPROACH, 0, 0, "initial"), (HEADING, 0, 12, "track"),
                          (ALTITUDE, 0, 7, "initial"), (ANGLE, 0, 0, "initial"), (SPEED, 0, 4, "initial"),
                          (ALTITUDE, 1, 5, "target"), (APPROACH, 2, 1, "clear"), (HEADING, 3, 14, "track"),
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
        with self.assertRaisesRegex(ValueError, "no heading word is said at step 4"):
            segment_of(reading(), HEADING, 4, LEAD)
        with self.assertRaisesRegex(ValueError, "not a step"):
            segment_of(reading(), HEADING, 10, LEAD)
        with self.assertRaisesRegex(ValueError, "not one of the six"):
            segment_of(reading(), 6, 0, LEAD)
        last = reading()
        last.words[9, SPEED] = 9
        last.instructions.append(Instruction(SPEED, 9, 9, "unspecified"))
        with self.assertRaisesRegex(ValueError, "the sentence's last, has no step after it to fly"):
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


class EndTest(unittest.TestCase):
    def test_the_segment_ends_on_the_first_cycle_that_starts_a_step_the_clock_puts_at_its_end(self):
        # the time clock: step 3 starts at cycle 6
        self.assertEqual(end_cycle(flown([float(c) for c in range(10)], 9), 3, 2.0), 6)
        # a clock that runs ahead: a step is heard only on a cycle that starts one (cycles 0, 2, 4), as the judge reads it
        self.assertEqual(end_cycle(flown([0.0, 0.5, 1.2, 3.9, 4.2, 6.0], 5), 2, 2.0), 4)

    def test_a_flight_that_ends_before_its_segment_does_not_reach_it(self):
        self.assertIsNone(end_cycle(flown([float(c) for c in range(10)], 5), 3, 2.0))

    def test_a_cut_flight_ends_at_the_cut(self):
        short = cut(flown([float(c) for c in range(10)], 9), 6)
        self.assertEqual(tuple(short.states.shape), (1, 7, 7))
        self.assertEqual(tuple(short.commands.shape), (1, 6, 3))
        self.assertEqual(tuple(short.sentence_s.shape), (1, 6))
        self.assertTrue(all(value.shape == (1, 6) for value in {**short.limits, **short.modes}.values()))
        self.assertEqual(int(short.done_cycle[0]), 5)


SPEC = SimpleNamespace(heading_lead_s=4.0, heading_tolerance_deg=4.5)
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
                              segment, SPEC, WORDS)
        outside = word_verdict(verdict(heading=[{"row": 0, "rows": 4, "inside": 3}]), segment, SPEC, WORDS)
        unjudged = word_verdict(verdict(heading=[{"row": 0, "rows": 0, "inside": 0}]), segment, SPEC, WORDS)
        left = word_verdict(verdict(heading=[{"row": 0, "rows": 4, "inside": 4}], intercepting_off_word_cycles=3),
                            segment, SPEC, WORDS)
        self.assertEqual([inside["status"], outside["status"], unjudged["status"], left["status"]],
                         ["inside", "outside", "not judged", "outside"])
        self.assertEqual((outside["checks"][0]["inside"], outside["checks"][0]["rows"]), (3, 4))

    def test_the_clearance_is_its_capture_turn_and_corridor(self):
        segment = segment_of(reading(), APPROACH, 5, LEAD)
        judged = word_verdict(verdict(capture_turn={"progress_ok": True, "rate_ok": True},
                                      corridor={"cleared": True, "entered": True, "rows": 5, "inside": 4}),
                              segment, SPEC, WORDS)
        self.assertEqual(judged["status"], "outside")
        self.assertEqual([check["ok"] for check in judged["checks"]], [True, True, True, False])
        self.assertEqual(word_verdict(verdict(), segment_of(reading(), APPROACH, 0, LEAD), SPEC, WORDS)["status"], "no check")

    def test_an_altitude_word_is_its_tube_and_an_angle_word_every_tube_it_anchors(self):
        tubes = [{"row": 0, "rows": 3, "inside": 3, "contained": True, "target_m": 210.0},
                 {"row": 2, "rows": 5, "inside": 4, "contained": False, "target_m": None}]
        altitude = word_verdict(verdict(vertical=tubes[:1]), segment_of(reading(), ALTITUDE, 4, LEAD), SPEC, WORDS)
        angle = word_verdict(verdict(vertical=tubes), segment_of(reading(), ANGLE, 7, LEAD), SPEC, WORDS)
        self.assertEqual(altitude["status"], "inside")
        self.assertEqual(angle["status"], "outside")
        self.assertEqual([check["ok"] for check in angle["checks"]], [True, False])
        self.assertEqual([check["name"] for check in angle["checks"]],
                         ["in the tube of the altitude word 210 m", "in the tube of the altitude word descend to land"])

    def test_a_speed_word_is_its_transition_and_band_and_the_pilots_own_speed_has_none(self):
        span = {"row": 0, "transition_ok": True, "cut_before_arrival": False, "band_rows": 4, "band_inside": 4,
                "contained": True}
        self.assertEqual(word_verdict(verdict(speed=[span]), segment_of(reading(), SPEED, 8, LEAD), SPEC, WORDS)["status"],
                         "inside")
        own = reading()
        own.words[8, SPEED] = 9
        own.instructions[-1] = Instruction(SPEED, 9, 8, "unspecified")
        self.assertEqual(word_verdict(verdict(), segment_of(own, SPEED, 8, LEAD), SPEC, WORDS)["status"], "no check")

    def test_a_flown_segment_the_gate_refuses_judges_nothing(self):
        refused = Verdict("timeout", 12, None, {}, None, flown_rows=13, refused="too short")
        self.assertEqual(word_verdict(refused, segment_of(reading(), HEADING, 3, LEAD), SPEC, WORDS)["status"], "not judged")

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
        run = replace_flown(base, states=states, commands=commands)
        segment = segment_of(reading(), HEADING, 3, LEAD)
        result = FlownSegment(segment=segment, reading=segment_reading(reading(), segment), signals=None, flown=run,
                              verdict=Verdict(outcome, cycles, None, {}, None, flown_rows=cycles + 1), reached_end=True)
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


def replace_flown(base: Flown, **changes) -> Flown:
    from dataclasses import replace
    return replace(base, **changes)


class BackendTest(unittest.TestCase):
    def write_set(self, root: Path, kind: str = "vocabulary-readback") -> None:
        training = root / "KXXX" / "training"
        (training / "a_set").mkdir(parents=True)
        (training / "index.json").write_text(json.dumps({"sets": [{"id": "a_set", "kind": kind, "file": "a_set/sample.json"}]}))
        (training / "a_set" / "sample.json").write_text(json.dumps({
            "producedBy": {"artefact": "4dTrajectory/outputs/POOLED/instruction_language/an_artefact"},
            "cohort": {"split": "val"}, "flights": [{"flightKey": "F_23R_abc_T", "datasetId": "KXXX:F_23R_abc_T"}]}))

    def test_a_set_names_its_artefact_and_split(self):
        with TemporaryDirectory() as tmp:
            self.write_set(Path(tmp))
            backend = AutopilotSegmentBackend(airports_root=Path(tmp), executor_root=Path(tmp) / "none")
            artefact, split, sample = backend.training_set("KXXX", "a_set")
            self.assertEqual((artefact.parts[-2:], split), (("instruction_language", "an_artefact"), "val"))
            with self.assertRaises(FileNotFoundError):
                backend.training_set("KXXX", "another_set")

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

    def test_a_request_names_a_column_and_an_integer_step(self):
        backend = AutopilotSegmentBackend(airports_root=Path("/nonexistent"), executor_root=Path("/nonexistent"))
        request = {"airport": "KXXX", "setId": "a_set", "flightKey": "F", "column": "heading", "row": 3}
        with self.assertRaisesRegex(ValueError, "none of"):
            backend.fly({**request, "column": "track"})
        with self.assertRaisesRegex(ValueError, "integer step"):
            backend.fly({**request, "row": 3.0})
        with self.assertRaisesRegex(ValueError, "no 'row'"):
            backend.fly({key: value for key, value in request.items() if key != "row"})


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

    def test_a_set_or_flight_that_is_not_there_is_a_404_and_a_refusal_is_raised(self):
        status, payload, _ = self.app(FakeAutopilot(FileNotFoundError("no set"))).handle_post("/autopilot/segment", {})
        self.assertEqual((status, payload), (404, {"ok": False, "error": "no set"}))
        with self.assertRaisesRegex(ValueError, "refused"):
            self.app(FakeAutopilot(ValueError("refused"))).handle_post("/autopilot/segment", {})


if __name__ == "__main__":
    unittest.main()
