"""The live executor of the Training view of stage A (`aeroviz_backend.autopilot_segment`; vocabulary §12.1 A23): which
segment a word is, a live segment against the export's flight (the same states, the same outcome and crossing), the
answer, the service's requests and warm-up, the HTTP mapping, the frontend fixture of an answer and the frontend's
mirrors. Synthetic flights only (`ts_transformer.tests.support.closed_loop_flight`); a published set is checked by
`check_live`."""

import json
import os
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import aeroviz_backend.autopilot_segment  # noqa: F401 — puts `ts_transformer` (under 4dTrajectory/) on the path
from ts_transformer.autopilot import replay
from ts_transformer.autopilot.conformance import STATE_BOUND_M
from ts_transformer.autopilot.judge import OUTCOMES, flown_track
from ts_transformer.experiments import training_export as export
from ts_transformer.instructions import training_files
from ts_transformer.instructions.spec import READING_RULE
from ts_transformer.instructions.words import COLUMNS, HEADING, UNCHANGED
from ts_transformer.tests.support import closed_loop_flight
from ts_transformer.tests.test_training_export import FIXTURE_SET, FIXTURES, stage_a_fixture

from aeroviz_backend.autopilot_segment.backend import AutopilotSegmentBackend, SetFlown
from aeroviz_backend.autopilot_segment.errors import NotListed, RequestRefused, Superseded
from aeroviz_backend.autopilot_segment.fly import apart_from_stored, fly_segment, segment_of
from aeroviz_backend.autopilot_segment.payload import SCHEMA, SEGMENT_END, segment_payload
from aeroviz_backend.http_server import AeroVizBackendApp

CPU = torch.device("cpu")
NEVER = lambda: False  # noqa: E731
FRONTEND = Path(__file__).resolve().parents[2] / "aeroviz-4d" / "src" / "data"
_FLIGHTS: dict[float, object] = {}


def flight(interval: float):
    """`closed_loop_flight` at ``interval``, built once for the module (each is read in closed loop)."""
    if interval not in _FLIGHTS:
        _FLIGHTS[interval] = closed_loop_flight(interval)
    return _FLIGHTS[interval]


def words_said(sentence) -> list[tuple[int, int]]:
    """Every ``(row, column)`` a closed-loop sentence says a word at."""
    return [(int(r), int(c)) for r, c in zip(*np.nonzero(sentence.grid != UNCHANGED))]


class SegmentTest(unittest.TestCase):
    def test_a_word_runs_to_the_next_word_of_its_column_a_heading_word_a_lead_later_and_the_last_to_the_outcome(self):
        one = flight(4.0)
        grid, every = one.sentence.grid, 4
        lead = int(round(one.words.spec.heading_lead_s / one.params.cycle_s))
        for row, column in words_said(one.sentence):
            segment = segment_of(one.sentence, column, row, 4.0, one.params, one.words)
            later = [r for r in range(row + 1, len(grid)) if grid[r, column] != UNCHANGED]
            expected = None if not later else later[0] * every + (lead if column == HEADING else 0)
            self.assertEqual((segment.start_cycle, segment.stop_cycle, segment.word),
                             (row * every, expected, int(grid[row, column])))
            self.assertEqual(segment.correction, bool(one.sentence.correction[row, column]))

    def test_a_row_that_says_no_word_of_the_column_is_refused(self):
        one = flight(2.0)
        silent = [(r, c) for r in range(len(one.sentence.grid)) for c in range(len(COLUMNS))
                  if one.sentence.grid[r, c] == UNCHANGED][0]
        with self.assertRaisesRegex(RequestRefused, "says no word"):
            segment_of(one.sentence, silent[1], silent[0], 2.0, one.params, one.words)
        with self.assertRaisesRegex(RequestRefused, "says no word"):
            segment_of(one.sentence, 0, len(one.sentence.grid), 2.0, one.params, one.words)


class LiveEqualsExportTest(unittest.TestCase):
    def test_every_word_flown_live_is_the_exported_flight_and_its_last_word_ends_as_the_export(self):
        """Outline §6 item 6: a live segment is the export's flown states from the same state with the same words (the
        executor conformance's bound), and a word flown to its outcome ends with the export's outcome and crossing."""
        for interval in (2.0, 4.0, 8.0):
            one = flight(interval)
            batch, (verdict,) = replay.fly_batch(one.batch, one.params, one.words, device=CPU)
            exported = export.replay_payload(batch, 0, verdict, one.batch, one.sentence, {"outcome": verdict.outcome},
                                             one.inputs.aero_params[0].numpy(), one.words.spec, one.words)
            reference = flown_track(batch.states[0].numpy(), one.geometry)
            last = {column: row for row, column in words_said(one.sentence)}
            for row, column in words_said(one.sentence):
                with self.subTest(interval=interval, row=row, column=COLUMNS[column]):
                    result = fly_segment(one.batch, one.inputs, 0, one.sentence, column, row, one.params, one.words,
                                         NEVER)
                    end = int(result.flown.done_cycle[0]) + 2
                    live = flown_track(result.flown.states[0, :end].numpy(), one.geometry)
                    for name in ("e", "n", "height"):
                        self.assertLess(float(np.abs(live[name] - reference[name][:end]).max()), STATE_BOUND_M)
                    apart = apart_from_stored(result, one.sentence, one.batch, 0, one.words.spec.step_s)
                    self.assertLess(max(apart["horizontalM"], apart["verticalM"]), STATE_BOUND_M)
                    # stopped exactly when its stop comes before the flight's end; the column's last word never
                    stop = result.segment.stop_cycle
                    self.assertEqual(result.stopped, stop is not None and stop <= int(batch.done_cycle[0]))
                    self.assertTrue(stop is None or last[column] != row)
                    if last[column] == row:
                        answer = segment_payload(result, one.geometry, -33.0, one.inputs.aero_params[0].numpy(), apart)
                        self.assertEqual(answer["segment"]["end"], exported["outcome"])
                        self.assertEqual(answer["crossing"], exported["crossing"])


class PayloadTest(unittest.TestCase):
    def answer(self, column: int, row: int):
        one = flight(2.0)
        result = fly_segment(one.batch, one.inputs, 0, one.sentence, column, row, one.params, one.words, NEVER)
        apart = apart_from_stored(result, one.sentence, one.batch, 0, one.words.spec.step_s)
        return result, segment_payload(result, one.geometry, -33.0, one.inputs.aero_params[0].numpy(), apart)

    def test_a_stopped_segment_is_drawn_from_its_word_to_its_stop_with_one_command_fewer_than_states(self):
        one = flight(2.0)
        row, column = [(r, c) for r, c in words_said(one.sentence) if c == HEADING and r > 0][0]
        result, answer = self.answer(column, row)
        track, segment = answer["track"], answer["segment"]
        self.assertEqual((segment["end"], answer["crossing"]), (SEGMENT_END, None))
        self.assertEqual(track["cycle"][0], segment["startCycle"])
        self.assertEqual(track["cycle"][-1], segment["stopCycle"] - 1 + 1)
        self.assertEqual(track["tS"], [float(c) for c in track["cycle"]])
        self.assertEqual(len(track["thrustFraction"]), len(track["cycle"]) - 1)
        self.assertEqual(len(track["attitude"]["headingDeg"]), len(track["cycle"]))
        self.assertTrue(np.allclose(np.subtract(track["altitudeHaeM"], track["altitudeMslM"]), -33.0, atol=0.02))

    def test_the_last_word_is_drawn_to_its_outcome_with_its_crossing_and_decision_altitude_check(self):
        one = flight(2.0)
        row = max(r for r, c in words_said(one.sentence) if c == HEADING)
        result, answer = self.answer(HEADING, row)
        self.assertEqual(answer["segment"]["end"], result.verdict.outcome)
        self.assertEqual(answer["segment"]["endCycle"], result.verdict.end_row)
        self.assertEqual(answer["track"]["cycle"][-1], result.verdict.end_row)
        decision = answer["crossing"]["decision"]
        self.assertEqual(decision["passed"], decision["lateralOk"] and decision["verticalOk"])


# ---- the service
def set_up_set(root: Path) -> dict:
    """The fixture set (`test_training_export.stage_a_fixture`, the export's own output) written under ``root``."""
    index, sample = stage_a_fixture()
    training = root / sample["airport"] / "training"
    training.mkdir(parents=True)
    training_files.write_set(training, sample["airport"], index["sets"][0], training_files.serialise(sample), [])
    return sample


class SyntheticBackend(AutopilotSegmentBackend):
    """The service on the fixture set, its flights the synthetic ones (a synthetic flight has no artefact)."""

    def executor_for(self, sample):
        one = flight(2.0)
        return Path("fixture/instruction_language"), Path("fixture/executor"), one.params, {"sha256": "fixture"}, one.words

    def set_flown(self, sample, split, interval_s, instructions, words, opened=None):
        one = flight(interval_s)
        return SetFlown(one.batch, [one.sentence])


class BackendTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.sample = set_up_set(self.root)
        self.backend = SyntheticBackend(airports_root=self.root)
        self.airport, self.key = self.sample["airport"], self.sample["flights"][0]["flightKey"]

    def tearDown(self):
        self.directory.cleanup()

    def request(self, **changes):
        events = self.sample["flights"][0]["closedLoop"]["2"]["events"]
        word = [e for e in events if e["column"] == HEADING and e["row"] > 0][0]
        return {"clientId": "page", "seq": 1, "airport": self.airport, "setId": FIXTURE_SET, "flightKey": self.key,
                "rowIntervalS": 2.0, "column": COLUMNS[word["column"]], "row": word["row"], **changes}

    def test_a_request_flies_its_word_and_names_what_it_flew(self):
        answer = self.backend.fly(self.request())
        self.assertEqual((answer["schema"], answer["setId"], answer["flightKey"], answer["rowIntervalS"]),
                         (SCHEMA, FIXTURE_SET, self.key, 2.0))
        self.assertEqual(answer["segment"]["end"], SEGMENT_END)
        self.assertEqual(answer["executor"]["specSha256"], "fixture")

    def test_a_request_the_view_cannot_make_is_refused_by_name(self):
        for changes, message in [({"column": "flaps"}, "none of"), ({"row": -1}, "whole number"),
                                 ({"row": True}, "whole number"), ({"seq": "1"}, "whole number"),
                                 ({"rowIntervalS": 3.0}, "none of the set's"), ({"rowIntervalS": "2"}, "a number"),
                                 ({"airport": "../x"}, "not an airport code")]:
            with self.subTest(changes=changes), self.assertRaisesRegex(RequestRefused, message):
                self.backend.fly({**self.request(clientId=str(changes)), **changes})
        request = self.request()
        del request["row"]
        with self.assertRaisesRegex(RequestRefused, "no 'row'"):
            self.backend.fly(request)

    def test_a_set_or_flight_not_listed_is_not_listed(self):
        with self.assertRaisesRegex(NotListed, "has no flight"):
            self.backend.fly(self.request(flightKey="nobody"))
        with self.assertRaisesRegex(NotListed, "lists no set"):
            self.backend.fly(self.request(setId="another", seq=2))
        with self.assertRaisesRegex(NotListed, "does not exist"):
            self.backend.fly(self.request(airport="KZZZ", seq=3))

    def test_a_page_s_later_request_supersedes_its_earlier_ones(self):
        self.backend.fly(self.request(seq=5))
        with self.assertRaisesRegex(Superseded, "request 5 came in before its request 4"):
            self.backend.fly(self.request(seq=4))
        self.backend.fly(self.request(seq=4, clientId="another page"))

    def test_the_warm_up_opens_every_set_and_says_what_it_skipped(self):
        broken = self.root / "KBBB" / "training"
        broken.mkdir(parents=True)
        (broken / training_files.INDEX_FILE).write_text('{"schema": "another", "airport": "KBBB", "sets": []}')
        lines = []
        self.backend.warm_up(log=lines.append)
        self.assertRegex(lines[0], r"^autopilot warm-up: KBBB skipped — .*is a another file")
        self.assertRegex(lines[1], rf"^autopilot warm-up: {self.airport} {FIXTURE_SET}: 1 flights opened in")
        self.assertRegex(lines[2], r"^autopilot warm-up: 1 sets ready in")


class SetUpTest(unittest.TestCase):
    """The service's own setup, not the synthetic one: the spec refused unless it is the set's, each split's flights
    drawn once for every Δ in the warm-up, and each (split, Δ) kept."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.sample = set_up_set(self.root)

    def tearDown(self):
        self.directory.cleanup()

    def test_the_spec_is_refused_unless_it_is_the_one_the_set_was_exported_with(self):
        from unittest import mock

        one = flight(2.0)
        backend = AutopilotSegmentBackend(airports_root=self.root)
        with mock.patch("ts_transformer.autopilot.replay.open_executor",
                        return_value=(one.params, {"sha256": "another"}, one.words)):
            with self.assertRaisesRegex(ValueError, "the set was exported with fixture"):
                backend.executor_for(self.sample)

    def test_the_warm_up_draws_each_split_once_for_every_row_interval_and_keeps_each(self):
        from unittest import mock

        from ts_transformer.experiments import training_flights

        drawn, built = [], []

        def open_flights(instructions, split, ids, words):
            drawn.append((split, tuple(ids)))
            return f"flights of {split}"

        def closed_loop_batch(flights, stored, interval, words):
            built.append((flights, interval))
            one = flight(interval)
            return one.batch, [one.sentence]

        backend = AutopilotSegmentBackend(airports_root=self.root)
        one = flight(2.0)
        with mock.patch.object(AutopilotSegmentBackend, "executor_for",
                               return_value=(Path("i"), Path("x"), one.params, {"sha256": "fixture"}, one.words)), \
                mock.patch.object(training_flights, "open_flights", open_flights), \
                mock.patch.object(training_flights, "stored_closed_loop", lambda *args: {}), \
                mock.patch.object(training_flights, "closed_loop_batch", closed_loop_batch):
            backend.warm_up(log=lambda line: None)
            key = self.sample["flights"][0]["datasetId"]
            self.assertEqual(drawn, [("train", (key,)), ("select", ())])
            self.assertEqual(built, [("flights of train", 2.0), ("flights of train", 4.0), ("flights of train", 8.0),
                                     ("flights of select", 2.0), ("flights of select", 4.0),
                                     ("flights of select", 8.0)])
            answer = backend.fly({"clientId": "p", "seq": 1, "airport": self.sample["airport"], "setId": FIXTURE_SET,
                                  "flightKey": self.sample["flights"][0]["flightKey"], "rowIntervalS": 4,
                                  "column": "heading", "row": 0})
            self.assertEqual((len(drawn), len(built), answer["rowIntervalS"]), (2, 6, 4.0))   # kept, not drawn again


class FakeAutopilot:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls, self.error = [], error

    def fly(self, payload):
        self.calls.append(payload)
        if self.error is not None:
            raise self.error
        return {"ok": True, "segment": {"end": SEGMENT_END}}


class EndpointTest(unittest.TestCase):
    def app(self, autopilot) -> AeroVizBackendApp:
        return AeroVizBackendApp(simulation_backend=object(), optimization_backend=object(),
                                 dynamics_comparison_backend=object(), observed_trajectory_backend=object(),
                                 autopilot_segment_backend=autopilot)

    def test_the_endpoint_delegates_the_request(self):
        autopilot = FakeAutopilot()
        status, payload, event = self.app(autopilot).handle_post("/autopilot/segment", {"row": 3})
        self.assertEqual((status, payload["segment"]["end"], event), (200, SEGMENT_END, None))
        self.assertEqual(autopilot.calls, [{"row": 3}])

    def test_a_bad_request_is_a_400_not_listed_a_404_superseded_a_409_and_anything_else_a_500_with_its_reason(self):
        post = lambda error: self.app(FakeAutopilot(error)).handle_post("/autopilot/segment", {})[:2]  # noqa: E731
        self.assertEqual(post(RequestRefused("no 'row'")), (400, {"ok": False, "error": "no 'row'"}))
        self.assertEqual(post(NotListed("no set")), (404, {"ok": False, "error": "no set"}))
        self.assertEqual(post(Superseded("a newer request")), (409, {"ok": False, "error": "a newer request"}))
        self.assertEqual(post(ValueError("re-read differs")), (500, {"ok": False, "error": "ValueError: re-read differs"}))


# ---- the frontend's fixture of an answer, and its mirrors
def autopilot_fixture() -> list[dict]:
    """Two answers on the fixture set at Δ = 2 s, written by the service: a heading word stopped at its stop, and the
    column's last heading word flown to its outcome — wall-clock fields fixed."""
    with tempfile.TemporaryDirectory() as directory:
        sample = set_up_set(Path(directory))
        backend = SyntheticBackend(airports_root=Path(directory))
        events = sample["flights"][0]["closedLoop"]["2"]["events"]
        heading = [e["row"] for e in events if e["column"] == HEADING]
        answers = []
        for seq, row in enumerate((heading[1], heading[-1]), start=1):
            answer = backend.fly({"clientId": "fixture", "seq": seq, "airport": sample["airport"], "setId": FIXTURE_SET,
                                  "flightKey": sample["flights"][0]["flightKey"], "rowIntervalS": 2.0,
                                  "column": COLUMNS[HEADING], "row": row})
            answer["computedUtc"] = "fixture"
            answer["timing"] = {name: 0 for name in answer["timing"]}
            answers.append(answer)
    return answers


class FixtureTest(unittest.TestCase):
    def test_the_frontend_fixture_of_an_answer_is_what_the_service_answers(self):
        text = json.dumps(autopilot_fixture(), separators=(",", ":"), allow_nan=False) + "\n"
        path = FIXTURES / "autopilot_segment.json"
        if os.environ.get("AEROVIZ_WRITE_FIXTURES") == "1":
            path.write_text(text, encoding="utf-8")
        self.assertEqual(path.read_text(encoding="utf-8"), text,
                         f"{path} is not what the service answers now: AEROVIZ_WRITE_FIXTURES=1 writes it again")


def ts_constant(path: Path, name: str) -> str:
    """A `export const NAME = "…"` of a frontend file."""
    match = re.search(rf'export const {name} = "([^"]*)"', path.read_text(encoding="utf-8"))
    if match is None:
        raise AssertionError(f"{path.name} has no string constant {name}")
    return match.group(1)


def ts_strings(path: Path, name: str) -> list[str]:
    """A `export const NAME = [ "…", … ] as const` of a frontend file."""
    match = re.search(rf"export const {name} = \[([^\]]*)\]", path.read_text(encoding="utf-8"))
    if match is None:
        raise AssertionError(f"{path.name} has no array constant {name}")
    return re.findall(r'"([^"]*)"', match.group(1))


class MirrorTest(unittest.TestCase):
    def test_the_frontend_reads_the_names_the_backend_and_the_export_write(self):
        autopilot, sample = FRONTEND / "trainingAutopilot.ts", FRONTEND / "trainingSample.ts"
        self.assertEqual(ts_constant(autopilot, "TRAINING_AUTOPILOT_SCHEMA"), SCHEMA)
        self.assertEqual(ts_constant(autopilot, "TRAINING_AUTOPILOT_SEGMENT_END"), SEGMENT_END)
        self.assertEqual(ts_constant(sample, "TRAINING_INDEX_SCHEMA"), training_files.INDEX_SCHEMA)
        self.assertEqual(ts_constant(sample, "TRAINING_INDEX_FILE"), training_files.INDEX_FILE)
        self.assertEqual(ts_constant(sample, "TRAINING_SAMPLE_SCHEMA"), training_files.SAMPLE_SCHEMA)
        self.assertEqual(ts_constant(sample, "TRAINING_SET_KIND"), training_files.SET_KIND)
        self.assertEqual(ts_constant(sample, "TRAINING_READING_RULE"), READING_RULE)
        self.assertEqual(ts_strings(sample, "TRAINING_OUTCOMES"), list(OUTCOMES))
        self.assertEqual(ts_strings(sample, "TRAINING_SPLITS"), list(training_files.SPLITS))
        self.assertEqual(ts_strings(sample, "TRAINING_STRATA"), list(export.STRATA))


if __name__ == "__main__":
    unittest.main()
