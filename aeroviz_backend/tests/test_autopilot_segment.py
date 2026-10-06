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
from ts_transformer.instructions import artefact, training_files
from ts_transformer.instructions.spec import READING_RULE
from ts_transformer.instructions.words import COLUMNS, HEADING, RUNWAY_GO_AROUND, UNCHANGED
from ts_transformer.tests.support import closed_loop_flight
from ts_transformer.tests.test_training_export import FIXTURE_SET, FIXTURES, stage_a_fixture

from aeroviz_backend.autopilot_segment.backend import AutopilotSegmentBackend, SetFlown
from aeroviz_backend.autopilot_segment.errors import ExecutorDiffers, NotListed, RequestRefused, Superseded
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
    return [(int(r), int(c)) for r, c in zip(*np.nonzero(sentence.rows.grid != UNCHANGED))]


class SegmentTest(unittest.TestCase):
    def test_a_word_runs_to_the_next_word_of_its_column_a_heading_word_a_lead_later_and_the_last_to_the_outcome(self):
        one = flight(4.0)
        grid, every = one.sentence.rows.grid, 4
        lead = int(round(one.words.spec.heading_lead_s / one.params.cycle_s))
        for row, column in words_said(one.sentence):
            segment = segment_of(one.sentence, column, row, 4.0, one.params, one.words)
            later = [r for r in range(row + 1, len(grid)) if grid[r, column] != UNCHANGED]
            expected = None if not later else later[0] * every + (lead if column == HEADING else 0)
            self.assertEqual((segment.start_cycle, segment.stop_cycle, segment.word),
                             (row * every, expected, int(grid[row, column])))
            self.assertEqual(segment.correction, bool(one.sentence.rows.correction[row, column]))

    def test_a_row_that_says_no_word_of_the_column_is_refused(self):
        one = flight(2.0)
        silent = [(r, c) for r in range(len(one.sentence.rows.grid)) for c in range(len(COLUMNS))
                  if one.sentence.rows.grid[r, c] == UNCHANGED][0]
        with self.assertRaisesRegex(RequestRefused, "says no word"):
            segment_of(one.sentence, silent[1], silent[0], 2.0, one.params, one.words)
        with self.assertRaisesRegex(RequestRefused, "says no word"):
            segment_of(one.sentence, 0, len(one.sentence.rows.grid), 2.0, one.params, one.words)


class LiveEqualsExportTest(unittest.TestCase):
    def test_every_word_flown_live_is_the_exported_flight_and_its_last_word_ends_as_the_export(self):
        """Outline §6 item 6: a live segment is the export's flown states from the same state with the same words (the
        executor conformance's bound), and a word flown to its outcome ends with the export's outcome and crossing."""
        for interval in (2.0, 4.0, 8.0):
            one = flight(interval)
            batch, (verdict,) = replay.fly_batch(one.batch, one.params, one.words, device=CPU)
            exported = export.replay_payload(batch, 0, verdict, one.batch, one.sentence,
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
def set_up_set(root: Path, split: str | None = None) -> dict:
    """The fixture set (`test_training_export.stage_a_fixture`, the export's own output) written under ``root``;
    ``split``: its flights' split, changed to it."""
    index, sample = stage_a_fixture()
    if split is not None:
        sample["flights"] = [{**item, "split": split} for item in sample["flights"]]
    training = root / sample["airport"] / "training"
    training.mkdir(parents=True)
    training_files.write_set(training, sample["airport"], index["sets"][0], training_files.serialise(sample), [])
    return sample


def set_up_sets(root: Path, airports: tuple[str, ...]) -> dict[str, dict]:
    """The fixture set written under ``root`` for each of ``airports``, its flights named by the airport (so that the
    sets differ, as the real ones do): each airport's sample."""
    index, sample = stage_a_fixture()
    samples = {}
    for airport in airports:
        flights = [{**item, "datasetId": f"{airport}:{item['datasetId']}"} for item in sample["flights"]]
        samples[airport] = {**sample, "airport": airport, "flights": flights}
        training = root / airport / "training"
        training.mkdir(parents=True)
        training_files.write_set(training, airport, {**index["sets"][0]}, training_files.serialise(samples[airport]), [])
    return samples


def signals_loaded(loads: list):
    """The warm-up's read of a split's signals (`backend.load_signals`) replaced by one that notes what it was asked for."""
    from unittest import mock

    return mock.patch("aeroviz_backend.autopilot_segment.backend.load_signals",
                      lambda instructions, split: loads.append((instructions, split)) or [])


class SyntheticBackend(AutopilotSegmentBackend):
    """The service on the fixture set, its flights the synthetic ones (a synthetic flight has no artefact)."""

    def executor_for(self, sample):
        one = flight(2.0)
        return Path("fixture/instruction_language"), Path("fixture/executor"), one.params, {"sha256": "fixture"}, one.words

    def set_flown(self, sample, split, interval_s, instructions, params, words, opened=None):
        one = flight(interval_s)
        return SetFlown(one.batch, [one.sentence], params)


class BackendTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.sample = set_up_set(self.root)
        self.backend = SyntheticBackend(splits=training_files.SPLITS, airports_root=self.root)
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
        backend = AutopilotSegmentBackend(splits=training_files.SPLITS, airports_root=self.root)
        with mock.patch.object(replay, "open_spec", return_value=(one.params, {"sha256": "another"}, one.words)):
            with self.assertRaisesRegex(ValueError, "the set was exported with fixture"):
                backend.executor_for(self.sample)

    def test_the_warm_up_draws_each_split_once_for_every_row_interval_and_keeps_each(self):
        from unittest import mock

        from ts_transformer.experiments import training_flights

        drawn, built, loads = [], [], []

        def open_flights(instructions, split, ids, words, signals=None):
            drawn.append((split, tuple(ids)))
            return f"flights of {split}"

        def closed_loop_batch(instructions, split, flights, stored, interval, params, words):
            built.append((flights, interval))
            one = flight(interval)
            return one.batch, [one.sentence]

        backend = AutopilotSegmentBackend(splits=training_files.SPLITS, airports_root=self.root)
        one = flight(2.0)
        with mock.patch.object(AutopilotSegmentBackend, "executor_for",
                               return_value=(Path("i"), Path("x"), one.params, {"sha256": "fixture"}, one.words)), \
                mock.patch.object(training_flights, "open_flights", open_flights), \
                mock.patch.object(training_flights, "stored_closed_loop", lambda *args: {}), \
                mock.patch.object(training_flights, "closed_loop_batch", closed_loop_batch), signals_loaded(loads):
            backend.warm_up(log=lambda line: None)
            key = self.sample["flights"][0]["datasetId"]
            self.assertEqual(drawn, [("train", (key,))])                 # none of a split it holds no flight of
            self.assertEqual(loads, [(Path("i"), "train")])              # its signals read one time for every Δ
            self.assertEqual(built, [("flights of train", 2.0), ("flights of train", 4.0), ("flights of train", 8.0)])
            answer = backend.fly({"clientId": "p", "seq": 1, "airport": self.sample["airport"], "setId": FIXTURE_SET,
                                  "flightKey": self.sample["flights"][0]["flightKey"], "rowIntervalS": 4,
                                  "column": "heading", "row": 0})
            self.assertEqual((len(drawn), len(built), answer["rowIntervalS"]), (1, 3, 4.0))   # kept, not drawn again


    def test_the_warm_up_reads_a_splits_signals_one_time_for_all_the_sets_of_an_artefact(self):
        """A43: the sets of the airports of one artefact share its signals; the warm-up reads each split of each artefact
        once, not once for each set; a request outside a warm-up reads what it needs by itself."""
        from unittest import mock

        from ts_transformer.experiments import training_flights

        shutil_rmtree = __import__("shutil").rmtree
        shutil_rmtree(self.root / self.sample["airport"])                  # the sets of this test: the two below
        set_up_sets(self.root, ("KAAA", "KBBB", "KCCC"))
        loads, taken = [], []
        backend = AutopilotSegmentBackend(splits=training_files.SPLITS, airports_root=self.root)
        one = flight(2.0)

        def executor_for(sample):          # KAAA and KBBB fly one artefact, KCCC another
            artefact_of = Path("j") if sample["airport"] == "KCCC" else Path("i")
            return artefact_of, Path("x"), one.params, {"sha256": "fixture"}, one.words

        with mock.patch.object(AutopilotSegmentBackend, "executor_for", lambda self, sample: executor_for(sample)), \
                mock.patch.object(training_flights, "open_flights",
                                  lambda instructions, split, ids, words, signals=None: taken.append(signals) or "flights"), \
                mock.patch.object(training_flights, "stored_closed_loop", lambda *args: {}), \
                mock.patch.object(training_flights, "closed_loop_batch",
                                  lambda *args: (flight(args[4]).batch, [flight(args[4]).sentence])), \
                signals_loaded(loads):
            backend.warm_up(log=lambda line: None)
            self.assertEqual(len(taken), 3)                                  # one draw for each set, not for each Δ
            self.assertEqual(sorted(loads), [(Path("i"), "train"), (Path("j"), "train")])
            self.assertTrue(all(signals == [] for signals in taken))
            taken.clear()
            sample = backend.training_set("KAAA", FIXTURE_SET)[1]
            backend.set_flown(sample, "train", 4.0, Path("i2"), one.params, one.words)
            self.assertEqual(taken, [None])                                  # no warm-up: `open_flights` reads its own


class RefusalTest(unittest.TestCase):
    def test_the_backend_runs_no_conformance_check_and_a_val_flight_is_refused(self):
        """A43 (D73): the spec opens without a check of the code — `executor_for` calls none of the labeller's, the
        executor's and the closed loop's checks, and opens the spec one time for a (artefact, executor spec); a set's
        flight of a split other than train and select is refused (outline §6 item 4)."""
        from unittest import mock

        one = flight(2.0)
        with tempfile.TemporaryDirectory() as name:
            sample = set_up_set(Path(name))
            backend = AutopilotSegmentBackend(splits=training_files.SPLITS, airports_root=Path(name))
            ran = AssertionError("a conformance check ran")
            spec = {"sha256": "fixture", "vocabulary_spec_sha256": one.words.spec.sha256}
            loaded = mock.Mock(return_value=(one.params, spec))
            with mock.patch("ts_transformer.autopilot.closed_loop.require_conforming_closed_loop", side_effect=ran), \
                    mock.patch("ts_transformer.autopilot.conformance.require_conforming_executor", side_effect=ran), \
                    mock.patch("ts_transformer.autopilot.replay.require_conforming_labeller", side_effect=ran), \
                    mock.patch.object(replay, "open_executor", side_effect=ran), \
                    mock.patch.object(replay, "load_spec", loaded), \
                    mock.patch.object(replay.artefact, "load_spec", return_value=one.words.spec):    # `open_spec` itself
                for _ in range(2):
                    params, record, words = backend.executor_for(sample)[2:]
                    self.assertEqual((params, record["sha256"]), (one.params, "fixture"))
                    self.assertIs(words.spec, one.words.spec)
            self.assertEqual(loaded.call_count, 1)
            with self.assertRaisesRegex(RequestRefused, "not 'val'"):
                backend.set_flown(sample, "val", 2.0, Path("i"), one.params, one.words)

    def test_a_flight_of_another_split_is_refused_before_any_check_runs(self):
        """A37: a request for a set's flight of a split other than train and select is refused by name before its
        artefact and executor spec are opened (the val days' data is opened for nothing)."""
        from unittest import mock

        with tempfile.TemporaryDirectory() as name:
            sample = set_up_set(Path(name), split="val")
            backend = AutopilotSegmentBackend(splits=training_files.SPLITS, airports_root=Path(name))
            checks = mock.Mock(side_effect=AssertionError("the spec was opened"))
            request = {"clientId": "page", "seq": 1, "airport": sample["airport"], "setId": FIXTURE_SET,
                       "flightKey": sample["flights"][0]["flightKey"], "rowIntervalS": 2.0, "column": "heading",
                       "row": 0}
            with mock.patch.object(replay, "open_spec", checks):
                with self.assertRaisesRegex(RequestRefused, "not 'val'"):
                    backend.fly(request)
            self.assertEqual(checks.call_count, 0)


class CallersSplitsTest(unittest.TestCase):
    """D109, A39: the splits a set's flights may be of come from whoever builds the service — stage A's sets give train
    and select and refuse a val flight (above); a caller that permits val opens a val flight and flies it live."""

    def test_the_splits_are_some_of_the_artefacts(self):
        for splits in ((), ("train", "test"), ("validation",)):
            with self.subTest(splits=splits), self.assertRaisesRegex(ValueError, "some of the artefact's splits"):
                AutopilotSegmentBackend(splits=splits, airports_root=Path("nowhere"))
        self.assertEqual(AutopilotSegmentBackend(splits=training_files.SPLITS, airports_root=Path("x")).splits,
                         ("train", "select"))

    def test_stage_a_s_own_services_give_train_and_select(self):
        """The server's service and `check_live`'s are stage A's (`stage_a_service`): a widened one would open val."""
        from unittest import mock

        from aeroviz_backend.autopilot_segment import check_live

        self.assertEqual(AeroVizBackendApp().autopilot_segment_backend().splits, training_files.SPLITS)
        seen = []

        def training_set(backend, airport, set_id):
            seen.append(backend.splits)
            raise NotListed("stop here")

        with tempfile.TemporaryDirectory() as name, mock.patch.object(AutopilotSegmentBackend, "training_set",
                                                                      training_set):
            with self.assertRaises(NotListed):
                check_live.main(["--set-id", "x", "--out", str(Path(name) / "out"), "--airport", "KXXX",
                                 "--root", name])
        self.assertEqual(seen, [training_files.SPLITS])

    def test_a_service_that_permits_val_warms_up_only_the_splits_a_set_holds(self):
        from unittest import mock

        from ts_transformer.experiments import training_flights

        one = flight(2.0)
        for split, expected in ((None, ["train"]), ("val", ["val"])):
            with self.subTest(split=split), tempfile.TemporaryDirectory() as name:
                set_up_set(Path(name), split=split)
                drawn = []
                backend = AutopilotSegmentBackend(splits=("train", "select", "val"), airports_root=Path(name))
                with signals_loaded([]), mock.patch.object(AutopilotSegmentBackend, "executor_for",
                                       return_value=(Path("i"), Path("x"), one.params, {"sha256": "f"}, one.words)), \
                        mock.patch.object(training_flights, "open_flights",
                                          lambda instructions, split, ids, words, signals=None: drawn.append(split) or split), \
                        mock.patch.object(training_flights, "stored_closed_loop", lambda *args: {}), \
                        mock.patch.object(training_flights, "closed_loop_batch",
                                          lambda *args: (flight(args[4]).batch, [flight(args[4]).sentence])):
                    backend.warm_up(log=lambda line: None)
                self.assertEqual(drawn, expected)

    def test_a_caller_that_permits_val_opens_a_val_flight_and_flies_it_as_the_set_flew_it(self):
        from unittest import mock

        from ts_transformer.experiments import training_flights

        with tempfile.TemporaryDirectory() as name:
            sample = set_up_set(Path(name), split="val")
            key = sample["flights"][0]["flightKey"]
            request = {"clientId": "page", "seq": 1, "airport": sample["airport"], "setId": FIXTURE_SET,
                       "flightKey": key, "rowIntervalS": 2.0, "column": "heading", "row": 0}
            # the service's own setup of a val flight: drawn from the artefact's val split and kept
            drawn = []

            def open_flights(instructions, split, ids, words, signals=None):
                drawn.append((split, tuple(ids)))
                return f"flights of {split}"

            def closed_loop_batch(instructions, split, flights, stored, interval, params, words):
                one = flight(interval)
                return one.batch, [one.sentence]

            one = flight(2.0)
            backend = AutopilotSegmentBackend(splits=("train", "select", "val"), airports_root=Path(name))
            with mock.patch.object(AutopilotSegmentBackend, "executor_for",
                                   return_value=(Path("i"), Path("x"), one.params, {"sha256": "fixture"}, one.words)), \
                    mock.patch.object(training_flights, "open_flights", open_flights), \
                    mock.patch.object(training_flights, "stored_closed_loop", lambda *args: {}), \
                    mock.patch.object(training_flights, "closed_loop_batch", closed_loop_batch):
                answer = backend.fly(request)
                backend.fly({**request, "seq": 2})
            self.assertEqual(drawn, [("val", (sample["flights"][0]["datasetId"],))])
            self.assertEqual((answer["flightKey"], answer["rowIntervalS"]), (key, 2.0))
            self.assertLess(max(answer["stored"]["horizontalM"], answer["stored"]["verticalM"]), STATE_BOUND_M)
            # the same set behind stage A's service: refused by name before any check runs
            stage_a = AutopilotSegmentBackend(splits=training_files.SPLITS, airports_root=Path(name))
            with self.assertRaisesRegex(RequestRefused, "not 'val'"):
                stage_a.fly(request)


class ExecutorDifferenceTest(unittest.TestCase):
    """A43 (D73): the backend checks what it shows by what it flies — an answer farther from the artefact's stored flown
    states than the executor conformance's bound is refused by name, in the one function both the service and stage B's
    hook call (`apart_from_stored`)."""

    def request(self, backend, sample):
        """The request of the last heading word: flown to the outcome, so the whole flight is compared."""
        events = sample["flights"][0]["closedLoop"]["2"]["events"]
        word = [e for e in events if e["column"] == HEADING][-1]
        return {"clientId": "page", "seq": 1, "airport": sample["airport"], "setId": FIXTURE_SET,
                "flightKey": sample["flights"][0]["flightKey"], "rowIntervalS": 2.0,
                "column": COLUMNS[word["column"]], "row": word["row"]}

    def served(self, one, params):
        """The synthetic service, its flights flown by ``params`` (the executor that the set was not exported with, when
        they differ from the fixture's)."""
        class Changed(SyntheticBackend):
            def executor_for(self, sample):
                return Path("fixture/instruction_language"), Path("fixture/executor"), params, {"sha256": "fixture"}, one.words

            def set_flown(self, sample, split, interval_s, instructions, parameters, words, opened=None):
                return SetFlown(one.batch, [one.sentence], params)

        return Changed

    def test_an_executor_whose_law_changed_is_refused_by_name_on_its_first_request(self):
        from dataclasses import replace

        one = flight(2.0)
        with tempfile.TemporaryDirectory() as name:
            sample = set_up_set(Path(name))
            same = self.served(one, one.params)(splits=training_files.SPLITS, airports_root=Path(name))
            answer = same.fly(self.request(same, sample))
            self.assertLessEqual(max(answer["stored"]["horizontalM"], answer["stored"]["verticalM"]), STATE_BOUND_M)
            for change in ({"bank_rate_deg_s": one.params.bank_rate_deg_s / 2.0},
                           {"path_time_constant_s": one.params.path_time_constant_s * 2.0}):
                with self.subTest(change=change):
                    drifted = self.served(one, replace(one.params, **change))(splits=training_files.SPLITS,
                                                                              airports_root=Path(name))
                    with self.assertRaisesRegex(ExecutorDiffers, rf"live executor flies .* m horizontally and .* m "
                                                                 rf"vertically from the artefact's stored states, past "
                                                                 rf"{STATE_BOUND_M:g} m: .*export the set again"):
                        drifted.fly(self.request(drifted, sample))

    def test_the_bound_is_the_conformance_bound_each_way(self):
        from dataclasses import replace

        one = flight(2.0)
        row, column = [(r, c) for r, c in words_said(one.sentence) if c == HEADING and r > 0][0]
        result = fly_segment(one.batch, one.inputs, 0, one.sentence, column, row, one.params, one.words, NEVER)
        step = one.words.spec.step_s
        def shifted(axis: int, metres: float):
            states = one.sentence.rows.states.copy()          # east (0) or MSL height (2), every row
            states[:, axis] += metres
            return replace(one.sentence, rows=replace(one.sentence.rows, states=states))

        for axis in (0, 2):
            with self.subTest(axis=axis):
                apart_from_stored(result, shifted(axis, 0.5 * STATE_BOUND_M), one.batch, 0, step)   # within: answered
                with self.assertRaises(ExecutorDiffers):
                    apart_from_stored(result, shifted(axis, 2.0 * STATE_BOUND_M), one.batch, 0, step)


    def test_a_flight_flown_to_its_outcome_that_ends_otherwise_than_the_stored_sentence_says_is_refused(self):
        from dataclasses import replace

        one = flight(2.0)
        row = max(r for r, c in words_said(one.sentence) if c == HEADING)
        result = fly_segment(one.batch, one.inputs, 0, one.sentence, HEADING, row, one.params, one.words, NEVER)
        self.assertIsNotNone(result.verdict)
        step = one.words.spec.step_s
        apart_from_stored(result, one.sentence, one.batch, 0, step)                      # as stored: answered
        other = next(name for name in OUTCOMES if name != one.sentence.withheld.outcome)
        changed = replace(one.sentence, withheld=replace(one.sentence.withheld, outcome=other))
        with self.assertRaisesRegex(ExecutorDiffers, f"ends {result.verdict.outcome}, the artefact's stored outcome is "
                                                     f"{other}"):
            apart_from_stored(result, changed, one.batch, 0, step)
        stopped = fly_segment(one.batch, one.inputs, 0, one.sentence, HEADING,
                              min(r for r, c in words_said(one.sentence) if c == HEADING and r > 0), one.params,
                              one.words, NEVER)
        self.assertIsNone(stopped.verdict)
        apart_from_stored(stopped, changed, one.batch, 0, step)         # a stopped segment has no outcome to compare


class OpeningTest(unittest.TestCase):
    """A43: a set is opened under its own lock and never under the request lock, so a request waits only for the opening
    of the set that it needs — and finds it open, not opened twice."""

    def test_the_warm_up_holds_no_request_lock_and_a_request_waits_only_for_its_own_set(self):
        import threading
        from unittest import mock

        from ts_transformer.experiments import training_flights

        one = flight(2.0)
        blocked, release, opened_of = threading.Event(), threading.Event(), []

        def open_flights(instructions, split, ids, words, signals=None):
            opened_of.append(ids[0].split(":")[0])
            if ids[0].startswith("KAAA"):                        # the warm-up reaches KAAA first and is held there
                self.assertFalse(backend._lock.locked())         # ... holding no request lock
                blocked.set()
                self.assertTrue(release.wait(30.0))
            return "flights"

        lines = []
        with tempfile.TemporaryDirectory() as name:
            samples = set_up_sets(Path(name), ("KAAA", "KBBB"))
            backend = AutopilotSegmentBackend(splits=training_files.SPLITS, airports_root=Path(name))
            executor = (Path("i"), Path("x"), one.params, {"sha256": "fixture"}, one.words)
            with mock.patch.object(AutopilotSegmentBackend, "executor_for", return_value=executor), \
                    mock.patch.object(training_flights, "open_flights", open_flights), \
                    mock.patch.object(training_flights, "stored_closed_loop", lambda *args: {}), \
                    mock.patch.object(training_flights, "closed_loop_batch",
                                      lambda *args: (flight(args[4]).batch, [flight(args[4]).sentence])), \
                    signals_loaded([]):
                warming = threading.Thread(target=backend.warm_up, kwargs={"log": lines.append})
                warming.start()
                same, kept = [], []
                waiting = threading.Thread(target=lambda: same.append(
                    backend.set_flown(samples["KAAA"], "train", 4.0, Path("i"), one.params, one.words)))
                try:
                    self.assertTrue(blocked.wait(30.0))              # the warm-up is opening KAAA
                    other = backend.set_flown(samples["KBBB"], "train", 2.0, Path("i"), one.params, one.words)
                    self.assertIs(other.batch, flight(2.0).batch)    # KBBB opened meanwhile, no waiting
                    self.assertTrue(warming.is_alive() and not backend._lock.locked())
                    waiting.start()                                  # a request for KAAA waits for its opening
                    waiting.join(0.3)
                    self.assertTrue(waiting.is_alive())
                finally:
                    release.set()
                    warming.join(30.0)
                    waiting.join(30.0)
                self.assertFalse(warming.is_alive() or waiting.is_alive())
                self.assertEqual(sorted(opened_of), ["KAAA", "KBBB"])  # each set drawn once, whoever asked
                self.assertEqual(len(same), 1)
                self.assertFalse([line for line in lines if "skipped" in line], lines)


class CheckLiveTest(unittest.TestCase):
    """`check_live.compare`: a live answer against the sample's closed-loop sentence of the same flight and Δ."""

    @staticmethod
    def case(offset_m: float = 0.0, end: str = "landed"):
        closed = {"flownFromRow": 2, "states": {"rows": 6, "eM": [0.0] * 6, "nM": [0.0] * 6, "heightMslM": [500.0] * 6},
                  "replay": {"outcome": "landed", "endCycle": 7, "crossing": {"heightM": 15.0}}}
        answer = {"executor": {"cycleS": 1.0}, "track": {"cycle": [0, 1, 2, 3, 4], "eM": [offset_m] * 5, "nM": [0.0] * 5,
                                                          "altitudeMslM": [500.0] * 5},
                  "segment": {"end": end, "endCycle": 7}, "crossing": {"heightM": 15.0}}
        return answer, closed

    def test_a_live_answer_differs_by_name_past_the_rounding_or_the_outcome(self):
        from aeroviz_backend.autopilot_segment.check_live import ROUNDING_M, compare

        self.assertEqual(compare(*self.case(offset_m=0.09), 2.0, True), [])
        self.assertRegex(compare(*self.case(offset_m=ROUNDING_M + 0.01), 2.0, False)[0], r"^cycle 0: .* row 2$")
        self.assertRegex(compare(*self.case(end="ground_contact"), 2.0, True)[0], "ended ground_contact at cycle 7")
        self.assertEqual(compare(*self.case(end="ground_contact"), 2.0, False), [])     # not its column's last word


class CheckLiveRunTest(unittest.TestCase):
    """A43: `check_live` is a runner — it runs the three checks for each set's artefact and executor spec before its work
    — and a refusal of the service is a differing segment in its record, not the end of the run."""

    def test_it_runs_the_checks_first_and_records_the_service_s_refusal_as_a_difference(self):
        from unittest import mock

        from ts_transformer.repo_layout import REPO_ROOT

        from aeroviz_backend.autopilot_segment import check_live

        class Refusing(SyntheticBackend):
            def fly(self, payload):
                raise ExecutorDiffers("flight: the live executor flies 3 m horizontally and 0 m vertically from the "
                                      "artefact's stored states")

        with tempfile.TemporaryDirectory() as name:
            sample = set_up_set(Path(name))
            gate = mock.Mock()
            with mock.patch.object(check_live, "stage_a_service",
                                   lambda root: Refusing(splits=training_files.SPLITS, airports_root=root)), \
                    mock.patch("ts_transformer.autopilot.closed_loop.require_conforming_closed_loop", gate):
                status = check_live.main(["--set-id", FIXTURE_SET, "--out", str(Path(name) / "out"),
                                          "--airport", sample["airport"], "--root", name])
            self.assertEqual(status, 1)
            gate.assert_called_once_with(REPO_ROOT / sample["source"]["instructions"],
                                         REPO_ROOT / sample["source"]["executor"])
            record = json.loads((Path(name, "out", "check.json")).read_text(encoding="utf-8"))
            self.assertGreater(record["segments"], 0)
            self.assertEqual(record["differing"], record["segments"])
            self.assertIn("3 m horizontally", record["segmentsDiffering"][0]["differ"][0])


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
        backend = SyntheticBackend(splits=training_files.SPLITS, airports_root=Path(directory))
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


def ts_number(path: Path, name: str) -> float:
    """A `export const NAME = <number>` of a frontend file."""
    match = re.search(rf"export const {name} = (-?[0-9.]+);", path.read_text(encoding="utf-8"))
    if match is None:
        raise AssertionError(f"{path.name} has no number constant {name}")
    return float(match.group(1))


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
        self.assertEqual(ts_strings(sample, "TRAINING_SET_SPLITS"), list(artefact.SPLITS))
        self.assertEqual(ts_strings(sample, "TRAINING_STRATA"), list(export.STRATA))
        self.assertEqual(ts_number(sample, "TRAINING_UNCHANGED"), UNCHANGED)
        self.assertEqual(ts_number(sample, "TRAINING_RUNWAY_GO_AROUND"), RUNWAY_GO_AROUND)


if __name__ == "__main__":
    unittest.main()
