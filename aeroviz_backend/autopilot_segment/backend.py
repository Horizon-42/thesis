"""``POST /autopilot/segment``: which set, its artefact and executor spec, the set's flights kept set up, one segment
flown at a time — and only each page's latest request: a page (``clientId``) numbers its requests (``seq``, rising), and
a request of a higher number supersedes the page's lower ones (`Superseded`) — one still waiting is not flown, one flying
stops before its next cycle, one that arrives after it is refused at once. The page's own numbers decide, not the order
requests happen to arrive in.

Which flight: the request names a Training set of stage A (``airport``, ``setId``: `training_files.listed_set`, the
airport's ``training/index_v4.json``) and a flight of it (``flightKey``); the set's sample names the instruction artefact
and the executor spec it was exported from, and the flight's split. Which sentence: the flight's closed-loop sentence at
``rowIntervalS`` (one of the set's Δ). Which word: Δ row ``row`` of ``column`` (`fly.segment_of`). The set's executor
spec opens without a check of the code: the backend produces no result that anyone compares with another, so it runs
none of the conformance checks at its start (vocabulary D73, A43; they run where the code changes, and in `check_live`).
It checks what it shows by what it flies: each answer carries the distance of the live flight from the artefact's stored
flown states, and one farther than `STATE_BOUND_M` is refused by name (`fly.apart_from_stored`: the executor code does
not fly the set as it was exported). The splits a set's flights may be of
come from whoever builds the service (D109): stage A's own sets give train and select (`training_files.SPLITS`, outline
§6 item 4), stage B's prior set of the claimed validation readout val too (prior B10); a flight of another split is
refused before its set is opened.

WARMED UP AT START (`warm_up`, which the server runs in a thread): every stage-A set is opened ahead of the first
request — its flights drawn and read again, each Δ's closed-loop sentences set up — which a first request would
otherwise do. A set (a split of it) is opened under its own lock, never under the request lock: a request for a set waits
only for the opening of that set, and opens it itself when the warm-up has not reached it.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ts_transformer.autopilot import replay
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.experiments import training_flights
from ts_transformer.instructions import training_files
from ts_transformer.instructions.artefact import SPLITS, ClosedLoopSentence, load_signals
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.words import COLUMNS, Words
from ts_transformer.io_utils import utc_now
from ts_transformer.repo_layout import COMPARISON_AIRPORTS_ROOT, REPO_ROOT

from aeroviz_backend.autopilot_segment.errors import NotListed, RequestRefused, Superseded
from aeroviz_backend.autopilot_segment.fly import apart_from_stored, fly_segment
from aeroviz_backend.autopilot_segment.payload import SCHEMA, segment_payload

#: An airport as the frontend's directories name it; the request's airport is a path segment.
AIRPORT_CODE = re.compile(r"[A-Z0-9]{3,4}")


def _field(record: dict[str, Any], name: str) -> Any:
    if name not in record:
        raise RequestRefused(f"the request has no {name!r}")
    return record[name]


def _require_split(split: str, splits: Sequence[str]) -> None:
    """A set's flights are of ``splits``, the service's caller's (D109): another split is refused by name."""
    if split not in splits:
        raise RequestRefused(f"a set's flights are of {tuple(splits)}, not {split!r}")


def _whole(value: Any, what: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RequestRefused(f"{what} must be a whole number, got {value!r}")
    return value


class SetFlown:
    """A set's flights of one split at one Δ, set up as the export set them up: the batch, each flight's stored
    closed-loop sentence, and where each dataset id sits in them."""

    def __init__(self, batch: replay.Batch, sentences: list[ClosedLoopSentence], params: ExecutorParams) -> None:
        self.batch, self.sentences = batch, sentences
        self.position = {flight.dataset_id: j for j, flight in enumerate(batch.signals)}
        self.inputs = batch.inputs(params.start_rule, training_flights.CPU)
        self.aero = self.inputs.aero_params.cpu().numpy()


@dataclass
class Opened:
    """What one warm-up keeps between the sets it opens: each split's signals, read one time for all the sets of an
    artefact (the sets of the five airports share one). It goes when the warm-up's loop ends, with the signals (~0.5 GB
    for train)."""

    signals: dict[tuple[Path, str], list[FlightSignals]] = field(default_factory=dict)

    def signals_of(self, instructions: Path, split: str) -> list[FlightSignals]:
        if (instructions, split) not in self.signals:
            self.signals[(instructions, split)] = load_signals(instructions, split)
        return self.signals[(instructions, split)]


def stage_a_service(airports_root: Path = COMPARISON_AIRPORTS_ROOT) -> AutopilotSegmentBackend:
    """The service of stage A's own sets (the server's, `check_live`'s): their flights are train's and select's
    (`training_files.SPLITS`, D109) — the one place that choice is written."""
    return AutopilotSegmentBackend(splits=training_files.SPLITS, airports_root=airports_root)


class AutopilotSegmentBackend:
    """``fly(payload)`` for ``POST /autopilot/segment``: ``{clientId, seq, airport, setId, flightKey, rowIntervalS,
    column, row}``. ``splits``: the splits the sets' flights may be of (D109; stage A's sets: `training_files.SPLITS`),
    of the artefact's (`instructions.artefact.SPLITS`)."""

    def __init__(self, *, splits: Sequence[str], airports_root: Path = COMPARISON_AIRPORTS_ROOT) -> None:
        if not splits or not set(splits) <= set(SPLITS):
            raise ValueError(f"a set's flights are of some of the artefact's splits {SPLITS}, not {tuple(splits)}")
        self.splits = tuple(splits)
        self.airports_root = Path(airports_root)
        self._lock = threading.Lock()
        # each page's highest request number (`clientId` → `seq`): a lower one waiting or flying is superseded by it
        self._latest: dict[str, int] = {}
        self._claims = threading.Lock()
        self._executors: dict[tuple[Path, Path], tuple[ExecutorParams, dict[str, Any], Words]] = {}
        self._flown: dict[tuple[Path, str, tuple[str, ...], float], SetFlown] = {}
        #: a set's flights of a split, drawn and read again one time for all its Δ (`set_flown`)
        self._drawn: dict[tuple[Path, str, tuple[str, ...]], training_flights.SetFlights] = {}
        #: one lock for each set and split being opened (`set_flown`): the request lock is not held by the warm-up
        self._opening: dict[tuple[Path, str, tuple[str, ...]], threading.Lock] = {}
        self._opening_guard = threading.Lock()

    def training_set(self, airport: str, set_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """The set's index entry and sample (`training_files.listed_set`)."""
        if not AIRPORT_CODE.fullmatch(airport):
            raise RequestRefused(f"airport {airport!r} is not an airport code")
        try:
            return training_files.listed_set(self.airports_root / airport / "training", airport, set_id)
        except training_files.NotListed as error:
            raise NotListed(str(error)) from None

    def executor_for(self, sample: dict[str, Any]) -> tuple[Path, Path, ExecutorParams, dict[str, Any], Words]:
        """The set's instruction artefact and executor spec (absolute), and the spec opened for them (`replay.open_spec`:
        refused unless it was measured against this artefact's vocabulary). No check of the code runs here (D73, A43):
        what the backend shows is checked in each answer (`fly.apart_from_stored`)."""
        instructions = REPO_ROOT / sample["source"]["instructions"]
        executor = REPO_ROOT / sample["source"]["executor"]
        key = (instructions, executor)
        if key not in self._executors:
            self._executors[key] = replay.open_spec(executor, instructions)
        params, record, words = self._executors[key]
        if record["sha256"] != sample["source"]["executorSpecSha256"]:
            raise ValueError(f"{executor.name} is spec {record['sha256'][:12]}, the set was exported with "
                             f"{sample['source']['executorSpecSha256'][:12]}")
        return instructions, executor, params, record, words

    def set_flown(self, sample: dict[str, Any], split: str, interval_s: float, instructions: Path, params: ExecutorParams,
                  words: Words, opened: Opened | None = None) -> SetFlown:
        """The set's flights of ``split`` on their closed-loop sentences at ``interval_s``, set up the first time they
        are asked for (``opened``: what the warm-up keeps between sets, `Opened`); refused for a split not in the
        service's ``splits``, its caller's (D109)."""
        _require_split(split, self.splits)
        ids = tuple(item["datasetId"] for item in sample["flights"] if item["split"] == split)
        key = (instructions, split, ids, interval_s)
        with self._opening_guard:
            opening = self._opening.setdefault(key[:3], threading.Lock())
        with opening:                       # one thread opens a set; another that needs it waits, then finds it open
            if key not in self._flown:
                drawn = self._drawn.get(key[:3])
                if drawn is None:
                    drawn = self._drawn[key[:3]] = training_flights.open_flights(
                        instructions, split, ids, words,
                        signals=None if opened is None else opened.signals_of(instructions, split))
                stored = training_flights.stored_closed_loop(instructions, split, interval_s, words)
                self._flown[key] = SetFlown(*training_flights.closed_loop_batch(instructions, split, drawn, stored,
                                                                                 interval_s, params, words), params)
        return self._flown[key]

    def warm_up(self, log: Callable[[str], None] = print) -> None:
        """Every stage-A set opened ahead of its first request: each airport's ``index_v4.json``, each set, each split and
        Δ (`set_flown`), each set under its own lock and never under the request lock, so a request waits at most for the
        opening of the set it needs. A set it cannot open is skipped with its reason (a request for it gets all of it)."""
        started = time.perf_counter()
        opened_sets = self._open_sets(log)
        log(f"autopilot warm-up: {opened_sets} sets ready in {time.perf_counter() - started:.1f} s")

    def _open_sets(self, log: Callable[[str], None]) -> int:
        """The loop of `warm_up`: the sets opened. What it keeps between sets (`Opened`) goes when it returns."""
        opened_sets, opened = 0, Opened()
        for index in sorted(self.airports_root.glob(f"*/training/{training_files.INDEX_FILE}")):
            airport = index.parent.parent.name
            try:
                sets = training_files.index_sets(json.loads(index.read_text(encoding="utf-8")), index, airport)
            except Exception as error:       # noqa: BLE001 — a prefetch: logged; a request gets it whole
                log(f"autopilot warm-up: {airport} skipped — {index}: {type(error).__name__}: {error}")
                continue
            for entry in sets:
                began = time.perf_counter()
                try:
                    _, sample = self.training_set(airport, entry["id"])
                    instructions, _, params, _, words = self.executor_for(sample)
                    held = {item["split"] for item in sample["flights"]}
                    for split in (split for split in self.splits if split in held):    # none it holds no flight of
                        for interval in sample["vocabulary"]["rowIntervalsS"]:
                            self.set_flown(sample, split, float(interval), instructions, params, words, opened)
                except Exception as error:   # noqa: BLE001 — a prefetch: logged by type; a request gets it whole
                    log(f"autopilot warm-up: {airport} {entry['id']} skipped — {type(error).__name__}: "
                        f"{str(error).split('; ')[0]}")
                    continue
                opened_sets += 1
                log(f"autopilot warm-up: {airport} {entry['id']}: {len(sample['flights'])} flights opened in "
                    f"{time.perf_counter() - began:.1f} s")
        return opened_sets

    def _claim(self, client: str, seq: int) -> None:
        """Make ``seq`` the page's latest request — unless the page already sent a later one."""
        with self._claims:
            if client in self._latest and seq <= self._latest[client]:
                raise Superseded(f"this page's request {self._latest[client]} came in before its request {seq}")
            self._latest[client] = seq

    def fly(self, payload: dict[str, Any]) -> dict[str, Any]:
        client = str(_field(payload, "clientId"))
        seq = _whole(_field(payload, "seq"), "seq")
        airport = str(_field(payload, "airport"))
        set_id = str(_field(payload, "setId"))
        flight_key = str(_field(payload, "flightKey"))
        interval = _field(payload, "rowIntervalS")
        column_name = _field(payload, "column")
        row = _whole(_field(payload, "row"), "row")
        if column_name not in COLUMNS:
            raise RequestRefused(f"column {column_name!r} is none of {COLUMNS}")
        if isinstance(interval, bool) or not isinstance(interval, (int, float)):
            raise RequestRefused(f"rowIntervalS must be a number, got {interval!r}")
        self._claim(client, seq)
        superseded = lambda: self._latest[client] != seq  # noqa: E731 — read each cycle
        asked = time.perf_counter()
        with self._lock:
            if superseded():
                raise Superseded("a newer request from this page came in while this one waited")
            started = time.perf_counter()
            _, sample = self.training_set(airport, set_id)
            if float(interval) not in [float(v) for v in sample["vocabulary"]["rowIntervalsS"]]:
                raise RequestRefused(f"rowIntervalS {interval} is none of the set's {sample['vocabulary']['rowIntervalsS']}")
            items = [item for item in sample["flights"] if item["flightKey"] == flight_key]
            if len(items) != 1:
                raise NotListed(f"Training set {set_id} at {airport} has no flight {flight_key}")
            item = items[0]
            _require_split(item["split"], self.splits)      # before the set is opened (A37)
            instructions, executor, params, record, words = self.executor_for(sample)
            flown_set = self.set_flown(sample, item["split"], float(interval), instructions, params, words)
            opened = time.perf_counter()
            j = flown_set.position[item["datasetId"]]
            sentence = flown_set.sentences[j]
            result = fly_segment(flown_set.batch, flown_set.inputs, j, sentence, COLUMNS.index(column_name), row, params,
                                 words, superseded)
            answering = time.perf_counter()
            apart = apart_from_stored(result, sentence, flown_set.batch, j, words.spec.step_s)
            body = segment_payload(result, flown_set.batch.geometries[j], float(item["haeMinusMslM"]),
                                   np.asarray(flown_set.aero[j]), apart)
            finished = time.perf_counter()
        return {
            "ok": True, "schema": SCHEMA, "airport": airport, "setId": set_id, "flightKey": flight_key,
            "datasetId": item["datasetId"], "rowIntervalS": float(interval), "computedUtc": utc_now(),
            # wall-clock seconds on the backend: waiting for the request before it (one at a time); then, adding up to
            # ``computeS``: the set found and opened (or kept), the executor's cycles (``cycles``), the judge, the answer
            "timing": {"waitS": round(started - asked, 3), "openS": round(opened - started, 3),
                       "flyS": round(result.fly_s, 3), "cycles": int(result.flown.commands.shape[1]),
                       "judgeS": round(result.judge_s, 3), "answerS": round(finished - answering, 3),
                       "computeS": round(finished - started, 3)},
            "executor": {"spec": executor.name, "specSha256": record["sha256"], "cycleS": params.cycle_s},
            "artefact": instructions.name,
            **body,
        }
