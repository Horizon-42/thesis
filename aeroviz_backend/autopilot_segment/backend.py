"""``POST /autopilot/segment``: which set, its artefact and executor spec, the set's flights kept set up, one segment
flown at a time — and only each page's latest request: a page (``clientId``) numbers its requests (``seq``, rising), and
a request of a higher number supersedes the page's lower ones (`Superseded`) — one still waiting is not flown, one flying
stops before its next cycle, one that arrives after it is refused at once. The page's own numbers decide, not the order
requests happen to arrive in.

Which flight: the request names a Training set of stage A (``airport``, ``setId``: `training_files.listed_set`, the
airport's ``training/index_v4.json``) and a flight of it (``flightKey``); the set's sample names the instruction artefact
and the executor spec it was exported from, and the flight's split. Which sentence: the flight's closed-loop sentence at
``rowIntervalS`` (one of the set's Δ). Which word: Δ row ``row`` of ``column`` (`fly.segment_of`). The executor spec
opens only for executor code that flies its reference tracks within the bounds (`replay.open_executor`), the
single-flight executor included.

WARMED UP AT START (`warm_up`, which the server runs in a thread): every stage-A set is opened ahead of the first
request — its flights drawn and read again, each Δ's closed-loop sentences set up — which a first request would
otherwise do.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ts_transformer.autopilot import replay
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.experiments import training_flights
from ts_transformer.instructions import training_files
from ts_transformer.instructions.artefact import ClosedLoopSentence
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


def _whole(value: Any, what: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RequestRefused(f"{what} must be a whole number, got {value!r}")
    return value


class SetFlown:
    """A set's flights of one split at one Δ, set up as the export set them up: the batch, each flight's stored
    closed-loop sentence, and where each dataset id sits in them."""

    def __init__(self, batch: replay.Batch, sentences: list[ClosedLoopSentence]) -> None:
        self.batch, self.sentences = batch, sentences
        self.position = {flight.dataset_id: j for j, flight in enumerate(batch.signals)}
        self.inputs = batch.inputs(training_flights.CPU)
        self.aero = self.inputs.aero_params.cpu().numpy()


class AutopilotSegmentBackend:
    """``fly(payload)`` for ``POST /autopilot/segment``: ``{clientId, seq, airport, setId, flightKey, rowIntervalS,
    column, row}``."""

    def __init__(self, *, airports_root: Path = COMPARISON_AIRPORTS_ROOT) -> None:
        self.airports_root = Path(airports_root)
        self._lock = threading.Lock()
        # each page's highest request number (`clientId` → `seq`): a lower one waiting or flying is superseded by it
        self._latest: dict[str, int] = {}
        self._claims = threading.Lock()
        self._executors: dict[tuple[Path, Path], tuple[ExecutorParams, dict[str, Any], Words]] = {}
        self._flown: dict[tuple[Path, str, tuple[str, ...], float], SetFlown] = {}

    def training_set(self, airport: str, set_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """The set's index entry and sample (`training_files.listed_set`)."""
        if not AIRPORT_CODE.fullmatch(airport):
            raise RequestRefused(f"airport {airport!r} is not an airport code")
        try:
            return training_files.listed_set(self.airports_root / airport / "training", airport, set_id)
        except training_files.NotListed as error:
            raise NotListed(str(error)) from None

    def executor_for(self, sample: dict[str, Any]) -> tuple[Path, Path, ExecutorParams, dict[str, Any], Words]:
        """The set's instruction artefact and executor spec (absolute), and the spec opened for them on this code."""
        instructions = REPO_ROOT / sample["source"]["instructions"]
        executor = REPO_ROOT / sample["source"]["executor"]
        key = (instructions, executor)
        if key not in self._executors:
            self._executors[key] = replay.open_executor(executor, instructions)
        params, record, words = self._executors[key]
        if record["sha256"] != sample["source"]["executorSpecSha256"]:
            raise ValueError(f"{executor.name} is spec {record['sha256'][:12]}, the set was exported with "
                             f"{sample['source']['executorSpecSha256'][:12]}")
        return instructions, executor, params, record, words

    def set_flown(self, sample: dict[str, Any], split: str, interval_s: float, instructions: Path, words: Words,
                  opened: dict[tuple[Path, str, tuple[str, ...]], training_flights.SetFlights] | None = None) -> SetFlown:
        """The set's flights of ``split`` on their closed-loop sentences at ``interval_s``, set up the first time they
        are asked for (``opened``: the flights drawn for another Δ of the same warm-up, reused)."""
        ids = tuple(item["datasetId"] for item in sample["flights"] if item["split"] == split)
        key = (instructions, split, ids, interval_s)
        if key not in self._flown:
            drawn = None if opened is None else opened.get(key[:3])
            if drawn is None:
                drawn = training_flights.open_flights(instructions, split, ids, words)
                if opened is not None:
                    opened[key[:3]] = drawn
            stored = training_flights.stored_closed_loop(instructions, split, interval_s, words)
            self._flown[key] = SetFlown(*training_flights.closed_loop_batch(drawn, stored, interval_s, words))
        return self._flown[key]

    def warm_up(self, log: Callable[[str], None] = print) -> None:
        """Every stage-A set opened ahead of its first request: each airport's ``index_v4.json``, each set, each split and
        Δ (`set_flown`), under the request lock, so a request waits at most for the set being opened. A set it cannot
        open is skipped with its reason (a request for it gets all of it)."""
        started, opened_sets = time.perf_counter(), 0
        for index in sorted(self.airports_root.glob(f"*/training/{training_files.INDEX_FILE}")):
            airport = index.parent.parent.name
            try:
                sets = training_files.index_sets(json.loads(index.read_text(encoding="utf-8")), index, airport)
            except Exception as error:       # noqa: BLE001 — a prefetch: logged; a request gets it whole
                log(f"autopilot warm-up: {airport} skipped — {index}: {type(error).__name__}: {error}")
                continue
            for entry in sets:
                began = time.perf_counter()
                with self._lock:
                    try:
                        _, sample = self.training_set(airport, entry["id"])
                        instructions, _, _, _, words = self.executor_for(sample)
                        opened: dict = {}
                        for split in training_files.SPLITS:
                            for interval in sample["vocabulary"]["rowIntervalsS"]:
                                self.set_flown(sample, split, float(interval), instructions, words, opened)
                    except Exception as error:   # noqa: BLE001 — a prefetch: logged by type; a request gets it whole
                        log(f"autopilot warm-up: {airport} {entry['id']} skipped — {type(error).__name__}: "
                            f"{str(error).split('; ')[0]}")
                        continue
                opened_sets += 1
                log(f"autopilot warm-up: {airport} {entry['id']}: {len(sample['flights'])} flights opened in "
                    f"{time.perf_counter() - began:.1f} s")
        log(f"autopilot warm-up: {opened_sets} sets ready in {time.perf_counter() - started:.1f} s")

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
            instructions, executor, params, record, words = self.executor_for(sample)
            flown_set = self.set_flown(sample, item["split"], float(interval), instructions, words)
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
