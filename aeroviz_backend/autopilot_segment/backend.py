"""``POST /autopilot/segment``: which set, which spec, the rebuilt flights kept, one flight flown at a time.

Which flight: the request names a Training set (``airport``, ``setId``) and a flight of it (``flightKey``); the set's
sample (under the frontend's airports root) names the artefact it was exported from and the split it was drawn from.
Its spec is the ONE executor spec this code accepts for the artefact (`replay.open_executor`), looked up again whenever
a spec is added, moved or rewritten.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from ts_transformer.autopilot import replay
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.experiments.instruction_training_export import KIND_READBACK, SAMPLE_SCHEMA, SPLIT
from ts_transformer.instructions.spec import READING_RULE
from ts_transformer.instructions.words import COLUMNS, Words
from ts_transformer.io_utils import utc_now
from ts_transformer.repo_layout import COMPARISON_AIRPORTS_ROOT, OPT_OUTPUTS_ROOT, REPO_ROOT

from aeroviz_backend.autopilot_segment.errors import NotListed, RequestRefused
from aeroviz_backend.autopilot_segment.fly import FlightContext, fly_segment, open_flight
from aeroviz_backend.autopilot_segment.payload import SCHEMA, segment_payload

#: Where executor specs are written (`run_ts.py executor_spec --dir`): each is a directory holding ``spec.json``.
DEFAULT_EXECUTOR_ROOT = OPT_OUTPUTS_ROOT / "POOLED" / "executor"
#: Flights kept rebuilt between requests (a rebuild opens the arrival manifest and the flight's track, ~1 s): the
#: segments of one flight are usually flown one after another.
FLIGHT_CACHE_SIZE = 8
#: An airport as the frontend's directories name it; the request's airport is a path segment.
AIRPORT_CODE = re.compile(r"[A-Z0-9]{3,4}")


def _field(record: dict[str, Any], name: str) -> Any:
    if name not in record:
        raise RequestRefused(f"the request has no {name!r}")
    return record[name]


class AutopilotSegmentBackend:
    """``fly(payload)`` for ``POST /autopilot/segment``: ``{airport, setId, flightKey, column, row}``."""

    def __init__(self, *, airports_root: Path = COMPARISON_AIRPORTS_ROOT,
                 executor_root: Path = DEFAULT_EXECUTOR_ROOT) -> None:
        self.airports_root = Path(airports_root)
        self.executor_root = Path(executor_root)
        self._lock = threading.Lock()
        # an artefact's spec, with the spec files it was chosen among (their paths and times of writing)
        self._executors: dict[Path, tuple[tuple[tuple[Path, int], ...], tuple[Path, ExecutorParams, dict[str, Any], Words]]] = {}
        self._flights: OrderedDict[tuple[Path, str], FlightContext] = OrderedDict()
        self._files: dict[Path, tuple[int, dict[str, Any]]] = {}

    def _json(self, path: Path) -> dict[str, Any]:
        """A Training file, parsed once per version on disk (a sample is several MB)."""
        version = path.stat().st_mtime_ns
        if path not in self._files or self._files[path][0] != version:
            self._files[path] = (version, json.loads(path.read_text(encoding="utf-8")))
        return self._files[path][1]

    def training_set(self, airport: str, set_id: str) -> tuple[Path, str, dict[str, Any]]:
        """The set's artefact (absolute), the split its flights were drawn from, and its sample — refused by name
        unless it is a read-back set of this vocabulary, drawn from the split the Training export draws from."""
        if not AIRPORT_CODE.fullmatch(airport):
            raise RequestRefused(f"airport {airport!r} is not an airport code")
        training = self.airports_root / airport / "training"
        if not (training / "index.json").is_file():
            raise NotListed(f"{airport} has no Training export")
        entries = [entry for entry in self._json(training / "index.json")["sets"] if entry["id"] == set_id]
        if len(entries) != 1:
            raise NotListed(f"{airport} lists no Training set {set_id!r}")
        (entry,) = entries
        if entry["kind"] != KIND_READBACK:
            raise ValueError(f"Training set {set_id} is a {entry['kind']} set: only a {KIND_READBACK} set's flights are "
                             "labelled flights of an instruction artefact")
        sample = self._json(training / entry["file"])
        rule = sample["vocabulary"]["readingRule"]
        if sample["schema"] != SAMPLE_SCHEMA or rule != READING_RULE:
            raise ValueError(f"Training set {set_id} is a {sample['schema']} sample read under {rule}: this backend flies "
                             f"{SAMPLE_SCHEMA} samples of {READING_RULE}")
        split = sample["cohort"]["split"]
        if split != SPLIT:
            raise ValueError(f"Training set {set_id} was drawn from {split}, not the {SPLIT} split the export draws from")
        return REPO_ROOT / sample["producedBy"]["artefact"], split, sample

    def executor_for(self, artefact: Path) -> tuple[Path, ExecutorParams, dict[str, Any], Words]:
        """The one executor spec written by this code for ``artefact``'s vocabulary (`replay.open_executor`); refused,
        naming every spec and why, when there is none or more than one. Chosen again when a spec is added, moved or
        rewritten."""
        listing = tuple((path.parent, path.stat().st_mtime_ns) for path in sorted(self.executor_root.glob("*/spec.json")))
        if artefact not in self._executors or self._executors[artefact][0] != listing:
            usable, refused = [], []
            for directory, _ in listing:
                try:
                    usable.append((directory, *replay.open_executor(directory, artefact)))
                except ValueError as error:
                    refused.append(f"{directory.name}: {error}")
            if len(usable) != 1:
                named = [directory.name for directory, *_ in usable]
                raise ValueError(f"{len(usable)} executor specs under {self.executor_root} fly {artefact.name} with this "
                                 f"code ({named}); one is needed. Refused: {refused}")
            self._executors[artefact] = (listing, usable[0])
        return self._executors[artefact][1]

    def flight(self, artefact: Path, split: str, dataset_id: str, words: Words) -> tuple[FlightContext, bool]:
        """The flight rebuilt, and whether it was kept from an earlier request."""
        key = (artefact, dataset_id)
        if key in self._flights:
            self._flights.move_to_end(key)
            return self._flights[key], True
        context = open_flight(artefact, split, dataset_id, words)
        self._flights[key] = context
        while len(self._flights) > FLIGHT_CACHE_SIZE:
            self._flights.popitem(last=False)
        return context, False

    def fly(self, payload: dict[str, Any]) -> dict[str, Any]:
        airport = str(_field(payload, "airport"))
        set_id = str(_field(payload, "setId"))
        flight_key = str(_field(payload, "flightKey"))
        column_name = _field(payload, "column")
        row = _field(payload, "row")
        if column_name not in COLUMNS:
            raise RequestRefused(f"column {column_name!r} is none of {COLUMNS}")
        if not isinstance(row, int) or isinstance(row, bool):
            raise RequestRefused(f"row must be an integer step, got {row!r}")
        asked = time.perf_counter()
        with self._lock:
            started = time.perf_counter()
            artefact, split, sample = self.training_set(airport, set_id)
            flights = [item for item in sample["flights"] if item["flightKey"] == flight_key]
            if len(flights) != 1:
                raise NotListed(f"Training set {set_id} at {airport} has no flight {flight_key}")
            dataset_id = flights[0]["datasetId"]
            directory, params, record, words = self.executor_for(artefact)
            opening = time.perf_counter()
            context, kept = self.flight(artefact, split, dataset_id, words)
            opened = time.perf_counter()
            result = fly_segment(context, params, words, COLUMNS.index(column_name), row)
            answering = time.perf_counter()
            body = segment_payload(result, context, words)
            finished = time.perf_counter()
        return {
            "ok": True, "schema": SCHEMA, "airport": airport, "setId": set_id, "flightKey": flight_key,
            "datasetId": dataset_id, "computedUtc": utc_now(),
            # wall-clock seconds on the backend: waiting for the flight before it (one at a time); then, adding up to
            # ``computeS``: the set and the executor spec found, the flight rebuilt (or kept), the segment set up (its
            # sentence, clock and the executor's physics), the executor's cycles (``cycles`` of them: what was
            # computed, which may run past the judged outcome), the judge, and the answer written
            "timing": {"waitS": round(started - asked, 3), "setupS": round(opening - started, 3),
                       "openS": round(opened - opening, 3), "flightKept": kept,
                       "prepareS": round(answering - opened - result.fly_s - result.judge_s, 3),
                       "flyS": round(result.fly_s, 3), "cycles": int(result.flown.commands.shape[1]),
                       "judgeS": round(result.judge_s, 3), "answerS": round(finished - answering, 3),
                       "computeS": round(finished - started, 3)},
            # the spec and the artefact by name: their shas say which (the spec's is its record's)
            "executor": {"spec": directory.name, "specSha256": record["sha256"],
                         "sourceSha256": record["source"]["executor_source_sha256"],
                         "wordClock": params.word_clock, "cycleS": params.cycle_s,
                         "timeoutFactor": params.timeout_factor},
            "artefact": artefact.name,
            "vocabularySpecSha256": words.spec.sha256,
            "group": context.group,
            **body,
        }
