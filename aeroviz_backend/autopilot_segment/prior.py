"""``POST /autopilot/prior-segment``: a word's segment of a sentence of a Training set of stage B (prior design §12 B6),
flown live by the same single-flight executor as stage A's (`backend.AutopilotSegmentBackend`, whose request lock, page
numbering, executor spec and flown-set caches this shares).

WHICH SENTENCE. The request names a prior set (``airport``, ``setId``: `prior.training_files.listed_set`, the airport's
``training/index_prior_v2.json``), a flight of it (``flightKey``) and ``sentence``: a sample number of the flight's
``prior`` list (the prior's own sentence) or ``"closedLoop"`` (the flight's closed-loop sentence at the prior's Δ, which
is stage A's). The sentence's Δ is the set's ``model.rowIntervalS``.

THE PRIOR'S SENTENCE IS FLOWN AS THE EXPORT FLEW IT. The export flew each sentence through the start of a closed loop
(`autopilot.start`) with its words, from the flight's first predicted step; here the same flight is set up on its stored
closed-loop sentence (`training_flights`, A23's setup: the observed flight moved to the first predicted step, its time
limit) and the sentence's words grid is replaced by the prior's words, so the executor hears the prior's words at the
cycles the export told them (a Δ row's words on the cycle that starts it). The answer is checked against the export's
flown track of the sentence (`stored`: the 2 s rows both have, the export's rounding included), and the backend test
holds the live flight to the readout's unrounded states within the executor conformance's bound.

WARMED UP AT START (`warm_up`, run beside stage A's): every listed prior set opened at its Δ.
"""

from __future__ import annotations

import dataclasses
import json
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from ts_transformer.autopilot import replay
from ts_transformer.autopilot.judge import flown_track
from ts_transformer.instructions.artefact import SEALED_READINGS
from ts_transformer.instructions.words import COLUMNS
from ts_transformer.io_utils import utc_now
from ts_transformer.prior.checkpoint import validation_claim
from ts_transformer.repo_layout import REPO_ROOT
from ts_transformer.prior import training_files as prior_files

from aeroviz_backend.autopilot_segment.errors import NotListed, RequestRefused, Superseded
from aeroviz_backend.autopilot_segment.fly import FlownSegment, apart_from_stored, fly_segment, last_state
from aeroviz_backend.autopilot_segment.payload import segment_payload

#: MIRROR of `aeroviz-4d/src/data/trainingPriorAutopilot.ts` (`TRAINING_PRIOR_AUTOPILOT_SCHEMA`); the reader refuses
#: anything else by name. A name changes with the payload's shape, on both sides, in one change. v1 (B6, 2026-10-05):
#: stage A's segment answer with the sentence it flew (a sample of the prior's, or the closed-loop sentence).
SCHEMA = "aeroviz-autopilot-prior-segment-v1"
#: The request's ``sentence`` that names the flight's closed-loop sentence. MIRROR of `trainingPriorAutopilot.ts`
#: (`TRAINING_PRIOR_CLOSED_LOOP`).
CLOSED_LOOP = "closedLoop"


def apart_from_exported(result: FlownSegment, exported: dict[str, Any], batch: replay.Batch, j: int, step_s: float
                        ) -> dict[str, float | int]:
    """The live flight against the export's flown track of the prior's sentence (``exported``: the sentence's ``track``,
    the 2 s rows from the first predicted step, written to 0.1 m) on the rows both have: how many and the largest
    horizontal and vertical distance."""
    step_cycles = int(round(step_s / result.flown.cycle_s))
    rows = min(exported["rows"], last_state(result) // step_cycles + 1)
    cycles = np.arange(rows) * step_cycles
    live = flown_track(result.flown.states[0, : cycles[-1] + 1].cpu().numpy(), batch.geometries[j])
    horizontal = float(np.hypot(live["e"][cycles] - np.array(exported["eM"][:rows]),
                                live["n"][cycles] - np.array(exported["nM"][:rows])).max())
    vertical = float(np.abs(live["height"][cycles] - np.array(exported["heightMslM"][:rows])).max())
    if not (np.isfinite(horizontal) and np.isfinite(vertical)):
        raise ValueError(f"the live flight is {horizontal} m / {vertical} m from the exported track: a non-finite state")
    return {"rows": int(rows), "horizontalM": horizontal, "verticalM": vertical}


def on_words(batch: replay.Batch, j: int, sentence: Any, grid: np.ndarray, words: Any) -> tuple[replay.Batch, Any]:
    """Flight ``j`` of ``batch`` (set up on its closed-loop sentence ``sentence``) told ``grid`` instead: the batch and the
    sentence with the prior's words (no word is a correction: the closed-loop reading added none)."""
    told = replay.Sentence(grid=grid, instructions=replay.instructions_of(grid, batch.geometries[j], words),
                           first_row=batch.sentences[j].first_row)
    sentences = list(batch.sentences)
    sentences[j] = told
    return (dataclasses.replace(batch, sentences=sentences),
            dataclasses.replace(sentence, rows=dataclasses.replace(sentence.rows, grid=grid,
                                                                   correction=np.zeros(grid.shape, dtype=bool))))


class PriorSegments:
    """``fly(payload)`` for ``POST /autopilot/prior-segment``: ``{clientId, seq, airport, setId, flightKey, sentence,
    column, row}``, on ``backend``'s caches, lock and page numbering."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self._val: Any = None
        self._val_guard = threading.Lock()      # the warm-up and a request may both make it first

    def splits_of(self, sample: dict[str, Any]) -> tuple[str, ...]:
        """The splits the set's flights may be of (outline D109): the claimed validation set's are the sealed readings
        (val) alone, every other set's are the service's. A claim on a set that is not val, or a val set with no claim,
        is refused by name."""
        claim, split = sample["source"]["validationClaim"], sample["cohort"]["split"]
        if claim is not None and split not in SEALED_READINGS:
            raise RequestRefused(f"set {sample['setId']} holds a validation claim but its cohort.split is {split!r}, "
                                 f"not one of {SEALED_READINGS}")
        if claim is None and split in SEALED_READINGS:
            raise RequestRefused(f"set {sample['setId']} is of split {split!r} but holds no validationClaim")
        if claim is None:
            return tuple(self.backend.splits)
        if claim["reader"] != prior_files.CLAIM_READER:
            raise RequestRefused(f"set {sample['setId']}: validationClaim.reader is {claim['reader']!r}, expected {prior_files.CLAIM_READER!r}")
        if claim["prior"] != sample["model"]["prior"]:
            raise RequestRefused(f"set {sample['setId']}: validationClaim.prior is {claim['prior']!r}, not the set's "
                                 f"model.prior {sample['model']['prior']!r}")
        if claim["readout"] != sample["source"]["readout"]:
            raise RequestRefused(f"set {sample['setId']}: validationClaim.readout is {claim['readout']!r}, not the set's "
                                 f"source.readout {sample['source']['readout']!r}")
        return SEALED_READINGS

    def val_service(self) -> Any:
        """The service that flies a claimed validation set's flights (outline D109): its splits are the sealed readings
        alone, its caches its own. Made on first use (`AutopilotSegmentBackend.__init__` builds this object)."""
        from aeroviz_backend.autopilot_segment.backend import AutopilotSegmentBackend

        with self._val_guard:
            if self._val is None:
                self._val = AutopilotSegmentBackend(splits=SEALED_READINGS, airports_root=self.backend.airports_root)
        return self._val

    def flying(self, sample: dict[str, Any]) -> Any:
        """The service whose splits are the set's: the validation service for a claimed set, else the backend. A claimed
        set is flown only when the prior's run holds the claim it names, on disk (D85)."""
        self.splits_of(sample)
        claim = sample["source"]["validationClaim"]
        if claim is None:
            return self.backend
        held = validation_claim(REPO_ROOT / claim["prior"], claim["reader"])      # repository-relative names
        if held != claim["readout"]:
            raise RequestRefused(f"set {sample['setId']}: the prior {claim['prior']} holds no claim of {claim['readout']} "
                                 f"by {claim['reader']} (it holds {held})")
        return self.val_service()

    def listed(self, airport: str, set_id: str) -> dict[str, Any]:
        """The prior set's sample (`prior.training_files.listed_set`)."""
        from aeroviz_backend.autopilot_segment.backend import AIRPORT_CODE

        if not AIRPORT_CODE.fullmatch(airport):
            raise RequestRefused(f"airport {airport!r} is not an airport code")
        try:
            return prior_files.listed_set(self.backend.airports_root / airport / "training", airport, set_id)[1]
        except prior_files.NotListed as error:
            raise NotListed(str(error)) from None

    def warm_up(self, log: Callable[[str], None] = print) -> None:
        """Every prior set opened at its Δ ahead of its first request (as stage A's warm-up does its sets): each set under
        its own lock (stage A's `set_flown`), never under the request lock; a set it cannot open is skipped with its
        reason."""
        started, opened_sets = time.perf_counter(), 0
        for index in sorted(self.backend.airports_root.glob(f"*/training/{prior_files.INDEX_FILE}")):
            airport = index.parent.parent.name
            try:
                sets = prior_files.index_sets(json.loads(index.read_text(encoding="utf-8")), index, airport)
            except Exception as error:       # noqa: BLE001 — a prefetch: logged; a request gets it whole
                log(f"prior warm-up: {airport} skipped — {index}: {type(error).__name__}: {error}")
                continue
            for entry in sets:
                began = time.perf_counter()
                try:
                    sample = self.listed(airport, entry["id"])
                    service = self.flying(sample)
                    instructions, _, params, _, words = self.backend.executor_for(sample)
                    service.set_flown(sample, sample["cohort"]["split"], float(sample["model"]["rowIntervalS"]),
                                      instructions, params, words)
                except Exception as error:   # noqa: BLE001 — a prefetch: logged by type; a request gets it whole
                    log(f"prior warm-up: {airport} {entry['id']} skipped — {type(error).__name__}: "
                        f"{str(error).split('; ')[0]}")
                    continue
                opened_sets += 1
                log(f"prior warm-up: {airport} {entry['id']}: {len(sample['flights'])} flights opened in "
                    f"{time.perf_counter() - began:.1f} s")
        log(f"prior warm-up: {opened_sets} sets ready in {time.perf_counter() - started:.1f} s")

    def fly(self, payload: dict[str, Any]) -> dict[str, Any]:
        from aeroviz_backend.autopilot_segment.backend import _field, _require_split, _whole

        backend = self.backend
        client = str(_field(payload, "clientId"))
        seq = _whole(_field(payload, "seq"), "seq")
        airport = str(_field(payload, "airport"))
        set_id = str(_field(payload, "setId"))
        flight_key = str(_field(payload, "flightKey"))
        which = _field(payload, "sentence")
        column_name = _field(payload, "column")
        row = _whole(_field(payload, "row"), "row")
        if column_name not in COLUMNS:
            raise RequestRefused(f"column {column_name!r} is none of {COLUMNS}")
        if which != CLOSED_LOOP:
            _whole(which, "sentence")
        backend._claim(client, seq)
        superseded = lambda: backend._latest[client] != seq  # noqa: E731 — read each cycle
        asked = time.perf_counter()
        with backend._lock:
            if superseded():
                raise Superseded("a newer request from this page came in while this one waited")
            started = time.perf_counter()
            sample = self.listed(airport, set_id)
            interval = float(sample["model"]["rowIntervalS"])
            items = [item for item in sample["flights"] if item["flightKey"] == flight_key]
            if len(items) != 1:
                raise NotListed(f"Training set {set_id} at {airport} has no flight {flight_key}")
            item = items[0]
            service = self.flying(sample)
            _require_split(item["split"], service.splits)       # before the set is opened (A37)
            if which == CLOSED_LOOP:
                said = None
            else:
                chosen = [s for s in item["prior"] if s["sample"] == which]
                if len(chosen) != 1:
                    raise NotListed(f"flight {flight_key} of set {set_id} has no prior sentence {which}")
                said = chosen[0]
            instructions, executor, params, record, words = backend.executor_for(sample)
            # a word said after the flight's outcome (the executor flew on to the sentence's last word: a crossing of another
            # runway, say) is never heard by the flight the judge read — it has no segment
            ended = said["track"]["lastCycle"] if said is not None else (
                item["closedLoop"][f"{interval:g}"]["replay"]["track"]["lastCycle"])
            heard = row * int(round(interval / params.cycle_s))
            if heard > ended:
                raise RequestRefused(f"the word at Δ row {row} is said at cycle {heard}, after the flight's outcome at cycle {ended}")
            flown_set = service.set_flown(sample, item["split"], interval, instructions, params, words)
            opened = time.perf_counter()
            j = flown_set.position[item["datasetId"]]
            batch, inputs, sentence = flown_set.batch, flown_set.inputs, flown_set.sentences[j]
            if said is not None:
                batch, sentence = on_words(batch, j, sentence, np.array(said["words"], dtype=np.int16), words)
            result = fly_segment(batch, inputs, j, sentence, COLUMNS.index(column_name), row, params, words, superseded)
            answering = time.perf_counter()
            if said is None:
                apart = apart_from_stored(result, sentence, batch, j, words.spec.step_s)
            else:
                apart = apart_from_exported(result, said["track"], batch, j, words.spec.step_s)
            body = segment_payload(result, batch.geometries[j], float(item["haeMinusMslM"]),
                                   np.asarray(flown_set.aero[j]), apart)
            finished = time.perf_counter()
        return {
            "ok": True, "schema": SCHEMA, "airport": airport, "setId": set_id, "flightKey": flight_key,
            "sentence": which, "datasetId": item["datasetId"], "rowIntervalS": interval, "computedUtc": utc_now(),
            "timing": {"waitS": round(started - asked, 3), "openS": round(opened - started, 3),
                       "flyS": round(result.fly_s, 3), "cycles": int(result.flown.commands.shape[1]),
                       "judgeS": round(result.judge_s, 3), "answerS": round(finished - answering, 3),
                       "computeS": round(finished - started, 3)},
            "executor": {"spec": executor.name, "specSha256": record["sha256"], "cycleS": params.cycle_s},
            "artefact": instructions.name,
            **body,
        }
